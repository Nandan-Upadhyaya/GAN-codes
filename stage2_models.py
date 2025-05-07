import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ResidualBlock(nn.Module):
    """Enhanced residual block with spectral normalization for stability"""
    def __init__(self, channel_num):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.BatchNorm2d(channel_num),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.BatchNorm2d(channel_num)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

# Self-Attention module for better spatial coherence
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = spectral_norm(nn.Conv2d(in_dim, in_dim//8, kernel_size=1))
        self.key_conv = spectral_norm(nn.Conv2d(in_dim, in_dim//8, kernel_size=1))
        self.value_conv = spectral_norm(nn.Conv2d(in_dim, in_dim, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))  # learnable parameter
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)  # batch matrix-matrix product
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out

class ConditioningAugmentation(nn.Module):
    """
    Conditioning Augmentation as described in StackGAN paper.
    Used in both Stage-I and Stage-II GANs.
    """
    def __init__(self, input_dim, output_dim):
        super(ConditioningAugmentation, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim
        
    def forward(self, x):
        # Get mean and log variance
        x = self.fc(x)
        mu = x[:, :self.output_dim]
        logvar = x[:, self.output_dim:]
        
        # Calculate KL divergence with numerical stability improvements
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp() + 1e-8)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        c_hat = mu + eps * std
        
        return c_hat, kl_loss

class StageIIGenerator(nn.Module):
    """
    Stage-II Generator as described in StackGAN paper.
    Takes low-resolution images from Stage-I (64x64) and generates high-resolution images (256x256).
    """
    def __init__(self, config):
        super(StageIIGenerator, self).__init__()
        self.config = config
        ngf = config.STAGE2_G_HDIM
        
        # Conditioning Augmentation
        self.ca = ConditioningAugmentation(config.EMBEDDING_DIM, config.CA_DIM)
        
        # Encode low-resolution image from Stage-I (from Section 3.2 of paper)
        self.encode_img = nn.Sequential(
            # Input is 64x64x3
            nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        
        # Joint processing with text embedding (concatenation as in paper)
        input_dim = ngf * 4 + config.CA_DIM
        # Per section 3.2: c_hat is spatially replicated to form a 16 x 16 x C tensor
        self.joint_conv = nn.Sequential(
            nn.Conv2d(input_dim, ngf * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks with more expressive capacity
        self.residual = nn.Sequential(
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4),
            SelfAttention(ngf * 4),  # Add self-attention after two residual blocks
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4)
        )
        
        # Use spectral norm in upsampling layers for stability
        self.upsample1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.upsample2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False)),  # 32x32 -> 64x64
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.upsample3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(ngf, ngf // 2, kernel_size=4, stride=2, padding=1, bias=False)),  # 64x64 -> 128x128
            nn.BatchNorm2d(ngf // 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.upsample4 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(ngf // 2, 3, kernel_size=4, stride=2, padding=1, bias=False)),  # 128x128 -> 256x256
            nn.Tanh()
        )
        
    def forward(self, x_input):
        stage1_img, text_embedding = x_input
        
        # Process text embedding through conditioning augmentation
        c_hat, kl_loss = self.ca(text_embedding)
        
        # Encode low-resolution image from Stage I
        encoded_img = self.encode_img(stage1_img)  # [batch, ngf*4, 16, 16]
        
        # Spatial replication of text embedding as in paper
        c_hat = c_hat.view(-1, self.config.CA_DIM, 1, 1)
        c_hat = c_hat.repeat(1, 1, 16, 16)  # [batch, CA_DIM, 16, 16]
        
        # Concatenate encoded image and text embedding along channel dimension
        concat = torch.cat([encoded_img, c_hat], dim=1)
        
        # Joint processing
        out = self.joint_conv(concat)
        
        # Residual blocks
        out = self.residual(out)
        
        # Upsampling to 256x256
        out = self.upsample1(out)
        out = self.upsample2(out)
        out = self.upsample3(out)
        out = self.upsample4(out)
        
        return out, kl_loss

class StageIIDiscriminator(nn.Module):
    """
    Stage-II Discriminator as described in StackGAN paper.
    Processes 256x256 images and text embeddings to determine if the image is real or fake.
    """
    def __init__(self, config):
        super(StageIIDiscriminator, self).__init__()
        self.config = config
        ndf = config.STAGE2_D_HDIM
        
        # Image encoder: extract image features (256x256 -> 4x4) as specified in paper
        self.img_encoder = nn.Sequential(
            # 256x256 -> 128x128
            spectral_norm(nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(ndf * 2),  # Instance norm instead of batch norm
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Add self-attention at ndf*4 (earlier in the network where channels match)
            SelfAttention(ndf * 4),
            
            # 32x32 -> 16x16
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            spectral_norm(nn.Conv2d(ndf * 16, ndf * 32, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Remove the incorrectly placed self-attention that was causing the error
            # SelfAttention(ndf * 8),  # This was causing the error due to dimension mismatch
        )
        
        # Text embedding encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM, ndf * 8),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final classifier over encoded image and text features
        self.joint_encoder = nn.Sequential(
            nn.Conv2d(ndf * 32 + ndf * 8, ndf * 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x_input):
        img, text_embedding = x_input
        
        # Extract image features
        img_features = self.img_encoder(img)  # [batch, ndf*32, 4, 4]
        
        # Process text embedding
        text_features = self.text_encoder(text_embedding)  # [batch, ndf*8]
        
        # Spatially replicate text features to match image feature dimensions
        text_features = text_features.view(-1, self.config.STAGE2_D_HDIM * 8, 1, 1)
        text_features = text_features.repeat(1, 1, 4, 4)  # [batch, ndf*8, 4, 4]
        
        # Concatenate along channel dimension
        combined_features = torch.cat([img_features, text_features], dim=1)  # [batch, ndf*40, 4, 4]
        
        # Process through joint encoder
        output = self.joint_encoder(combined_features)
        
        return output.view(-1)
