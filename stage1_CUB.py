# Implementation of Stage-I GAN from StackGAN paper
import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import sqrtm
import logging
import time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import BertTokenizer, BertModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('debug_gan.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('GAN_Debug')

# Configuration class
class Config:
    DATASET_NAME = 'birds'
    EMBEDDING_DIM = 768
    Z_DIM = 100
    STAGE1_G_LR = 0.0002
    STAGE1_D_LR = 0.0002
    STAGE1_G_HDIM = 64
    STAGE1_D_HDIM = 64
    STAGE1_IMAGE_SIZE = 64
    BATCH_SIZE = 64
    EPOCHS = 600
    NUM_EXAMPLES = 8
    CA_DIM = 128
    KL_WEIGHT = 0.1
    FEATURE_MATCHING_WEIGHT = 1.0
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CUB_FOLDER = "CUB"

# Dataset for CUB with correct mapping
class CUBCaptionDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        # Parse class mapping
        self.class_id_to_name = {}
        with open(os.path.join(root, 'allclasses.txt'), 'r') as f:
            for line in f:
                idx, name = line.strip().split('.', 1)
                self.class_id_to_name[idx.zfill(3)] = name.strip()

        # Parse split classes
        split_classes_file = {
            'train': 'trainclasses.txt',
            'val': 'valclasses.txt',
            'test': 'testclasses.txt'
        }[split]
        with open(os.path.join(root, split_classes_file), 'r') as f:
            self.split_classes = [line.strip().split('.', 1)[0].zfill(3) for line in f if line.strip()]

        # Build image-caption mapping (expand so each (image, caption) is a separate sample)
        self.samples = []
        for class_id in self.split_classes:
            class_folder = f"{class_id}.{self.class_id_to_name[class_id]}"
            img_dir = os.path.join(root, 'images', class_folder)
            cap_dir = os.path.join(root, 'text_c10', class_folder)
            if not os.path.isdir(img_dir) or not os.path.isdir(cap_dir):
                continue
            img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
            for img_path in img_files:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                cap_files = sorted(glob.glob(os.path.join(cap_dir, f"{img_name}*.txt")))
                for cap_file in cap_files:
                    with open(cap_file, 'r') as f:
                        captions = [line.strip() for line in f if line.strip()]
                    for caption in captions:
                        self.samples.append({
                            'img_path': img_path,
                            'caption': caption,
                            'class_id': class_id
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        caption = sample['caption']
        return img, caption, int(sample['class_id'])

# Text encoder using BERT
class TextEncoder:
    def __init__(self, device='cpu'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.model.eval()
        self.device = device

    def encode(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=32)
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            emb = outputs.last_hidden_state[:, 0, :]
        return emb

# Conditioning Augmentation with better KL regularization
class ConditioningAugmentation(nn.Module):
    def __init__(self, input_dim=768, ca_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.bn1 = nn.BatchNorm1d(input_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim // 2, ca_dim * 2)
        
        # Initialize weights with smaller values to keep distribution closer to normal
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)  # Reduced from 1.0
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.3)  # Reduced from 0.8
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, text_embedding):
        if torch.isnan(text_embedding).any() or torch.isinf(text_embedding).any():
            text_embedding = torch.randn_like(text_embedding)
        
        x = self.fc1(text_embedding)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        # Use tighter clamping for better KL divergence
        x = torch.clamp(x, -1.0, 1.0)  # Reduced from -1.5, 1.5

        mu = x[:, :x.size(1) // 2]
        logvar = x[:, x.size(1) // 2:]
        
        # Initialize with smaller logvar to reduce KL divergence
        logvar = torch.clamp(logvar, min=-2.0, max=0.5)  # Modified from -4.0, 4.0
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        c = mu + eps * std
        
        # Remove additional noise which makes distribution farther from normal
        # c = c + 0.02 * torch.randn_like(c)  # Removed this line
        
        return c, mu, logvar

# Residual Block for Generator
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.block(x))

# Stage-I Generator
class Stage1Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        ngf = config.STAGE1_G_HDIM
        self.ca = ConditioningAugmentation(input_dim=config.EMBEDDING_DIM, ca_dim=config.CA_DIM)
        self.fc = nn.Sequential(
            nn.Linear(config.Z_DIM + config.CA_DIM, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.05)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.05)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.01)
    def forward(self, noise, text_embedding):
        c, mu, logvar = self.ca(text_embedding)
        z_c = torch.cat((noise, c), dim=1)
        h = self.fc(z_c)
        h = h.view(-1, 64 * 8, 4, 4)
        h = self.upsample1(h)
        h = self.upsample2(h)
        h = self.upsample3(h)
        output = self.upsample4(h)
        return output, mu, logvar

# Stage-I Discriminator
class Stage1Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        ndf = config.STAGE1_D_HDIM
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM, ndf * 8),
            nn.BatchNorm1d(ndf * 8, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output = nn.Sequential(
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, image, text_embedding, return_features=False):
        image = torch.clamp(image, -1.0, 1.0)
        text_embedding = torch.clamp(text_embedding, -10.0, 10.0)
        img_features = self.img_encoder(image)
        text_features = self.text_encoder(text_embedding)
        text_features = text_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4)
        combined_features = torch.cat([img_features, text_features], dim=1)
        output = self.output(combined_features)
        output = output.squeeze()
        if return_features:
            return output, combined_features
        return output

# Fixed InceptionScoreFID with proper weight initialization
class InceptionScoreFID:
    def __init__(self, device):
        self.device = device
        # Fix deprecated warning by using weights parameter
        self.weights = models.Inception_V3_Weights.IMAGENET1K_V1
        self.model = models.inception_v3(weights=self.weights).to(device)
        self.model.eval()
        # Get proper transforms from the weights
        self.preprocess = self.weights.transforms()
        
    def calculate_inception_score(self, images, splits=1):
        # Process images in smaller batches to prevent OOM
        N = images.size(0)
        preds = []
        
        for i in range(0, N, 16):  # Smaller batch size
            batch = images[i:i+16].to(self.device)
            # Proper preprocessing
            batch = (batch + 1) / 2.0  # [-1,1] to [0,1]
            batch = torch.clamp(batch, 0, 1)
            
            # Resize to inception input size
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Apply inception preprocessing
            batch = self.preprocess(batch)
            
            # Forward pass
            with torch.no_grad():
                pred = self.model(batch)
                preds.append(F.softmax(pred, dim=1).cpu().numpy())
                
        preds = np.concatenate(preds, axis=0)
        
        # Calculate IS score properly
        py = np.mean(preds, axis=0)
        split_scores = []
        
        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            kl = part * (np.log(part + 1e-6) - np.log(py + 1e-6))
            kl = np.mean(np.sum(kl, axis=1))
            split_scores.append(np.exp(kl))
            
        return float(np.mean(split_scores)), float(np.std(split_scores))
    
    def calculate_fid(self, real_images, fake_images):
        # Pre-process images consistently
        def extract_features(images):
            features = []
            for i in range(0, len(images), 16):  # Smaller batch size
                batch = images[i:i+16].to(self.device)
                batch = (batch + 1) / 2.0  # [-1,1] to [0,1]
                batch = torch.clamp(batch, 0, 1)
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                batch = self.preprocess(batch)
                
                # Extract features from the last pooling layer
                with torch.no_grad():
                    # Get the penultimate layer, not the logits
                    self.model.fc = torch.nn.Identity()  # Replace classifier with identity
                    feat = self.model(batch)
                    features.append(feat.cpu().numpy())
                    
            return np.concatenate(features, axis=0)
        
        # Extract features
        real_feats = extract_features(real_images)
        fake_feats = extract_features(fake_images)
        
        # Calculate mean and covariance
        mu_real = np.mean(real_feats, axis=0)
        mu_fake = np.mean(fake_feats, axis=0)
        sigma_real = np.cov(real_feats, rowvar=False)
        sigma_fake = np.cov(fake_feats, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_fake
        # Add a small epsilon to diagonal for numerical stability
        sigma_real += np.eye(sigma_real.shape[0]) * 1e-6
        sigma_fake += np.eye(sigma_fake.shape[0]) * 1e-6
        
        covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2*covmean)
        return float(fid)

def get_datasets_and_loaders(cub_root, batch_size, num_workers=4):
    image_size = 64
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 8, image_size + 8)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = CUBCaptionDataset(cub_root, split='train', transform=train_transform)
    val_set = CUBCaptionDataset(cub_root, split='val', transform=eval_transform)
    test_set = CUBCaptionDataset(cub_root, split='test', transform=eval_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_set, val_set, test_set, train_loader, val_loader, test_loader

def train(config, num_epochs=None, debug=False, resume_epoch=0, resume_checkpoint=None):
    """
    Train the model with checkpointing and better sample visualization.
    Args:
        config: Configuration object
        num_epochs: Number of epochs to train
        debug: Enable debug mode
        resume_epoch: Epoch to resume from (0 for new training)
        resume_checkpoint: Path to checkpoint file to resume from
    """
    # Create folders for checkpoints and samples
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    
    cub_root = config.CUB_FOLDER
    batch_size = config.BATCH_SIZE
    device = config.DEVICE
    train_set, val_set, test_set, train_loader, val_loader, test_loader = get_datasets_and_loaders(cub_root, batch_size)
    text_encoder = TextEncoder(device=device)
    generator = Stage1Generator(config).to(device)
    discriminator = Stage1Discriminator(config).to(device)
    
    # Setup optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=config.STAGE1_G_LR, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config.STAGE1_D_LR * 0.05, betas=(0.5, 0.999))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    elif resume_epoch > 0:
        # Try to load from the resume_epoch checkpoint
        resume_path = os.path.join("checkpoints", f"checkpoint_epoch_{resume_epoch}.pth")
        if os.path.exists(resume_path):
            print(f"Loading checkpoint from epoch {resume_epoch}")
            checkpoint = torch.load(resume_path, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            start_epoch = resume_epoch
            print(f"Resuming from epoch {start_epoch}")
    
    g_losses, d_losses, kl_losses = [], [], []
    metric = InceptionScoreFID(device)
    
    # Add KL weight annealing
    kl_weight_scheduler = lambda epoch: min(0.01 * (epoch + 1), config.KL_WEIGHT)
    
    for epoch in range(start_epoch, num_epochs or config.EPOCHS):
        # Calculate current KL weight
        current_kl_weight = kl_weight_scheduler(epoch)
        
        generator.train()
        discriminator.train()
        epoch_g_losses, epoch_d_losses, epoch_kl_losses = [], [], []
        
        for imgs, captions, _ in train_loader:
            imgs = imgs.to(device)
            text_embs = text_encoder.encode(list(captions)).to(device)
            batch_size = imgs.size(0)
            noise = torch.randn(batch_size, config.Z_DIM, device=device)
            # Train Discriminator
            d_optimizer.zero_grad()
            with torch.no_grad():
                fake_imgs, _, _ = generator(noise, text_embs)
            real_logits = discriminator(imgs, text_embs)
            fake_logits = discriminator(fake_imgs, text_embs)
            real_labels = torch.ones_like(real_logits, device=device) * 0.9
            fake_labels = torch.zeros_like(fake_logits, device=device) + 0.1
            d_loss_real = F.binary_cross_entropy_with_logits(real_logits, real_labels)
            d_loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            # Train Generator
            g_optimizer.zero_grad()
            fake_imgs, mu, logvar = generator(noise, text_embs)
            fake_logits = discriminator(fake_imgs, text_embs)
            g_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits, device=device))
            kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=1).mean()
            # No need to clamp KL loss with our improved CA
            # kl_loss = torch.clamp(kl_loss, max=10.0)  # Remove this line
            total_loss = g_loss + current_kl_weight * kl_loss
            total_loss.backward()
            g_optimizer.step()
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            kl_losses.append(kl_loss.item())

        # --- Compute IS and FID for train ---
        generator.eval()
        real_imgs, fake_imgs, train_prompts = [], [], []
        with torch.no_grad():
            for imgs, captions, _ in train_loader:
                imgs = imgs.to(device)
                text_embs = text_encoder.encode(list(captions)).to(device)
                noise = torch.randn(imgs.size(0), config.Z_DIM, device=device)
                fake, _, _ = generator(noise, text_embs)
                real_imgs.append(imgs.cpu())
                fake_imgs.append(fake.cpu())
                train_prompts.extend(captions)
                if len(real_imgs) * batch_size > 1024: break  # limit for speed
        real_imgs = torch.cat(real_imgs, dim=0)[:1024]
        fake_imgs = torch.cat(fake_imgs, dim=0)[:1024]
        train_is, _ = metric.calculate_inception_score(fake_imgs)
        train_fid = metric.calculate_fid(real_imgs, fake_imgs)

        # --- Compute IS and FID for val ---
        real_imgs, fake_imgs, val_prompts = [], [], []
        with torch.no_grad():
            for imgs, captions, _ in val_loader:
                imgs = imgs.to(device)
                text_embs = text_encoder.encode(list(captions)).to(device)
                noise = torch.randn(imgs.size(0), config.Z_DIM, device=device)
                fake, _, _ = generator(noise, text_embs)
                real_imgs.append(imgs.cpu())
                fake_imgs.append(fake.cpu())
                val_prompts.extend(captions)
                if len(real_imgs) * batch_size > 1024: break  # limit for speed
        real_imgs = torch.cat(real_imgs, dim=0)[:1024]
        fake_imgs = torch.cat(fake_imgs, dim=0)[:1024]
        val_is, _ = metric.calculate_inception_score(fake_imgs)
        val_fid = metric.calculate_fid(real_imgs, fake_imgs)

        # --- Print metrics as requested ---
        print(f"Epoch [{epoch+1}/{num_epochs or config.EPOCHS}] (training) - G_loss: {np.mean(g_losses):.4f} D_loss: {np.mean(d_losses):.4f} KL_loss: {np.mean(kl_losses):.4f} KL_weight: {current_kl_weight:.4f} IS: {train_is:.4f} FID: {train_fid:.2f}")
        print(f"Epoch [{epoch+1}/{num_epochs or config.EPOCHS}] (validation) - IS: {val_is:.4f} FID: {val_fid:.2f}")

        # Save model checkpoint after each epoch
        checkpoint_path = os.path.join("checkpoints", f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'g_loss': np.mean(g_losses),
            'd_loss': np.mean(d_losses),
            'kl_loss': np.mean(kl_losses),
            'train_is': train_is,
            'train_fid': train_fid,
            'val_is': val_is,
            'val_fid': val_fid,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # --- Save a sample image and prompt with better visibility ---
        sample_img = fake_imgs[0]
        sample_prompt = val_prompts[0] if val_prompts else ""
        
        # Create a figure with more space for text
        plt.figure(figsize=(4, 4.5))
        plt.imshow(np.clip((sample_img.permute(1, 2, 0).numpy() + 1) / 2, 0, 1))
        
        # Add a white background for the prompt text for better visibility
        plt.figtext(0.5, 0.01, sample_prompt, wrap=True, horizontalalignment='center', 
                   fontsize=9, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.axis('off')
        plt.tight_layout()
        sample_path = os.path.join("samples", f"sample_epoch_{epoch+1}.png")
        plt.savefig(sample_path)
        plt.close()
        print(f"Sample saved to {sample_path}")

    # Final model save
    final_path = os.path.join("checkpoints", "final_model.pth")
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }, final_path)
    print(f"Final model saved to {final_path}")

# KL loss is high because the Conditioning Augmentation network's output distribution (mu, logvar)
# is far from standard normal at the start of training. This is expected early on.
# As training progresses, KL loss should decrease as the CA learns to regularize its output.

def generate_images_from_text(model_path, text_descriptions, config=None, num_images=4):
    if config is None:
        config = Config()
    device = config.DEVICE
    generator = Stage1Generator(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    text_encoder = TextEncoder(device=device)
    text_embeddings = text_encoder.encode(text_descriptions).to(device)
    all_images = []
    with torch.no_grad():
        for embedding in text_embeddings:
            embedding = embedding.unsqueeze(0).repeat(num_images, 1)
            noise = torch.randn(num_images, config.Z_DIM, device=device)
            fake_images, _ = generator(noise, embedding)
            fake_images = (fake_images + 1) / 2.0
            all_images.append(fake_images.cpu())
    fig, axes = plt.subplots(len(text_descriptions), num_images, figsize=(num_images * 3, len(text_descriptions) * 3))
    for i, images in enumerate(all_images):
        for j in range(num_images):
            ax = axes[i, j] if len(text_descriptions) > 1 else axes[j]
            img = images[j].numpy().transpose(1, 2, 0)
            ax.imshow(img)
            if j == 0:
                ax.set_ylabel(text_descriptions[i][:30], fontsize=8)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"generated_samples_{int(time.time())}.png", dpi=300)
    plt.show()
    return all_images

# Update the main function to support resuming training
def main():
    config = Config()
    print(f"CUB dataset path: {config.CUB_FOLDER}")
    print(f"Image size: {config.STAGE1_IMAGE_SIZE}x{config.STAGE1_IMAGE_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    
    # Automatically find latest checkpoint
    resume_epoch = 0
    resume_checkpoint = None
    
    if os.path.exists("checkpoints"):
        checkpoint_files = glob.glob(os.path.join("checkpoints", "checkpoint_epoch_*.pth"))
        if checkpoint_files:
            # Extract epoch numbers from filenames
            epoch_numbers = []
            for f in checkpoint_files:
                try:
                    epoch_num = int(f.split("checkpoint_epoch_")[1].split(".pth")[0])
                    epoch_numbers.append(epoch_num)
                except:
                    continue
            
            if epoch_numbers:
                # Get the latest epoch
                latest_epoch = max(epoch_numbers)
                resume_checkpoint = os.path.join("checkpoints", f"checkpoint_epoch_{latest_epoch}.pth")
                print(f"Found latest checkpoint: {resume_checkpoint} (Epoch {latest_epoch})")
    
    print("Running training mode...")
    if resume_checkpoint:
        print(f"Resuming from latest checkpoint (Epoch {latest_epoch})")
    train(config, num_epochs=config.EPOCHS, debug=False, resume_epoch=0, resume_checkpoint=resume_checkpoint)

if __name__ == "__main__":
    main()
