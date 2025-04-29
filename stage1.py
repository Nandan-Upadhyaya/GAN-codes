# Implementation of Stage-I GAN from StackGAN paper
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import scipy
from scipy import linalg
import os
import time
import pickle
import random
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
 print(f"PyTorch version: {torch.__version__}")
 print(f"Torchvision version: {torchvision.__version__}")
 print(f"NumPy version: {np.__version__}")
 print(f"SciPy version: {scipy.__version__}")
 print(f"Matplotlib version: {matplotlib.__version__}")
 print(f"CUDA available: {torch.cuda.is_available()}")

 if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Set random seeds for reproducibility
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class Config:
    # Data parameters
    DATASET_NAME = 'birds'  # 'birds' or 'flowers'
    EMBEDDING_DIM = 10240  # dimension of the text embedding for birds dataset
    Z_DIM = 100  # dimension of the noise vector as in the paper
    LOCAL_DATASET_PATH = None  # will be set later
    
    # Hyperparameters from StackGAN paper
    STAGE1_G_LR = 0.0002  # learning rate for stage 1 generator (Adam) - paper value
    STAGE1_D_LR = 0.0002  # learning rate for stage 1 discriminator (Adam) - paper value
    STAGE1_G_HDIM = 64  # base filter size for generator as in paper
    STAGE1_D_HDIM = 64  # base filter size for discriminator as in paper
    STAGE1_IMAGE_SIZE = 64  # output size for Stage-I (64x64 as in paper)
    
    # Training parameters from paper
    BATCH_SIZE = 128  # batch size from the StackGAN paper
    EPOCHS = 600      # total epochs for training
    SNAPSHOT_INTERVAL = 1
    NUM_EXAMPLES = 6  # number of examples to visualize
    
    # Conditioning Augmentation parameters as in paper
    CA_DIM = 128  # dimension of the CA output as in paper
    
    # Adam optimizer parameters from paper
    BETA1 = 0.5    # paper value
    BETA2 = 0.999  # paper value
    
    # KL divergence regularization weight from paper
    LAMBDA = 2.0   # paper value
    
    # Metrics parameters
    EVAL_INTERVAL = 1
    FID_SAMPLE_SIZE = 500
    INCEPTION_SAMPLE_SIZE = 500
    R_PRECISION_K = 5
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, images_path=None, train_path=None, test_path=None):
        if images_path and train_path and test_path:
            self.LOCAL_DATASET_PATH = {
                'images': images_path,
                'train': train_path,
                'test': test_path
            }

class GANMetrics:
    """Class for computing standard GAN evaluation metrics"""
    def __init__(self, config, lazy_load=True):
        self.config = config
        self.device = config.DEVICE
        self.inception_model = None
        self.inception_classifier = None
        
        if not lazy_load:
            self._load_inception_models()
    
    def _load_inception_models(self):
        if self.inception_model is not None and self.inception_classifier is not None:
            return  # Already loaded
            
        print("Loading Inception v3 model for metrics calculation...")
        start_time = time.time()
        
        # Model for feature extraction (FID) - with progress indication
        print("  - Loading feature extraction model...")
        self.inception_model = models.inception_v3(pretrained=True, progress=True)
        self.inception_model.fc = nn.Identity()  # Remove classification layer
        self.inception_model.eval()
        self.inception_model = self.inception_model.to(self.device)
        print(f"  - Feature extraction model loaded in {time.time() - start_time:.1f}s")
        
        # Model for classification scores (Inception Score) - with progress indication
        class_time = time.time()
        print("  - Loading classification model...")
        self.inception_classifier = models.inception_v3(pretrained=True, progress=True)
        self.inception_classifier.eval()
        self.inception_classifier = self.inception_classifier.to(self.device)
        print(f"  - Classification model loaded in {time.time() - class_time:.1f}s")
        
        print(f"Inception models loaded successfully in {time.time() - start_time:.1f}s total")
    
    def _get_activations(self, images, model, batch_size=50):
        """Get Inception activations for a batch of images"""
        # Make sure models are loaded before calculation
        if self.inception_model is None or self.inception_classifier is None:
            self._load_inception_models()
            
        n_batches = len(images) // batch_size + (1 if len(images) % batch_size != 0 else 0)
        activations = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx].to(self.device)
            
            # Resize for Inception v3 (299x299)
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=True)
            
            # Get activations
            with torch.no_grad():
                batch_activations = model(batch)
                activations.append(batch_activations.cpu().detach())
                
        return torch.cat(activations, dim=0)
    
    def calculate_fid(self, real_images, fake_images):
        """Calculate FID score between real and fake images"""
        print("Calculating FID score...")
        # Make sure models are loaded before calculation
        if self.inception_model is None:
            self._load_inception_models()
            
        # Get activations
        real_activations = self._get_activations(real_images, self.inception_model)
        fake_activations = self._get_activations(fake_images, self.inception_model)
        
        # Convert to NumPy arrays
        real_activations = real_activations.numpy()
        fake_activations = fake_activations.numpy()
        
        # Calculate mean and covariance statistics
        real_mu = np.mean(real_activations, axis=0)
        fake_mu = np.mean(fake_activations, axis=0)
        real_sigma = np.cov(real_activations, rowvar=False)
        fake_sigma = np.cov(fake_activations, rowvar=False)
        
        # Calculate FID
        mu_diff = real_mu - fake_mu
        mu_diff_sq = np.dot(mu_diff, mu_diff)
        covmean = linalg.sqrtm(np.dot(real_sigma, fake_sigma))
        
        # Check for numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        trace_term = np.trace(real_sigma + fake_sigma - 2.0 * covmean)
        fid = mu_diff_sq + trace_term
        
        return fid
    
    def calculate_inception_score(self, fake_images, splits=10):
        """Calculate Inception Score for fake images"""
        print("Calculating Inception Score...")
        # Make sure models are loaded before calculation
        if self.inception_classifier is None:
            self._load_inception_models()
            
        # Get predictions
        preds = []
        n_batches = len(fake_images) // 50 + (1 if len(fake_images) % 50 != 0 else 0)
        
        for i in range(n_batches):
            start_idx = i * 50
            end_idx = min((i + 1) * 50, len(fake_images))
            batch = fake_images[start_idx:end_idx].to(self.device)
            
            # Resize for Inception v3 (299x299)
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=True)
            
            # Get predictions
            with torch.no_grad():
                pred = F.softmax(self.inception_classifier(batch), dim=1)
                preds.append(pred.cpu().numpy())
                
        preds = np.concatenate(preds, axis=0)
        
        # Calculate mean kl-divergence
        split_scores = []
        n_images = len(preds)
        split_size = n_images // splits
        
        for i in range(splits):
            part = preds[i * split_size:(i + 1) * split_size]
            kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True) + 1e-10) + 1e-10)
            kl = np.mean(np.sum(kl, axis=1))
            split_scores.append(np.exp(kl))
            
        # Return mean and std of IS
        is_mean = np.mean(split_scores)
        is_std = np.std(split_scores)
        
        return is_mean, is_std
    
    def calculate_r_precision(self, image_embeddings, text_embeddings, k=5):
        """Calculate R-precision for text-image matching"""
        print("Calculating R-precision...")
        # Convert to NumPy arrays if they are torch tensors
        if isinstance(image_embeddings, torch.Tensor):
            image_embeddings = image_embeddings.cpu().numpy()
        if isinstance(text_embeddings, torch.Tensor):
            text_embeddings = text_embeddings.cpu().numpy()
            
        # Normalize embeddings for cosine similarity
        image_norm = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        text_norm = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        image_embeddings = image_embeddings / (image_norm + 1e-8)
        text_embeddings = text_embeddings / (text_norm + 1e-8)
        
        # Calculate similarity matrix
        similarity = np.matmul(image_embeddings, text_embeddings.T)
        
        # Calculate R-precision
        r_precision = 0.0
        for i in range(len(similarity)):
            # Get top-k indices
            top_k_indices = np.argsort(similarity[i])[-k:]
            # Check if true match (index i) is in top k
            if i in top_k_indices:
                r_precision += 1.0
                
        # Normalize by number of samples
        r_precision /= len(similarity)
        
        return r_precision
    
class BirdsDataset(Dataset):
    def __init__(self, config, image_dir, embeddings, filenames, transform=None):
        self.config = config
        self.image_dir = image_dir
        self.embeddings = embeddings
        self.filenames = filenames
        self.image_size = config.STAGE1_IMAGE_SIZE
        self.transform = transform or transforms.Compose([
            transforms.Resize((config.STAGE1_IMAGE_SIZE, config.STAGE1_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        try:
            # Get embedding
            embedding = self.embeddings[idx]
            # Load image
            filename = self.filenames[idx]
            image = self.load_image(filename)
            
            # Convert embedding to tensor
            if isinstance(embedding, list) or (isinstance(embedding, np.ndarray) and len(embedding.shape) > 1):
                embedding = np.array(embedding).flatten()
            embedding = torch.tensor(embedding, dtype=torch.float32)
            
            return image, embedding
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            # Return a placeholder in case of error
            dummy_img = torch.zeros(3, self.image_size, self.image_size)
            dummy_emb = torch.zeros(self.config.EMBEDDING_DIM)
            return dummy_img, dummy_emb
    
    def load_image(self, filename):
        """Load and preprocess image"""
        # Normalize path separators for Windows
        filename = filename.replace('/', os.path.sep)
        # Extract bird class name and image name from filename
        parts = filename.split(os.path.sep)
        if len(parts) > 1:
            bird_class = parts[0]  # e.g., "002.Laysan_Albatross"
            img_name = parts[1]    # e.g., "Laysan_Albatross_0002_1027"
        else:
            # If no path separator, try to extract class from filename convention
            img_name = filename
            bird_class = None
        # Check if the bird class directory exists directly
        if bird_class:
            class_dir = os.path.join(self.image_dir, bird_class)
            if os.path.isdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.exists(img_path):
                    return self._process_image(img_path)
            bird_class = None
        # If not found, search all class directories
        for dir_name in os.listdir(self.image_dir):
            class_path = os.path.join(self.image_dir, dir_name)
            if not os.path.isdir(class_path):
                continue
            # Try matching by full image name
            img_path = os.path.join(class_path, img_name)
            if os.path.exists(img_path):
                return self._process_image(img_path)
            # Try matching by base name without extension
            base_img_name = os.path.splitext(img_name)[0]
            for img_file in os.listdir(class_path):
                # Remove extension for comparison
                base_file = os.path.splitext(img_file)[0]
                # Compare base names
                if base_img_name == base_file:
                    img_path = os.path.join(class_path, img_file)
                    return self._process_image(img_path)
        # If we reach here, image was not found
        print(f"Warning: Image not found: {filename}")
        # Return a black image as placeholder
        black_img = Image.new('RGB', (self.image_size, self.image_size), 'black')
        return self.transform(black_img)

    def _process_image(self, img_path):
        """Process an image given its path"""
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            return self.transform(img)
        return img
    


class DataManager:
    def __init__(self, config):
        self.config = config
        self.image_size = config.STAGE1_IMAGE_SIZE
        
        # Check if LOCAL_DATASET_PATH is a dictionary with required keys
        if not isinstance(config.LOCAL_DATASET_PATH, dict) or not all(k in config.LOCAL_DATASET_PATH for k in ['images', 'train', 'test']):
            raise ValueError("LOCAL_DATASET_PATH must be a dictionary with 'images', 'train', and 'test' keys")
        
        # Set directories directly from the dictionary
        self.image_dir = config.LOCAL_DATASET_PATH['images']
        self.train_dir = config.LOCAL_DATASET_PATH['train']
        self.test_dir = config.LOCAL_DATASET_PATH['test']
        
        # Use the pickle files directly from train folder
        self.embedding_path = os.path.join(self.train_dir, 'char-CNN-RNN-embeddings.pickle')
        self.filenames_path = os.path.join(self.train_dir, 'filenames.pickle')
        
        print(f"Using dataset structure:")
        print(f"Images directory: {self.image_dir}")
        print(f"Train directory: {self.train_dir}")
        print(f"Test directory: {self.test_dir}")
        
        # Verify files exist
        if not os.path.exists(self.embedding_path):
            raise FileNotFoundError(f"Embeddings file not found: {self.embedding_path}")
        if not os.path.exists(self.filenames_path):
            raise FileNotFoundError(f"Filenames file not found: {self.filenames_path}")
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Images directory not found: {self.image_dir}")
            
        # Load text embeddings
        with open(self.embedding_path, 'rb') as f:
            self.embeddings = pickle.load(f, encoding='latin1')
            
        # Load filenames
        with open(self.filenames_path, 'rb') as f:
            self.filenames = pickle.load(f, encoding='latin1')
            
        self.dataset_size = len(self.filenames)
        print(f"Dataset size: {self.dataset_size}")
        print(f"Sample filename: {self.filenames[0] if self.dataset_size > 0 else 'No files found'}")
        
        # Create transform for image processing
        self.transform = transforms.Compose([
            transforms.Resize((config.STAGE1_IMAGE_SIZE, config.STAGE1_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

    def get_data(self):
        """Return a PyTorch DataLoader for training"""
        print("Setting up data streaming pipeline...")
        # Check embeddings format
        is_embeddings_list = isinstance(self.embeddings, list)
        if is_embeddings_list:
            print(f"Embeddings length: {len(self.embeddings)}")
            print(f"Filenames length: {len(self.filenames)}")
            
            # Create PyTorch Dataset
            train_dataset = BirdsDataset(
                self.config,
                self.image_dir,
                self.embeddings,
                self.filenames,
                self.transform
            )
            
            # Create DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True if self.config.DEVICE.type == 'cuda' else False,
                drop_last=True
            )
            
            # Estimate dataset size
            estimated_size = min(len(self.filenames), len(self.embeddings))
            
        else:
            # Dictionary-based embeddings not implemented yet
            raise ValueError("Dictionary-based embeddings not implemented yet")
            
        return train_loader, estimated_size

    def get_test_data(self):
        """Return a PyTorch DataLoader for testing/validation"""
        print("Setting up test data streaming pipeline...")
        # Load test embeddings and filenames
        test_embedding_path = os.path.join(self.test_dir, 'char-CNN-RNN-embeddings.pickle')
        test_filenames_path = os.path.join(self.test_dir, 'filenames.pickle')
        
        # Verify files exist
        if not os.path.exists(test_embedding_path):
            raise FileNotFoundError(f"Test embeddings file not found: {test_embedding_path}")
        if not os.path.exists(test_filenames_path):
            raise FileNotFoundError(f"Test filenames file not found: {test_filenames_path}")
            
        # Load test text embeddings
        with open(test_embedding_path, 'rb') as f:
            test_embeddings = pickle.load(f, encoding='latin1')
            
        # Load test filenames
        with open(test_filenames_path, 'rb') as f:
            test_filenames = pickle.load(f, encoding='latin1')
            
        test_dataset_size = len(test_filenames)
        print(f"Test dataset size: {test_dataset_size}")
        
        # Create PyTorch Dataset
        test_dataset = BirdsDataset(
            self.config,
            self.image_dir,
            test_embeddings,
            test_filenames,
            self.transform
        )
        
        # Create DataLoader (no shuffling for validation)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.config.DEVICE.type == 'cuda' else False,
            drop_last=True
        )
        
        # Estimate dataset size
        estimated_size = min(len(test_filenames), len(test_embeddings))
        
        return test_loader, estimated_size

    def visualize_samples(self, num_samples=3):
        """Visualize a few samples to verify text-image pairs"""
        print("Visualizing sample text-image pairs for verification...")
        # Create a dataset for visualization
        viz_dataset = BirdsDataset(
            self.config,
            self.image_dir,
            self.embeddings[:num_samples],
            self.filenames[:num_samples],
            self.transform
        )
        
        for i in range(num_samples):
            try:
                # Get image and embedding
                image, embedding = viz_dataset[i]
                
                # Convert tensor to numpy for plotting
                image_np = image.numpy().transpose(1, 2, 0)
                image_np = (image_np + 1) / 2.0  # Convert from [-1,1] to [0,1]
                
                plt.figure(figsize=(8, 4))
                # Image display
                plt.subplot(1, 2, 1)
                plt.imshow(image_np)
                plt.title(f"Image: {self.filenames[i]}")
                plt.axis('off')
                
                # Embedding visualization (show first 20 values)
                embedding_np = embedding.numpy()
                embedding_values = embedding_np[:20]
                
                plt.subplot(1, 2, 2)
                plt.bar(range(len(embedding_values)), embedding_values)
                plt.title(f"First 20 embedding values\nShape: {embedding_np.shape}")
                plt.tight_layout()
                plt.savefig(f"sample_pair_{i}.png")
                plt.close()
                print(f"Sample {i+1}: Image shape {image_np.shape}, Embedding shape {embedding_np.shape}")
            except Exception as e:
                print(f"Error visualizing sample {i}: {e}")

class ConditioningAugmentation(nn.Module):
    """
    Conditioning Augmentation module as described in StackGAN paper.
    Transforms text embeddings into conditioning variables following a Gaussian distribution.
    """
    def __init__(self, input_dim, output_dim):
        super(ConditioningAugmentation, self).__init__()
        # First reduce input dimension to make it tractable
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),  # *2 for mean and log variance
        )
        self.output_dim = output_dim
        
    def forward(self, x):
        # Get mean and log variance
        x = self.fc(x)
        mu = x[:, :self.output_dim]
        logvar = x[:, self.output_dim:]
        
        # Calculate KL divergence loss as in paper: -1/2 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0)  # normalize by batch size
        
        # Reparameterization trick: sample from N(mu, sigma^2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        c_hat = mu + eps * std
        
        return c_hat, kl_loss

class StageIGenerator(nn.Module):
    """
    Generator for Stage-I GAN as described in StackGAN paper.
    Generates 64x64 images based on text embeddings and random noise.
    """
    def __init__(self, config):
        super(StageIGenerator, self).__init__()
        self.config = config
        ngf = config.STAGE1_G_HDIM
        
        # Conditioning Augmentation as in paper
        self.ca = ConditioningAugmentation(config.EMBEDDING_DIM, config.CA_DIM)
        
        # Initial dense layer to project and reshape
        self.fc = nn.Linear(config.Z_DIM + config.CA_DIM, ngf * 8 * 4 * 4)
        
        # Upsampling layers exactly as in paper: 4x4 -> 64x64
        self.upsample1 = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 4x4 -> 8x8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 8x8 -> 16x16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 16x16 -> 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),  # 32x32 -> 64x64
            nn.Tanh()  # Output in range [-1, 1] as in paper
        )
        
    def forward(self, noise, text_embedding):
        # Process through conditioning augmentation
        c_hat, kl_loss = self.ca(text_embedding)
        
        # Concatenate noise and conditioning variable
        z_c = torch.cat([noise, c_hat], dim=1)
        
        # Project and reshape
        out = self.fc(z_c)
        out = out.view(-1, self.config.STAGE1_G_HDIM * 8, 4, 4)
        
        # Upsampling to 64x64 as in paper
        out = self.upsample1(out)
        out = self.upsample2(out)
        out = self.upsample3(out)
        out = self.upsample4(out)
        
        return out, kl_loss

class StageIDiscriminator(nn.Module):
    """
    Discriminator for Stage-I GAN as described in StackGAN paper.
    Classifies 64x64 images as real or fake based on the image and text embedding.
    """
    def __init__(self, config):
        super(StageIDiscriminator, self).__init__()
        self.config = config
        ndf = config.STAGE1_D_HDIM
        
        # Image encoder: extract image features (64x64 -> 4x4)
        self.img_encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Text embedding compression as in paper
        self.text_encoder = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM, ndf * 8),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final classification layer
        self.joint_encoder = nn.Sequential(
            nn.Conv2d(ndf * 16, ndf * 8, 1, 1, 0, bias=False),  # Compress channels
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # 4x4 -> 1x1
            nn.Sigmoid()  # Output probability as in original GAN
        )
        
    def forward(self, img, text_embedding):
        # Extract image features
        img_features = self.img_encoder(img)  # [batch, ndf*8, 4, 4]
        
        # Process text embedding
        text_features = self.text_encoder(text_embedding)  # [batch, ndf*8]
        
        # Spatially replicate text features to match image feature dimensions
        text_features = text_features.view(-1, self.config.STAGE1_D_HDIM * 8, 1, 1)
        text_features = text_features.repeat(1, 1, 4, 4)  # [batch, ndf*8, 4, 4]
        
        # Concatenate along channel dimension
        combined_features = torch.cat([img_features, text_features], dim=1)  # [batch, ndf*16, 4, 4]
        
        # Process through joint encoder
        output = self.joint_encoder(combined_features)
        
        return output.view(-1)

class GANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.checkpoint_dir = os.path.join('checkpoints', config.DATASET_NAME)
        self.sample_dir = os.path.join('samples', config.DATASET_NAME)
        self.metrics_dir = os.path.join('metrics', config.DATASET_NAME)
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize models
        self.generator = StageIGenerator(config).to(self.device)
        self.discriminator = StageIDiscriminator(config).to(self.device)
        
        # Initialize optimizers as in paper (Adam with beta1=0.5, beta2=0.999)
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.STAGE1_G_LR,
            betas=(config.BETA1, config.BETA2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.STAGE1_D_LR,
            betas=(config.BETA1, config.BETA2)
        )
        
        # Create fixed noise vector for visualization
        self.fixed_noise = torch.randn(config.NUM_EXAMPLES, config.Z_DIM, device=self.device)
        
        # Initialize metrics tracking
        self.metrics = {
            'g_losses': [],
            'd_losses': [],
            'kl_losses': [],
            'fid_scores': [],
            'inception_scores': [],
            'r_precision': []
        }
        
        # Initialize metrics calculator
        print("Initializing metrics calculator (models will be loaded when needed)")
        self.metrics_calculator = GANMetrics(config, lazy_load=True)
        
        # Create log directory
        self.log_dir = os.path.join('logs', config.DATASET_NAME)
        os.makedirs(self.log_dir, exist_ok=True)
        
        print("Trainer initialization complete")
        
    def train_discriminator(self, real_images, embeddings, noise):
        """Train the discriminator for one step using binary cross-entropy loss as in original GAN paper"""
        self.d_optimizer.zero_grad()
        
        batch_size = real_images.size(0)
        
        # Generate fake images
        fake_images, _ = self.generator(noise, embeddings)
        
        # Real images: label = 1
        real_labels = torch.ones(batch_size, device=self.device)
        real_logits = self.discriminator(real_images, embeddings)
        d_loss_real = F.binary_cross_entropy(real_logits, real_labels)
        
        # Fake images: label = 0
        fake_labels = torch.zeros(batch_size, device=self.device)
        fake_logits = self.discriminator(fake_images.detach(), embeddings)
        d_loss_fake = F.binary_cross_entropy(fake_logits, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        
        # Backpropagation and optimization
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def train_generator(self, embeddings, noise):
        """Train the generator for one step using binary cross-entropy loss + KL divergence regularization"""
        self.g_optimizer.zero_grad()
        
        batch_size = embeddings.size(0)
        
        # Generate fake images
        fake_images, kl_loss = self.generator(noise, embeddings)
        
        # Compute generator loss - fool the discriminator
        real_labels = torch.ones(batch_size, device=self.device)
        fake_logits = self.discriminator(fake_images, embeddings)
        g_loss = F.binary_cross_entropy(fake_logits, real_labels)
        
        # Add KL divergence loss with lambda regularization as in paper
        total_loss = g_loss + self.config.LAMBDA * kl_loss
        
        # Backpropagation and optimization
        total_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), kl_loss.item()
    
    def save_samples(self, epoch, fixed_embeddings):
        """Save sample generated images for visualization"""
        self.generator.eval()
        with torch.no_grad():
            fake_images, _ = self.generator(self.fixed_noise, fixed_embeddings)
            
            # Convert to numpy and denormalize
            fake_images = fake_images.cpu().numpy()
            fake_images = (fake_images + 1) / 2.0  # [-1, 1] -> [0, 1]
            
            # Save as grid
            fig, axs = plt.subplots(2, 3, figsize=(12, 8))
            for i, ax in enumerate(axs.flatten()):
                if i < fake_images.shape[0]:
                    ax.imshow(np.transpose(fake_images[i], (1, 2, 0)))
                    ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.sample_dir, f'samples_epoch_{epoch+1}.png'))
            plt.close()
        self.generator.train()
    
    def compute_metrics(self, epoch, real_images, text_embeddings, num_samples=500):
        """Compute evaluation metrics"""
        print(f"\nComputing metrics for epoch {epoch}...")
        
        # Generate fake images for metrics calculation
        self.generator.eval()
        with torch.no_grad():
            # Sample random noise and text embeddings
            noise = torch.randn(num_samples, self.config.Z_DIM, device=self.device)
            sampled_embeddings = text_embeddings[:num_samples].to(self.device)
            
            # Generate fake images
            fake_images, _ = self.generator(noise, sampled_embeddings)
            
            # Move to CPU for metrics calculation
            fake_images = fake_images.cpu()
            real_images_sample = real_images[:num_samples].cpu()
            
            # Calculate FID score
            try:
                fid_score = self.metrics_calculator.calculate_fid(real_images_sample, fake_images)
                self.metrics['fid_scores'].append(fid_score)
                print(f"FID Score: {fid_score:.4f}")
            except Exception as e:
                print(f"Error calculating FID score: {e}")
            
            # Calculate Inception Score
            try:
                is_mean, is_std = self.metrics_calculator.calculate_inception_score(fake_images)
                self.metrics['inception_scores'].append(is_mean)
                print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
            except Exception as e:
                print(f"Error calculating Inception Score: {e}")
            
            # Calculate R-precision
            try:
                # Get feature embeddings of generated images
                r_precision = self.metrics_calculator.calculate_r_precision(
                    fake_images.detach().cpu().numpy(),
                    text_embeddings[:num_samples].cpu().numpy(),
                    k=self.config.R_PRECISION_K
                )
                self.metrics['r_precision'].append(r_precision)
                print(f"R-precision: {r_precision:.4f}")
            except Exception as e:
                print(f"Error calculating R-precision: {e}")
        
        self.generator.train()
    
    def plot_metrics(self, epoch):
        """Plot and save metrics"""
        # Only plot if we have enough data
        if len(self.metrics['g_losses']) < 2:
            return
            
        # Create directory
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Plot losses
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.metrics['g_losses'], label='Generator')
        plt.plot(self.metrics['d_losses'], label='Discriminator')
        plt.title('GAN Losses')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(self.metrics['kl_losses'], label='KL Divergence')
        plt.title('KL Divergence Loss')
        plt.legend()
        
        # Plot evaluation metrics if available
        if len(self.metrics['fid_scores']) > 0:
            plt.subplot(1, 3, 3)
            plt.plot(self.metrics['fid_scores'], label='FID Score')
            plt.title('FID Score (lower is better)')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, f'metrics_epoch_{epoch+1}.png'))
        plt.close()
        
        # Save metrics as pickle
        with open(os.path.join(self.metrics_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(self.metrics, f)

def save_model(model, path):
    """Save model weights"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load model weights"""
    model.load_state_dict(torch.load(path, map_location=model.device))


def train(config, num_epochs=10, skip_metrics=False):
    print(f"Training on device: {config.DEVICE}")
    
    # Create data manager
    data_manager = DataManager(config)
    
    # Visualize some samples
    data_manager.visualize_samples(num_samples=3)
    
    # Get training and validation data loaders
    train_loader, train_size = data_manager.get_data()
    val_loader, val_size = data_manager.get_test_data()
    
    print(f"Training dataset size: {train_size}")
    print(f"Validation dataset size: {val_size}")
    
    # Create trainer
    trainer = GANTrainer(config)
    
    # Get fixed embeddings for visualization
    fixed_embeddings = None
    real_images_sample = None  # For metrics calculation
    text_embeddings_sample = None  # For metrics calculation
    for real_images, embeddings in train_loader:
        # Store fixed examples for visualization
        fixed_embeddings = embeddings[:config.NUM_EXAMPLES].to(config.DEVICE)
        # Store samples for metrics calculation
        real_images_sample = real_images
        text_embeddings_sample = embeddings
        break
    
    # Start training loop
    steps_per_epoch = train_size // config.BATCH_SIZE
    print(f"Steps per epoch: {steps_per_epoch}")
    
    print("\n======= Starting training =======\n")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n===== EPOCH {epoch+1}/{num_epochs} =====")
        
        # Training phase
        trainer.generator.train()
        trainer.discriminator.train()
        
        g_losses = []
        d_losses = []
        kl_losses = []
        
        for step, (real_images, embeddings) in enumerate(train_loader):
            # Move data to device
            real_images = real_images.to(config.DEVICE)
            embeddings = embeddings.to(config.DEVICE)
            
            batch_size = real_images.size(0)
            
            # Train discriminator
            noise = torch.randn(batch_size, config.Z_DIM, device=config.DEVICE)
            d_loss = trainer.train_discriminator(real_images, embeddings, noise)
            d_losses.append(d_loss)
            
            # Train generator
            noise = torch.randn(batch_size, config.Z_DIM, device=config.DEVICE)
            g_loss, kl_loss = trainer.train_generator(embeddings, noise)
            g_losses.append(g_loss)
            kl_losses.append(kl_loss)
            
            # Print progress
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{steps_per_epoch}, "
                      f"G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, KL Loss: {kl_loss:.4f}")
        
        # Calculate epoch average losses
        g_loss_avg = np.mean(g_losses)
        d_loss_avg = np.mean(d_losses)
        kl_loss_avg = np.mean(kl_losses)
        
        # Store metrics
        trainer.metrics['g_losses'].append(g_loss_avg)
        trainer.metrics['d_losses'].append(d_loss_avg)
        trainer.metrics['kl_losses'].append(kl_loss_avg)
        
        # Compute evaluation metrics periodically if not skipped
        if not skip_metrics and (epoch + 1) % config.EVAL_INTERVAL == 0:
            trainer.compute_metrics(epoch + 1, real_images_sample, text_embeddings_sample, 
                                   num_samples=min(500, len(real_images_sample)))
        
        # Save generated samples
        trainer.save_samples(epoch, fixed_embeddings)
        
        # Plot metrics
        trainer.plot_metrics(epoch)
        
        # Save model checkpoint
        if (epoch + 1) % config.SNAPSHOT_INTERVAL == 0 or epoch == num_epochs - 1:
            save_model(trainer.generator, 
                      os.path.join(trainer.checkpoint_dir, f'generator_epoch_{epoch+1}.pt'))
            save_model(trainer.discriminator, 
                      os.path.join(trainer.checkpoint_dir, f'discriminator_epoch_{epoch+1}.pt'))
            print(f"Saved model checkpoint at epoch {epoch+1}")
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Average losses - G: {g_loss_avg:.4f}, D Loss: {d_loss_avg:.4f}, KL: {kl_loss_avg:.4f}")
    
    print("\n======= Training Complete =======")
    
    # Final evaluation if not skipped
    if not skip_metrics:
        print("\n======= Final Evaluation =======\n")
        trainer.compute_metrics(num_epochs, real_images_sample, text_embeddings_sample, 
                               num_samples=min(1000, len(real_images_sample)))

def main():
    # Set your dataset paths here
    images_path = "images"  # Directory containing bird class folders with images
    train_path = "train"    # Directory with training data embeddings
    test_path = "test"      # Directory with test data embeddings

    config = Config(images_path, train_path, test_path)
    
    # Print key configuration values for verification
    print("\nConfiguration:")
    print(f"Embedding dimension: {config.EMBEDDING_DIM}")
    print(f"Conditioning Augmentation dimension: {config.CA_DIM}")
    print(f"Batch size: {config.BATCH_SIZE} (as per paper)")
    print(f"Learning rate: {config.STAGE1_G_LR} (as per paper)")
    print(f"Image size: {config.STAGE1_IMAGE_SIZE}x{config.STAGE1_IMAGE_SIZE} (as per paper)")
    
    # Use the exact batch size from the paper without automatic adjustment
    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    
    # Adding skip_metrics=True to avoid inception model loading during initial testing
    train(config, num_epochs=5, skip_metrics=True)  # Start with a small number for testing

if __name__ == "__main__":
    # This is the important fix for the multiprocessing issue
    import multiprocessing
    multiprocessing.freeze_support()
    main()
