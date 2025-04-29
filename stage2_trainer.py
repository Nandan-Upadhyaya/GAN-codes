import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import scipy
import torchvision
import matplotlib

from stage2_models import StageIIGenerator, StageIIDiscriminator
from stage1 import StageIGenerator, Config, GANMetrics


    
class Stage2Trainer:
    def __init__(self, config, stage1_checkpoint=None):
        self.config = config
        self.device = config.DEVICE
        self.checkpoint_dir = os.path.join('checkpoints', config.DATASET_NAME, 'stage2')  # Separate folder for Stage-II
        self.sample_dir = os.path.join('samples', config.DATASET_NAME, 'stage2')  # Separate folder for Stage-II
        self.metrics_dir = os.path.join('metrics', config.DATASET_NAME, 'stage2')  # Separate folder for Stage-II
        self.stage1_checkpoint_dir = os.path.join('checkpoints', config.DATASET_NAME)
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Store the specified checkpoint name
        self.stage1_checkpoint = stage1_checkpoint
        
        # Initialize models
        self.stage1_generator = StageIGenerator(config).to(self.device)
        self.generator = StageIIGenerator(config).to(self.device)
        self.discriminator = StageIIDiscriminator(config).to(self.device)
        
        # Initialize optimizers as in paper (Adam with beta1=0.5, beta2=0.999)
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.STAGE2_G_LR,
            betas=(config.BETA1, config.BETA2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.STAGE2_D_LR,
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
        self.metrics_calculator = GANMetrics(config, lazy_load=True)
        
        # Load pre-trained Stage-I Generator
        self._load_stage1_generator()
        
        print("Stage-II Trainer initialization complete")
        
    def _load_stage1_generator(self):
        """Load pre-trained Stage-I Generator"""
        # If specific checkpoint is provided, use it
        if self.stage1_checkpoint:
            checkpoint_path = os.path.join(self.stage1_checkpoint_dir, self.stage1_checkpoint)
            if os.path.exists(checkpoint_path):
                print(f"Loading specified Stage-I Generator from: {checkpoint_path}")
            else:
                print(f"Specified checkpoint {checkpoint_path} not found, falling back to default selection")
                self.stage1_checkpoint = None
        
        # Otherwise use default logic to find best checkpoint
        if not self.stage1_checkpoint:
            # Find latest checkpoint in the checkpoint directory
            stage1_checkpoints = [f for f in os.listdir(self.stage1_checkpoint_dir) 
                                if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
            
            if not stage1_checkpoints:
                raise FileNotFoundError(f"No Stage-I generator checkpoints found in {self.stage1_checkpoint_dir}")
                
            # Find the best checkpoint or latest one
            best_path = os.path.join(self.stage1_checkpoint_dir, 'best_model.pt')
            if os.path.exists(best_path):
                checkpoint_path = best_path
            else:
                # Sort checkpoints by epoch number
                epochs = [int(f.split('_')[-1].split('.')[0]) for f in stage1_checkpoints]
                latest_idx = np.argmax(epochs)
                checkpoint_path = os.path.join(self.stage1_checkpoint_dir, stage1_checkpoints[latest_idx])
        
        print(f"Loading Stage-I Generator from: {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load Stage-I Generator weights
        if 'generator_state' in checkpoint:
            self.stage1_generator.load_state_dict(checkpoint['generator_state'])
        else:
            raise KeyError("Could not find generator_state in checkpoint")
            
        # Set to evaluation mode
        self.stage1_generator.eval()
        
        print("Stage-I Generator loaded successfully")
    
    def train_discriminator(self, real_images, stage1_images, embeddings):
        """Train the discriminator for one step using binary cross-entropy loss"""
        self.d_optimizer.zero_grad()
        
        batch_size = real_images.size(0)
        
        # Generate high-resolution fake images
        fake_images, _ = self.generator([stage1_images, embeddings])
        
        # Real images: label = 1
        real_labels = torch.ones(batch_size, device=self.device)
        real_logits = self.discriminator([real_images, embeddings])
        d_loss_real = F.binary_cross_entropy(real_logits, real_labels)
        
        # Fake images: label = 0
        fake_labels = torch.zeros(batch_size, device=self.device)
        fake_logits = self.discriminator([fake_images.detach(), embeddings])
        d_loss_fake = F.binary_cross_entropy(fake_logits, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        
        # Backpropagation and optimization
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def train_generator(self, stage1_images, embeddings):
        """Train the generator for one step using binary cross-entropy loss + KL divergence regularization"""
        self.g_optimizer.zero_grad()
        
        batch_size = embeddings.size(0)
        
        # Generate high-resolution fake images
        fake_images, kl_loss = self.generator([stage1_images, embeddings])
        
        # Compute generator loss - fool the discriminator
        real_labels = torch.ones(batch_size, device=self.device)
        fake_logits = self.discriminator([fake_images, embeddings])
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
        self.stage1_generator.eval()
        
        with torch.no_grad():
            # Generate Stage-I images
            stage1_images, _ = self.stage1_generator(self.fixed_noise, fixed_embeddings)
            
            # Generate Stage-II images
            fake_images, _ = self.generator([stage1_images, fixed_embeddings])
            
            # Convert to numpy and denormalize
            stage1_images = stage1_images.cpu().numpy()
            fake_images = fake_images.cpu().numpy()
            
            stage1_images = (stage1_images + 1) / 2.0  # [-1, 1] -> [0, 1]
            fake_images = (fake_images + 1) / 2.0  # [-1, 1] -> [0, 1]
            
            # Save as grid showing Stage-I and Stage-II side by side
            fig, axs = plt.subplots(self.config.NUM_EXAMPLES, 2, figsize=(8, 2*self.config.NUM_EXAMPLES))
            
            for i in range(self.config.NUM_EXAMPLES):
                # Stage-I image
                axs[i, 0].imshow(np.transpose(stage1_images[i], (1, 2, 0)))
                axs[i, 0].set_title("Stage-I (64x64)")
                axs[i, 0].axis('off')
                
                # Stage-II image
                axs[i, 1].imshow(np.transpose(fake_images[i], (1, 2, 0)))
                axs[i, 1].set_title("Stage-II (256x256)")
                axs[i, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.sample_dir, f'stage2_samples_epoch_{epoch}.png'))
            plt.close()
            
        self.generator.train()
    
    def compute_metrics(self, epoch, real_images, text_embeddings, stage1_images, num_samples=500):
        """Compute evaluation metrics"""
        print(f"\nComputing metrics for epoch {epoch}...")
        
        # Generate fake images for metrics calculation
        self.generator.eval()
        with torch.no_grad():
            # Generate high-resolution fake images
            fake_images, _ = self.generator([stage1_images[:num_samples], text_embeddings[:num_samples]])
            
            # Move to CPU for metrics calculation
            fake_images = fake_images.cpu()
            real_images_sample = real_images[:num_samples].cpu()
            
            fid_score = None
            is_mean = None
            is_std = None
            
            # Calculate FID score
            try:
                fid_score = self.metrics_calculator.calculate_fid(real_images_sample, fake_images)
                self.metrics['fid_scores'].append(fid_score)
                # Remove individual print statement
            except Exception as e:
                print(f"Error calculating FID score: {e}")
            
            # Calculate Inception Score
            try:
                is_mean, is_std = self.metrics_calculator.calculate_inception_score(fake_images)
                self.metrics['inception_scores'].append(is_mean)
                # Remove individual print statement
            except Exception as e:
                print(f"Error calculating Inception Score: {e}")
            
            # Print metrics in a single line if available
            if fid_score is not None and is_mean is not None:
                print(f"Epoch {epoch} metrics: Inception Score: {is_mean:.4f} ± {is_std:.4f}, FID: {fid_score:.4f}")
                
        self.generator.train()
    
    def save_checkpoint(self, epoch):
        """Save checkpoint for resuming training"""
        checkpoint = {
            'epoch': epoch,
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'metrics': self.metrics
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")
        
        # Also save as latest checkpoint for easy resuming
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best model based on FID score (lower is better)
        if len(self.metrics['fid_scores']) > 0:
            current_fid = self.metrics['fid_scores'][-1]
            if len(self.metrics['fid_scores']) == 1 or current_fid < min(self.metrics['fid_scores'][:-1]):
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                print(f"Saved best model with FID score: {current_fid:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        
        # Load optimizer states
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state'])
        
        # Load metrics history and other tracking variables
        self.metrics = checkpoint['metrics']
        
        return checkpoint['epoch']

def resize_images(images, target_size):
    """Resize a batch of images from Stage-I (64x64) to Stage-II input size (256x256)"""
    resized = F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=True)
    return resized

def train_stage2_gan(config, stage1_loader, stage2_loader, resume_checkpoint=None, 
                    stage1_checkpoint='best_model.pt', verbose=True):
    # Only print system and configuration information if verbose is True
    if verbose:
        import torch
        import torchvision
        import numpy as np
        import scipy
        import matplotlib
        
    
    print(f"Training Stage-II GAN on device: {config.DEVICE}")
    
    # Create trainer with specified Stage-I checkpoint
    trainer = Stage2Trainer(config, stage1_checkpoint)
    
    # Handle resume training
    start_epoch = 0
    checkpoint_dir = os.path.join('checkpoints', config.DATASET_NAME, 'stage2')
    
    # If specific checkpoint is provided, use it
    if resume_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, resume_checkpoint)
        if os.path.exists(checkpoint_path):
            start_epoch = trainer.load_checkpoint(checkpoint_path)
            print(f"Resuming training from epoch {start_epoch+1}")
        else:
            print(f"Checkpoint {checkpoint_path} not found, starting from epoch 0")
    # Otherwise, try to load the latest checkpoint automatically
    else:
        latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(latest_path):
            start_epoch = trainer.load_checkpoint(latest_path)
            print(f"Automatically resuming from latest checkpoint (epoch {start_epoch+1})")
        else:
            print("No checkpoint found, starting from epoch 0")
    
    # Get fixed embeddings and images for visualization
    fixed_embeddings = None
    real_images_sample = None
    text_embeddings_sample = None
    stage1_images_sample = None
    
    for stage1_data, stage2_data in zip(stage1_loader, stage2_loader):
        # Get Stage-I data
        _, stage1_embeddings = stage1_data
        
        # Get Stage-II data
        stage2_real_images, stage2_embeddings = stage2_data
        
        # Store fixed examples for visualization
        fixed_embeddings = stage1_embeddings[:config.NUM_EXAMPLES].to(config.DEVICE)
        
        # Store samples for metrics calculation
        with torch.no_grad():
            # Generate Stage-I images
            noise = torch.randn(stage1_embeddings.size(0), config.Z_DIM, device=config.DEVICE)
            stage1_fake_images, _ = trainer.stage1_generator(noise, stage1_embeddings.to(config.DEVICE))
            
        real_images_sample = stage2_real_images
        text_embeddings_sample = stage2_embeddings
        stage1_images_sample = stage1_fake_images.cpu()
        
        break
    
    # Start training loop
    total_epochs = config.EPOCHS
    print("\n======= Starting Stage-II GAN Training =======\n")
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        
        # Initialize metrics
        g_losses = []
        d_losses = []
        kl_losses = []
        
        # Iterate through both Stage-I and Stage-II dataloaders together
        for stage1_data, stage2_data in zip(stage1_loader, stage2_loader):
            # Get Stage-I data (we only need embeddings)
            _, stage1_embeddings = stage1_data
            stage1_embeddings = stage1_embeddings.to(config.DEVICE)
            
            # Get Stage-II data
            stage2_real_images, stage2_embeddings = stage2_data
            stage2_real_images = stage2_real_images.to(config.DEVICE)
            stage2_embeddings = stage2_embeddings.to(config.DEVICE)
            
            batch_size = stage2_real_images.size(0)
            
            # Generate Stage-I images
            noise = torch.randn(batch_size, config.Z_DIM, device=config.DEVICE)
            with torch.no_grad():
                stage1_fake_images, _ = trainer.stage1_generator(noise, stage1_embeddings)
            
            # Train discriminator
            d_loss = trainer.train_discriminator(stage2_real_images, stage1_fake_images, stage2_embeddings)
            d_losses.append(d_loss)
            
            # Train generator
            g_loss, kl_loss = trainer.train_generator(stage1_fake_images, stage2_embeddings)
            g_losses.append(g_loss)
            kl_losses.append(kl_loss)
        
        # Calculate epoch average losses
        g_loss_avg = np.mean(g_losses)
        d_loss_avg = np.mean(d_losses)
        kl_loss_avg = np.mean(kl_losses)
        
        # Store metrics
        trainer.metrics['g_losses'].append(g_loss_avg)
        trainer.metrics['d_losses'].append(d_loss_avg)
        trainer.metrics['kl_losses'].append(kl_loss_avg)
        
        # Print training information
        print(f"Epoch {epoch+1} (training) : Gen loss: {g_loss_avg:.4f}, disc loss: {d_loss_avg:.4f}, KL: {kl_loss_avg:.4f}")
        
        # Save generated samples after every epoch
        trainer.save_samples(epoch+1, fixed_embeddings)
        
        # Compute evaluation metrics periodically
        if (epoch + 1) % config.EVAL_INTERVAL == 0:
            trainer.compute_metrics(
                epoch + 1,
                real_images_sample.to(config.DEVICE),
                text_embeddings_sample.to(config.DEVICE),
                stage1_images_sample.to(config.DEVICE)
            )
            
            # Fix conditional check to not require r_precision
            if trainer.metrics['inception_scores'] and trainer.metrics['fid_scores']:
                is_score = trainer.metrics['inception_scores'][-1]
                fid_score = trainer.metrics['fid_scores'][-1]
                print(f"Epoch {epoch+1} (validation) : Inception Score: {is_score:.4f}, FID: {fid_score:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % config.SNAPSHOT_INTERVAL == 0 or epoch == total_epochs - 1:
            trainer.save_checkpoint(epoch + 1)
        
        # Print epoch completion information
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{total_epochs} completed in {epoch_time:.2f}s")
    
    print("\n======= Stage-II GAN Training Complete =======")
