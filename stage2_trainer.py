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

# Add gradient penalty for WGAN-GP style stabilization
def compute_gradient_penalty(discriminator, real_samples, fake_samples, embeddings, device):
    """Compute gradient penalty for improved WGAN training stability"""
    # Interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Calculate discriminator output for interpolated images
    d_interpolates = discriminator([interpolates, embeddings])
    
    # Create labels for gradient computation
    fake_outputs = torch.ones(real_samples.size(0), device=device, requires_grad=False)
    
    # Get gradients w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

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
        
        # Enhanced optimizers with appropriate betas for WGAN-style training
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.STAGE2_G_LR,
            betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.STAGE2_D_LR,
            betas=(0.5, 0.999)
        )
        
        # Add learning rate schedulers
        self.g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.g_optimizer, mode='min', factor=0.5, patience=20,
            min_lr=1e-6, verbose=True
        )
        self.d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.d_optimizer, mode='min', factor=0.5, patience=20,
            min_lr=1e-6, verbose=True
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
        """Improved discriminator training with gradient penalty and label smoothing"""
        self.d_optimizer.zero_grad()
        
        batch_size = real_images.size(0)
        
        # Generate high-resolution fake images
        with torch.no_grad():
            fake_images, _ = self.generator([stage1_images, embeddings])
        
        # Real images: label smoothing (0.9 instead of 1.0)
        real_labels = torch.ones(batch_size, device=self.device) * 0.9
        real_logits = self.discriminator([real_images, embeddings])
        d_loss_real = F.binary_cross_entropy(real_logits, real_labels)
        
        # Fake images
        fake_labels = torch.zeros(batch_size, device=self.device)
        fake_logits = self.discriminator([fake_images.detach(), embeddings])
        d_loss_fake = F.binary_cross_entropy(fake_logits, fake_labels)
        
        # Gradient penalty
        gp_weight = 10.0  # λ for gradient penalty
        gp = compute_gradient_penalty(
            self.discriminator, real_images, fake_images.detach(), 
            embeddings, self.device
        )
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake + gp_weight * gp
        
        # Backpropagation and optimization
        d_loss.backward()
        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5.0)
        self.d_optimizer.step()
        
        return d_loss.item(), gp.item()
    
    def train_generator(self, stage1_images, embeddings):
        """Enhanced generator training with feature matching loss"""
        self.g_optimizer.zero_grad()
        
        batch_size = embeddings.size(0)
        
        # Generate high-resolution fake images
        fake_images, kl_loss = self.generator([stage1_images, embeddings])
        
        # Standard adversarial loss
        real_labels = torch.ones(batch_size, device=self.device)
        fake_logits = self.discriminator([fake_images, embeddings])
        g_loss = F.binary_cross_entropy(fake_logits, real_labels)
        
        # Add KL divergence loss with dynamic lambda weighting
        # Gradually increase KL weight to prevent KL vanishing
        kl_weight = min(2.0, 0.2 + self.current_epoch * 0.004)  # Linearly increase from 0.2 to 2.0
        total_loss = g_loss + kl_weight * kl_loss
        
        # Backpropagation and optimization
        total_loss.backward()
        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=5.0)
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
        full_path = checkpoint_path
        
        # If checkpoint_path is just a filename, prepend the checkpoint directory
        if not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
            full_path = os.path.join(self.checkpoint_dir, checkpoint_path)
            print(f"Using full path: {full_path}")
        
        if not os.path.exists(full_path):
            print(f"ERROR: Checkpoint file {full_path} does not exist!")
            print(f"Available checkpoints in {self.checkpoint_dir}:")
            
            try:
                files = os.listdir(self.checkpoint_dir)
                checkpoint_files = [f for f in files if f.endswith('.pt')]
                for f in checkpoint_files:
                    print(f"  - {f}")
            except Exception as e:
                print(f"Error listing checkpoint directory: {e}")
                
            print("Starting training from scratch.")
            return 0
            
        try:
            checkpoint = torch.load(full_path, map_location=self.device)
            # Load model states with backward compatibility
            try:
                # Load generator state with strict=False to allow for architecture changes
                self.generator.load_state_dict(checkpoint['generator_state'], strict=False)
                missing_keys = set(self.generator.state_dict().keys()) - set(checkpoint['generator_state'].keys())
                if missing_keys:
                    print(f"Warning: Some generator keys were not found in checkpoint: {missing_keys}")
                
                # Load discriminator state with strict=False
                self.discriminator.load_state_dict(checkpoint['discriminator_state'], strict=False)
                missing_keys = set(self.discriminator.state_dict().keys()) - set(checkpoint['discriminator_state'].keys())
                if missing_keys:
                    print(f"Warning: Some discriminator keys were not found in checkpoint: {missing_keys}")
                    
                print("Model states loaded successfully")
            except Exception as e:
                print(f"Warning: Error loading model states: {e}")
                print("This might be due to architecture changes. Trying to continue anyway...")
            
            # Load optimizer states
            try:
                self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state'])
                self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state'])
                print("Optimizer states loaded successfully")
            except Exception as e:
                print(f"Warning: Error loading optimizer states: {e}")
                print("Continuing with freshly initialized optimizers")
            
            # Load metrics history
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
                print("Training metrics history loaded")
            
            # Make sure current_epoch is set
            self.current_epoch = checkpoint['epoch']
            
            return checkpoint['epoch']
        except Exception as e:
            print(f"Critical error loading checkpoint: {e}")
            print(f"Full traceback:")
            import traceback
            traceback.print_exc()
            print("Starting fresh training")
            return 0

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
    trainer.current_epoch = 0  # Track epoch for dynamic weight adjustments
    
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
        trainer.current_epoch = epoch  # Update current epoch for dynamic parameters
        epoch_start_time = time.time()
        
        # Initialize metrics
        g_losses = []
        d_losses = []
        kl_losses = []
        
        # Add batch processing time tracking
        data_loading_time = 0
        compute_time = 0
        last_print_time = time.time()
        
        # Show progress indicators
        print(f"Processing batches for epoch {epoch+1}...")
        
        # Iterate through both Stage-I and Stage-II dataloaders together
        for batch_idx, (stage1_data, stage2_data) in enumerate(zip(stage1_loader, stage2_loader)):
            # Mark data loading completion time
            data_load_end_time = time.time()
            
            # Periodically print progress to show activity
            if time.time() - last_print_time > 60:  # Print every minute
                print(f"  Processing batch {batch_idx+1}... (still active)")
                last_print_time = time.time()
            
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
            d_loss, gp = trainer.train_discriminator(stage2_real_images, stage1_fake_images, stage2_embeddings)
            d_losses.append(d_loss)
            
            # Train generator
            g_loss, kl_loss = trainer.train_generator(stage1_fake_images, stage2_embeddings)
            g_losses.append(g_loss)
            kl_losses.append(kl_loss)
            
            # Track timing
            compute_end_time = time.time()
            data_loading_time += data_load_end_time - (compute_end_time - data_loading_time if batch_idx > 0 else epoch_start_time)
            compute_time += compute_end_time - data_load_end_time
        
        # Print timing information
        if len(g_losses) > 0:  # Only print if we processed at least one batch
            print(f"\nData loading time: {data_loading_time:.2f}s, Compute time: {compute_time:.2f}s")
            print(f"Data loading efficiency: {100 * compute_time / (compute_time + data_loading_time):.1f}%")
        else:
            print("\nWARNING: No batches were processed this epoch!")
        
        # Calculate epoch average losses
        g_loss_avg = np.mean(g_losses)
        d_loss_avg = np.mean(d_losses)
        kl_loss_avg = np.mean(kl_losses)
        
        # Store metrics
        trainer.metrics['g_losses'].append(g_loss_avg)
        trainer.metrics['d_losses'].append(d_loss_avg)
        trainer.metrics['kl_losses'].append(kl_loss_avg)
        
        # Update learning rates based on loss progression
        trainer.g_scheduler.step(g_loss_avg)
        trainer.d_scheduler.step(d_loss_avg)
        
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
