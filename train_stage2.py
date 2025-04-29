import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys

from stage1 import Config, DataManager
from stage2_trainer import train_stage2_gan

def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage-II GAN for StackGAN')
    parser.add_argument('--images_path', type=str, default='images',
                        help='Path to the images directory')
    parser.add_argument('--train_path', type=str, default='train',
                        help='Path to training data')
    parser.add_argument('--test_path', type=str, default='test',
                        help='Path to test data')
    parser.add_argument('--epochs', type=int, default=600,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--stage1_checkpoint', type=str, default='best_model.pt',
                        help='Specific Stage-I checkpoint to use (e.g., checkpoint_epoch_553.pt for IS or checkpoint_epoch_546.pt for FID)')
    return parser.parse_args()

def extend_config_for_stage2(config):
    """Add Stage-II specific parameters to config"""
    # Set up Stage-II hyperparameters exactly as in StackGAN paper
    config.STAGE2_G_HDIM = 128  # Generator filter sizes
    config.STAGE2_D_HDIM = 64   # Discriminator filter sizes
    config.STAGE2_G_LR = 0.0002  # Learning rate from paper
    config.STAGE2_D_LR = 0.0002  # Learning rate from paper
    config.STAGE2_IMAGE_SIZE = 256  # High-resolution output size
    config.LAMBDA = 2.0  # KL regularization parameter
    
    # Same as Stage-I for consistency
    config.CA_DIM = 128  # Conditioning augmentation output dimension
    config.BETA1 = 0.5  # Adam optimizer beta1 parameter
    config.BETA2 = 0.999  # Adam optimizer beta2 parameter
    
    return config

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Configure data paths
    images_path = args.images_path
    train_path = args.train_path
    test_path = args.test_path
    
    # Create and configure config object
    config = Config(images_path, train_path, test_path)
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config = extend_config_for_stage2(config)
    
    print("\nStage-II GAN Configuration:")
    print(f"Embedding dimension: {config.EMBEDDING_DIM}")
    print(f"Conditioning Augmentation dimension: {config.CA_DIM}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rates: G={config.STAGE2_G_LR}, D={config.STAGE2_D_LR}")
    print(f"Image sizes: Stage-I={config.STAGE1_IMAGE_SIZE}x{config.STAGE1_IMAGE_SIZE}, "
          f"Stage-II={config.STAGE2_IMAGE_SIZE}x{config.STAGE2_IMAGE_SIZE}\n")
    
    # Create data manager and get dataloaders
    data_manager = DataManager(config)
    
    # Create Stage-I dataloader for getting low-res images
    train_stage1_loader, _ = data_manager.get_data()
    
    # Create custom transform for Stage-II (high-res) images
    from torchvision import transforms
    stage2_transform = transforms.Compose([
        transforms.Resize((config.STAGE2_IMAGE_SIZE, config.STAGE2_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Create Stage-II dataloader with high-res images
    from torch.utils.data import DataLoader
    from stage1 import BirdsDataset
    
    # Create Stage-II dataset with higher resolution
    train_stage2_dataset = BirdsDataset(
        config,
        data_manager.image_dir,
        data_manager.embeddings,
        data_manager.filenames,
        stage2_transform
    )
    
    # Create DataLoader
    train_stage2_loader = DataLoader(
        train_stage2_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Increase from 4 to 8
        pin_memory=True,
        prefetch_factor=2,  # Add prefetching 
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True
    )
    
    # Start training Stage-II GAN
    train_stage2_gan(config, train_stage1_loader, train_stage2_loader, resume_checkpoint=args.resume, stage1_checkpoint=args.stage1_checkpoint)

if __name__ == "__main__":
    main()
