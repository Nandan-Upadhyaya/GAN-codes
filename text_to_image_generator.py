"""
Text-to-Image generator using trained Stage-I GAN and char-CNN-RNN text encoder.
This script allows generating 64x64 bird images from text descriptions.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
from PIL import Image

# Import from stage1.py for consistent model definitions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage1 import StageIGenerator, Config

# Import char-CNN-RNN implementation
from char_cnn_rnn.char_cnn_rnn import char_cnn_rnn, prepare_text

class TextEncoder:
    """Text encoder using char-CNN-RNN PyTorch model"""
    def __init__(self, device=None):
        """Initialize the char-CNN-RNN model"""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for text encoding")
        
        # Initialize char-CNN-RNN model for birds dataset using ICML model type
        print("Initializing char-CNN-RNN model...")
        self.model = char_cnn_rnn('birds', 'icml').to(self.device)
        self.model.eval()  # Set to evaluation mode
        print("Text encoder initialized")
    
    def encode_text(self, descriptions):
        """Encode text descriptions to embeddings"""
        embeddings = []
        
        # Process each description
        for desc in descriptions:
            # Prepare text (convert to one-hot tensor)
            one_hot = prepare_text(desc)
            one_hot = one_hot.to(self.device)
            
            # Add batch dimension
            one_hot = one_hot.unsqueeze(0)
            
            # Forward pass to get embedding
            with torch.no_grad():
                embedding = self.model(one_hot)
            
            # Collect embedding
            embeddings.append(embedding.cpu())
        
        # Stack embeddings
        return torch.cat(embeddings, dim=0)

def expand_embedding_dim(embeddings, target_dim=10240):
    """
    Expand embeddings from char-CNN-RNN output dimension (1024) to
    StackGAN expected dimension (10240).
    
    This function simulates the 10x concatenation that was likely used in training.
    """
    # Current dimension should be 1024
    current_dim = embeddings.size(1)
    
    # Calculate repeat factor
    repeat_factor = target_dim // current_dim
    if target_dim % current_dim != 0:
        print(f"Warning: Target dimension {target_dim} is not a multiple of current dimension {current_dim}")
        repeat_factor = target_dim // current_dim + 1
    
    # Repeat embeddings and truncate to target dimension
    expanded = embeddings.repeat(1, repeat_factor)[:, :target_dim]
    
    return expanded

class ImageGenerator:
    """Class for generating images from text using trained Stage-I GAN"""
    def __init__(self, checkpoint_path, device=None):
        """Initialize with path to best checkpoint"""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for image generation")
        
        # Create config
        self.config = Config()
        
        # Load generator model
        self.generator = StageIGenerator(self.config).to(self.device)
        self.load_checkpoint(checkpoint_path)
        self.generator.eval()
        
    def load_checkpoint(self, checkpoint_path):
        """Load generator weights from checkpoint"""
        print(f"Loading generator from checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract generator state dict
            if 'generator_state' in checkpoint:
                self.generator.load_state_dict(checkpoint['generator_state'])
                print("Generator loaded from full checkpoint")
            elif 'state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['state_dict'])
                print("Generator loaded from state_dict")
            else:
                self.generator.load_state_dict(checkpoint)
                print("Generator loaded from direct weights")
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
            
    def generate_images(self, embeddings, num_images=1, noise=None):
        """Generate images using text embeddings"""
        # Move embeddings to device
        embeddings = embeddings.to(self.device)
        
        # Generate multiple images per text if requested
        batch_size = embeddings.size(0)
        images_list = []
        
        with torch.no_grad():
            for i in range(num_images):
                # Generate random noise if not provided
                if noise is None:
                    z = torch.randn(batch_size, self.config.Z_DIM, device=self.device)
                else:
                    z = noise.to(self.device)
                    
                # Generate images
                fake_images, _ = self.generator(z, embeddings)
                
                # Detach and move to CPU
                fake_images = fake_images.cpu().detach()
                
                # Denormalize from [-1, 1] to [0, 1]
                fake_images = (fake_images + 1) / 2.0
                
                images_list.append(fake_images)
        
        # Return tensor of shape [num_images, batch_size, C, H, W]
        return torch.stack(images_list)
        
def display_images(images, texts=None, save_path=None):
    """Display generated images with their text descriptions"""
    # Convert from tensor to numpy if needed
    if isinstance(images, torch.Tensor):
        images = images.numpy()
    
    # Handle different input shapes
    if images.ndim == 5:  # [num_per_text, batch_size, C, H, W]
        num_per_text, batch_size = images.shape[:2]
    else:  # [batch_size, C, H, W]
        num_per_text, batch_size = 1, images.shape[0]
        images = images.reshape(1, batch_size, *images.shape[1:])
    
    # Create figure with proper layout
    fig, axes = plt.subplots(batch_size, num_per_text, 
                             figsize=(num_per_text * 3, batch_size * 3.5))
    
    # Handle single image case
    if batch_size == 1 and num_per_text == 1:
        axes = np.array([[axes]])
    elif batch_size == 1:
        axes = axes.reshape(1, -1)
    elif num_per_text == 1:
        axes = axes.reshape(-1, 1)
        
    # Display each image
    for i in range(batch_size):
        for j in range(num_per_text):
            # Get image and convert from [C,H,W] to [H,W,C]
            img = np.transpose(images[j, i], (1, 2, 0))
            # Clip to valid range
            img = np.clip(img, 0, 1)
            
            # Display image
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            
            # Add text description as title for first image in each row
            if j == 0 and texts is not None and i < len(texts):
                axes[i, j].set_title(texts[i], fontsize=10, wrap=True)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Images saved to {save_path}")
        
    plt.show()

def main():
    """Main function for text-to-image generation"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate bird images from text descriptions')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/birds/best_model.pt',
                        help='Path to the checkpoint file')
    parser.add_argument('--descriptions', nargs='*', type=str,
                        help='Text descriptions to generate images from')
    parser.add_argument('--num_images', type=int, default=3,
                        help='Number of images to generate per description')
    
    args = parser.parse_args()
    
    # 1. Initialize text encoder
    text_encoder = TextEncoder()
    
    # 2. Initialize image generator
    image_generator = ImageGenerator(args.checkpoint_path)
    
    # 3. Get text descriptions from user if not provided
    descriptions = args.descriptions
    if not descriptions:
        print("\nEnter bird descriptions (one per line, empty line to finish):")
        descriptions = []
        while True:
            line = input()
            if not line:
                break
            descriptions.append(line)
    
    if not descriptions:
        descriptions = [
            "a yellow bird with black wings",
            "a bright red bird with a pointy beak",
            "a small blue bird with a short beak"
        ]
        print("\nUsing default descriptions:")
        for desc in descriptions:
            print(f"- {desc}")
    
    # 4. Encode text descriptions
    print("\nEncoding text descriptions...")
    embeddings = text_encoder.encode_text(descriptions)
    
    # 5. Expand embeddings from 1024-dim to 10240-dim
    print("Expanding embeddings to match training dimension...")
    expanded_embeddings = expand_embedding_dim(embeddings, target_dim=10240)
    
    # 6. Generate images
    print(f"\nGenerating {args.num_images} image(s) per description...")
    images = image_generator.generate_images(expanded_embeddings, num_images=args.num_images)
    
    # 7. Display results
    print("\nDisplaying generated images...")
    timestamp = int(time.time())
    save_path = f"generated_birds_{timestamp}.png"
    display_images(images, descriptions, save_path=save_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
