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
import re

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
        
        # Initialize color enhancement parameters
        self.color_terms = {
            'black': 0.95, 'white': 0.95, 'red': 0.9, 'green': 0.9, 'blue': 0.9,
            'yellow': 0.9, 'orange': 0.9, 'purple': 0.9, 'brown': 0.85, 'grey': 0.85,
            'gray': 0.85, 'pink': 0.8, 'tan': 0.8, 'golden': 0.8, 'silver': 0.8,
            'dark': 0.7, 'light': 0.7, 'bright': 0.7, 'pale': 0.7
        }
    
    def encode_text(self, descriptions):
        """Encode text descriptions to embeddings"""
        embeddings = []
        color_weights = []
        
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
            
            # Extract color weight for this description
            color_weight = self._extract_color_weight(desc)
            color_weights.append(color_weight)
            
            # Collect embedding
            embeddings.append(embedding.cpu())
        
        # Stack embeddings
        raw_embeddings = torch.cat(embeddings, dim=0)
        
        # Apply normalization and feature enhancement
        enhanced_embeddings = self._enhance_embeddings(raw_embeddings, color_weights)
        
        return enhanced_embeddings
        
    def _extract_color_weight(self, text):
        """Extract color importance weight from text description"""
        text = text.lower()
        weight = 0.0
        
        # Check for color terms
        for color, color_weight in self.color_terms.items():
            if color in text:
                # Higher weight if color appears at the beginning of the text
                position_factor = 1.2 if re.search(r'^\s*\w*\s*' + color, text) else 1.0
                weight = max(weight, color_weight * position_factor)
                
        # Return at least a minimum weight
        return max(weight, 0.5)
    
    def _enhance_embeddings(self, embeddings, color_weights):
        """Apply normalization and enhancement techniques to embeddings"""
        # 1. Feature-wise normalization (similar to batch norm without learned parameters)
        mean = embeddings.mean(dim=1, keepdim=True)
        std = embeddings.std(dim=1, keepdim=True) + 1e-8
        normalized = (embeddings - mean) / std
        
        # 2. Apply non-linear activation to enhance feature separation
        enhanced = torch.tanh(normalized * 1.2)
        
        # 3. Apply color weighting
        color_weights = torch.tensor(color_weights).view(-1, 1)
        color_enhanced = enhanced * (1.0 + 0.5 * color_weights)
        
        # 4. Secondary normalization to control magnitude
        final_mean = color_enhanced.mean(dim=1, keepdim=True)
        final_std = color_enhanced.std(dim=1, keepdim=True) + 1e-8
        final_normalized = (color_enhanced - final_mean) / final_std
        
        return final_normalized

def enhanced_embedding_expansion(embeddings, target_dim=10240):
    """
    Enhanced embedding expansion that better preserves semantic information.
    This improves upon simple repetition with a more sophisticated approach
    that might better match what was done in the original paper.
    """
    # Current dimension should be 1024
    current_dim = embeddings.size(1)
    batch_size = embeddings.size(0)
    
    # Calculate number of splits for feature groups
    num_splits = 8  # Split into 8 feature groups
    split_size = current_dim // num_splits
    
    # Create output tensor
    expanded = torch.zeros(batch_size, target_dim)
    
    # Process each feature group differently
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_splits - 1 else current_dim
        features = embeddings[:, start_idx:end_idx]
        
        # Different processing for different feature groups
        if i % 4 == 0:  # Group 1: Simple repeat (for global features)
            section = features.repeat(1, target_dim // current_dim * 2)[:, :target_dim // num_splits]
        elif i % 4 == 1:  # Group 2: Scale and repeat (for color features)
            scaled = features * 1.5  # Emphasize these features
            section = scaled.repeat(1, target_dim // current_dim * 2)[:, :target_dim // num_splits]
        elif i % 4 == 2:  # Group 3: Power scaling for non-linear emphasis
            powered = torch.sign(features) * torch.abs(features) ** 0.8
            section = powered.repeat(1, target_dim // current_dim * 2)[:, :target_dim // num_splits]
        else:  # Group 4: Normalization and repeat
            norm = torch.nn.functional.normalize(features, p=2, dim=1)
            section = norm.repeat(1, target_dim // current_dim * 2)[:, :target_dim // num_splits]
        
        # Place in output tensor
        target_start = i * (target_dim // num_splits)
        target_end = (i + 1) * (target_dim // num_splits)
        expanded[:, target_start:target_end] = section
    
    # Final normalization
    output = torch.nn.functional.normalize(expanded, p=2, dim=1) * 10.0
    
    return output

class ImageGenerator:
    """Class for generating images from text using trained Stage-I GAN"""
    def __init__(self, checkpoint_path1, checkpoint_path2=None, device=None):
        """Initialize with path to best checkpoint(s)"""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for image generation")
        
        # Create config
        self.config = Config()
        
        # Load primary generator model
        self.generator1 = StageIGenerator(self.config).to(self.device)
        self.load_checkpoint(self.generator1, checkpoint_path1)
        self.generator1.eval()
        print(f"Primary generator loaded from {checkpoint_path1}")
        
        # Load secondary generator model if provided
        self.generator2 = None
        if checkpoint_path2:
            self.generator2 = StageIGenerator(self.config).to(self.device)
            self.load_checkpoint(self.generator2, checkpoint_path2)
            self.generator2.eval()
            print(f"Secondary generator loaded from {checkpoint_path2}")
        
    def load_checkpoint(self, model, checkpoint_path):
        """Load generator weights from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract generator state dict
            if 'generator_state' in checkpoint:
                model.load_state_dict(checkpoint['generator_state'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
            
    def generate_images(self, embeddings, num_images=1, noise=None, use_both_models=True):
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
                
                # Generate with primary model
                fake_images1, _ = self.generator1(z, embeddings)
                fake_images1_cpu = fake_images1.cpu().detach()
                fake_images1_cpu = (fake_images1_cpu + 1) / 2.0  # Convert to [0, 1]
                
                # Use both models if available and requested
                if self.generator2 is not None and use_both_models:
                    # Generate with secondary model
                    fake_images2, _ = self.generator2(z, embeddings)
                    fake_images2_cpu = fake_images2.cpu().detach()
                    fake_images2_cpu = (fake_images2_cpu + 1) / 2.0  # Convert to [0, 1]
                    
                    # Select better image based on color variance heuristic
                    selected_images = self._select_better_images(fake_images1_cpu, fake_images2_cpu)
                    images_list.append(selected_images)
                else:
                    images_list.append(fake_images1_cpu)
        
        # Return tensor of shape [num_images, batch_size, C, H, W]
        return torch.stack(images_list)
    
    def _select_better_images(self, images1, images2):
        """Select better images based on color variance heuristic"""
        batch_size = images1.size(0)
        selected = []
        
        for i in range(batch_size):
            # Extract per-image tensors
            img1 = images1[i]
            img2 = images2[i]
            
            # Calculate color variance for each image
            var1 = torch.var(img1, dim=[1, 2]).sum()
            var2 = torch.var(img2, dim=[1, 2]).sum()
            
            # Calculate color contrast
            contrast1 = torch.max(img1) - torch.min(img1)
            contrast2 = torch.max(img2)
            
            # Combined score - higher is better
            score1 = var1 * 0.7 + contrast1 * 0.3
            score2 = var2 * 0.7 + contrast2 * 0.3
            
            # Select the better image
            selected.append(img1 if score1 >= score2 else img2)
        
        return torch.stack(selected)
        
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
    parser.add_argument('--checkpoint1', type=str, default='checkpoints/birds/checkpoint_epoch_546.pt',
                        help='Path to the primary checkpoint file')
    parser.add_argument('--checkpoint2', type=str, default='checkpoints/birds/checkpoint_epoch_553.pt',
                        help='Path to the secondary checkpoint file (optional)')
    parser.add_argument('--descriptions', nargs='*', type=str,
                        help='Text descriptions to generate images from')
    parser.add_argument('--num_images', type=int, default=3,
                        help='Number of images to generate per description')
    parser.add_argument('--use_both_models', action='store_true',
                        help='Use both models and select better images')
    
    args = parser.parse_args()
    
    # 1. Initialize text encoder
    text_encoder = TextEncoder()
    
    # 2. Initialize image generator with both checkpoints
    image_generator = ImageGenerator(args.checkpoint1, args.checkpoint2)
    
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
    
    # 4. Encode text descriptions with enhanced encoding
    print("\nEncoding text descriptions...")
    embeddings = text_encoder.encode_text(descriptions)
    
    # 5. Apply enhanced embedding expansion
    print("Applying enhanced embedding expansion...")
    expanded_embeddings = enhanced_embedding_expansion(embeddings, target_dim=10240)
    
    # 6. Generate images using both checkpoints if available
    print(f"\nGenerating {args.num_images} image(s) per description...")
    images = image_generator.generate_images(expanded_embeddings, 
                                            num_images=args.num_images, 
                                            use_both_models=args.use_both_models)
    
    # 7. Display results
    print("\nDisplaying generated images...")
    timestamp = int(time.time())
    save_path = f"generated_birds_{timestamp}.png"
    display_images(images, descriptions, save_path=save_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
