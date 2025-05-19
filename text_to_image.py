import torch
import matplotlib.pyplot as plt
import argparse
import os
import sys
import time
from PIL import Image
import numpy as np

# Add parent directory to path to import from stage1_CUB
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage1_CUB import Config, StageIGenerator, TextEncoder, generate_images_from_text

def parse_args():
    parser = argparse.ArgumentParser(description='Generate bird images from text descriptions')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/birds/generator_best.pt',
                        help='Path to generator checkpoint')
    parser.add_argument('--prompts', type=str, nargs='+', 
                        default=['A small bird with red breast and white wings'],
                        help='Text prompts to generate images from')
    parser.add_argument('--num_images', type=int, default=4,
                        help='Number of images to generate per prompt')
    parser.add_argument('--output', type=str, default='generated_images',
                        help='Output directory for generated images')
    parser.add_argument('--truncation', type=float, default=0.8,
                        help='Truncation value for noise (0-1, lower = better quality but less diversity)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize configuration
    config = Config()
    
    print("Generating images for the following prompts:")
    for i, prompt in enumerate(args.prompts):
        print(f"{i+1}. {prompt}")
    
    # Generate images
    all_images = generate_images_from_text(
        args.checkpoint, 
        args.prompts,
        config,
        args.num_images,
        args.truncation
    )
    
    # Save individual images
    timestamp = int(time.time())
    for i, images_batch in enumerate(all_images):
        for j, img_tensor in enumerate(images_batch):
            # Convert to numpy and transpose
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            
            # Convert to PIL and save
            img_pil = Image.fromarray(img_np)
            prompt_slug = args.prompts[i].replace(' ', '_')[:30]
            img_path = os.path.join(args.output, f"{timestamp}_{prompt_slug}_{j+1}.png")
            img_pil.save(img_path)
            print(f"Saved image to {img_path}")
    
    print(f"All images generated and saved to {args.output}")

if __name__ == "__main__":
    main()
