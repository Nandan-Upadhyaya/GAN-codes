import tensorflow as tf
import argparse
from config import Config
from train import Trainer
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train StackGAN')
    parser.add_argument('--stage', type=int, default=1, help='Stage to train (1 or 2)')
    parser.add_argument('--dataset', type=str, default='birds', help='Dataset name (birds or flowers)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=600, help='Number of epochs')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Create configuration
    config = Config()
    config.DATASET_NAME = args.dataset
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    
    # Set up memory growth to avoid occupying all GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train based on stage
    if args.stage == 1:
        print("Training Stage-I GAN...")
        trainer.train_stage1()
    elif args.stage == 2:
        print("Training Stage-II GAN...")
        trainer.train_stage2()
    else:
        print("Invalid stage. Please specify 1 or 2.")

if __name__ == '__main__':
    main()
