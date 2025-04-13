import tensorflow as tf
import numpy as np # Fixed typo from 'npp' to 'np'
import os
import time
import pickle
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import requests
import zipfile
import tarfile
import shutil
from model import StageIGenerator, StageIDiscriminator

# Ensure compatibility with TensorFlow 2.10.1
print(f"TensorFlow version: {tf.__version__}")

class Config:
    # Data parameters
    DATASET_NAME = 'birds'  # 'birds' or 'flowers'
    EMBEDDING_DIM = 1024  # dimension of the text embedding
    Z_DIM = 100  # dimension of the noise vector
    LOCAL_DATASET_PATH = None  # path to local dataset

    # Stage I parameters
    STAGE1_G_LR = 0.0002  # learning rate for stage 1 generator
    STAGE1_D_LR = 0.0002  # learning rate for stage 1 discriminator
    STAGE1_G_HDIM = 128  # hidden dimension for stage 1 generator
    STAGE1_D_HDIM = 64   # hidden dimension for stage 1 discriminator
    STAGE1_IMAGE_SIZE = 64  # size of image in stage 1
    
    # Training parameters
    BATCH_SIZE = 128  # Increased from 64 to better utilize GPU
    EPOCHS = 600
    SNAPSHOT_INTERVAL = 50
    NUM_EXAMPLES = 6  # number of examples to visualize during training
    
    # Conditioning Augmentation
    CA_DIM = 128  # dimension of the augmented conditioning vector
    
    # Adam optimizer parameters
    BETA1 = 0.5
    BETA2 = 0.999
    
    # KL divergence regularization weight
    LAMBDA = 1.0
    
    # Dataset URLs
    BIRDS_DATASET_URL = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    # Direct link to pre-processed embeddings for birds dataset
    BIRDS_EMBEDDING_URL = r"C:\Users\nanda\OneDrive\Desktop\StackGAN\char-CNN-RNN-embeddings.pickle"
    # Backup direct link for train/test splits
    BIRDS_TRAIN_TEST_URL = "https://github.com/hanzhanggit/StackGAN/raw/master/Data/birds/train-test-split.pickle"

    # Memory management
    SHUFFLE_BUFFER_SIZE = 4000  # Increased from 1000 to better utilize RAM
    PREFETCH_BUFFER_SIZE = 12    # Increased from 2 to improve pipeline efficiency

def download_and_prepare_dataset(config):
    """Use the local dataset without downloading anything"""
    # Only use the provided local dataset path
    if config.LOCAL_DATASET_PATH and os.path.exists(config.LOCAL_DATASET_PATH):
        print(f"Using local dataset from: {config.LOCAL_DATASET_PATH}")
        return config.LOCAL_DATASET_PATH
    else:
        raise ValueError("Please provide a valid local dataset path using --data_path argument. No downloads will be performed.")

class Dataset:
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
        self.class_info_path = os.path.join(self.train_dir, 'class_info.pickle')
        
        print(f"Using dataset structure:")
        print(f"Images directory: {self.image_dir}")
        print(f"Train directory: {self.train_dir}")
        print(f"Test directory: {self.test_dir}")
        print(f"Embeddings file: {self.embedding_path}")
        print(f"Filenames file: {self.filenames_path}")
        
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
        
        # If not found, or no class in filename, search all class directories
        for dir_name in os.listdir(self.image_dir):
            class_path = os.path.join(self.image_dir, dir_name)
            if not os.path.isdir(class_path):
                continue
                
            # Try matching by full image name
            img_path = os.path.join(class_path, img_name)
            if os.path.exists(img_path):
                return self._process_image(img_path)
                
            # Try matching by base name without extension
            for img_file in os.listdir(class_path):
                # Remove extension for comparison
                base_img_name = os.path.splitext(img_name)[0]
                base_file = os.path.splitext(img_file)[0]
                
                # Compare base names
                if base_img_name == base_file:
                    img_path = os.path.join(class_path, img_file)
                    return self._process_image(img_path)
                    
                # If still not found, try more relaxed matching
                # This might catch cases where naming conventions differ slightly
                if base_img_name.replace('_', '') == base_file.replace('_', ''):
                    img_path = os.path.join(class_path, img_file)
                    return self._process_image(img_path)
        
        # If we reach here, image was not found
        print(f"Warning: Image not found: {filename}")
        # Return a black image as placeholder
        return np.zeros((self.image_size, self.image_size, 3))
    
    def _process_image(self, img_path):
        """Process an image given its path"""
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
        return img

    def get_data(self):
        """Return a TensorFlow dataset that streams data efficiently"""
        print("Setting up data streaming pipeline for memory efficiency...")
        
        # Check embeddings format
        is_embeddings_list = isinstance(self.embeddings, list)
        if is_embeddings_list:
            print("Detected embeddings as a list structure. Using index-based matching.")
            print(f"Embeddings length: {len(self.embeddings)}")
            print(f"Filenames length: {len(self.filenames)}")
            
            # Check embedding shape
            sample_embedding = self.embeddings[0]
            print(f"Sample embedding shape: {np.array(sample_embedding).shape}")
            
            # Create a generator function that yields one sample at a time
            def data_generator():
                for i, filename in enumerate(self.filenames):
                    if i >= len(self.embeddings):
                        continue
                        
                    try:
                        image = self.load_image(filename)
                        embedding = self.embeddings[i]
                        
                        # Handle 3D embeddings - flatten to 2D
                        embedding_arr = np.array(embedding)
                        if len(embedding_arr.shape) > 2:
                            print(f"Reshaping embeddings from {embedding_arr.shape} to 2D") if i == 0 else None
                            # If shape is [seq_len, emb_dim], flatten it
                            embedding_arr = embedding_arr.flatten()
                        
                        yield image, embedding_arr
                        
                        # Print progress occasionally
                        if (i+1) % 100 == 0:
                            print(f"Streamed {i+1}/{self.dataset_size} images")
                            
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
            
            # Create output signature for the dataset
            sample_embedding_arr = np.array(self.embeddings[0])
            if len(sample_embedding_arr.shape) > 2:
                # If 3D, convert to flattened 1D
                emb_shape = sample_embedding_arr.size
                embedding_shape = tf.TensorShape([emb_shape])
            else:
                embedding_shape = tf.TensorShape(sample_embedding_arr.shape)
            
            # Create TensorFlow dataset from generator
            dataset = tf.data.Dataset.from_generator(
                data_generator,
                output_signature=(
                    tf.TensorSpec(shape=(self.image_size, self.image_size, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=embedding_shape, dtype=tf.float32)
                )
            )
            
            # Estimate dataset size (may be slightly smaller due to skipped errors)
            estimated_size = min(len(self.filenames), len(self.embeddings))
        else:
            # Dictionary-based embeddings
            raise ValueError("Dictionary-based embeddings not implemented yet")
        
        # Apply dataset transformations
        dataset = dataset.shuffle(buffer_size=4000)  # Smaller buffer to save memory
        dataset = dataset.batch(self.config.BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset, estimated_size

def save_images(images, path):
    """Save images to a single figure"""
    images = (images + 1) / 2.0  # Rescale to [0, 1]
    n_images = images.shape[0]
    rows = int(np.sqrt(n_images))
    cols = int(np.ceil(n_images / rows))
    plt.figure(figsize=(cols * 2, rows * 2))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_model(model, path):
    """Save model weights"""
    model.save_weights(path)

def load_model(model, path):
    """Load model weights"""
    model.load_weights(path)

class Stage1Trainer:
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = os.path.join('checkpoints', config.DATASET_NAME)
        self.sample_dir = os.path.join('samples', config.DATASET_NAME)
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        # Initialize models:
        self.generator = StageIGenerator(config)
        self.discriminator = StageIDiscriminator(config)
        # Initialize optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.STAGE1_G_LR, beta_1=config.BETA1, beta_2=config.BETA2
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.STAGE1_D_LR, beta_1=config.BETA1, beta_2=config.BETA2
        )
        # Create fixed noise vector for visualization
        self.fixed_noise = tf.random.normal([config.NUM_EXAMPLES, config.Z_DIM])

    @tf.function
    def train_generator(self, embeddings, noise):
        # Ensure embeddings have the right shape (should be 2D)
        if len(embeddings.shape) > 2:
            embeddings = tf.reshape(embeddings, [tf.shape(embeddings)[0], -1])
            
        with tf.GradientTape() as tape:
            # Generate fake images
            fake_images, kl_loss = self.generator([noise, embeddings], training=True)
            # Compute discriminator outputs for fake images
            fake_logits = self.discriminator([fake_images, embeddings], training=True)
            # Calculate generator loss
            g_loss = -tf.reduce_mean(fake_logits)
            # Add KL divergence loss
            total_loss = g_loss + self.config.LAMBDA * kl_loss
        # Compute gradients and update generator
        gradients = tape.gradient(total_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return g_loss, kl_loss

    @tf.function
    def train_discriminator(self, real_images, embeddings, noise):
        # Ensure embeddings have the right shape (should be 2D)
        if len(embeddings.shape) > 2:
            embeddings = tf.reshape(embeddings, [tf.shape(embeddings)[0], -1])
            
        with tf.GradientTape() as tape:
            # Generate fake images
            fake_images, _ = self.generator([noise, embeddings], training=True)
            # Compute discriminator outputs for real and fake images
            real_logits = self.discriminator([real_images, embeddings], training=True)
            fake_logits = self.discriminator([fake_images, embeddings], training=True)
            # Calculate discriminator loss (Wasserstein loss)
            d_loss_real = -tf.reduce_mean(real_logits)
            d_loss_fake = tf.reduce_mean(fake_logits)
            d_loss = d_loss_real + d_loss_fake
            # Add gradient penalty (WGAN-GP)
            alpha = tf.random.uniform([real_images.shape[0], 1, 1, 1], 0., 1.)
            interpolated = alpha * real_images + (1 - alpha) * fake_images
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                interp_logits = self.discriminator([interpolated, embeddings], training=True)
            grads = gp_tape.gradient(interp_logits, interpolated)
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1.0))
            d_loss += 10.0 * gradient_penalty
        # Compute gradients and update discriminator
        gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        return d_loss

    def save_samples(self, epoch, embeddings):
        # Ensure embeddings have the right shape
        if len(embeddings.shape) > 2:
            embeddings = tf.reshape(embeddings, [tf.shape(embeddings)[0], -1])
            
        # Generate images
        fake_images, _ = self.generator([self.fixed_noise, embeddings], training=False)
        # Save images
        save_path = os.path.join(self.sample_dir, f'stage1_epoch_{epoch}.png')
        save_images(fake_images, save_path)

    def train(self, dataset, dataset_size):
        steps_per_epoch = dataset_size // self.config.BATCH_SIZE
        # Get fixed embeddings for visualization
        for _, embeddings in dataset.take(1):
            fixed_embeddings = embeddings[:self.config.NUM_EXAMPLES]
            # Ensure fixed embeddings have the right shape
            if len(fixed_embeddings.shape) > 2:
                fixed_embeddings = tf.reshape(fixed_embeddings, [self.config.NUM_EXAMPLES, -1])
        # Start training
        for epoch in range(self.config.EPOCHS):
            start_time = time.time()
            # Initialize metrics
            g_losses = []
            d_losses = []
            kl_losses = []
            for step, (real_images, embeddings) in enumerate(dataset):
                batch_size = real_images.shape[0]
                # Generate random noise
                noise = tf.random.normal([batch_size, self.config.Z_DIM])
                # Train discriminator
                d_loss = self.train_discriminator(real_images, embeddings, noise)
                d_losses.append(d_loss)
                # Train generator
                noise = tf.random.normal([batch_size, self.config.Z_DIM])
                g_loss, kl_loss = self.train_generator(embeddings, noise)
                g_losses.append(g_loss)
                kl_losses.append(kl_loss)
                # Print progress
                if (step + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.config.EPOCHS}, "
                          f"Step {step+1}/{steps_per_epoch}, "
                          f"G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, KL Loss: {kl_loss:.4f}")
            # Calculate epoch averages
            g_loss_avg = sum(g_losses) / len(g_losses)
            d_loss_avg = sum(d_losses) / len(d_losses)
            kl_loss_avg = sum(kl_losses) / len(kl_losses)
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}, "
                  f"G Loss: {g_loss_avg:.4f}, D Loss: {d_loss_avg:.4f}, KL Loss: {kl_loss_avg:.4f}, "
                  f"Time: {time.time() - start_time:.2f}s")
            # Save samples
            if (epoch + 1) % self.config.SNAPSHOT_INTERVAL == 0:
                self.save_samples(epoch + 1, fixed_embeddings)
            # Save model
            if (epoch + 1) % self.config.SNAPSHOT_INTERVAL == 0:
                save_model(self.generator, os.path.join(self.checkpoint_dir, f'stage1_generator_{epoch+1}'))
                save_model(self.discriminator, os.path.join(self.checkpoint_dir, f'stage1_discriminator_{epoch+1}'))

def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage-I GAN')
    parser.add_argument('--dataset', type=str, default='birds', 
                        help='Dataset name (birds or flowers)')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=600, 
                        help='Number of epochs')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='GPU to use')
    parser.add_argument('--images_path', type=str, required=True,
                        help='Path to the directory containing bird class folders with images')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to the train directory with pickle files')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to the test directory with pickle files')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
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
    
    # Create configuration
    config = Config()
    config.DATASET_NAME = args.dataset
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    # Store all paths in a dictionary in LOCAL_DATASET_PATH
    config.LOCAL_DATASET_PATH = {
        'images': args.images_path,
        'train': args.train_path,
        'test': args.test_path
    }
    
    # Check if all paths exist
    for path_type, path in config.LOCAL_DATASET_PATH.items():
        if not os.path.exists(path):
            raise ValueError(f"The {path_type} path does not exist: {path}")
    
    print(f"Using dataset paths:")
    print(f"  Images: {config.LOCAL_DATASET_PATH['images']}")
    print(f"  Train: {config.LOCAL_DATASET_PATH['train']}")
    print(f"  Test: {config.LOCAL_DATASET_PATH['test']}")
    
    # Create dataset
    dataset = Dataset(config)
    train_dataset, dataset_size = dataset.get_data()
    
    # Create trainer
    trainer = Stage1Trainer(config)
    
    # Train Stage-I GAN
    print("Training Stage-I GAN...")
    trainer.train(train_dataset, dataset_size)

if __name__ == '__main__':
    main()