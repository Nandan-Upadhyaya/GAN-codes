import tensorflow as tf
import numpy as np
import os
import time
import pickle
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from model import StageIGenerator, StageIIGenerator, StageIIDiscriminator

# Ensure compatibility with TensorFlow 2.10.1
print(f"TensorFlow version: {tf.__version__}")

class Config:
    # Data parameters
    DATASET_NAME = 'birds'  # 'birds' or 'flowers'
    EMBEDDING_DIM = 1024  # dimension of the text embedding
    Z_DIM = 100  # dimension of the noise vector

    # Stage I parameters
    STAGE1_G_LR = 0.0002  # learning rate for stage 1 generator
    STAGE1_D_LR = 0.0002  # learning rate for stage 1 discriminator
    STAGE1_G_HDIM = 128  # hidden dimension for stage 1 generator
    STAGE1_D_HDIM = 64   # hidden dimension for stage 1 discriminator
    STAGE1_IMAGE_SIZE = 64  # size of image in stage 1
    
    # Stage II parameters
    STAGE2_G_LR = 0.0002  # learning rate for stage 2 generator
    STAGE2_D_LR = 0.0002  # learning rate for stage 2 discriminator
    STAGE2_G_HDIM = 128  # hidden dimension for stage 2 generator
    STAGE2_D_HDIM = 64   # hidden dimension for stage 2 discriminator
    STAGE2_IMAGE_SIZE = 256  # size of image in stage 2
    
    # Training parameters
    BATCH_SIZE = 64
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

class Dataset:
    def __init__(self, config, stage='stage2'):
        self.config = config
        self.stage = stage
        self.image_size = (
            config.STAGE1_IMAGE_SIZE if stage == 'stage1' else config.STAGE2_IMAGE_SIZE
        )
        
        # Paths should be modified based on your dataset location
        self.data_dir = f'data/{config.DATASET_NAME}'
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.embedding_path = os.path.join(self.data_dir, 'embeddings.pickle')
        self.filenames_path = os.path.join(self.data_dir, 'filenames.pickle')
        
        # Load text embeddings
        with open(self.embedding_path, 'rb') as f:
            self.embeddings = pickle.load(f, encoding='latin1')
        
        # Load filenames
        with open(self.filenames_path, 'rb') as f:
            self.filenames = pickle.load(f, encoding='latin1')
            
        self.dataset_size = len(self.filenames)
        print(f"Dataset size: {self.dataset_size}")
    
    def load_image(self, filename):
        """Load and preprocess image"""
        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
        return img
    
    def get_data(self):
        """Return a TensorFlow dataset"""
        # Create lists to hold data
        images = []
        embeddings = []
        
        # Load all images and embeddings
        for filename in self.filenames:
            image = self.load_image(filename)
            embedding = self.embeddings[filename]
            
            images.append(image)
            embeddings.append(embedding)
        
        # Convert to numpy arrays
        images = np.array(images)
        embeddings = np.array(embeddings)
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, embeddings))
        dataset = dataset.shuffle(buffer_size=self.dataset_size)
        dataset = dataset.batch(self.config.BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset, self.dataset_size
    
    def get_stage1_data(self):
        """Return Stage-I data"""
        self.stage = 'stage1'
        self.image_size = self.config.STAGE1_IMAGE_SIZE
        return self.get_data()
    
    def get_stage2_data(self):
        """Return Stage-II data"""
        self.stage = 'stage2'
        self.image_size = self.config.STAGE2_IMAGE_SIZE
        return self.get_data()

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

class Stage2Trainer:
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = os.path.join('checkpoints', config.DATASET_NAME)
        self.sample_dir = os.path.join('samples', config.DATASET_NAME)
        
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Initialize models
        self.stage1_generator = StageIGenerator(config)
        self.generator = StageIIGenerator(config)
        self.discriminator = StageIIDiscriminator(config)
        
        # Initialize optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.STAGE2_G_LR, beta_1=config.BETA1, beta_2=config.BETA2
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.STAGE2_D_LR, beta_1=config.BETA1, beta_2=config.BETA2
        )
        
        # Create fixed noise vector for visualization
        self.fixed_noise = tf.random.normal([config.NUM_EXAMPLES, config.Z_DIM])
        
        # Load Stage-I Generator
        self._load_stage1_generator()
    
    def _load_stage1_generator(self):
        """Load pre-trained Stage-I Generator"""
        checkpoint_dir = os.path.join(self.checkpoint_dir)
        
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('stage1_generator_')]
        if not checkpoints:
            raise FileNotFoundError(f"No Stage-I Generator checkpoint found in {checkpoint_dir}")
        
        # Get latest checkpoint
        epochs = [int(f.split('_')[-1]) for f in checkpoints]
        latest_epoch = max(epochs)
        latest_checkpoint = os.path.join(checkpoint_dir, f'stage1_generator_{latest_epoch}')
        
        # Load weights
        print(f"Loading Stage-I Generator from {latest_checkpoint}")
        load_model(self.stage1_generator, latest_checkpoint)
    
    @tf.function
    def train_generator(self, stage1_images, embeddings):
        with tf.GradientTape() as tape:
            # Generate high-resolution fake images
            fake_images, kl_loss = self.generator([stage1_images, embeddings], training=True)
            
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
    def train_discriminator(self, real_images, stage1_images, embeddings):
        with tf.GradientTape() as tape:
            # Generate high-resolution fake images
            fake_images, _ = self.generator([stage1_images, embeddings], training=True)
            
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
        # Generate low-resolution images using Stage-I Generator
        low_res_fake, _ = self.stage1_generator([self.fixed_noise, embeddings], training=False)
        
        # Generate high-resolution images using Stage-II Generator
        fake_images, _ = self.generator([low_res_fake, embeddings], training=False)
        
        # Save combined samples (low-res and high-res)
        combined_images = []
        for i in range(self.config.NUM_EXAMPLES):
            # Upscale low-res to high-res size for better visualization
            upscaled_low_res = tf.image.resize(
                low_res_fake[i:i+1], 
                [self.config.STAGE2_IMAGE_SIZE, self.config.STAGE2_IMAGE_SIZE]
            )[0]
            combined_images.append(upscaled_low_res)
            combined_images.append(fake_images[i])
        
        combined_images = tf.stack(combined_images)
        save_path = os.path.join(self.sample_dir, f'stage2_epoch_{epoch}.png')
        save_images(combined_images, save_path)
    
    def train(self, dataset_stage1, dataset_stage2, dataset_size):
        steps_per_epoch = dataset_size // self.config.BATCH_SIZE
        
        # Get fixed embeddings for visualization
        for _, embeddings in dataset_stage2.take(1):
            fixed_embeddings = embeddings[:self.config.NUM_EXAMPLES]
        
        # Start training
        for epoch in range(self.config.EPOCHS):
            start_time = time.time()
            
            # Initialize metrics
            g_losses = []
            d_losses = []
            kl_losses = []
            
            # Create iterator for both datasets
            dataset_combined = tf.data.Dataset.zip((dataset_stage1, dataset_stage2))
            
            for step, ((_, stage1_embeddings), (stage2_images, stage2_embeddings)) in enumerate(
                dataset_combined):
                
                batch_size = stage2_images.shape[0]
                
                # Generate low-resolution images using stage1 generator
                noise = tf.random.normal([batch_size, self.config.Z_DIM])
                low_res_fake, _ = self.stage1_generator([noise, stage1_embeddings], training=False)
                
                # Train discriminator
                d_loss = self.train_discriminator(stage2_images, low_res_fake, stage2_embeddings)
                d_losses.append(d_loss)
                
                # Train generator
                g_loss, kl_loss = self.train_generator(low_res_fake, stage2_embeddings)
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
                save_model(self.generator, os.path.join(self.checkpoint_dir, f'stage2_generator_{epoch+1}'))
                save_model(self.discriminator, os.path.join(self.checkpoint_dir, f'stage2_discriminator_{epoch+1}'))

def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage-II GAN')
    parser.add_argument('--dataset', type=str, default='birds', 
                        help='Dataset name (birds or flowers)')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=600, 
                        help='Number of epochs')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='GPU to use')
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
    
    # Create dataset
    dataset = Dataset(config)
    train_dataset_stage1, _ = dataset.get_stage1_data()
    train_dataset_stage2, dataset_size = dataset.get_stage2_data()
    
    # Create trainer
    trainer = Stage2Trainer(config)
    
    # Train Stage-II GAN
    print("Training Stage-II GAN...")
    trainer.train(train_dataset_stage1, train_dataset_stage2, dataset_size)

if __name__ == '__main__':
    main()
