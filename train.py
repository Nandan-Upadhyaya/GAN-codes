import tensorflow as tf
import numpy as np
import time
import os
from model import StageIGenerator, StageIDiscriminator, StageIIGenerator, StageIIDiscriminator
from data_loader import Dataset
from utils import save_images, save_model, load_model
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = os.path.join('checkpoints', config.DATASET_NAME)
        self.sample_dir = os.path.join('samples', config.DATASET_NAME)
        
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Create dataset
        self.dataset = Dataset(config)
        
        # Initialize models
        self.stage1_generator = StageIGenerator(config)
        self.stage1_discriminator = StageIDiscriminator(config)
        self.stage2_generator = StageIIGenerator(config)
        self.stage2_discriminator = StageIIDiscriminator(config)
        
        # Initialize optimizers
        self.g1_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.STAGE1_G_LR, beta_1=config.BETA1, beta_2=config.BETA2
        )
        self.d1_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.STAGE1_D_LR, beta_1=config.BETA1, beta_2=config.BETA2
        )
        self.g2_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.STAGE2_G_LR, beta_1=config.BETA1, beta_2=config.BETA2
        )
        self.d2_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.STAGE2_D_LR, beta_1=config.BETA1, beta_2=config.BETA2
        )
        
        # Create fixed noise vector for visualization
        self.fixed_noise = tf.random.normal([config.NUM_EXAMPLES, config.Z_DIM])
    
    def train_stage1(self):
        # Get dataset
        dataset, dataset_size = self.dataset.get_stage1_data()
        steps_per_epoch = dataset_size // self.config.BATCH_SIZE
        
        # Get fixed embeddings for visualization
        for _, embeddings in dataset.take(1):
            fixed_embeddings = embeddings[:self.config.NUM_EXAMPLES]
        
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
                d_loss = self.train_stage1_discriminator(real_images, embeddings, noise)
                d_losses.append(d_loss)
                
                # Train generator
                noise = tf.random.normal([batch_size, self.config.Z_DIM])
                g_loss, kl_loss = self.train_stage1_generator(embeddings, noise)
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
                self.save_stage1_samples(epoch + 1, fixed_embeddings)
            
            # Save model
            if (epoch + 1) % self.config.SNAPSHOT_INTERVAL == 0:
                save_model(self.stage1_generator, os.path.join(self.checkpoint_dir, f'stage1_generator_{epoch+1}'))
                save_model(self.stage1_discriminator, os.path.join(self.checkpoint_dir, f'stage1_discriminator_{epoch+1}'))
    
    @tf.function
    def train_stage1_generator(self, embeddings, noise):
        with tf.GradientTape() as tape:
            # Generate fake images
            fake_images, kl_loss = self.stage1_generator([noise, embeddings], training=True)
            
            # Compute discriminator outputs for fake images
            fake_logits = self.stage1_discriminator([fake_images, embeddings], training=True)
            
            # Calculate generator loss
            g_loss = -tf.reduce_mean(fake_logits)
            
            # Add KL divergence loss
            total_loss = g_loss + self.config.LAMBDA * kl_loss
        
        # Compute gradients and update generator
        gradients = tape.gradient(total_loss, self.stage1_generator.trainable_variables)
        self.g1_optimizer.apply_gradients(zip(gradients, self.stage1_generator.trainable_variables))
        
        return g_loss, kl_loss
    
    @tf.function
    def train_stage1_discriminator(self, real_images, embeddings, noise):
        with tf.GradientTape() as tape:
            # Generate fake images
            fake_images, _ = self.stage1_generator([noise, embeddings], training=True)
            
            # Compute discriminator outputs for real and fake images
            real_logits = self.stage1_discriminator([real_images, embeddings], training=True)
            fake_logits = self.stage1_discriminator([fake_images, embeddings], training=True)
            
            # Calculate discriminator loss (Wasserstein loss)
            d_loss_real = -tf.reduce_mean(real_logits)
            d_loss_fake = tf.reduce_mean(fake_logits)
            d_loss = d_loss_real + d_loss_fake
            
            # Add gradient penalty (WGAN-GP)
            alpha = tf.random.uniform([real_images.shape[0], 1, 1, 1], 0., 1.)
            interpolated = alpha * real_images + (1 - alpha) * fake_images
            
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                interp_logits = self.stage1_discriminator([interpolated, embeddings], training=True)
                
            grads = gp_tape.gradient(interp_logits, interpolated)
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1.0))
            
            d_loss += 10.0 * gradient_penalty
        
        # Compute gradients and update discriminator
        gradients = tape.gradient(d_loss, self.stage1_discriminator.trainable_variables)
        self.d1_optimizer.apply_gradients(zip(gradients, self.stage1_discriminator.trainable_variables))
        
        return d_loss
    
    def save_stage1_samples(self, epoch, fixed_embeddings):
        # Generate images
        fake_images, _ = self.stage1_generator([self.fixed_noise, fixed_embeddings], training=False)
        
        # Save images
        save_path = os.path.join(self.sample_dir, f'stage1_epoch_{epoch}.png')
        save_images(fake_images, save_path)
    
    def train_stage2(self):
        # Get stage1 and stage2 datasets
        dataset_stage1, _ = self.dataset.get_stage1_data()
        dataset_stage2, dataset_size = self.dataset.get_stage2_data()
        steps_per_epoch = dataset_size // self.config.BATCH_SIZE
        
        # Make sure stage1 generator is loaded
        try:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint:
                stage1_checkpoint = os.path.join(self.checkpoint_dir, 'stage1_generator')
                load_model(self.stage1_generator, stage1_checkpoint)
                print("Loaded stage1 generator checkpoint")
            else:
                print("No checkpoint found, please train stage1 first")
                return
        except:
            print("Error loading stage1 generator, please train stage1 first")
            return
        
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
            
            for step, ((_, stage1_embeddings), (stage2_images, stage2_embeddings)) in enumerate(
                zip(dataset_stage1, dataset_stage2)):
                
                batch_size = stage2_images.shape[0]
                
                # Generate low-resolution images using stage1 generator
                noise = tf.random.normal([batch_size, self.config.Z_DIM])
                low_res_fake, _ = self.stage1_generator([noise, stage1_embeddings], training=False)
                
                # Train discriminator
                d_loss = self.train_stage2_discriminator(stage2_images, low_res_fake, stage2_embeddings)
                d_losses.append(d_loss)
                
                # Train generator
                g_loss, kl_loss = self.train_stage2_generator(low_res_fake, stage2_embeddings)
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
                self.save_stage2_samples(epoch + 1, fixed_embeddings)
            
            # Save model
            if (epoch + 1) % self.config.SNAPSHOT_INTERVAL == 0:
                save_model(self.stage2_generator, os.path.join(self.checkpoint_dir, f'stage2_generator_{epoch+1}'))
                save_model(self.stage2_discriminator, os.path.join(self.checkpoint_dir, f'stage2_discriminator_{epoch+1}'))
    
    @tf.function
    def train_stage2_generator(self, low_res_fake, embeddings):
        with tf.GradientTape() as tape:
            # Generate high-resolution fake images
            fake_images, kl_loss = self.stage2_generator([low_res_fake, embeddings], training=True)
            
            # Compute discriminator outputs for fake images
            fake_logits = self.stage2_discriminator([fake_images, embeddings], training=True)
            
            # Calculate generator loss
            g_loss = -tf.reduce_mean(fake_logits)
            
            # Add KL divergence loss
            total_loss = g_loss + self.config.LAMBDA * kl_loss
        
        # Compute gradients and update generator
        gradients = tape.gradient(total_loss, self.stage2_generator.trainable_variables)
        self.g2_optimizer.apply_gradients(zip(gradients, self.stage2_generator.trainable_variables))
        
        return g_loss, kl_loss
    
    @tf.function
    def train_stage2_discriminator(self, real_images, low_res_fake, embeddings):
        with tf.GradientTape() as tape:
            # Generate high-resolution fake images
            fake_images, _ = self.stage2_generator([low_res_fake, embeddings], training=True)
            
            # Compute discriminator outputs for real and fake images
            real_logits = self.stage2_discriminator([real_images, embeddings], training=True)
            fake_logits = self.stage2_discriminator([fake_images, embeddings], training=True)
            
            # Calculate discriminator loss (Wasserstein loss)
            d_loss_real = -tf.reduce_mean(real_logits)
            d_loss_fake = tf.reduce_mean(fake_logits)
            d_loss = d_loss_real + d_loss_fake
            
            # Add gradient penalty (WGAN-GP)
            alpha = tf.random.uniform([real_images.shape[0], 1, 1, 1], 0., 1.)
            interpolated = alpha * real_images + (1 - alpha) * fake_images
            
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                interp_logits = self.stage2_discriminator([interpolated, embeddings], training=True)
                
            grads = gp_tape.gradient(interp_logits, interpolated)
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1.0))
            
            d_loss += 10.0 * gradient_penalty
        
        # Compute gradients and update discriminator
        gradients = tape.gradient(d_loss, self.stage2_discriminator.trainable_variables)
        self.d2_optimizer.apply_gradients(zip(gradients, self.stage2_discriminator.trainable_variables))
        
        return d_loss
    
    def save_stage2_samples(self, epoch, fixed_embeddings):
        # Generate low-resolution images
        noise = tf.random.normal([self.config.NUM_EXAMPLES, self.config.Z_DIM])
        low_res_fake, _ = self.stage1_generator([noise, fixed_embeddings], training=False)
        
        # Generate high-resolution images
        fake_images, _ = self.stage2_generator([low_res_fake, fixed_embeddings], training=False)
        
        # Save images
        save_path = os.path.join(self.sample_dir, f'stage2_epoch_{epoch}.png')
        save_images(fake_images, save_path)
