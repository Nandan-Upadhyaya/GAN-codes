import tensorflow as tf
import numpy as np
import os
import pickle
from PIL import Image

class Dataset:
    def __init__(self, config, stage='stage1'):
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
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
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
