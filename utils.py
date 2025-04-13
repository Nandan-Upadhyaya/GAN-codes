import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def save_images(images, path):
    """Save images to a single figure"""
    images = (images + 1) / 2.0  # Rescale to [0, 1]
    n_images = images.shape[0]
    
    rows = int(math.sqrt(n_images))
    cols = int(math.ceil(n_images / rows))
    
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

def generate_interpolated_embeddings(embedding1, embedding2, n_steps=5):
    """Generate interpolated embeddings between two text embeddings"""
    alphas = np.linspace(0, 1, n_steps)
    embeddings = []
    
    for alpha in alphas:
        embedding = embedding1 * (1 - alpha) + embedding2 * alpha
        embeddings.append(embedding)
    
    return np.array(embeddings)

def generate_images_from_embeddings(generator, embeddings, noise=None, stage=1):
    """Generate images from text embeddings"""
    n_samples = embeddings.shape[0]
    
    if noise is None:
        noise = tf.random.normal([n_samples, 100])
    
    if stage == 1:
        fake_images, _ = generator([noise, embeddings], training=False)
        return fake_images
    else:
        # For stage 2, we need to generate low-res images first
        stage1_generator = generator[0]
        stage2_generator = generator[1]
        
        low_res_fake, _ = stage1_generator([noise, embeddings], training=False)
        fake_images, _ = stage2_generator([low_res_fake, embeddings], training=False)
        return fake_images

def preprocess_text(text, text_encoder):
    """Preprocess text and convert it to embeddings using a pre-trained text encoder"""
    # This is a placeholder - actual implementation would depend on your text encoder
    # For example, with BERT:
    # tokenized_text = tokenizer(text, return_tensors='tf')
    # embeddings = text_encoder(tokenized_text)
    pass
