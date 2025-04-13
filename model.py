import tensorflow as tf
from keras import layers, Model, optimizers
import numpy as np

class ConditioningAugmentation(layers.Layer):
    def __init__(self, ca_dim):
        super(ConditioningAugmentation, self).__init__()
        self.ca_dim = ca_dim
        self.dense_mean = layers.Dense(ca_dim)
        self.dense_log_sigma = layers.Dense(ca_dim)
        
    def call(self, text_embedding, training=True):
        mean = self.dense_mean(text_embedding)
        log_sigma = self.dense_log_sigma(text_embedding)
        
        if training:
            epsilon = tf.random.normal(shape=tf.shape(mean))
            c = mean + tf.exp(log_sigma) * epsilon
        else:
            c = mean
            
        kl_loss = -0.5 * tf.reduce_mean(
            1 + 2 * log_sigma - tf.square(mean) - tf.exp(2 * log_sigma)
        )
        
        return c, kl_loss

class StageIGenerator(Model):
    def __init__(self, config):
        super(StageIGenerator, self).__init__()
        self.config = config
        self.conditioning_augmentation = ConditioningAugmentation(config.CA_DIM)
        
        self.fc = layers.Dense(4 * 4 * config.STAGE1_G_HDIM * 8)
        self.batch_norm0 = layers.BatchNormalization()
        
        self.upsample1 = layers.Conv2DTranspose(
            config.STAGE1_G_HDIM * 4, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm1 = layers.BatchNormalization()
        
        self.upsample2 = layers.Conv2DTranspose(
            config.STAGE1_G_HDIM * 2, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm2 = layers.BatchNormalization()
        
        self.upsample3 = layers.Conv2DTranspose(
            config.STAGE1_G_HDIM, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm3 = layers.BatchNormalization()
        
        self.upsample4 = layers.Conv2DTranspose(
            3, kernel_size=4, strides=2, padding='same', activation='tanh',
            kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        
    def call(self, inputs, training=True):
        z_code, text_embedding = inputs
        
        c_code, kl_loss = self.conditioning_augmentation(text_embedding, training)
        
        z_c_code = tf.concat([z_code, c_code], axis=1)
        
        x = self.fc(z_c_code)
        x = self.batch_norm0(x, training=training)
        x = tf.nn.relu(x)
        x = tf.reshape(x, [-1, 4, 4, self.config.STAGE1_G_HDIM * 8])
        
        x = self.upsample1(x)
        x = self.batch_norm1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.upsample2(x)
        x = self.batch_norm2(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.upsample3(x)
        x = self.batch_norm3(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.upsample4(x)
        
        return x, kl_loss

class StageIDiscriminator(Model):
    def __init__(self, config):
        super(StageIDiscriminator, self).__init__()
        self.config = config
        
        self.conv1 = layers.Conv2D(
            config.STAGE1_D_HDIM, kernel_size=4, strides=2, padding='same',
            kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        
        self.conv2 = layers.Conv2D(
            config.STAGE1_D_HDIM * 2, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(
            config.STAGE1_D_HDIM * 4, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm3 = layers.BatchNormalization()
        
        self.conv4 = layers.Conv2D(
            config.STAGE1_D_HDIM * 8, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm4 = layers.BatchNormalization()
        
        # Text embedding projection
        self.text_projection = layers.Dense(4 * 4 * config.STAGE1_D_HDIM * 8)
        
        # Output layer
        self.output_layer = layers.Conv2D(
            1, kernel_size=4, strides=1, padding='valid',
            kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        
    def call(self, inputs, training=True):
        image, text_embedding = inputs
        
        x = self.conv1(image)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv3(x)
        x = self.batch_norm3(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv4(x)
        x = self.batch_norm4(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        # Project text embedding and spatially replicate
        text_projection = self.text_projection(text_embedding)
        text_projection = tf.reshape(text_projection, [-1, 4, 4, self.config.STAGE1_D_HDIM * 8])
        
        # Concatenate image features with text projection
        x = tf.concat([x, text_projection], axis=3)
        
        # Output layer
        x = self.output_layer(x)
        x = tf.squeeze(x, axis=[1, 2, 3])
        
        return x

class StageIIGenerator(Model):
    def __init__(self, config):
        super(StageIIGenerator, self).__init__()
        self.config = config
        self.conditioning_augmentation = ConditioningAugmentation(config.CA_DIM)
        
        # Encoder for Stage-I images
        self.encoder = [
            layers.Conv2D(
                config.STAGE2_G_HDIM, kernel_size=3, strides=1, padding='same',
                kernel_initializer=tf.random_normal_initializer(0, 0.02)
            ),
            layers.ReLU(),
            layers.Conv2D(
                config.STAGE2_G_HDIM * 2, kernel_size=4, strides=2, padding='same',
                use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(
                config.STAGE2_G_HDIM * 4, kernel_size=4, strides=2, padding='same',
                use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
            ),
            layers.BatchNormalization(),
            layers.ReLU()
        ]
        
        # Residual blocks
        self.residual_blocks = []
        for _ in range(4):
            self.residual_blocks.append(self._build_residual_block(config.STAGE2_G_HDIM * 4))
        
        # Upsampling layers
        self.upsample1 = layers.Conv2DTranspose(
            config.STAGE2_G_HDIM * 2, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm1 = layers.BatchNormalization()
        
        self.upsample2 = layers.Conv2DTranspose(
            config.STAGE2_G_HDIM, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm2 = layers.BatchNormalization()
        
        self.upsample3 = layers.Conv2DTranspose(
            config.STAGE2_G_HDIM // 2, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm3 = layers.BatchNormalization()
        
        self.upsample4 = layers.Conv2DTranspose(
            3, kernel_size=4, strides=2, padding='same', activation='tanh',
            kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        
    def _build_residual_block(self, dim):
        def block(inputs, training=True):
            x = layers.Conv2D(
                dim, kernel_size=3, strides=1, padding='same', use_bias=False,
                kernel_initializer=tf.random_normal_initializer(0, 0.02)
            )(inputs)
            x = layers.BatchNormalization()(x, training=training)
            x = tf.nn.relu(x)
            
            x = layers.Conv2D(
                dim, kernel_size=3, strides=1, padding='same', use_bias=False,
                kernel_initializer=tf.random_normal_initializer(0, 0.02)
            )(x)
            x = layers.BatchNormalization()(x, training=training)
            
            return layers.add([x, inputs])
        
        return block
    
    def call(self, inputs, training=True):
        stage1_img, text_embedding = inputs
        
        c_code, kl_loss = self.conditioning_augmentation(text_embedding, training)
        
        # Encode Stage-I image
        x = stage1_img
        for layer in self.encoder:
            if isinstance(layer, layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Spatial replication of conditioning vector
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, 16, 16, 1])
        
        # Concatenate encoded image with conditioning
        x = tf.concat([x, c_code], axis=3)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x, training=training)
        
        # Upsample to higher resolution
        x = self.upsample1(x)
        x = self.batch_norm1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.upsample2(x)
        x = self.batch_norm2(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.upsample3(x)
        x = self.batch_norm3(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.upsample4(x)
        
        return x, kl_loss

class StageIIDiscriminator(Model):
    def __init__(self, config):
        super(StageIIDiscriminator, self).__init__()
        self.config = config
        
        self.conv1 = layers.Conv2D(
            config.STAGE2_D_HDIM, kernel_size=4, strides=2, padding='same',
            kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        
        self.conv2 = layers.Conv2D(
            config.STAGE2_D_HDIM * 2, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(
            config.STAGE2_D_HDIM * 4, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm3 = layers.BatchNormalization()
        
        self.conv4 = layers.Conv2D(
            config.STAGE2_D_HDIM * 8, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm4 = layers.BatchNormalization()
        
        self.conv5 = layers.Conv2D(
            config.STAGE2_D_HDIM * 16, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        self.batch_norm5 = layers.BatchNormalization()
        
        # Text embedding projection
        self.text_projection = layers.Dense(8 * 8 * config.STAGE2_D_HDIM * 8)
        
        # Output layer
        self.output_layer = layers.Conv2D(
            1, kernel_size=4, strides=1, padding='valid',
            kernel_initializer=tf.random_normal_initializer(0, 0.02)
        )
        
    def call(self, inputs, training=True):
        image, text_embedding = inputs
        
        x = self.conv1(image)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv3(x)
        x = self.batch_norm3(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv4(x)
        x = self.batch_norm4(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv5(x)
        x = self.batch_norm5(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        # Project text embedding and spatially replicate
        text_projection = self.text_projection(text_embedding)
        text_projection = tf.reshape(text_projection, [-1, 8, 8, self.config.STAGE2_D_HDIM * 8])
        
        # Concatenate image features with text projection
        x = tf.concat([x, text_projection], axis=3)
        
        # Output layer
        x = self.output_layer(x)
        x = tf.squeeze(x, axis=[1, 2, 3])
        
        return x
