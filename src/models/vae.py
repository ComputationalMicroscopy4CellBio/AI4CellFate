import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def reparameterize(z_mean, z_log_var):
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE_Encoder_(Model):
    def __init__(self, latent_dim):
        super(VAE_Encoder_, self).__init__()
        self.conv1 = layers.Conv2D(32, kernel_size=3, strides=2, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, kernel_size=3, strides=2, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, z

class VAE_Decoder_(Model):
    def __init__(self):
        super(VAE_Decoder_, self).__init__()
        self.dense = layers.Dense(7 * 7 * 64, activation='relu')
        self.reshape = layers.Reshape((7, 7, 64))
        self.deconv1 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same')
        self.deconv2 = layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same')
        self.output_layer = layers.Conv2DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')
    
    def call(self, z):
        x = self.dense(z)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return self.output_layer(x)

# VAE Model
class VAE(Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(latent_dim)
        self.decoder = VAE_Decoder()
    
    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var, z

# def vae_loss(x, reconstructed_x, z_mean, z_log_var):
#     recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, reconstructed_x))
#     kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#     return recon_loss + kl_loss