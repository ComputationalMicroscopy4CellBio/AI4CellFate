import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, GaussianNoise,concatenate, Dense, Dropout, Conv2D, Add, UpSampling2D, Dot, Conv2DTranspose, Activation, Reshape, InputSpec, LeakyReLU, Flatten, BatchNormalization, SpectralNormalization, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

# Apply Spectral Normalization to Conv2D
ConvSN2D = lambda *args, **kwargs: SpectralNormalization(Conv2D(*args, **kwargs))

# Apply Spectral Normalization to Conv2DTranspose
ConvSN2DTranspose = lambda *args, **kwargs: SpectralNormalization(Conv2DTranspose(*args, **kwargs))

def build_simple_autoencoder(img_shape, latent_dim):
    encoder = tf.keras.Sequential([
        ConvSN2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu", input_shape=img_shape),
        ConvSN2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        Flatten(),
        Dense(latent_dim, activation="relu")
    ])
    decoder = tf.keras.Sequential([
        Dense(5 * 5 * 64, activation="relu"),
        Reshape((5, 5, 64)),
        ConvSN2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        ConvSN2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="sigmoid"),
        ConvSN2D(1, (1, 1), activation="sigmoid")
    ])
    return encoder, decoder

def build_deeper_autoencoder(img_shape, latent_dim):
    encoder = tf.keras.Sequential([
        ConvSN2D(32, (3, 3), strides=(1, 1), padding="same", activation="relu", input_shape=img_shape),
        ConvSN2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        BatchNormalization(),
        ConvSN2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        Flatten(),
        Dense(latent_dim, activation="relu")
    ])
    decoder = tf.keras.Sequential([
        Dense(5 * 5 * 128, activation="relu"),
        Reshape((5, 5, 128)),
        ConvSN2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        BatchNormalization(),
        ConvSN2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        ConvSN2D(1, (1, 1), activation="sigmoid")
    ])
    return encoder, decoder

def build_residual_autoencoder(img_shape, latent_dim):
    class ResidualBlock(tf.keras.layers.Layer):
        def call(self, inputs):
            x = ConvSN2D(64, (3, 3), padding="same", activation="relu")(inputs)
            x = ConvSN2D(64, (3, 3), padding="same")(x)
            return tf.keras.layers.Add()([inputs, x])

    encoder = tf.keras.Sequential([
        ConvSN2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu", input_shape=img_shape),
        ResidualBlock(),
        ConvSN2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        ResidualBlock(),
        Flatten(),
        Dense(latent_dim, activation="relu")
    ])
    decoder = tf.keras.Sequential([
        Dense(5 * 5 * 64, activation="relu"),
        Reshape((5, 5, 64)),
        ResidualBlock(),
        ConvSN2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        ResidualBlock(),
        ConvSN2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="sigmoid"),
        ConvSN2D(1, (1, 1), activation="sigmoid")
    ])
    return encoder, decoder