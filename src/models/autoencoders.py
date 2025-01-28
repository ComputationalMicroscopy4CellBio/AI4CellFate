import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, GaussianNoise,concatenate, Dense, Dropout, Conv2D, Add, UpSampling2D, Dot, Conv2DTranspose, Activation, Reshape, InputSpec, LeakyReLU, Flatten, BatchNormalization, SpectralNormalization, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

# Apply Spectral Normalization to Conv2D
ConvSN2D = lambda *args, **kwargs: SpectralNormalization(Conv2D(*args, **kwargs))

# Apply Spectral Normalization to Conv2DTranspose
ConvSN2DTranspose = lambda *args, **kwargs: SpectralNormalization(Conv2DTranspose(*args, **kwargs))

def build_simple_autoencoder(img_shape, latent_dim):
    enc_input = Input(shape=(img_shape[0], img_shape[1], img_shape[2]), name='encoder_input')

    # Initial Convolution
    X = SpectralNormalization(Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu'))(enc_input)
    X = MaxPooling2D(pool_size=2)(X)  # Downsample by 2x

    # Second Convolution Block
    X = SpectralNormalization(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))(X)
    X = Dropout(0.3)(X)
    X = MaxPooling2D(pool_size=2)(X)  # Downsample further by 2x

    # Flatten and Dense Layers
    X = Flatten()(X)
   # X = GaussianNoise(stddev=gaussian_noise_std)(X)
    X = Dense(64, activation='swish')(X)

    # Latent Space
    z = SpectralNormalization(Dense(latent_dim, activation=None))(X)
   # z = GaussianNoise(stddev=self.gaussian_noise_std)(z)

    # Model
    encoder_model = Model(enc_input, z, name='encoder')

    dec_input = Input(shape=(latent_dim,), name='decoder_input')

    last_conv_shape = (5, 5, 32)  # Reduced number of filters for simplicity
    X = Dense(last_conv_shape[0] * last_conv_shape[1] * last_conv_shape[2])(dec_input)
    #X = GaussianNoise(stddev=self.gaussian_noise_std)(X)
    X = Reshape((last_conv_shape[0], last_conv_shape[1], last_conv_shape[2]))(X)

    # First Upsampling Block
    X = UpSampling2D(size=(2, 2), interpolation="nearest")(X)
    X = SpectralNormalization(Conv2D(16, kernel_size=3, padding='same', activation='relu'))(X)

    # Second Upsampling Block
    X = Dropout(0.3)(X)
    X = UpSampling2D(size=(2, 2), interpolation="nearest")(X)
    X = SpectralNormalization(Conv2D(16, kernel_size=3, padding='same', activation='relu'))(X)

    # Output layer with sigmoid activation to match the reconstructed image shape
    X = SpectralNormalization(Conv2D(img_shape[2], kernel_size=3, padding='same', activation='sigmoid'))(X)

    # Build and return the model
    decoder_model = Model(dec_input, X, name='decoder')

    return encoder_model, decoder_model

# def build_simple_autoencoder(img_shape, latent_dim):
#     # Encoder
#     encoder = tf.keras.Sequential([
#         ConvSN2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu", input_shape=img_shape),
#         ConvSN2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
#         Flatten(),
#         Dense(latent_dim, activation="relu")
#     ])
#     # Decoder
#     decoder = tf.keras.Sequential([
#         Dense(5 * 5 * 64, activation="relu"),
#         Reshape((5, 5, 64)),
#         ConvSN2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
#         ConvSN2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu"),
#         ConvSN2D(1, (1, 1), activation="sigmoid")  # Output is single-channel (grayscale)
#     ])
    
#     return encoder, decoder

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