import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, GaussianNoise,concatenate, Dense, Dropout, Conv2D, Add, UpSampling2D, Dot, Conv2DTranspose, Activation, Reshape, InputSpec, LeakyReLU, Flatten, BatchNormalization, SpectralNormalization, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential


class Decoder:
    def __init__(self, latent_dim, img_shape, gaussian_noise_std):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.gaussian_noise_std = gaussian_noise_std
        self.model = self.build_generator()

    def build_generator(self):
        dec_input = Input(shape=(self.latent_dim,), name='decoder_input')
        last_conv_shape = (5, 5, 64)
        X = SpectralNormalization(Dense(last_conv_shape[0] * last_conv_shape[1] * last_conv_shape[2]))(dec_input)
        X = GaussianNoise(stddev=self.gaussian_noise_std)(X)
        X = Reshape((last_conv_shape[0], last_conv_shape[1], last_conv_shape[2]))(X)

        X = self.res_block_up(X, 64)
        X = Dropout(0.3)(X)
        X = self.res_block_up(X, 32)
        X = Dropout(0.3)(X)

        X = SpectralNormalization(Conv2D(self.img_shape[2], (3, 3), strides=(1, 1), padding='same', activation='sigmoid'))(X)
        decoder_model = Model(dec_input, X, name='decoder')

        return decoder_model

    def res_block_up(self, layer_input, filters):
        d = BatchNormalization()(layer_input)
        d = Activation('relu')(d)
        d = UpSampling2D(size=(2, 2), interpolation="nearest")(d)
        d = SpectralNormalization(Conv2D(filters, kernel_size=3, strides=1, padding='same'))(d)
        d = GaussianNoise(stddev=self.gaussian_noise_std)(d)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)
        d = SpectralNormalization(Conv2D(filters, kernel_size=3, strides=1, padding='same'))(d)
        d = GaussianNoise(stddev=self.gaussian_noise_std)(d)
        d_res = UpSampling2D(size=(2, 2), interpolation="nearest")(layer_input)
        d_res = SpectralNormalization(Conv2D(filters, kernel_size=1, strides=1, padding='same'))(d_res)
        d = Add()([d, d_res])
        return d

# class Decoder:
#     def __init__(self, latent_dim, img_shape, gaussian_noise_std):
#         self.latent_dim = latent_dim
#         self.img_shape = img_shape
#         self.gaussian_noise_std = gaussian_noise_std
#         self.model = self.build_generator()

#     def build_generator(self):
#         # Input layer
#         dec_input = Input(shape=(self.latent_dim,), name='decoder_input')

#         # Fully connected layer to reshape into a feature map
#         last_conv_shape = (5, 5, 32)  # Reduced number of filters for simplicity
#         X = Dense(last_conv_shape[0] * last_conv_shape[1] * last_conv_shape[2])(dec_input)
#         X = GaussianNoise(stddev=self.gaussian_noise_std)(X)
#         X = Reshape((last_conv_shape[0], last_conv_shape[1], last_conv_shape[2]))(X)

#         # First Upsampling Block
#         X = UpSampling2D(size=(2, 2), interpolation="nearest")(X)
#         X = SpectralNormalization(Conv2D(16, kernel_size=3, padding='same', activation='relu'))(X)

#         # Second Upsampling Block
#         X = Dropout(0.3)(X)
#         X = UpSampling2D(size=(2, 2), interpolation="nearest")(X)
#         X = SpectralNormalization(Conv2D(16, kernel_size=3, padding='same', activation='relu'))(X)

#         # Output layer with sigmoid activation to match the reconstructed image shape
#         X = SpectralNormalization(Conv2D(self.img_shape[2], kernel_size=3, padding='same', activation='sigmoid'))(X)

#         # Build and return the model
#         decoder_model = Model(dec_input, X, name='decoder')

#         return decoder_model