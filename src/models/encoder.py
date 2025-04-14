import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, GaussianNoise,concatenate, Dense, Dropout, Conv2D, Add, UpSampling2D, Dot, Conv2DTranspose, Activation, Reshape, InputSpec, LeakyReLU, Flatten, BatchNormalization, SpectralNormalization, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

class Encoder:
    def __init__(self, img_shape, latent_dim, num_classes, gaussian_noise_std):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.gaussian_noise_std = gaussian_noise_std
        self.model = self.build_encoder()

    def build_encoder(self):
        enc_input = Input(shape=(self.img_shape[0], self.img_shape[1], self.img_shape[2]), name='encoder_input')
        X = SpectralNormalization(Conv2D(16, kernel_size=3, padding='same', activation='relu'))(enc_input)
        X = self.res_block_down(X, 32)
        X = Dropout(0.3)(X)
        X = self.res_block_down(X, 64)
        X = Dropout(0.3)(X)

        X = Flatten()(X)
        X = GaussianNoise(stddev=self.gaussian_noise_std)(X)
        X = Activation('swish')(X)

        z = SpectralNormalization(Dense(self.latent_dim))(X)
        z = GaussianNoise(stddev=self.gaussian_noise_std)(z)
        z = BatchNormalization()(z)

        encoder_model = Model(enc_input, z, name='encoder') 

        return encoder_model


    def res_block_down(self, layer_input, filters):
        d = BatchNormalization()(layer_input)
        d = Activation('relu')(d)
        d = SpectralNormalization(Conv2D(filters, kernel_size=3, strides=2, padding='same'))(d)
        d = GaussianNoise(stddev=self.gaussian_noise_std)(d)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)
        d = SpectralNormalization(Conv2D(filters, kernel_size=3, strides=1, padding='same'))(d)
        d = GaussianNoise(stddev=self.gaussian_noise_std)(d)
        d_res = SpectralNormalization(Conv2D(filters, kernel_size=1, strides=2, padding='same'))(layer_input)
        d = Add()([d, d_res])
        return d
