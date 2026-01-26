import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, GaussianNoise,concatenate, Dense, Dropout, Conv2D, Add, UpSampling2D, Dot, Conv2DTranspose, Activation, Reshape, InputSpec, LeakyReLU, Flatten, BatchNormalization, SpectralNormalization, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential


class Decoder:
    """
    Decoder component of an Adversarial Autoencoder (AAE).
    
    This class implements the decoder part of an AAE, which is responsible for generating images
    from latent space representations. It uses residual blocks with upsampling to progressively
    increase the spatial dimensions while maintaining feature quality.
    
    Attributes:
        latent_dim (int): Dimension of the latent space representation.
        img_shape (tuple): Shape of the output image (height, width, channels).
        gaussian_noise_std (float): Standard deviation for Gaussian noise added during generation.
        model (tf.keras.Model): The built decoder model.
    """
    
    def __init__(self, latent_dim, img_shape, gaussian_noise_std):
        """
        Initialize the Decoder.
        
        Args:
            latent_dim (int): Dimension of the latent space representation.
            img_shape (tuple): Shape of the output image (height, width, channels).
            gaussian_noise_std (float): Standard deviation for Gaussian noise added during generation.
        """
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.gaussian_noise_std = gaussian_noise_std
        self.model = self.build_generator()

    def build_generator(self):
        """
        Build the decoder model architecture with controlled feature expansion.
        
        Dimensional flow for 144x144 images (mirrors encoder in reverse):
        - Input: latent_dim features
        - Dense: latent_dim → 20736 features (36×36×16)
        - Reshape: 20736 → (36, 36, 16) (restore spatial structure)
        - res_block_up: (36, 36, 16) → (72, 72, 8)
        - res_block_up: (72, 72, 8) → (144, 144, 2)
        - Conv: (144, 144, 2) → (144, 144, 1) (final image)
        
        Key features:
        1. Symmetric with encoder for optimal reconstruction
        2. Exactly 2 upsampling blocks to get 36→72→144
        3. Controlled channel progression maintains quality
        
        Returns:
            tf.keras.Model: The built decoder model.
        """
        dec_input = Input(shape=(self.latent_dim,), name='decoder_input')
        last_conv_shape = (36, 36, 16)
        X = SpectralNormalization(Dense(last_conv_shape[0] * last_conv_shape[1] * last_conv_shape[2]))(dec_input)
        X = GaussianNoise(stddev=self.gaussian_noise_std)(X)
        X = Reshape((last_conv_shape[0], last_conv_shape[1], last_conv_shape[2]))(X)

        X = self.res_block_up(X, 8)
        X = Dropout(0.3)(X)
        X = self.res_block_up(X, 2)
        X = Dropout(0.3)(X)

        X = SpectralNormalization(Conv2D(self.img_shape[2], (3, 3), strides=(1, 1), padding='same', activation='sigmoid'))(X)
        decoder_model = Model(dec_input, X, name='decoder')

        return decoder_model

    def res_block_up(self, layer_input, filters):
        """
        Create a residual upsampling block.
        
        This block performs:
        1. Batch normalization and ReLU activation
        2. Upsampling by a factor of 2
        3. Two convolutional layers with spectral normalization
        4. Skip connection with upsampling and 1x1 convolution
        5. Addition of main and skip paths
        
        Args:
            layer_input (tf.Tensor): Input tensor to the residual block.
            filters (int): Number of filters in the convolutional layers.
            
        Returns:
            tf.Tensor: Output tensor after applying the residual block.
        """
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
