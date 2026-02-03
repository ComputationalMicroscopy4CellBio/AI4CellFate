import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, GaussianNoise,concatenate, Dense, Dropout, Conv2D, Add, UpSampling2D, Dot, Conv2DTranspose, Activation, Reshape, InputSpec, LeakyReLU, Flatten, BatchNormalization, SpectralNormalization, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

class Encoder:
    """
    Encoder component of an Adversarial Autoencoder (AAE).
    
    This class implements the encoder part of an AAE, which maps input images to latent space
    representations. It uses residual blocks with downsampling to progressively reduce spatial
    dimensions while extracting meaningful features.
    
    Attributes:
        img_shape (tuple): Shape of the input image (height, width, channels).
        latent_dim (int): Dimension of the latent space representation.
        num_classes (int): Number of classes for classification tasks.
        gaussian_noise_std (float): Standard deviation for Gaussian noise added during encoding.
        model (tf.keras.Model): The built encoder model.
    """

    def __init__(self, img_shape, latent_dim, num_classes, gaussian_noise_std):
        """
        Initialize the Encoder.
        
        Args:
            img_shape (tuple): Shape of the input image (height, width, channels).
            latent_dim (int): Dimension of the latent space representation.
            num_classes (int): Number of classes for classification tasks.
            gaussian_noise_std (float): Standard deviation for Gaussian noise added during encoding.
        """
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.gaussian_noise_std = gaussian_noise_std
        self.model = self.build_encoder()

    def build_encoder(self):
        enc_input = Input(shape=self.img_shape, name='encoder_input')

        X = SpectralNormalization(
            Conv2D(16, 3, padding='same', activation='relu')
        )(enc_input)

        X = self.res_block_down(X, 32)
        X = self.res_block_down(X, 48)

        # Global Average Pooling to reduce the spatial dimensions to a single vector
        X = GlobalAveragePooling2D()(X)

        X = GaussianNoise(self.gaussian_noise_std)(X)
        X = Activation('swish')(X)

        z = SpectralNormalization(Dense(self.latent_dim))(X)
        z = GaussianNoise(self.gaussian_noise_std)(z)

        return Model(enc_input, z, name="encoder")



    def res_block_down(self, layer_input, filters):
        """
        Create a residual downsampling block.
        
        This block performs:
        1. Batch normalization and ReLU activation
        2. Two convolutional layers with spectral normalization and stride=2 for downsampling
        3. Skip connection with 1x1 convolution and stride=2
        4. Addition of main and skip paths
        
        Args:
            layer_input (tf.Tensor): Input tensor to the residual block.
            filters (int): Number of filters in the convolutional layers.
            
        Returns:
            tf.Tensor: Output tensor after applying the residual block.
        """
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
