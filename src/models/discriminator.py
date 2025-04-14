import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, GaussianNoise,concatenate, Dense, Dropout, Conv2D, Add, UpSampling2D, Dot, Conv2DTranspose, Activation, Reshape, InputSpec, LeakyReLU, Flatten, BatchNormalization, SpectralNormalization, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

class Discriminator:
    """
    Discriminator component of an Adversarial Autoencoder (AAE).
    
    This class implements the discriminator part of an AAE, which aims to distinguish between
    samples from the prior distribution and the encoder's output distribution in latent space.
    It consists of several dense layers with LeakyReLU activations and dropout for regularization.
    
    Attributes:
        latent_dim (int): Dimension of the latent space representation.
        model (tf.keras.Model): The built discriminator model.
    """

    def __init__(self, latent_dim):
        """
        Initialize the Discriminator.
        
        Args:
            latent_dim (int): Dimension of the latent space representation.
        """
        self.latent_dim = latent_dim
        self.model = self.build_discriminator()

    def build_discriminator(self):
        """
        Build the discriminator model architecture.
        
        The model consists of:
        1. Three dense layers with 512 units each
        2. LeakyReLU activations with alpha=0.2
        3. Batch normalization and dropout for regularization
        4. Final sigmoid output layer for binary classification
        
        Returns:
            tf.keras.Model: The built discriminator model.
        """
        model = Sequential()
        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(BatchNormalization(scale=False))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(1, activation="sigmoid"))

        # Create the input layer and connect it to the model
        encoded_repr = Input(shape=(self.latent_dim,))
        validity = model(encoded_repr)
        
        # Create a full model that takes the encoded representation and outputs validity
        discriminator_model = Model(encoded_repr, validity, name='discriminator')

        return discriminator_model