import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
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
        Build the discriminator model architecture - simplified for 2D latent space.
        
        The model consists of:
        1. Two small dense layers (32, 16 units)
        2. LeakyReLU activations with alpha=0.2
        3. Minimal regularization appropriate for 2D input
        4. Final sigmoid output layer for binary classification
        
        Architecture: 2 → 32 → 16 → 1 (~600 parameters vs 525K)
        
        Key improvements:
        - Appropriate complexity for 2D latent classification task
        - Balanced with encoder for stable adversarial training
        - Much faster training and better gradient flow
        - Prevents discriminator from being too powerful
        
        Returns:
            tf.keras.Model: The built discriminator model.
        """
        model = Sequential()
        
        # First hidden layer: 2 → 32
        model.add(Dense(32, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        
        # Second hidden layer: 32 → 16
        model.add(Dense(16))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))  # Light regularization
        
        # Output layer: 16 → 1
        model.add(Dense(1, activation="sigmoid"))

        # Create the input layer and connect it to the model
        encoded_repr = Input(shape=(self.latent_dim,))
        validity = model(encoded_repr)
        
        # Create a full model that takes the encoded representation and outputs validity
        discriminator_model = Model(encoded_repr, validity, name='discriminator')

        return discriminator_model