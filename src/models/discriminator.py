import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, GaussianNoise,concatenate, Dense, Dropout, Conv2D, Add, UpSampling2D, Dot, Conv2DTranspose, Activation, Reshape, InputSpec, LeakyReLU, Flatten, BatchNormalization, SpectralNormalization, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

class Discriminator:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.model = self.build_discriminator()

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(BatchNormalization(scale=False))
        model.add(LeakyReLU(alpha=0.2))
        
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        
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