from tensorflow.keras import layers, Sequential
import tensorflow as tf
from tensorflow.keras.regularizers import l2

# A small MLP for classification from the latent space
def mlp_classifier(latent_dim):
    return Sequential([
        layers.Input(shape=(latent_dim,)),
        # layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)), #
        # layers.Dropout(0.3),
        # layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        # layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')
    ])