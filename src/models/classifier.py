from tensorflow.keras import layers, Sequential

# A small MLP for classification from the latent space
def create_mlp(latent_dim):
    return Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(2, activation='softmax')  # Output probabilities for two classes
    ])