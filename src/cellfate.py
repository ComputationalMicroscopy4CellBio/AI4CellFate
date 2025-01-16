import numpy as np
import os
import tensorflow as tf
from src.training.train import train_model 

# Main function for training
def main():
    # Load the data
    x_train = np.load('./data/stretched_x_train.npy') #TODO later: replace with data loader

    # Config for training
    config = {
        'batch_size': 30,
        'epochs': 3,
        'learning_rate': 0.001,
        'seed': 42,
        'latent_dim': 20,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5, 
        'lambda_adv': 0.05,
    }

    # Train the autoencoder
    train_model(config, x_train)

if __name__ == '__main__':
    main()
