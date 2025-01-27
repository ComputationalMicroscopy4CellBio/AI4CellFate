import numpy as np
import os
import tensorflow as tf
from src.training.train import train_autoencoder, train_cellfate, train_cov
#from src.training.optimised_train import train_cov
from src.evaluation.evaluate import Evaluation
from src.models import Encoder, Decoder, Discriminator
from src.models.autoencoders import *
from src.training.loss_functions import cov_loss_terms
from src.training.optimised_train import train_cov_scaled, train_autoencoder_scaled

def load_data():
    """Load training and testing data."""
    x_train = np.load('./data/stretched_x_train.npy')  # TODO: replace with data loader later
    x_test = np.load('./data/stretched_x_test.npy')
    y_train = np.load('./data/train_labels.npy')
    y_test = np.load('./data/test_labels.npy')
    return x_train, x_test, y_train, y_test

def evaluate_model(encoder, decoder, classifier, x_train, y_train, x_test, y_test, full_evaluation=False, output_dir="./results/optimisation"):
    """Evaluate the trained model."""
    evaluator = Evaluation(output_dir)
    
    print("HERE")
    z_imgs, _ = encoder.predict(x_train)
    recon_imgs = decoder.predict(z_imgs)
 
    evaluator.reconstruction_images(x_train, recon_imgs[:,:,:,0], epoch=0)
    
    if full_evaluation: 
        # Visualize latent space
        evaluator.visualize_latent_space(z_imgs, y_train, epoch=0)

        # Covariance matrix
        cov_matrix = cov_loss_terms(z_imgs)[0]
        evaluator.plot_cov_matrix(cov_matrix, epoch=0)

        # KL divergence
        print("KL Divergences in each dimension: ", evaluator.calculate_kl_divergence(z_imgs))


# Main function
def main():
    """Main function with the full workflow of the CellFate project."""
    
    # Load data
    x_train, x_test, y_train, y_test = load_data()

    # Config for training
    config = {
        'batch_size': 30,
        'epochs': 10,
        'learning_rate': 0.001,
        'seed': 42,
        'latent_dim': 10,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 1, 
        'lambda_adv': 0.18156,
        'lambda_clf': 0.5, #0.05 for the mlp
        'lambda_cov': 0.1,
    }

    # Train the autoencoder model
    autoencoder_results = train_autoencoder_scaled(config, x_train)
    encoder = autoencoder_results['encoder']
    decoder = autoencoder_results['decoder']
    discriminator = autoencoder_results['discriminator']

    evaluate_model(encoder, decoder, None, x_train, y_train, x_test, y_test)

    config = {
        'batch_size': 30,
        'epochs': 50,
        'learning_rate': 0.001,
        'seed': 42,
        'latent_dim': 10,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 0.7085, 
        'lambda_adv': 0.1094,
        'lambda_clf': 0.1,
        'lambda_cov': 0.1821,
    }

    #Train the model with cov only
    cov_model_results = train_cov_scaled(config, encoder, decoder, discriminator, x_train, y_train)
    encoder_cov = cov_model_results['encoder']
    decoder_cov = cov_model_results['decoder']
    discriminator_cov = cov_model_results['discriminator']

    evaluate_model(encoder_cov, decoder_cov, 0, x_train, y_train, x_test, y_test, full_evaluation=True)
