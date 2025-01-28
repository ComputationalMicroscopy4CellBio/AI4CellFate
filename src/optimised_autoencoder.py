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

def evaluate_model(encoder, decoder, classifier, x_train, y_train, x_test, y_test, full_evaluation=False, output_dir="./results/diff_autoencoders"):
    """Evaluate the trained model."""
    evaluator = Evaluation(output_dir)
    
    z_imgs = encoder.predict(x_train)
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


def train_with_multiple_architectures(x_train, architectures):

    results_dir = "./results/diff_autoencoders"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    x_train, x_test, y_train, y_test = load_data()

    # Config for training autoencoder
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
    
    img_shape = (x_train.shape[1], x_train.shape[2], 1) # Assuming grayscale images
    #x_train = tf.expand_dims(x_train, axis=-1)

    for arch_name, build_autoencoder in architectures.items():
        print(f"Training with architecture: {arch_name}")

        # Build architecture-specific encoder and decoder
        encoder, decoder = build_autoencoder(img_shape, config['latent_dim'])
        discriminator = Discriminator(latent_dim=config['latent_dim']).model

        # Train autoencoder
        arch_dir = os.path.join(results_dir, arch_name)
        os.makedirs(arch_dir, exist_ok=True)

        trained_models = train_autoencoder_scaled(config, x_train, encoder, decoder, discriminator, save_loss_plot=False, save_model_weights=False)
        print("AUTOENCODER TRAINED")
        evaluate_model(encoder, decoder, None, x_train, y_train, x_test, y_test)

        # Train CellFate model

    #     config = {
    #     'batch_size': 30,
    #     'epochs': 10,
    #     'learning_rate': 0.001,
    #     'seed': 42,
    #     'latent_dim': 10,
    #     'GaussianNoise_std': 0.003,
    #     'lambda_recon': 0.7085, 
    #     'lambda_adv': 0.1094,
    #     'lambda_clf': 0.1,
    #     'lambda_cov': 0.1821,
    # }

        cov_results = train_cov_scaled(config, trained_models['encoder'], trained_models['decoder'], trained_models['discriminator'], 
                  x_train, None, save_loss_plot=True, save_model_weights=True, save_every_epoch=False)
        
        evaluate_model(cov_results['encoder'], cov_results['decoder'], 0, x_train, y_train, x_test, y_test, full_evaluation=True)

        print(f"Results saved for {arch_name} under {arch_dir}")


if __name__ == '__main__':
    architectures = {
    "simple_autoencoder": build_simple_autoencoder,
    "deeper_autoencoder": build_deeper_autoencoder,
    "residual_autoencoder": build_residual_autoencoder
    }
    x_train, x_test, y_train, y_test = load_data()

    train_with_multiple_architectures(x_train=x_train, architectures=architectures)
