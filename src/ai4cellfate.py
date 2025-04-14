import numpy as np
from src.training.train import *
from src.evaluation.evaluate import evaluate_model
from src.utils import *

# Function to load data
def load_data():
    """Load training and testing data."""
    # TODO: replace with data loader

    x_train = np.load('./data/images/train_images_augmented.npy')[:,0,:,:]
    y_train = np.load('./data/labels/train_labels_augmented.npy')
    x_test = np.load('./data/images/test_images.npy')[:,0,:,:]
    y_test = np.load('./data/labels/test_labels.npy')
    
    return x_train, x_test, y_train, y_test


# Main function
def main():
    """Main function with the full workflow of the AI4CellFate project."""
    
    # Load data
    x_train, x_test, y_train, y_test = load_data() 


    ##### STAGE 1 #####
    # Train Autoencoder (To wait for the reconstruction losses to converge before training the AI4CellFate model)

    config_autoencoder = {
        'batch_size': 30,
        'epochs': 15,
        'learning_rate': 0.001,
        'seed': 42,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5,
        'lambda_adv': 1,
    }

    lambda_autoencoder_results = train_autoencoder(config_autoencoder, x_train)
    encoder = lambda_autoencoder_results['encoder']
    decoder = lambda_autoencoder_results['decoder']
    discriminator = lambda_autoencoder_results['discriminator']

    # Evaluate the trained model (store latent space and reconstructed images)
    evaluate_model(encoder, decoder, x_train, y_train, output_dir="./results/optimisation/autoencoder")

    ##### STAGE 2#####
    # Train AI4CellFate: Autoencoder + Covariance + Contrastive (Engineered Latent Space)

    config_ai4cellfate = {
        'batch_size': 30,
        'epochs': 100,
        'learning_rate': 0.001,
        'seed': 42,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 6,
        'lambda_adv': 4,
        'lambda_cov': 0.0001,
        'lambda_contra': 8,
    }
 
    lambda_ae_cov_results = train_cellfate(config_ai4cellfate, encoder, decoder, discriminator, x_train, y_train, x_test, y_test) 
    encoder = lambda_ae_cov_results['encoder']
    decoder = lambda_ae_cov_results['decoder']
    discriminator = lambda_ae_cov_results['discriminator']

    print(lambda_ae_cov_results['good_conditions_stop'])
    save_model_weights_to_disk(encoder, decoder, discriminator, output_dir="./results/models/autoencoder_cov")

    # Evaluate the trained model (store latent space and reconstructed images)
    evaluate_model(lambda_ae_cov_results['encoder'], lambda_ae_cov_results['decoder'], x_train, y_train, output_dir="./results/optimisation/autoencoder_cov")
    

if __name__ == '__main__':
    main()