import numpy as np
from src.training.new_optimised_train import *
from src.evaluation.evaluate import Evaluation

# Function to load data
def load_data():
    """Load training and testing data."""
    x_train = np.load('./data/stretched_x_train.npy')  # TODO: replace with data loader later
    x_test = np.load('./data/stretched_x_test.npy')
    y_train = np.load('./data/train_labels.npy')
    y_test = np.load('./data/test_labels.npy')
    return x_train, x_test, y_train, y_test

def evaluate_model(encoder, decoder, x_train, y_train, full_evaluation=False, output_dir="./results/evaluation"):
    """Evaluate the trained model."""
    evaluator = Evaluation(output_dir)
    
    print("HERE")
    z_imgs = encoder.predict(x_train)
    recon_imgs = decoder.predict(z_imgs)
    #print(recon_imgs)
 
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
    }

    # Train the lambda optimisation autoencoder
    lambda_autoencoder_results = train_lambdas_autoencoder(config, x_train)
    encoder = lambda_autoencoder_results['encoder']
    decoder = lambda_autoencoder_results['decoder']
    discriminator = lambda_autoencoder_results['discriminator']
    reconstruction_losses = lambda_autoencoder_results['recon_loss']
    adversarial_losses = lambda_autoencoder_results['adv_loss']

    # Train the autoencoder starting from the optimal lambdas
    scaled_autoencoder_results = train_autoencoder_scaled(config, x_train, reconstruction_losses, adversarial_losses, encoder, decoder, discriminator)


    #evaluate_model(encoder_cov, decoder_cov, 0, x_train, y_train, x_test, y_test, full_evaluation=True)
 
if __name__ == '__main__':
    main()