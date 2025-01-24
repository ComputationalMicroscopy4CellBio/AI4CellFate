import numpy as np
import os
import tensorflow as tf
from src.training.train import train_autoencoder, train_cellfate, train_cov
#from src.training.optimised_train import train_cov
from src.evaluation.evaluate import Evaluation
from src.models import Encoder, Decoder, Discriminator
from src.training.loss_functions import cov_loss_terms
from src.training.optimised_train import train_cov_scaled, train_autoencoder_scaled

# Function to load data
def load_data():
    """Load training and testing data."""
    x_train = np.load('./data/stretched_x_train.npy')  # TODO: replace with data loader later
    x_test = np.load('./data/stretched_x_test.npy')
    y_train = np.load('./data/train_labels.npy')
    y_test = np.load('./data/test_labels.npy')
    return x_train, x_test, y_train, y_test

# Function to evaluate the model
def evaluate_model(encoder, decoder, classifier, x_train, y_train, x_test, y_test, full_evaluation=False, output_dir="./results/evaluation"):
    """Evaluate the trained model."""
    evaluator = Evaluation(output_dir)
    
    print("HERE")
    z_imgs, _ = encoder.predict(x_train)
    recon_imgs = decoder.predict(z_imgs)
    #print(recon_imgs)
 
    evaluator.reconstruction_images(x_train, recon_imgs[:,:,:,0], epoch=0)
    
    if full_evaluation: 
        # Predict labels and plot confusion matrix
        # test_latent_space, _ = encoder.predict(x_test)
        # y_pred = classifier.predict(test_latent_space) 
        # evaluator.plot_confusion_matrix(y_test, y_pred, num_classes=2)
        
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
        # 'lambda_recon': 0.7054, 
        # 'lambda_adv': 0.1097,
        # 'lambda_clf': 0.1,
        # 'lambda_cov': 0.1848,
    }

    #Train the model with cov only
    cov_model_results = train_cov_scaled(config, encoder, decoder, discriminator, x_train, y_train)
    encoder_cov = cov_model_results['encoder']
    decoder_cov = cov_model_results['decoder']
    discriminator_cov = cov_model_results['discriminator']

    evaluate_model(encoder_cov, decoder_cov, 0, x_train, y_train, x_test, y_test, full_evaluation=True)

    # img_shape = (x_train.shape[1], x_train.shape[2], 1)
    # encoder = Encoder(img_shape=img_shape, latent_dim=config['latent_dim'], num_classes=2, gaussian_noise_std=config['GaussianNoise_std']).model
    # decoder = Decoder(latent_dim=config['latent_dim'], img_shape=img_shape, gaussian_noise_std=config['GaussianNoise_std']).model
    # discriminator = Discriminator(latent_dim=config['latent_dim']).model

    # encoder.load_weights("/Users/inescunha/Downloads/No_classifier_1000epochs/encoder.weights.h5")
    # decoder.load_weights("/Users/inescunha/Downloads/No_classifier_1000epochs/decoder.weights.h5")
    # discriminator.load_weights("/Users/inescunha/Downloads/No_classifier_1000epochs/discriminator.weights.h5")

   # Train the full model
#     full_model_results = train_cellfate(config, encoder, decoder, discriminator, x_train, y_train, x_test, y_test)
#     final_encoder = full_model_results['encoder']
#     final_decoder = full_model_results['decoder']
#     classifier = full_model_results['classifier']

#   #  Evaluate the model
#     evaluate_model(final_encoder, final_decoder, classifier, x_train, y_train, x_test, y_test, full_evaluation=True)

if __name__ == '__main__':
    main()