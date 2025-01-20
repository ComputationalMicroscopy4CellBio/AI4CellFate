import numpy as np
import os
import tensorflow as tf
#from src.training.train import train_model, train_cellfate, train_cov
from src.training.optimised_train import train_cov
from src.evaluation.evaluate import Evaluation
from src.training.loss_functions import cov_loss_terms

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
 
    evaluator.reconstruction_images(x_train, recon_imgs[:,:,:,0])
    
    if full_evaluation: 
        # Predict labels and plot confusion matrix
        # test_latent_space, _ = encoder.predict(x_test)
        # y_pred = classifier.predict(test_latent_space) 
        # evaluator.plot_confusion_matrix(y_test, y_pred, num_classes=2)
        
        # Visualize latent space
        evaluator.visualize_latent_space(z_imgs, y_train)

        # Covariance matrix
        cov_matrix = cov_loss_terms(z_imgs)[0]
        evaluator.plot_cov_matrix(cov_matrix)

        # KL divergence
        print("KL Divergences in each dimension: ", evaluator.calculate_kl_divergence(z_imgs))

# Main function
def main():
    """Main function with the full workflow of the CellFate project."""
    
    # Load data
    x_train, x_test, y_train, y_test = load_data()

    # Config for training
    # config = {
    #     'batch_size': 30,
    #     'epochs': 10,
    #     'learning_rate': 0.001,
    #     'seed': 69,
    #     'latent_dim': 10,
    #     'GaussianNoise_std': 0.003,
    #     'lambda_recon': 5, 
    #     'lambda_adv': 0.05,
    #     'lambda_clf': 0.05,
    #     'lambda_cov': 0.1,
    # }

    # # Train the autoencoder model
    # autoencoder_results = train_model(config, x_train)
    # encoder = autoencoder_results['encoder']
    # decoder = autoencoder_results['decoder']
    # discriminator = autoencoder_results['discriminator']

    #evaluate_model(encoder, decoder, None, x_train, y_train, x_test, y_test)

    config = {
        'batch_size': 30,
        'epochs': 10,
        'learning_rate': 0.001,
        'seed': 69,
        'latent_dim': 10,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5, 
        'lambda_adv': 0.05,
        'lambda_clf': 0.05,
        'lambda_cov': 0.1,
    }

    # Train the model with cov only
    full_model_results = train_cov(config, x_train, y_train)
    final_encoder = full_model_results['encoder']
    final_decoder = full_model_results['decoder']

    evaluate_model(final_encoder, final_decoder, 0, x_train, y_train, x_test, y_test, full_evaluation=True)

    # Train the full model
    # full_model_results = train_cellfate(config, encoder, decoder, discriminator, x_train, y_train, x_test, y_test)
    # final_encoder = full_model_results['encoder']
    # final_decoder = full_model_results['decoder']
    # classifier = full_model_results['classifier']

    # Evaluate the model
    #evaluate_model(final_encoder, final_decoder, classifier, x_train, y_train, x_test, y_test, full_evaluation=True)

if __name__ == '__main__':
    main()