import numpy as np
import os
import tensorflow as tf
from src.training.train import train_model, train_cellfate
from src.evaluation.evaluate import Evaluation

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
    print(recon_imgs)
 
    evaluator.reconstruction_images(x_train, recon_imgs[:,:,:,0])
    
    if full_evaluation: ##TODO later
        # Predict labels and plot confusion matrix
        test_latent_space, _ = encoder.predict(x_test)
        y_pred = classifier.predict(test_latent_space) 
        evaluator.plot_confusion_matrix(y_test, y_pred, num_classes=2)
        
        # Visualize latent space
        evaluator.visualize_latent_space(z_imgs, y_train)

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
        'latent_dim': 20,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5, 
        'lambda_adv': 0.05,
        'lambda_clf': 0.05,
    }

    # Train the autoencoder model
    autoencoder_results = train_model(config, x_train)
    encoder = autoencoder_results['encoder']
    decoder = autoencoder_results['decoder']
    discriminator = autoencoder_results['discriminator']

    config = {
        'batch_size': 30,
        'epochs': 30,
        'learning_rate': 0.0001,
        'seed': 42,
        'latent_dim': 20,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5, 
        'lambda_adv': 0.05,
        'lambda_clf': 0.05,
    }

    # Train the full model
    full_model_results = train_cellfate(config, encoder, decoder, discriminator, x_train, y_train, x_test, y_test)
    final_encoder = full_model_results['encoder']
    final_decoder = full_model_results['decoder']
    classifier = full_model_results['classifier']

    # Evaluate the model
    evaluate_model(encoder, final_decoder, classifier, x_train, y_train, x_test, y_test, full_evaluation=True)

if __name__ == '__main__':
    main()