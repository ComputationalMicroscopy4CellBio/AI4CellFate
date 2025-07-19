import numpy as np
from .training.train import *
from .evaluation.evaluate import evaluate_model
from .utils import *
from .preprocessing.preprocessing_functions import augment_dataset, augmentations

# Function to load data
def load_data():
    """Load training and testing data."""
    # TODO: replace with data loader

    # Augmented data
    # x_train = np.load('./data/images/train_images_augmented.npy')[:,0,:,:]
    # y_train = np.load('./data/labels/train_labels_augmented.npy')
    # x_test = np.load('./data/images/test_time_norm.npy')[:,0,:,:]
    # y_test = np.load('./data/labels/test_labels.npy')

    # Non augmented data
    x_train = np.load('./data/images/train_no_aug_time_norm.npy')[:,0,:,:]  
    y_train = np.load('./data/labels/train_labels.npy')  
    x_test = np.load('./data/images/test_time_norm.npy')[:,0,:,:]
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
        'learning_rate': 0.0001,
        'seed': 42,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5,
        'lambda_adv': 1,
    }

    # Split training data into train and validation sets
    x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Augment training set
    augmented_x_train, augmented_y_train = augment_dataset(
                x_train_, 
                y_train_, 
                augmentations, 
                augment_times=5,
                seed=42
            )

    lambda_autoencoder_results = train_autoencoder(config_autoencoder, augmented_x_train, x_val)
    encoder = lambda_autoencoder_results['encoder']
    decoder = lambda_autoencoder_results['decoder']
    discriminator = lambda_autoencoder_results['discriminator']

    # Evaluate the trained model (store latent space and reconstructed images)
    evaluate_model(encoder, decoder, augmented_x_train, augmented_y_train, output_dir="./results/optimisation/autoencoder")

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
 
    lambda_ae_cov_results = train_cellfate(config_ai4cellfate, encoder, decoder, discriminator, augmented_x_train, augmented_y_train, x_val, y_val, x_test, y_test) 
    encoder = lambda_ae_cov_results['encoder']
    decoder = lambda_ae_cov_results['decoder']
    discriminator = lambda_ae_cov_results['discriminator']

    print(lambda_ae_cov_results['good_conditions_stop'])
    save_model_weights_to_disk(encoder, decoder, discriminator, output_dir="./results/models/autoencoder_cov")

    # Evaluate the trained model (store latent space and reconstructed images)
    evaluate_model(lambda_ae_cov_results['encoder'], lambda_ae_cov_results['decoder'], augmented_x_train, augmented_y_train, output_dir="./results/optimisation/autoencoder_cov")
    

if __name__ == '__main__':
    main()