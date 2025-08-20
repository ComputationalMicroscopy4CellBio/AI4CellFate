import numpy as np
from .training.train import *
from .evaluation.evaluate import evaluate_model
from .utils import *
from .preprocessing.preprocessing_functions import augment_dataset, augmentations

# Function to load data
def load_data():
    """Load training and testing data."""
    # TODO: replace with data loader

    # Augmented data - FIRST FRAME ONLY
    # augmented_x_train = np.load('./data/final_split/augmented_x_train.npy')
    # augmented_y_train = np.load('./data/final_split/augmented_y_train.npy')
    # x_val = np.load('./data/final_split/x_val.npy')
    # y_val = np.load('./data/final_split/y_val.npy')
    # x_test = np.load('./data/final_split/x_test.npy')
    # y_test = np.load('./data/final_split/y_test.npy')

    augmented_x_train = np.load('/Users/inescunha/Documents/GitHub/AI4CellFate/data/second_generation/first_gen_augmented_images.npy')
    augmented_y_train = np.load('/Users/inescunha/Documents/GitHub/AI4CellFate/data/second_generation/first_gen_augmented_labels.npy')
    # x_val = np.load('/Users/inescunha/Documents/GitHub/AI4CellFate/data/second_generation/first_gen_val_images.npy')
    # y_val = np.load('/Users/inescunha/Documents/GitHub/AI4CellFate/data/second_generation/first_gen_val_labels.npy')
    second_gen_images = np.load('/Users/inescunha/Documents/GitHub/AI4CellFate/data/second_generation/stretched_second_gen.npy')
    second_gen_labels = np.load('/Users/inescunha/Documents/GitHub/AI4CellFate/data/second_generation/second_gen_labels.npy')

    x_val, x_test, y_val, y_test = train_test_split(
    second_gen_images, second_gen_labels,
    test_size=0.5,  # 50% of 40% = 20% of total
    random_state=42,
    stratify=second_gen_labels  # Keep class balance
)
    
    return augmented_x_train, x_val, x_test, augmented_y_train, y_val, y_test


# Main function
def main():
    """Main function with the full workflow of the AI4CellFate project."""
    
    # Load data
    augmented_x_train, x_val, x_test, augmented_y_train, y_val, y_test = load_data() 

    ##### STAGE 1 #####
    # Train Autoencoder (To wait for the reconstruction losses to converge before training the AI4CellFate model)

    config_autoencoder = {
        'batch_size': 30,
        'epochs': 50, 
        'learning_rate': 0.0001,
        'seed': 42,
        'latent_dim': 3,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5,
        'lambda_adv': 1,
    }

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
        'latent_dim': 3,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 6,
        'lambda_adv': 4,
        'lambda_cov': 1,
        'lambda_contra': 20,
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