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
    augmented_x_train = np.load('./data/final_split/augmented_x_train.npy')
    augmented_y_train = np.load('./data/final_split/augmented_y_train.npy')
    x_val = np.load('./data/final_split/x_val.npy')
    y_val = np.load('./data/final_split/y_val.npy')
    x_test = np.load('./data/final_split/x_test.npy')
    y_test = np.load('./data/final_split/y_test.npy')
    
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
        'epochs': 35, 
        'learning_rate': 0.0001,
        'seed': 43,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5,
        'lambda_adv': 1,
    }

    ##### STAGE 2#####
    # Train AI4CellFate: Autoencoder + Covariance + Contrastive (Engineered Latent Space)

    config_ai4cellfate = {
        'batch_size': 30,
        'epochs': 100,
        'learning_rate': 0.001,
        'seed': 43,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 6,
        'lambda_adv': 4,
        'lambda_cov': 1,
        'lambda_contra': 8, 
    }

    # Create parameter-based folder name
    folder_name = (f"s1_ep{config_autoencoder['epochs']}_lr{config_autoencoder['lambda_recon']}"
                   f"_la{config_autoencoder['lambda_adv']}_seed{config_autoencoder['seed']}"
                   f"_ldim{config_autoencoder['latent_dim']}_s2_lr{config_ai4cellfate['lambda_recon']}"
                   f"_la{config_ai4cellfate['lambda_adv']}_lc{config_ai4cellfate['lambda_cov']}"
                   f"_lcon{config_ai4cellfate['lambda_contra']}")
    
    output_base_dir = f"./results/{folder_name}"
    print(f"Saving results to: {output_base_dir}")

    lambda_autoencoder_results = train_autoencoder(config_autoencoder, augmented_x_train, x_val, output_dir=output_base_dir)
    encoder = lambda_autoencoder_results['encoder']
    decoder = lambda_autoencoder_results['decoder']
    discriminator = lambda_autoencoder_results['discriminator']

    save_model_weights_to_disk(encoder, decoder, discriminator, output_dir=f"{output_base_dir}/models_stage1")
    # Evaluate the trained model (store latent space and reconstructed images)
    evaluate_model(encoder, decoder, augmented_x_train, augmented_y_train, output_dir=f"{output_base_dir}/stage1")
    
    lambda_ae_cov_results = train_cellfate(config_ai4cellfate, encoder, decoder, discriminator, augmented_x_train, augmented_y_train, x_val, y_val, x_test, y_test, output_dir=output_base_dir) 
    encoder = lambda_ae_cov_results['encoder']
    decoder = lambda_ae_cov_results['decoder']
    discriminator = lambda_ae_cov_results['discriminator']

    print(lambda_ae_cov_results['good_conditions_stop'])
    save_model_weights_to_disk(encoder, decoder, discriminator, output_dir=f"{output_base_dir}/models")

    # Evaluate the trained model (store latent space and reconstructed images)
    evaluate_model(lambda_ae_cov_results['encoder'], lambda_ae_cov_results['decoder'], augmented_x_train, augmented_y_train, output_dir=f"{output_base_dir}/stage2")
    

if __name__ == '__main__':
    main()