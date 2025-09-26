import numpy as np
from sklearn.model_selection import train_test_split
from ..training.train import *
from ..evaluation.evaluate import evaluate_model
from ..utils import *
from ..preprocessing.preprocessing_functions import augment_dataset, augmentations

# Function to load data
def load_data():
    """Load training and testing data."""
    # TODO: replace with data loader

    x_train = np.load('./data/images/train_no_aug_time_norm.npy')[:,0,:,:] # FIRST FRAME ONLY
    y_train = np.load('./data/labels/train_labels.npy')  
    x_test = np.load('./data/images/test_time_norm.npy')[:,0,:,:] # FIRST FRAME ONLY
    y_test = np.load('./data/labels/test_labels.npy')
    
    print(f"Train set: {x_train.shape[0]} samples")
    print(f"Train labels: {y_train.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]} samples")
    print(f"Test labels: {y_test.shape[0]} samples")
    
    return x_train, x_test, y_train, y_test


# Main function
def main():
    """Main function with the full workflow of the AI4CellFate project."""
    
    # Load data
    x_train, x_test, y_train, y_test = load_data() 

    config_autoencoder = {
        'batch_size': 30,
        'epochs': 35, 
        'learning_rate': 0.0001,
        'seed': 42,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5,
        'lambda_adv': 1,
    }

    config_ai4cellfate = {
        'batch_size': 30,
        'epochs': 100,
        'learning_rate': 0.001,
        'seed': 42,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 6,
        'lambda_adv': 4,
        'lambda_cov': 1,
        'lambda_contra': 8, 
    }

    dataset_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for size in dataset_size:

        ###### Split into train, validation and test set ####

        x_all = np.concatenate([x_train, x_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)

        print(f"Combined data shape: {x_all.shape}")
        print(f"Combined labels shape: {y_all.shape}")

        # First split: 60% train, 40% temp (which will become 20% val + 20% test)
        x_train_new, x_temp, y_train_new, y_temp = train_test_split(
            x_all, y_all, 
            test_size=0.4,  # 40% for temp (val + test)
            random_state=42, 
            stratify=y_all  # Keep class balance
        )

        # Second split: Split the 40% temp into 20% val + 20% test
        x_val, x_test_new, y_val, y_test_new = train_test_split(
            x_temp, y_temp,
            test_size=0.5,  # 50% of 40% = 20% of total
            random_state=42,
            stratify=y_temp  # Keep class balance
        )

        #### NOW WE HAVE x_train_new, x_val, x_test_new, y_train_new, y_val, y_test_new ####
        np.random.seed(42)
        less_indexes = np.random.choice(np.arange(len(y_train_new)), int(size * len(y_train_new)), replace=False)

        smaller_x_train = x_train_new[np.sort(less_indexes)]
        smaller_y_train = y_train_new[np.sort(less_indexes)]

        augmented_x_train, augmented_y_train = augment_dataset(
                smaller_x_train, 
                smaller_y_train, 
                augmentations, 
                augment_times=5,
                seed=42
            )

        # Create parameter-based folder name
        folder_name = (f"data_size_study{size}_s1_ep{config_autoencoder['epochs']}_lr{config_autoencoder['lambda_recon']}"
                    f"_la{config_autoencoder['lambda_adv']}_seed{config_autoencoder['seed']}"
                    f"_ldim{config_autoencoder['latent_dim']}_s2_lr{config_ai4cellfate['lambda_recon']}"
                    f"_la{config_ai4cellfate['lambda_adv']}_lc{config_ai4cellfate['lambda_cov']}"
                    f"_lcon{config_ai4cellfate['lambda_contra']}")
        
        output_base_dir = f"./results/{folder_name}"
        print(f"Saving results to: {output_base_dir}")


        ##### STAGE 1 #####
        # Train Autoencoder (To wait for the reconstruction losses to converge before training the AI4CellFate model)

        lambda_autoencoder_results = train_autoencoder(config_autoencoder, augmented_x_train, x_val, output_dir=output_base_dir)
        encoder = lambda_autoencoder_results['encoder']
        decoder = lambda_autoencoder_results['decoder']
        discriminator = lambda_autoencoder_results['discriminator']

        save_model_weights_to_disk(encoder, decoder, discriminator, output_dir=f"{output_base_dir}/models_stage1")
        # Evaluate the trained model (store latent space and reconstructed images)
        evaluate_model(encoder, decoder, augmented_x_train, augmented_y_train, output_dir=f"{output_base_dir}/stage1")

        ##### STAGE 2#####
        # Train AI4CellFate: Autoencoder + Covariance + Contrastive (Engineered Latent Space)

        lambda_ae_cov_results = train_cellfate(config_ai4cellfate, encoder, decoder, discriminator, augmented_x_train, augmented_y_train, x_val, y_val, x_test_new, y_test_new, output_dir=output_base_dir) 
        encoder = lambda_ae_cov_results['encoder']
        decoder = lambda_ae_cov_results['decoder']
        discriminator = lambda_ae_cov_results['discriminator']

        print(lambda_ae_cov_results['good_conditions_stop'])
        save_model_weights_to_disk(encoder, decoder, discriminator, output_dir=f"{output_base_dir}/models")

        # Evaluate the trained model (store latent space and reconstructed images)
        evaluate_model(lambda_ae_cov_results['encoder'], lambda_ae_cov_results['decoder'], augmented_x_train, augmented_y_train, output_dir=f"{output_base_dir}/stage2")
    
if __name__ == '__main__':
    main()