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

    first_gen_images = np.load('/Users/inescunha/Documents/GitHub/AI4CellFate/data/second_generation/stretched_first_gen.npy')
    first_gen_labels = np.load('/Users/inescunha/Documents/GitHub/AI4CellFate/data/second_generation/first_gen_labels.npy')
    second_gen_images = np.load('/Users/inescunha/Documents/GitHub/AI4CellFate/data/second_generation/stretched_second_gen.npy')
    second_gen_labels = np.load('/Users/inescunha/Documents/GitHub/AI4CellFate/data/second_generation/second_gen_labels.npy')

#     augmented_x_train, augmented_y_train = augment_dataset(second_gen_images, second_gen_labels, augmentations)

#     x_val, x_test, y_val, y_test = train_test_split(
#     first_gen_images, first_gen_labels,
#     test_size=0.5,  # 50% of 40% = 20% of total
#     random_state=42,
#     stratify=first_gen_labels  # Keep class balance
# )

    combined_images = second_gen_images
    combined_labels = second_gen_labels

    # Combine first and second generation data
    #combined_images = np.concatenate([first_gen_images, second_gen_images], axis=0)
    #combined_labels = np.concatenate([first_gen_labels, second_gen_labels], axis=0)

    print(f"Combined dataset shape: {combined_images.shape}")
    print(f"Combined labels shape: {combined_labels.shape}")
    print(f"Label distribution: {np.bincount(combined_labels)}")

    # First split: 60% train, 40% temp (which will be split into 20% val, 20% test)
    x_train, x_temp, y_train, y_temp = train_test_split(
        combined_images, combined_labels,
        test_size=0.4,  # 40% for temp (val + test)
        random_state=42,
        stratify=combined_labels  # Keep class balance
    )

    # Second split: Split the temp 40% into 20% val and 20% test (50/50 split of temp)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp,
        test_size=0.5,  # 50% of 40% = 20% of total for test, 20% for val
        random_state=42,
        stratify=y_temp  # Keep class balance
    )

    print(f"Train set: {x_train.shape[0]} samples ({x_train.shape[0]/len(combined_images)*100:.1f}%)")
    print(f"Val set: {x_val.shape[0]} samples ({x_val.shape[0]/len(combined_images)*100:.1f}%)")
    print(f"Test set: {x_test.shape[0]} samples ({x_test.shape[0]/len(combined_images)*100:.1f}%)")

    # Augment only the training set
    augmented_x_train, augmented_y_train = augment_dataset(x_train, y_train, augmentations)

    print(f"Augmented train set: {augmented_x_train.shape[0]} samples")
    
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
        'epochs': 40, 
        'learning_rate': 0.0001,
        'seed': 42,
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
        'seed': 42,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 6,
        'lambda_adv': 4,
        'lambda_cov': 1,
        'lambda_contra': 14, 
    }

    # Create parameter-based folder name
    folder_name = (f"s1_ep{config_autoencoder['epochs']}_lr{config_autoencoder['lambda_recon']}"
                   f"_la{config_autoencoder['lambda_adv']}_seed{config_autoencoder['seed']}"
                   f"_ldim{config_autoencoder['latent_dim']}_s2_lr{config_ai4cellfate['lambda_recon']}"
                   f"_la{config_ai4cellfate['lambda_adv']}_lc{config_ai4cellfate['lambda_cov']}"
                   f"_lcon{config_ai4cellfate['lambda_contra']}")
    
    output_base_dir = f"./results/{folder_name}"
    print(f"Saving results to: {output_base_dir}")

    lambda_autoencoder_results = train_autoencoder(config_autoencoder, augmented_x_train, x_val)
    encoder = lambda_autoencoder_results['encoder']
    decoder = lambda_autoencoder_results['decoder']
    discriminator = lambda_autoencoder_results['discriminator']

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