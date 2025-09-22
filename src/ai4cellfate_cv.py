import numpy as np
from .training.cross_validation import CrossValidation
from .training.train import train_autoencoder, train_cellfate
from .evaluation.evaluate import evaluate_model
from .utils import *

def load_original_data():
    """Load original training data (before augmentation) for cross-validation."""
    # Load original training data (not augmented)
    x_train_orig = np.load('./data/final_split/x_train.npy') # Use original, not augmented
    y_train_orig = np.load('./data/final_split/y_train.npy')  # Use original, not augmented
    x_val = np.load('./data/final_split/x_val.npy')
    y_val = np.load('./data/final_split/y_val.npy')
    # Combine train and val
    combined_x_train = np.concatenate((x_train_orig, x_val), axis=0)
    combined_y_train = np.concatenate((y_train_orig, y_val), axis=0)

    # Load test data (this remains unchanged)
    x_test = np.load('./data/final_split/x_test.npy')
    y_test = np.load('./data/final_split/y_test.npy')
    
    return combined_x_train, combined_y_train, x_test, y_test

def run_cross_validation(k_folds=5, random_state=42):
    """
    Run k-fold cross-validation for AI4CellFate model.
    
    Args:
        k_folds (int): Number of folds for cross-validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Cross-validation results
    """
    
    # Load original training data (before augmentation)
    x_train_orig, y_train_orig, x_test, y_test = load_original_data()
    
    print(f"Original training data shape: {x_train_orig.shape}")
    print(f"Original training labels shape: {y_train_orig.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Configuration for autoencoder training
    config_autoencoder = {
        'batch_size': 30,
        'epochs': 35,
        'learning_rate': 0.0001,
        'seed': random_state,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5,
        'lambda_adv': 1,
    }
    
    # Configuration for AI4CellFate training
    config_ai4cellfate = {
        'batch_size': 30,
        'epochs': 100,
        'learning_rate': 0.001,
        'seed': random_state,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 6,
        'lambda_adv': 4,
        'lambda_cov': 1,
        'lambda_contra': 8,
    }
    
    # Initialize cross-validation
    cv = CrossValidation(
        k_folds=k_folds, 
        random_state=random_state,
        save_individual_models=True  # Set to True if you want to save models from each fold
    )
    
    # Run cross-validation
    cv_results = cv.run_cross_validation(
        x_train_orig, 
        y_train_orig,
        x_test,
        y_test,
        config_autoencoder,
        config_ai4cellfate,
        output_dir="./results/cross_validation",
        apply_augmentation=True  # Apply augmentation to each fold
    )

    # TODO: ADD EVALUATION TO FINAL TRAINED ENCODER AND DECODER Evaluate the trained model (store latent space and reconstructed images)
    #evaluate_model(lambda_ae_cov_results['encoder'], lambda_ae_cov_results['decoder'], x_train, y_train, output_dir="./results/optimisation/autoencoder_cov")
    
    
    return cv_results

# def run_final_model_with_cv_best_params(cv_results):
#     """
#     Train final model using best parameters from cross-validation.
    
#     Args:
#         cv_results (dict): Results from cross-validation
        
#     Returns:
#         dict: Final model results
#     """
    
#     # Load all training data (including augmented)
#     x_train = np.load('./data/images/train_images_augmented.npy')[:,0,:,:]
#     y_train = np.load('./data/labels/train_labels_augmented.npy')
#     x_test = np.load('./data/images/test_images.npy')[:,0,:,:]
#     y_test = np.load('./data/labels/test_labels.npy')
    
#     print(f"\nTraining final model with all training data...")
#     print(f"Training data shape: {x_train.shape}")
#     print(f"Test data shape: {x_test.shape}")
    
#     # Use the same configurations as in cross-validation
#     # In a real scenario, you might want to adjust these based on CV results
#     config_autoencoder = {
#         'batch_size': 30,
#         'epochs': 15,
#         'learning_rate': 0.001,
#         'seed': 42,
#         'latent_dim': 2,
#         'GaussianNoise_std': 0.003,
#         'lambda_recon': 5,
#         'lambda_adv': 1,
#     }
    
#     config_ai4cellfate = {
#         'batch_size': 30,
#         'epochs': 100,
#         'learning_rate': 0.001,
#         'seed': 42,
#         'latent_dim': 2,
#         'GaussianNoise_std': 0.003,
#         'lambda_recon': 6,
#         'lambda_adv': 4,
#         'lambda_cov': 0.0001,
#         'lambda_contra': 8,
#     }
    
#     # STAGE 1: Train Autoencoder
#     print("\nStage 1: Training final autoencoder...")
#     autoencoder_results = train_autoencoder(config_autoencoder, x_train)
#     encoder = autoencoder_results['encoder']
#     decoder = autoencoder_results['decoder']
#     discriminator = autoencoder_results['discriminator']
    
#     # Evaluate the autoencoder
#     evaluate_model(encoder, decoder, x_train, y_train, output_dir="./results/final_model/autoencoder")
    
#     # STAGE 2: Train AI4CellFate
#     print("\nStage 2: Training final AI4CellFate model...")
#     ai4cellfate_results = train_cellfate(
#         config_ai4cellfate, 
#         encoder, 
#         decoder, 
#         discriminator, 
#         x_train, 
#         y_train, 
#         x_test, 
#         y_test
#     )
    
#     encoder = ai4cellfate_results['encoder']
#     decoder = ai4cellfate_results['decoder']
#     discriminator = ai4cellfate_results['discriminator']
    
#     print(ai4cellfate_results['good_conditions_stop'])
    
#     # Save final model
#     save_model_weights_to_disk(encoder, decoder, discriminator, output_dir="./results/final_model/ai4cellfate")
    
#     # Evaluate final model
#     evaluate_model(encoder, decoder, x_train, y_train, output_dir="./results/final_model/evaluation")
    
#     return ai4cellfate_results

# def compare_cv_with_simple_split():
#     """
#     Compare cross-validation results with simple train/test split.
    
#     Returns:
#         dict: Comparison results
#     """
#     print("=== Comparing Cross-Validation with Simple Train/Test Split ===\n")
    
#     # Run cross-validation
#     print("1. Running Cross-Validation...")
#     cv_results = run_cross_validation(k_folds=5, random_state=42)
    
#     # Run simple train/test split (original method)
#     print("\n2. Running Simple Train/Test Split...")
#     x_train = np.load('./data/images/train_images_augmented.npy')[:,0,:,:]
#     y_train = np.load('./data/labels/train_labels_augmented.npy')
#     x_test = np.load('./data/images/test_images.npy')[:,0,:,:]
#     y_test = np.load('./data/labels/test_labels.npy')
    
#     config_autoencoder = {
#         'batch_size': 30,
#         'epochs': 15,
#         'learning_rate': 0.001,
#         'seed': 42,
#         'latent_dim': 2,
#         'GaussianNoise_std': 0.003,
#         'lambda_recon': 5,
#         'lambda_adv': 1,
#     }
    
#     config_ai4cellfate = {
#         'batch_size': 30,
#         'epochs': 100,
#         'learning_rate': 0.001,
#         'seed': 42,
#         'latent_dim': 2,
#         'GaussianNoise_std': 0.003,
#         'lambda_recon': 6,
#         'lambda_adv': 4,
#         'lambda_cov': 0.0001,
#         'lambda_contra': 8,
#     }
    
#     # Train models
#     autoencoder_results = train_autoencoder(config_autoencoder, x_train)
#     ai4cellfate_results = train_cellfate(
#         config_ai4cellfate, 
#         autoencoder_results['encoder'], 
#         autoencoder_results['decoder'], 
#         autoencoder_results['discriminator'], 
#         x_train, 
#         y_train, 
#         x_test, 
#         y_test
#     )
    
#     # Compare results
#     print("\n=== Results Comparison ===")
#     print(f"Cross-Validation Results:")
#     print(f"  Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
#     print(f"  Mean F1-Score: {cv_results['mean_f1_score']:.4f} ± {cv_results['std_f1_score']:.4f}")
    
#     print(f"\nSimple Train/Test Split:")
#     print(f"  Test performance would need to be calculated on the test set")
#     print(f"  (This gives you a single performance estimate)")
    
#     return {
#         'cv_results': cv_results,
#         'simple_split_results': ai4cellfate_results
#     }

def main():
    """
    Main function with cross-validation workflow.
    Choose between different execution modes.
    """
    
    import argparse
    parser = argparse.ArgumentParser(description='AI4CellFate with Cross-Validation')
    parser.add_argument('--mode', choices=['cv', 'final', 'compare'], default='cv',
                       help='Execution mode: cv (cross-validation), final (final model), compare (comparison)')
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    #if args.mode == 'cv':
    # Run only cross-validation
    print(f"Running cross-validation mode cv with {args.k_folds} folds and random state {args.random_state}")
    cv_results = run_cross_validation(k_folds=args.k_folds, random_state=args.random_state)
    print(f"\nCross-validation completed. Results saved to ./results/cross_validation/")
        
    # elif args.mode == 'final':
    #     # Run cross-validation first, then final model
    #     print(f"Running cross-validation mode final with {args.k_folds} folds and random state {args.random_state}")
    #     cv_results = run_cross_validation(k_folds=args.k_folds, random_state=args.random_state)
    #     final_results = run_final_model_with_cv_best_params(cv_results)
    #     print(f"\nFinal model training completed. Results saved to ./results/final_model/")
        
    # elif args.mode == 'compare':
    #     # Compare CV with simple split
    #     print(f"Running comparison mode compare")
    #     comparison_results = compare_cv_with_simple_split()
    #     print(f"\nComparison completed. Results saved to respective directories.")

if __name__ == '__main__':
    main() 