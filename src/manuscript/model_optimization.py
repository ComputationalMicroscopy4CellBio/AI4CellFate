import numpy as np
import itertools
from ..training.train import *
from ..evaluation.evaluate import evaluate_model
from ..utils import *
from ..preprocessing.preprocessing_functions import augment_dataset, augmentations

frame_index = 1

# Function to load data
def load_data():
    """Load training and testing data."""
    
    augmented_x_train = np.load('/proj/cmcb/projects/AI4CellFate/AI4CellFate/data/train_images_aug.npy')[:, frame_index, :, :]
    augmented_y_train = np.load('/proj/cmcb/projects/AI4CellFate/AI4CellFate/data/train_labels_aug.npy')
    x_val = np.load('/proj/cmcb/projects/AI4CellFate/AI4CellFate/data/test_images.npy')[:, frame_index, :, :]
    y_val = np.load('/proj/cmcb/projects/AI4CellFate/AI4CellFate/data/test_labels.npy')
    x_test = np.load('/proj/cmcb/projects/AI4CellFate/AI4CellFate/data/test_images.npy')[:, frame_index, :, :]
    y_test = np.load('/proj/cmcb/projects/AI4CellFate/AI4CellFate/data/test_labels.npy')
    
    print(f"Augmented train set: {augmented_x_train.shape[0]} samples")
    print(f"Augmented train labels: {augmented_y_train.shape[0]} samples")
    print(f"Validation set: {x_val.shape[0]} samples")
    print(f"Validation labels: {y_val.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]} samples")
    print(f"Test labels: {y_test.shape[0]} samples")
    
    return augmented_x_train, x_val, x_test, augmented_y_train, y_val, y_test


def run_single_configuration(config_autoencoder, config_ai4cellfate, augmented_x_train, x_val, x_test, augmented_y_train, y_val, y_test):
    """Run a single AI4CellFate configuration (both stages) and save all results."""
    
    # Create parameter-based folder name
    folder_name = (f"s1_ep{config_autoencoder['epochs']}_lr{config_autoencoder['lambda_recon']}"
                   f"_la{config_autoencoder['lambda_adv']}_seed{config_autoencoder['seed']}"
                   f"_ldim{config_autoencoder['latent_dim']}_s2_lr{config_ai4cellfate['lambda_recon']}"
                   f"_la{config_ai4cellfate['lambda_adv']}_lc{config_ai4cellfate['lambda_cov']}"
                   f"_lcon{config_ai4cellfate['lambda_contra']}_frame{frame_index}")
    
    output_base_dir = f"./results/model_optimisation/{folder_name}"
    print(f"Running configuration: {folder_name}")
    print(f"Saving results to: {output_base_dir}")

    try:
        ##### STAGE 1 #####
        print("Starting Stage 1 training...")
        lambda_autoencoder_results = train_autoencoder(config_autoencoder, augmented_x_train, x_val, output_dir=output_base_dir)
        encoder = lambda_autoencoder_results['encoder']
        decoder = lambda_autoencoder_results['decoder']
        discriminator = lambda_autoencoder_results['discriminator']

        save_model_weights_to_disk(encoder, decoder, discriminator, output_dir=f"{output_base_dir}/models_stage1")
        # Evaluate the trained model (store latent space and reconstructed images)
        evaluate_model(encoder, decoder, augmented_x_train, augmented_y_train, output_dir=f"{output_base_dir}/stage1")
        
        ##### STAGE 2 #####
        print("Starting Stage 2 training...")
        lambda_ae_cov_results = train_cellfate(config_ai4cellfate, encoder, decoder, discriminator, augmented_x_train, augmented_y_train, x_val, y_val, x_test, y_test, output_dir=output_base_dir) 
        encoder = lambda_ae_cov_results['encoder']
        decoder = lambda_ae_cov_results['decoder']
        discriminator = lambda_ae_cov_results['discriminator']

        print(f"Good conditions stopped at epochs: {lambda_ae_cov_results['good_conditions_stop']}")
        save_model_weights_to_disk(encoder, decoder, discriminator, output_dir=f"{output_base_dir}/models")

        # Evaluate the trained model (store latent space and reconstructed images)
        evaluate_model(lambda_ae_cov_results['encoder'], lambda_ae_cov_results['decoder'], augmented_x_train, augmented_y_train, output_dir=f"{output_base_dir}/stage2")
        
        print(f"✅ Successfully completed configuration: {folder_name}")
        return True, lambda_ae_cov_results
        
    except Exception as e:
        print(f"❌ Failed configuration {folder_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def run_model_optimization():
    print(">>> model_optimization.py: starting run_model_optimization()", flush=True)
    """
    Run comprehensive model optimization across different hyperparameters.
    
    Sweeps over:
    - latent_dim: 2 to 10
    - lambda_contra: 0.1 to 2
    - lambda_cov: 0.1 to 5
    """
    
    print("🚀 Starting AI4CellFate Model Optimization")
    print("="*60)
    
    # Load data once
    augmented_x_train, x_val, x_test, augmented_y_train, y_val, y_test = load_data()
    
    # Define hyperparameter search space
    latent_dims = [2, 3] #2, 
    lambda_contras = [0.2, 0.4, 0.5, 0.7, 1.0] #0.01, 0.05, 0.2, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
    lambda_covs = [0.5, 0.7, 1.0, 2.0] #0.1, , 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
    
    # Base configuration for Stage 1 (autoencoder)
    base_config_autoencoder = {
        'batch_size': 30,
        'epochs': 100, 
        'learning_rate': 0.0001,
        'seed': 42,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5,
        'lambda_adv': 1,
    }
    
    # Base configuration for Stage 2 (AI4CellFate)
    base_config_ai4cellfate = {
        'batch_size': 30,
        'epochs': 100,
        'learning_rate': 0.0001,
        'seed': 42,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 6,
        'lambda_adv': 2,
    }
    
    # Track results
    results_summary = []
    
    # Total number of configurations
    total_configs = len(latent_dims) * len(lambda_contras) * len(lambda_covs)
    print(f"Total configurations to run: {total_configs}")
    print(f"  - Latent dims: {latent_dims}")
    print(f"  - Lambda contra: {lambda_contras}")
    print(f"  - Lambda cov: {lambda_covs}")
    print("="*60)
    
    current_config = 0
    
    # Grid search over all parameter combinations
    for latent_dim, lambda_contra, lambda_cov in itertools.product(latent_dims, lambda_contras, lambda_covs):
        current_config += 1
        
        print(f"\n{'='*60}")
        print(f"🔧 Configuration {current_config}/{total_configs}")
        print(f"   Latent Dim: {latent_dim}, Lambda Contra: {lambda_contra}, Lambda Cov: {lambda_cov}")
        print(f"{'='*60}")
        
        # Create configurations for this run
        config_autoencoder = base_config_autoencoder.copy()
        config_autoencoder['latent_dim'] = latent_dim
        
        config_ai4cellfate = base_config_ai4cellfate.copy()
        config_ai4cellfate['latent_dim'] = latent_dim
        config_ai4cellfate['lambda_contra'] = lambda_contra
        config_ai4cellfate['lambda_cov'] = lambda_cov
        
        # Run the configuration
        success, results = run_single_configuration(
            config_autoencoder, config_ai4cellfate,
            augmented_x_train, x_val, x_test, 
            augmented_y_train, y_val, y_test
        )
        
        # Track results
        result_entry = {
            'config_id': current_config,
            'latent_dim': latent_dim,
            'lambda_contra': lambda_contra,
            'lambda_cov': lambda_cov,
            'success': success,
        }
        
        if success and results:
            result_entry['good_epochs'] = results.get('good_conditions_stop', [])
            result_entry['final_recon_loss'] = float(results['recon_loss'][-1]) if results['recon_loss'] else None
            result_entry['final_adv_loss'] = float(results['adv_loss'][-1]) if results['adv_loss'] else None
            result_entry['final_cov_loss'] = float(results['cov_loss'][-1]) if results['cov_loss'] else None
            result_entry['final_contra_loss'] = float(results['contra_loss'][-1]) if results['contra_loss'] else None
            
            # Extract confusion matrix if available
            if 'confusion_matrix' in results:
                cm = results['confusion_matrix']
                result_entry['confusion_matrix_diag'] = float(np.mean(np.diag(cm)))
        else:
            result_entry['good_epochs'] = []
            result_entry['error'] = 'Training failed'
        
        results_summary.append(result_entry)
        
        # Save intermediate results after each configuration
        import json
        summary_path = "./results/model_optimization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"📊 Intermediate results saved to: {summary_path}")
    
    # Print final summary
    print("\n" + "="*80)
    print("🎯 MODEL OPTIMIZATION SUMMARY")
    print("="*80)
    
    print(f"\nTotal configurations run: {len(results_summary)}")
    successful_runs = sum(1 for r in results_summary if r['success'])
    print(f"Successful runs: {successful_runs}/{len(results_summary)}")
    
    # Group by latent dimension
    print("\n📊 Results by Latent Dimension:")
    for latent_dim in latent_dims:
        dim_results = [r for r in results_summary if r['latent_dim'] == latent_dim]
        successful_dim = sum(1 for r in dim_results if r['success'])
        print(f"\n  Latent Dim {latent_dim}: {successful_dim}/{len(dim_results)} successful")
        
        # Show best configuration for this latent_dim (by confusion matrix diagonal)
        successful_dim_results = [r for r in dim_results if r['success'] and 'confusion_matrix_diag' in r]
        if successful_dim_results:
            best = max(successful_dim_results, key=lambda x: x.get('confusion_matrix_diag', 0))
            print(f"    Best: λ_contra={best['lambda_contra']}, λ_cov={best['lambda_cov']}, "
                  f"acc={best['confusion_matrix_diag']:.3f}")
    
    # Group by lambda_contra
    print("\n📊 Results by Lambda Contra:")
    for lambda_contra in lambda_contras:
        contra_results = [r for r in results_summary if r['lambda_contra'] == lambda_contra]
        successful_contra = sum(1 for r in contra_results if r['success'])
        print(f"  Lambda Contra {lambda_contra}: {successful_contra}/{len(contra_results)} successful")
    
    # Group by lambda_cov
    print("\n📊 Results by Lambda Cov:")
    for lambda_cov in lambda_covs:
        cov_results = [r for r in results_summary if r['lambda_cov'] == lambda_cov]
        successful_cov = sum(1 for r in cov_results if r['success'])
        print(f"  Lambda Cov {lambda_cov}: {successful_cov}/{len(cov_results)} successful")
    
    # Find overall best configuration
    successful_results = [r for r in results_summary if r['success'] and 'confusion_matrix_diag' in r]
    if successful_results:
        best_overall = max(successful_results, key=lambda x: x.get('confusion_matrix_diag', 0))
        print("\n🏆 Best Overall Configuration:")
        print(f"  Latent Dim: {best_overall['latent_dim']}")
        print(f"  Lambda Contra: {best_overall['lambda_contra']}")
        print(f"  Lambda Cov: {best_overall['lambda_cov']}")
        print(f"  Accuracy (mean diagonal): {best_overall['confusion_matrix_diag']:.4f}")
        print(f"  Good epochs: {best_overall['good_epochs']}")
    
    # Save final results summary
    summary_path = "./results/model_optimization_summary_final.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n📊 Final results summary saved to: {summary_path}")
    print("\n🎉 Model optimization completed!")
    
    return results_summary


# Main function for model optimization
def main():
    """Main function to run model optimization studies."""
    results = run_model_optimization()
    return results


if __name__ == '__main__':
    main()
