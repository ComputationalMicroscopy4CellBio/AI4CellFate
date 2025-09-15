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

def run_single_configuration(config_autoencoder, config_ai4cellfate, augmented_x_train, x_val, x_test, augmented_y_train, y_val, y_test):
    """Run a single AI4CellFate configuration (both stages) and save all results."""
    
    # Create parameter-based folder name
    folder_name = (f"s1_ep{config_autoencoder['epochs']}_lr{config_autoencoder['lambda_recon']}"
                   f"_la{config_autoencoder['lambda_adv']}_seed{config_autoencoder['seed']}"
                   f"_ldim{config_autoencoder['latent_dim']}_s2_lr{config_ai4cellfate['lambda_recon']}"
                   f"_la{config_ai4cellfate['lambda_adv']}_lc{config_ai4cellfate['lambda_cov']}"
                   f"_lcon{config_ai4cellfate['lambda_contra']}")
    
    output_base_dir = f"./results/{folder_name}"
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
        return True, lambda_ae_cov_results['good_conditions_stop']
        
    except Exception as e:
        print(f"❌ Failed configuration {folder_name}: {str(e)}")
        return False, []

def run_ablation_studies():
    """Run comprehensive ablation studies across different configurations and seeds."""
    
    print("🚀 Starting AI4CellFate Ablation Studies")
    print("="*60)
    
    # Load data once
    augmented_x_train, x_val, x_test, augmented_y_train, y_val, y_test = load_data()
    
    # Define seeds to test
    seeds = [42, 43, 44]
    
    # Define base configuration (your current best config)
    base_config_autoencoder = {
        'batch_size': 30,
        'epochs': 35, 
        'learning_rate': 0.0001,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5,
        'lambda_adv': 1,
    }
    
    base_config_ai4cellfate = {
        'batch_size': 30,
        'epochs': 100,
        'learning_rate': 0.001,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 6,
        'lambda_adv': 4,
        'lambda_cov': 1,
        'lambda_contra': 8, 
    }
    
    # Define ablation configurations
    ablation_configs = [
        {
            'name': 'Full_Model',
            'description': 'Complete AI4CellFate model (baseline)',
            'stage1_changes': {},  # No changes
            'stage2_changes': {}   # No changes
        },
        {
            'name': 'No_Adversarial',
            'description': 'Remove adversarial loss (both stages)',
            'stage1_changes': {'lambda_adv': 0},
            'stage2_changes': {'lambda_adv': 0}
        },
        {
            'name': 'No_Covariance',
            'description': 'Remove covariance loss (stage 2)',
            'stage1_changes': {},
            'stage2_changes': {'lambda_cov': 0}
        },
        {
            'name': 'No_Contrastive',
            'description': 'Remove contrastive loss (stage 2)',
            'stage1_changes': {},
            'stage2_changes': {'lambda_contra': 0}
        },
        {
            'name': 'Reconstruction_Only',
            'description': 'Only reconstruction loss (basic autoencoder)',
            'stage1_changes': {'lambda_adv': 0},
            'stage2_changes': {'lambda_adv': 0, 'lambda_cov': 0, 'lambda_contra': 0}
        }
    ]
    
    # Track results
    results_summary = []
    
    # Run ablation studies
    for config in ablation_configs:
        print(f"\n📋 Running ablation: {config['name']}")
        print(f"Description: {config['description']}")
        print("-" * 40)
        
        for seed in seeds:
            print(f"\n🌱 Seed: {seed}")
            
            # Create modified configurations
            config_autoencoder = base_config_autoencoder.copy()
            config_autoencoder['seed'] = seed
            config_autoencoder.update(config['stage1_changes'])
            
            config_ai4cellfate = base_config_ai4cellfate.copy()
            config_ai4cellfate['seed'] = seed
            config_ai4cellfate.update(config['stage2_changes'])
            
            # Run the configuration
            success, good_epochs = run_single_configuration(
                config_autoencoder, config_ai4cellfate,
                augmented_x_train, x_val, x_test, 
                augmented_y_train, y_val, y_test
            )
            
            # Track results
            results_summary.append({
                'ablation': config['name'],
                'seed': seed,
                'success': success,
                'good_epochs': good_epochs,
                'config_s1': config_autoencoder,
                'config_s2': config_ai4cellfate
            })
    
    # Print final summary
    print("\n" + "="*60)
    print("🎯 ABLATION STUDIES SUMMARY")
    print("="*60)
    
    for config in ablation_configs:
        print(f"\n{config['name']}:")
        config_results = [r for r in results_summary if r['ablation'] == config['name']]
        for result in config_results:
            status = "✅" if result['success'] else "❌"
            epochs_info = f"(stopped at: {result['good_epochs']})" if result['good_epochs'] else "(no good epochs)"
            print(f"  Seed {result['seed']}: {status} {epochs_info}")
    
    print(f"\nTotal configurations run: {len(results_summary)}")
    successful_runs = sum(1 for r in results_summary if r['success'])
    print(f"Successful runs: {successful_runs}/{len(results_summary)}")
    
    # Save results summary
    import json
    summary_path = "./results/ablation_summary.json"
    with open(summary_path, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        json_results = []
        for result in results_summary:
            json_result = result.copy()
            # Convert any numpy types to regular types
            for key, value in json_result.items():
                if hasattr(value, 'tolist'):
                    json_result[key] = value.tolist()
            json_results.append(json_result)
        json.dump(json_results, f, indent=2)
    
    print(f"📊 Results summary saved to: {summary_path}")
    print("\n🎉 Ablation studies completed!")
    
    return results_summary

# Main function for ablation studies
def main():
    """Main function to run ablation studies."""
    results = run_ablation_studies()
    return results

if __name__ == '__main__':
    main()