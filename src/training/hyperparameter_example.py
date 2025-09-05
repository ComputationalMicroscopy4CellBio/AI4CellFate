"""
Example usage of hyperparameter optimization for tabular features.
Add this code to your notebook to run comprehensive hyperparameter optimization.
"""

# Import the hyperparameter optimization function
import sys
import os
sys.path.append(os.path.abspath("../.."))
from src.training.hyperparameter_optimization import run_fair_feature_comparison
import numpy as np

def run_tabular_hyperparameter_optimization(train_tracks_features, 
                                           train_labels, 
                                           test_tracks_features, 
                                           test_labels,
                                           feature_names):
    """
    Run hyperparameter optimization for tabular features.
    
    This will:
    1. Test many different MLP architectures
    2. Use 5-fold cross-validation for robust evaluation
    3. Find the best hyperparameters for each feature pair
    4. Return comprehensive results
    """
    
    print("🚀 Starting Comprehensive Hyperparameter Optimization")
    print("=" * 60)
    print(f"Features: {len(feature_names)}")
    print(f"Feature pairs to test: {len(feature_names) * (len(feature_names) - 1) // 2}")
    print(f"Training samples: {train_tracks_features.shape[0]}")
    print(f"Test samples: {test_tracks_features.shape[0]}")
    print()
    
    # Run fair feature comparison (two-step process)
    results = run_fair_feature_comparison(
        train_features=train_tracks_features,
        train_labels=train_labels,
        test_features=test_tracks_features,
        test_labels=test_labels,
        feature_names=feature_names,
        sample_pairs_for_optimization=15,  # Use 15 pairs to find best architecture
        cv_folds=5,                        # 5-fold cross-validation
        max_combinations=50,               # Test 50 hyperparameter combinations
        epochs=50,                         # Train each model for 50 epochs
        batch_size=32,                     # Batch size for training
        random_state=42,                   # For reproducibility
        save_results=True,                 # Save numpy arrays automatically
        verbose=1                          # Show progress
    )
    
    return results

def analyze_optimization_results(results):
    """
    Analyze and visualize the hyperparameter optimization results.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Extract results for analysis
    all_results = []
    for key, result in results['all_results'].items():
        if result['all_results']:  # Check if optimization succeeded
            best_result = result['all_results'][0]  # Best result for this feature pair
            all_results.append({
                'feature_pair': ' + '.join(result['feature_names']),
                'cv_accuracy': best_result['cv_accuracy_mean'],
                'cv_std': best_result['cv_accuracy_std'],
                'test_accuracy': best_result['test_accuracy'],
                'test_f1': best_result['test_f1'],
                'hidden_layers': str(best_result['params']['hidden_layers']),
                'dropout_rate': best_result['params']['dropout_rate'],
                'learning_rate': best_result['params']['learning_rate'],
                'l2_reg': best_result['params']['l2_reg'],
                'activation': best_result['params']['activation'],
                'batch_norm': best_result['params']['batch_norm']
            })
    
    # Create DataFrame for easy analysis
    df = pd.DataFrame(all_results)
    df = df.sort_values('cv_accuracy', ascending=False)
    
    print("\n📊 DETAILED ANALYSIS")
    print("=" * 60)
    
    # Top 10 feature pairs
    print("\n🏆 Top 10 Feature Pairs:")
    print(df[['feature_pair', 'cv_accuracy', 'test_accuracy', 'test_f1']].head(10).to_string(index=False))
    
    # Architecture analysis
    print(f"\n🏗️  Architecture Analysis:")
    print("Most common hidden layer configurations:")
    print(df['hidden_layers'].value_counts().head(5))
    
    print(f"\nMost common dropout rates:")
    print(df['dropout_rate'].value_counts().head(5))
    
    print(f"\nMost common learning rates:")  
    print(df['learning_rate'].value_counts().head(5))
    
    # Performance statistics
    print(f"\n📈 Performance Statistics:")
    print(f"Mean CV accuracy: {df['cv_accuracy'].mean():.4f} ± {df['cv_accuracy'].std():.4f}")
    print(f"Mean test accuracy: {df['test_accuracy'].mean():.4f} ± {df['test_accuracy'].std():.4f}")
    print(f"Best CV accuracy: {df['cv_accuracy'].max():.4f}")
    print(f"Best test accuracy: {df['test_accuracy'].max():.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Top 20 feature pairs
    top_20 = df.head(20)
    axes[0, 0].barh(range(len(top_20)), top_20['cv_accuracy'])
    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels(top_20['feature_pair'], fontsize=8)
    axes[0, 0].set_xlabel('CV Accuracy')
    axes[0, 0].set_title('Top 20 Feature Pairs (CV Accuracy)')
    axes[0, 0].invert_yaxis()
    
    # 2. CV vs Test accuracy
    axes[0, 1].scatter(df['cv_accuracy'], df['test_accuracy'], alpha=0.6)
    axes[0, 1].plot([df['cv_accuracy'].min(), df['cv_accuracy'].max()], 
                    [df['cv_accuracy'].min(), df['cv_accuracy'].max()], 'r--')
    axes[0, 1].set_xlabel('CV Accuracy')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('CV vs Test Accuracy')
    
    # 3. Architecture frequency
    arch_counts = df['hidden_layers'].value_counts().head(10)
    axes[1, 0].bar(range(len(arch_counts)), arch_counts.values)
    axes[1, 0].set_xticks(range(len(arch_counts)))
    axes[1, 0].set_xticklabels(arch_counts.index, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Most Common Architectures')
    
    # 4. Hyperparameter correlation with performance
    param_performance = df.groupby('dropout_rate')['cv_accuracy'].mean().sort_index()
    axes[1, 1].plot(param_performance.index, param_performance.values, 'o-')
    axes[1, 1].set_xlabel('Dropout Rate')
    axes[1, 1].set_ylabel('Mean CV Accuracy')
    axes[1, 1].set_title('Dropout Rate vs Performance')
    
    plt.tight_layout()
    plt.show()
    
    return df

"""
# Run hyperparameter optimization
results = run_tabular_hyperparameter_optimization(
    train_tracks_features, 
    train_labels, 
    test_tracks_features, 
    test_labels,
    feature_names
)

# Analyze results
df_results = analyze_optimization_results(results)

# Save results
import pickle
with open('hyperparameter_optimization_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Get the best model for further use
best_features = results['best_overall']['features']
best_params = results['best_overall']['params']
print(f"Best feature combination: {best_features}")
print(f"Best hyperparameters: {best_params}")
"""
