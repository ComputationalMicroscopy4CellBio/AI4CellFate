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
        sample_pairs_for_optimization=10,  # Use 15 pairs to find best architecture
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
    Analyze the hyperparameter optimization results and create a summary DataFrame.
    
    Args:
        results: Dictionary returned from run_fair_feature_comparison
        
    Returns:
        pandas.DataFrame: Summary of results for each feature pair
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Extract results for analysis from the new format
    all_results = []
    
    # Get data from the new results structure
    pair_names = results['pair_names']
    cv_accuracies = results['cv_accuracies']
    cv_stds = results['cv_stds']
    test_accuracies = results['test_accuracies']
    test_precisions = results['test_precisions']
    
    # Get the fixed architecture used
    best_params = results['best_global_params']
    
    for i, pair_name in enumerate(pair_names):
        # Skip failed pairs (NaN values)
        if not (np.isnan(cv_accuracies[i]) or np.isnan(test_accuracies[i])):
            all_results.append({
                'feature_pair': pair_name,
                'cv_accuracy': cv_accuracies[i],
                'cv_std': cv_stds[i],
                'test_accuracy': test_accuracies[i],
                'test_precision': test_precisions[i],
                'hidden_layers': str(best_params['hidden_layers']),
                'dropout_rate': best_params['dropout_rate'],
                'learning_rate': best_params['learning_rate']
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("❌ No successful optimization results found!")
        return df
    
    # Sort by test accuracy (descending)
    df = df.sort_values('test_accuracy', ascending=False).reset_index(drop=True)
    
    print(f"\n📊 HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Total feature pairs analyzed: {len(df)}")
    print(f"Best test accuracy: {df['test_accuracy'].max():.4f}")
    print(f"Best feature pair: {df.iloc[0]['feature_pair']}")
    print(f"Mean test accuracy: {df['test_accuracy'].mean():.4f} ± {df['test_accuracy'].std():.4f}")
    print(f"Architecture used: {best_params}")
    
    # Show top 10 results
    print(f"\n🏆 TOP 10 FEATURE PAIRS:")
    print("-" * 80)
    for i, row in df.head(10).iterrows():
        print(f"{i+1:2d}. {row['feature_pair']:30s} | Test Acc: {row['test_accuracy']:.4f} | CV: {row['cv_accuracy']:.4f}±{row['cv_std']:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Top 20 feature pairs
    top_20 = df.head(20)
    axes[0, 0].barh(range(len(top_20)), top_20['test_accuracy'])
    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels(top_20['feature_pair'], fontsize=8)
    axes[0, 0].set_xlabel('Test Accuracy')
    axes[0, 0].set_title('Top 20 Feature Pairs (Test Accuracy)')
    axes[0, 0].invert_yaxis()
    
    # 2. CV vs Test accuracy
    axes[0, 1].scatter(df['cv_accuracy'], df['test_accuracy'], alpha=0.6)
    axes[0, 1].plot([df['cv_accuracy'].min(), df['cv_accuracy'].max()], 
                    [df['cv_accuracy'].min(), df['cv_accuracy'].max()], 'r--')
    axes[0, 1].set_xlabel('CV Accuracy')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('CV vs Test Accuracy')
    
    # 3. Test accuracy distribution
    axes[1, 0].hist(df['test_accuracy'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(df['test_accuracy'].mean(), color='red', linestyle='--', label=f'Mean: {df["test_accuracy"].mean():.3f}')
    axes[1, 0].set_xlabel('Test Accuracy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Test Accuracy Distribution')
    axes[1, 0].legend()
    
    # 4. Test accuracy vs precision
    axes[1, 1].scatter(df['test_accuracy'], df['test_precision'], alpha=0.6)
    axes[1, 1].set_xlabel('Test Accuracy')
    axes[1, 1].set_ylabel('Test Precision (Class 0)')
    axes[1, 1].set_title('Accuracy vs Precision')
    
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
