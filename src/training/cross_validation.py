import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
import os
import json
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from ..preprocessing.preprocessing_functions import augment_dataset, augmentations
from ..models import Encoder, Decoder, Discriminator, mlp_classifier
from ..utils import set_seed, convert_namespace_to_dict
from .train import train_autoencoder, train_cellfate
from ..evaluation.evaluate import Evaluation, visualize_latent_space


class CrossValidation:
    """
    K-fold cross-validation for AI4CellFate model training and evaluation.
    
    This class handles the entire cross-validation pipeline including:
    - Data splitting with stratification
    - Per-fold data augmentation
    - Model training on each fold
    - Performance aggregation and reporting
    """
    
    def __init__(self, k_folds=5, random_state=42, save_individual_models=False):
        """
        Initialize cross-validation setup.
        
        Args:
            k_folds (int): Number of folds for cross-validation
            random_state (int): Random seed for reproducibility
            save_individual_models (bool): Whether to save models from each fold
        """
        self.k_folds = k_folds
        self.random_state = random_state
        self.save_individual_models = save_individual_models
        self.fold_results = []
        
    def prepare_fold_data(self, x_train, y_train, train_idx, val_idx, apply_augmentation=True):
        """
        Prepare training and validation data for a single fold.
        
        Args:
            x_train (np.ndarray): Original training images
            y_train (np.ndarray): Original training labels
            train_idx (np.ndarray): Training indices for this fold
            val_idx (np.ndarray): Validation indices for this fold
            apply_augmentation (bool): Whether to apply data augmentation
            
        Returns:
            tuple: (x_fold_train, y_fold_train, x_fold_val, y_fold_val)
        """
        # Split data for this fold
        x_fold_train = x_train[train_idx]
        y_fold_train = y_train[train_idx]
        x_fold_val = x_train[val_idx]
        y_fold_val = y_train[val_idx]
        
        # Apply data augmentation only to training data
        if apply_augmentation:
            print(f"Applying data augmentation to fold training data...")
            x_fold_train, y_fold_train = augment_dataset(
                x_fold_train, 
                y_fold_train, 
                augmentations, 
                augment_times=5,
                seed=self.random_state
            )
            print(f"Augmented training data shape: {x_fold_train.shape}")
        
        return x_fold_train, y_fold_train, x_fold_val, y_fold_val
    
    def train_fold(self, fold_num, x_fold_train, y_fold_train, x_fold_val, y_fold_val, x_test, y_test,
                   config_autoencoder, config_ai4cellfate, output_dir):
        """
        Train models for a single fold.
        
        Args:
            fold_num (int): Current fold number
            x_fold_train (np.ndarray): Training images for this fold
            y_fold_train (np.ndarray): Training labels for this fold
            x_fold_val (np.ndarray): Validation images for this fold
            y_fold_val (np.ndarray): Validation labels for this fold
            config_autoencoder (dict): Configuration for autoencoder training
            config_ai4cellfate (dict): Configuration for AI4CellFate training
            output_dir (str): Directory to save results
            
        Returns:
            dict: Results for this fold including models and metrics
        """
        print(f"\n=== Training Fold {fold_num + 1}/{self.k_folds} ===")
        
        # Set fold-specific seed
        fold_seed = self.random_state + fold_num
        config_autoencoder['seed'] = fold_seed
        config_ai4cellfate['seed'] = fold_seed
        
        fold_output_dir = os.path.join(output_dir, f"fold_{fold_num + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # STAGE 1: Train Autoencoder
        print("Stage 1: Training Autoencoder...")
        autoencoder_results = train_autoencoder(config_autoencoder, x_fold_train, x_fold_val)
        encoder = autoencoder_results['encoder']
        decoder = autoencoder_results['decoder']
        discriminator = autoencoder_results['discriminator']
        
        # STAGE 2: Train AI4CellFate
        print("Stage 2: Training AI4CellFate...")
        ai4cellfate_results = train_cellfate(
            config_ai4cellfate, 
            encoder, 
            decoder, 
            discriminator, 
            x_fold_train, 
            y_fold_train, 
            x_fold_val, 
            y_fold_val, 
            x_fold_val,  
            y_fold_val
        )
        
        # Get final models
        final_encoder = ai4cellfate_results['encoder']
        final_decoder = ai4cellfate_results['decoder']
        final_discriminator = ai4cellfate_results['discriminator']
        confusion_matrix = ai4cellfate_results['confusion_matrix']
        
        # Save latent space on this fold
        evaluator = Evaluation(fold_output_dir)
        latent_space = final_encoder.predict(x_fold_train)
        evaluator.visualize_latent_space(latent_space, y_fold_train, epoch=0)
        
        ##### Calculate metrics ####
        mean_diagonal = np.mean(np.diag(confusion_matrix))

        # Calculate precision from confusion matrix with safe division
        denominator = confusion_matrix[0,0] + confusion_matrix[1,0]
        if denominator > 0:
            precision = confusion_matrix[0,0] / denominator
        else:
            precision = 0.0
        
        recall = confusion_matrix[1,1] / (confusion_matrix[1,0] + confusion_matrix[1,1])
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        
        print(f"Fold {fold_num + 1} Validation Results:")
        print(f"  Accuracy: {mean_diagonal:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1_score:.4f}")
        
        # Save fold results
        fold_results = {
            'fold_num': fold_num + 1,
            'validation_metrics': {
                'accuracy': mean_diagonal,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'confusion_matrix': confusion_matrix.tolist(),
            },
            'models': {
                'encoder': final_encoder,
                'decoder': final_decoder,
                'discriminator': final_discriminator,
            }
        }
        
        # Save individual models if requested
        if self.save_individual_models:
            models_dir = os.path.join(fold_output_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            final_encoder.save_weights(os.path.join(models_dir, "encoder.weights.h5"))
            final_decoder.save_weights(os.path.join(models_dir, "decoder.weights.h5"))
            final_discriminator.save_weights(os.path.join(models_dir, "discriminator.weights.h5"))
        
        # Save fold metrics
        with open(os.path.join(fold_output_dir, "metrics.json"), 'w') as f:
            json.dump({
                'validation_metrics': {
                    'accuracy': mean_diagonal,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'confusion_matrix': confusion_matrix.tolist(),
                }
            }, f, indent=2)
        
        return fold_results
    
    def run_cross_validation(self, x_train, y_train, x_test, y_test, config_autoencoder, config_ai4cellfate, 
                           output_dir="./results/cross_validation", apply_augmentation=True):
        """
        Run complete k-fold cross-validation.
        
        Args:
            x_train (np.ndarray): Original training images (before augmentation)
            y_train (np.ndarray): Original training labels
            config_autoencoder (dict): Configuration for autoencoder training
            config_ai4cellfate (dict): Configuration for AI4CellFate training
            output_dir (str): Directory to save all results
            apply_augmentation (bool): Whether to apply data augmentation
            
        Returns:
            dict: Complete cross-validation results
        """
        print(f"Starting {self.k_folds}-fold cross-validation...")
        print(f"Training data shape: {x_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        
        # Store all fold results
        self.fold_results = []
        
        # Perform cross-validation
        for fold_num, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
            print(f"\nFold {fold_num + 1}: Train size={len(train_idx)}, Val size={len(val_idx)}")
            
            # Prepare fold data
            x_fold_train, y_fold_train, x_fold_val, y_fold_val = self.prepare_fold_data(
                x_train, y_train, train_idx, val_idx, apply_augmentation
            )
            
            # Train fold
            fold_results = self.train_fold(
                fold_num, x_fold_train, y_fold_train, x_fold_val, y_fold_val,
                x_test, y_test, config_autoencoder.copy(), config_ai4cellfate.copy(), output_dir
            )
            
            self.fold_results.append(fold_results)
        
        # Aggregate results
        cv_results = self.aggregate_results(output_dir)
        
        return cv_results
    
    def aggregate_results(self, output_dir):
        """
        Aggregate and analyze results from all folds.
        
        Args:
            output_dir (str): Directory to save aggregated results
            
        Returns:
            dict: Aggregated cross-validation results
        """
        print(f"\n=== Cross-Validation Results Summary ===")
        
        # Extract metrics from all folds
        accuracies = [fold['validation_metrics']['accuracy'] for fold in self.fold_results]
        precisions = [fold['validation_metrics']['precision'] for fold in self.fold_results]
        recalls = [fold['validation_metrics']['recall'] for fold in self.fold_results]
        f1_scores = [fold['validation_metrics']['f1_score'] for fold in self.fold_results]
        
        # Calculate statistics
        cv_results = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls),
            'mean_f1_score': np.mean(f1_scores),
            'std_f1_score': np.std(f1_scores),
            'fold_accuracies': accuracies,
            'fold_precisions': precisions,
            'fold_recalls': recalls,
            'fold_f1_scores': f1_scores,
        }
        
        # Print results
        print(f"Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"Precision: {cv_results['mean_precision']:.4f} ± {cv_results['std_precision']:.4f}")
        print(f"Recall: {cv_results['mean_recall']:.4f} ± {cv_results['std_recall']:.4f}")
        print(f"F1-score: {cv_results['mean_f1_score']:.4f} ± {cv_results['std_f1_score']:.4f}")
        
        # Save aggregated results
        with open(os.path.join(output_dir, "cv_summary.json"), 'w') as f:
            json.dump({k: v for k, v in cv_results.items() if k not in ['fold_accuracies', 'fold_precisions', 'fold_recalls', 'fold_f1_scores']}, f, indent=2)
        
        # Create visualization
        self.plot_cv_results(cv_results, output_dir)
        
        return cv_results
    
    def plot_cv_results(self, cv_results, output_dir):
        """
        Create visualizations of cross-validation results.
        
        Args:
            cv_results (dict): Cross-validation results
            output_dir (str): Directory to save plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score',]
        titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score',]
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//3, i%3]
            # Handle the irregular pluralization
            if metric == 'accuracy':
                fold_values = cv_results['fold_accuracies']
                mean_val = cv_results['mean_accuracy']
                std_val = cv_results['std_accuracy']
            elif metric == 'f1_score':
                fold_values = cv_results['fold_f1_scores']
                mean_val = cv_results['mean_f1_score']
                std_val = cv_results['std_f1_score']
            else:
                fold_values = cv_results[f'fold_{metric}s']
                mean_val = cv_results[f'mean_{metric}']
                std_val = cv_results[f'std_{metric}']
            
            ax.bar(range(1, self.k_folds + 1), fold_values, alpha=0.7, color='skyblue')
            ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
            
            ax.set_xlabel('Fold', fontsize=16, fontname='Arial')
            ax.set_ylabel(title, fontsize=16, fontname='Arial')
            ax.set_title(f'{title} across {self.k_folds} folds', fontsize=18, fontname='Arial')
            ax.set_xticks(range(1, self.k_folds + 1))
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            # Set Arial font for tick labels
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Arial')
            
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
        
        # Hide the empty subplot
        if len(metrics) < 6:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cv_results.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'cv_results.eps'), format='eps', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cross-validation results plot saved to {os.path.join(output_dir, 'cv_results.png')}") 