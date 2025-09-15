import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras import layers, Sequential
import itertools
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class MLPHyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization for MLP classifiers on tabular features.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = []
        self.best_global_params = None
        
    def create_mlp_model(self, 
                        input_dim: int,
                        hidden_layers: List[int],
                        dropout_rate: float = 0.3,
                        learning_rate: float = 0.001):
        """
        Create MLP model with specified architecture.
        Always uses ReLU activation and batch norm only at input.
        """
        def build_model():
            model = Sequential()
            model.add(layers.Input(shape=(input_dim,)))
            
            # Batch normalization at input (always applied)
            model.add(layers.BatchNormalization())
            
            # Hidden layers (always use ReLU activation, no L2 regularization)
            for i, units in enumerate(hidden_layers):
                model.add(layers.Dense(units, activation='relu'))
                model.add(layers.Dropout(dropout_rate))
            
            # Output layer (softmax for classification)
            model.add(layers.Dense(2, activation='softmax'))
            
            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        
        return build_model
    
    def get_hyperparameter_grid(self):
        """
        Define comprehensive hyperparameter search space.
        """
        return {
            # Architecture variations
            'hidden_layers': [
                [16],           # Single layer - small
                [32],           # Single layer - medium  
                [16, 8],        # Two layers - small
                [32, 16],       # Two layers - medium
                [32, 16, 8],    # Three layers - medium
                [16, 16],       # Equal layers - small
                [32, 32],       # Equal layers - medium
            ],
            
            # Regularization
            'dropout_rate': [0.3, 0.4, 0.5], #[0.2, 0.3, 0.4, 0.5],
            
            # Training parameters
            'learning_rate': [0.0001, 0.001, 0.01], #, 0.01
        }
    
    def optimize_single_feature_pair(self, 
                                   X_train: np.ndarray, 
                                   y_train: np.ndarray,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   feature_names: List[str],
                                   cv_folds: int = 5,
                                   max_combinations: int = 30,
                                   epochs: int = 100,
                                   batch_size: int = 32,
                                   verbose: int = 0) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a single feature pair using cross-validation.
        """
        
        # Set random seeds
        tf.keras.utils.set_random_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Get hyperparameter grid
        param_grid = self.get_hyperparameter_grid()
        
        # Sample random combinations if grid is too large
        param_combinations = list(itertools.product(*param_grid.values()))
        if len(param_combinations) > max_combinations:
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:max_combinations]
        
        # Class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        
        best_score = 0
        best_params = None
        best_model = None
        cv_results = []
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_grid.keys(), params))
            
            if verbose > 0:
                print(f"Testing combination {i+1}/{len(param_combinations)}: {param_dict}")
            
            try:
                # Cross-validation scores
                fold_scores = []
                fold_predictions = []
                fold_true_labels = []
                
                for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                    try:
                        # Split data
                        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                        
                        # Aggressive memory cleanup
                        tf.keras.backend.clear_session()
                        import gc
                        gc.collect()
                        
                        # Create model
                        model_builder = self.create_mlp_model(
                            input_dim=X_train.shape[1],
                            **param_dict
                        )
                        model = model_builder()
                        
                        # Use smaller batch size to avoid memory issues
                        effective_batch_size = min(batch_size, len(X_fold_train) // 4)
                        effective_batch_size = max(8, effective_batch_size)  # Minimum batch size of 8
                        
                        # Train model
                        history = model.fit(
                            X_fold_train, y_fold_train,
                            validation_data=(X_fold_val, y_fold_val),
                            epochs=epochs,
                            batch_size=effective_batch_size,
                            class_weight=class_weight_dict,
                            verbose=0
                        )
                        
                        # Evaluate on validation fold
                        y_pred_proba = model.predict(X_fold_val, verbose=0)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        
                        # Calculate metrics from confusion matrix only
                        cm = confusion_matrix(y_fold_val, y_pred)
                        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        
                        # Accuracy as mean diagonal of normalized confusion matrix
                        accuracy = np.mean(np.diag(cm_normalized))
                        fold_scores.append(accuracy)
                        fold_predictions.extend(y_pred)
                        fold_true_labels.extend(y_fold_val)
                        
                        # Clean up model explicitly
                        del model
                        tf.keras.backend.clear_session()
                        
                    except Exception as fold_e:
                        if verbose > 0:
                            print(f"   Fold {fold} failed: {fold_e}")
                        # Use a default poor score for failed folds
                        fold_scores.append(0.3)
                        tf.keras.backend.clear_session()
                        continue
                
                # Calculate cross-validation metrics
                cv_accuracy_mean = np.mean(fold_scores)
                cv_accuracy_std = np.std(fold_scores)
                
                # Train final model on full training set for test evaluation
                tf.keras.backend.clear_session()
                final_model = self.create_mlp_model(
                    input_dim=X_train.shape[1],
                    **param_dict
                )()
                
                final_model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=class_weight_dict,
                    verbose=0
                )
                
                # Test set evaluation
                y_test_pred_proba = final_model.predict(X_test, verbose=0)
                y_test_pred = np.argmax(y_test_pred_proba, axis=1)
                
                # Calculate test metrics from confusion matrix only
                cm = confusion_matrix(y_test, y_test_pred)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                # Accuracy as mean diagonal of normalized confusion matrix
                test_accuracy = np.mean(np.diag(cm_normalized))
                
                # Precision for class 0 
                test_precision = cm_normalized[0, 0] / (cm_normalized[0, 0] + cm_normalized[1, 0]) if (cm_normalized[0, 0] + cm_normalized[1, 0]) > 0 else 0
                
                # Keep other metrics for compatibility
                test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                
                # Store results
                result = {
                    'params': param_dict,
                    'cv_accuracy_mean': cv_accuracy_mean,
                    'cv_accuracy_std': cv_accuracy_std,
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1,
                    'confusion_matrix': cm,
                    'confusion_matrix_normalized': cm_normalized,
                    'feature_names': feature_names
                }
                
                cv_results.append(result)
                
                # Track best model based on CV score
                if cv_accuracy_mean > best_score:
                    best_score = cv_accuracy_mean
                    best_params = param_dict
                    best_model = final_model
                
            except Exception as e:
                if verbose > 0:
                    print(f"Failed combination {i+1}: {e}")
                continue
        
        # Sort results by CV accuracy
        cv_results.sort(key=lambda x: x['cv_accuracy_mean'], reverse=True)
        
        return {
            'best_params': best_params,
            'best_cv_score': best_score,
            'best_model': best_model,
            'all_results': cv_results,
            'feature_names': feature_names
        }
    
    def find_global_best_hyperparameters(self,
                                       train_features: np.ndarray,
                                       train_labels: np.ndarray,
                                       test_features: np.ndarray,
                                       test_labels: np.ndarray,
                                       feature_names: List[str],
                                       sample_pairs: int = 5,
                                       cv_folds: int = 5,
                                       max_combinations: int = 50,
                                       epochs: int = 100,
                                       batch_size: int = 32,
                                       verbose: int = 1) -> Dict[str, Any]:
        """
        Find globally best hyperparameters using a sample of feature pairs.
        """
        
        # Combine original train and test data
        X_combined = np.vstack([train_features, test_features])
        y_combined = np.hstack([train_labels, test_labels])
        
        # Generate feature pairs and sample them
        feature_pairs = list(itertools.combinations(range(len(feature_names)), 2))
        np.random.seed(self.random_state)
        sampled_pairs = np.random.choice(len(feature_pairs), min(sample_pairs, len(feature_pairs)), replace=False)
        
        if verbose > 0:
            print(f"🔍 Finding global best hyperparameters using {len(sampled_pairs)} feature pairs...")
        
        all_results = []
        
        for i, pair_idx in enumerate(sampled_pairs):
            f1, f2 = feature_pairs[pair_idx]
            if verbose > 0:
                print(f"  Testing pair {i+1}/{len(sampled_pairs)}: [{feature_names[f1]}, {feature_names[f2]}]")
            
            # Extract feature pair data
            X_pair = X_combined[:, [f1, f2]]
            pair_feature_names = [feature_names[f1], feature_names[f2]]
            
            # Split combined data back into train/test for this method
            split_idx = len(train_features)  # Original train size
            X_train_pair = X_pair[:split_idx]
            y_train_pair = y_combined[:split_idx]
            X_test_pair = X_pair[split_idx:]
            y_test_pair = y_combined[split_idx:]
            
            # Optimize this feature pair
            result = self.optimize_single_feature_pair(
                X_train_pair, y_train_pair, X_test_pair, y_test_pair,
                pair_feature_names, cv_folds, max_combinations, epochs, batch_size, verbose-1
            )
            
            if result['all_results']:
                all_results.extend(result['all_results'])
        
        if not all_results:
            raise ValueError("No successful hyperparameter optimization results!")
        
        # Sort all results by CV accuracy and find global best
        all_results.sort(key=lambda x: x['cv_accuracy_mean'], reverse=True)
        global_best = all_results[0]
        
        self.best_global_params = global_best['params']
        
        if verbose > 0:
            print(f"🏆 Global best hyperparameters found:")
            print(f"   CV Accuracy: {global_best['cv_accuracy_mean']:.4f} ± {global_best['cv_accuracy_std']:.4f}")
            print(f"   Parameters: {self.best_global_params}")
        
        return {
            'best_params': self.best_global_params,
            'best_cv_score': global_best['cv_accuracy_mean'],
            'all_sampled_results': all_results,
            'sampled_pairs': [(feature_pairs[i][0], feature_pairs[i][1]) for i in sampled_pairs]
        }
    
    def evaluate_all_pairs_with_fixed_model(self,
                                           train_features: np.ndarray,
                                           train_labels: np.ndarray,
                                           test_features: np.ndarray,
                                           test_labels: np.ndarray,
                                           feature_names: List[str],
                                           fixed_params: Dict[str, Any] = None,
                                           cv_folds: int = 5,
                                           epochs: int = 100,
                                           batch_size: int = 32,
                                           verbose: int = 1) -> Dict[str, Any]:
        """
        Evaluate all feature pairs using fixed hyperparameters for fair comparison.
        """
        
        # Use global best params if none provided
        if fixed_params is None:
            if self.best_global_params is None:
                raise ValueError("No global best parameters found! Run find_global_best_hyperparameters first.")
            fixed_params = self.best_global_params
        
        # Combine original train and test data
        X_combined = np.vstack([train_features, test_features])
        y_combined = np.hstack([train_labels, test_labels])
        
        # Generate all feature pairs
        feature_pairs = list(itertools.combinations(range(len(feature_names)), 2))
        
        if verbose > 0:
            print(f"🔬 Evaluating all {len(feature_pairs)} feature pairs with fixed architecture:")
            print(f"   Architecture: {fixed_params}")
        
        # Storage for results
        pair_names = []
        cv_accuracies = []
        cv_stds = []
        test_accuracies = []
        test_precisions = []
        confusion_matrices = []
        
        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_combined), y=y_combined)
        class_weight_dict = dict(enumerate(class_weights))
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for i, (f1, f2) in enumerate(feature_pairs):
            pair_name = f"{feature_names[f1]} + {feature_names[f2]}"
            pair_names.append(pair_name)
            
            if verbose > 0 and i % 10 == 0:
                print(f"   Progress: {i+1}/{len(feature_pairs)} pairs...")
            
            # Extract feature pair data
            X_pair = X_combined[:, [f1, f2]]
            
            # Split into development and test sets
            X_dev, X_test_final, y_dev, y_test_final = train_test_split(
                X_pair, y_combined, test_size=0.2, stratify=y_combined, random_state=self.random_state
            )
            
            try:
                # Cross-validation on development set
                fold_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(cv.split(X_dev, y_dev)):
                    try:
                        X_fold_train, X_fold_val = X_dev[train_idx], X_dev[val_idx]
                        y_fold_train, y_fold_val = y_dev[train_idx], y_dev[val_idx]
                        
                        # Aggressive memory cleanup
                        tf.keras.backend.clear_session()
                        import gc
                        gc.collect()
                        
                        # Create model with fixed params
                        model_builder = self.create_mlp_model(input_dim=2, **fixed_params)
                        model = model_builder()
                        
                        # Use smaller batch size to avoid memory issues
                        effective_batch_size = min(batch_size, len(X_fold_train) // 4)
                        effective_batch_size = max(8, effective_batch_size)  # Minimum batch size of 8
                        
                        # Train model with timeout and error handling
                        model.fit(
                            X_fold_train, y_fold_train,
                            validation_data=(X_fold_val, y_fold_val),
                            epochs=epochs,
                            batch_size=effective_batch_size,
                            class_weight=class_weight_dict,
                            verbose=0
                        )
                        
                        # Evaluate on validation fold
                        y_pred_proba = model.predict(X_fold_val, verbose=0)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        
                        # Calculate accuracy from confusion matrix only
                        cm = confusion_matrix(y_fold_val, y_pred)
                        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        accuracy = np.mean(np.diag(cm_normalized))
                        fold_scores.append(accuracy)
                        
                        # Clean up model explicitly
                        del model
                        tf.keras.backend.clear_session()
                        
                    except Exception as fold_e:
                        if verbose > 0:
                            print(f"   Fold {fold} failed: {fold_e}")
                        # Use a default poor score for failed folds
                        fold_scores.append(0.3)
                        tf.keras.backend.clear_session()
                        continue
                
                # CV metrics
                cv_acc_mean = np.mean(fold_scores)
                cv_acc_std = np.std(fold_scores)
                cv_accuracies.append(cv_acc_mean)
                cv_stds.append(cv_acc_std)
                
                # Final model on full development set
                tf.keras.backend.clear_session()
                final_model = self.create_mlp_model(input_dim=2, **fixed_params)()
                final_model.fit(
                    X_dev, y_dev,
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=class_weight_dict,
                    verbose=0
                )
                
                # Test set evaluation
                y_test_pred_proba = final_model.predict(X_test_final, verbose=0)
                y_test_pred = np.argmax(y_test_pred_proba, axis=1)
                
                # Calculate metrics from confusion matrix only
                cm = confusion_matrix(y_test_final, y_test_pred)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                # Accuracy from normalized confusion matrix (mean diagonal)
                test_accuracy = np.mean(np.diag(cm_normalized))
                
                # Precision for class 0 (as per your convention)
                precision_class0 = cm_normalized[0, 0] / (cm_normalized[0, 0] + cm_normalized[1, 0]) if (cm_normalized[0, 0] + cm_normalized[1, 0]) > 0 else 0
                
                test_accuracies.append(test_accuracy)
                test_precisions.append(precision_class0)
                confusion_matrices.append(cm_normalized)
                
            except Exception as e:
                if verbose > 0:
                    print(f"   Failed pair {pair_name}: {e}")
                cv_accuracies.append(np.nan)
                cv_stds.append(np.nan)
                test_accuracies.append(np.nan)
                test_precisions.append(np.nan)
                confusion_matrices.append(np.full((2, 2), np.nan))
        
        # Convert to numpy arrays
        cv_accuracies = np.array(cv_accuracies)
        cv_stds = np.array(cv_stds)
        test_accuracies = np.array(test_accuracies)
        test_precisions = np.array(test_precisions)
        confusion_matrices = np.array(confusion_matrices)
        
        if verbose > 0:
            print(f"✅ Completed evaluation of all feature pairs!")
            print(f"   Mean CV accuracy: {np.nanmean(cv_accuracies):.4f} ± {np.nanstd(cv_accuracies):.4f}")
            print(f"   Mean test accuracy: {np.nanmean(test_accuracies):.4f} ± {np.nanstd(test_accuracies):.4f}")
            print(f"   Best test accuracy: {np.nanmax(test_accuracies):.4f}")
        
        return {
            'pair_names': pair_names,
            'cv_accuracies': cv_accuracies,
            'cv_stds': cv_stds,
            'test_accuracies': test_accuracies,
            'test_precisions': test_precisions,
            'confusion_matrices': confusion_matrices,
            'fixed_params': fixed_params,
            'feature_pairs': feature_pairs,
            'feature_names': feature_names
        }
    
    def optimize_all_feature_pairs(self,
                                 train_features: np.ndarray,
                                 train_labels: np.ndarray,
                                 test_features: np.ndarray,
                                 test_labels: np.ndarray,
                                 feature_names: List[str],
                                 cv_folds: int = 5,
                                 max_combinations: int = 50,
                                 epochs: int = 100,
                                 batch_size: int = 32,
                                 verbose: int = 1) -> Dict[str, Any]:
        """
        Optimize hyperparameters for all feature pairs.
        """
        
        # Generate all feature pairs
        feature_pairs = list(itertools.combinations(range(len(feature_names)), 2))
        
        all_results = {}
        best_overall = {'score': 0, 'params': None, 'features': None}
        
        for i, (f1, f2) in enumerate(feature_pairs):
            if verbose > 0:
                print(f"\n=== Optimizing Feature Pair {i+1}/{len(feature_pairs)}: "
                      f"[{feature_names[f1]}, {feature_names[f2]}] ===")
            
            # Extract feature pair data
            X_train = train_features[:, [f1, f2]]
            X_test = test_features[:, [f1, f2]]
            pair_feature_names = [feature_names[f1], feature_names[f2]]
            
            # Optimize this feature pair
            result = self.optimize_single_feature_pair(
                X_train, train_labels, X_test, test_labels,
                pair_feature_names, cv_folds, max_combinations, 
                epochs, batch_size, verbose-1
            )
            
            all_results[f"pair_{i}_{f1}_{f2}"] = result
            
            # Track best overall
            if result['best_cv_score'] > best_overall['score']:
                best_overall = {
                    'score': result['best_cv_score'],
                    'params': result['best_params'],
                    'features': pair_feature_names,
                    'feature_indices': (f1, f2),
                    'full_result': result
                }
            
            if verbose > 0:
                print(f"Best CV accuracy: {result['best_cv_score']:.4f}")
                print(f"Best params: {result['best_params']}")
        
        return {
            'all_results': all_results,
            'best_overall': best_overall,
            'feature_names': feature_names
        }
    
    def print_summary(self, results: Dict[str, Any], top_k: int = 10):
        """
        Print summary of optimization results.
        """
        print("\n" + "="*80)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*80)
        
        # Extract all results
        all_pairs = []
        for key, result in results['all_results'].items():
            if result['all_results']:  # Check if optimization succeeded
                best_result = result['all_results'][0]  # Already sorted by CV accuracy
                all_pairs.append({
                    'features': result['feature_names'],
                    'cv_accuracy': best_result['cv_accuracy_mean'],
                    'cv_std': best_result['cv_accuracy_std'],
                    'test_accuracy': best_result['test_accuracy'],
                    'test_f1': best_result['test_f1'],
                    'params': best_result['params']
                })
        
        # Sort by CV accuracy
        all_pairs.sort(key=lambda x: x['cv_accuracy'], reverse=True)
        
        print(f"\nTop {top_k} Feature Pairs:")
        print("-" * 80)
        for i, pair in enumerate(all_pairs[:top_k]):
            print(f"{i+1:2d}. {pair['features']}")
            print(f"    CV Acc: {pair['cv_accuracy']:.4f} ± {pair['cv_std']:.4f}")
            print(f"    Test Acc: {pair['test_accuracy']:.4f}, Test F1: {pair['test_f1']:.4f}")
            print(f"    Best params: {pair['params']}")
            print()
        
        print(f"\nBest Overall:")
        print(f"Features: {results['best_overall']['features']}")
        print(f"CV Accuracy: {results['best_overall']['score']:.4f}")
        print(f"Best params: {results['best_overall']['params']}")


def run_fair_feature_comparison(train_features: np.ndarray,
                               train_labels: np.ndarray,
                               test_features: np.ndarray,
                               test_labels: np.ndarray,
                               feature_names: List[str],
                               sample_pairs_for_optimization: int = 5,
                               cv_folds: int = 5,
                               max_combinations: int = 50,
                               epochs: int = 50,
                               batch_size: int = 32,
                               random_state: int = 42,
                               save_results: bool = True,
                               verbose: int = 1) -> Dict[str, Any]:
    """
    Run fair comparison of all feature pairs using the same optimized model architecture.
    
    This is a two-step process:
    1. Find globally best hyperparameters using a sample of feature pairs
    2. Evaluate all feature pairs using the same fixed architecture for fair comparison
    
    Args:
        train_features: Training feature matrix (n_samples, n_features)
        train_labels: Training labels (n_samples,)
        test_features: Test feature matrix (n_samples, n_features)  
        test_labels: Test labels (n_samples,)
        feature_names: List of feature names
        sample_pairs_for_optimization: Number of feature pairs to sample for hyperparameter optimization
        cv_folds: Number of cross-validation folds
        max_combinations: Maximum hyperparameter combinations to try
        epochs: Training epochs per model
        batch_size: Training batch size
        random_state: Random seed
        save_results: Whether to save results as numpy arrays
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
    
    Returns:
        Dictionary containing all results including numpy arrays
    """
    
    optimizer = MLPHyperparameterOptimizer(random_state=random_state)
    
    if verbose > 0:
        print("🚀 FEATURE PAIR COMPARISON")
        print("=" * 50)
        print("Step 1: Finding globally best hyperparameters...")
    
    # Step 1: Find globally best hyperparameters
    global_optimization = optimizer.find_global_best_hyperparameters(
        train_features, train_labels, test_features, test_labels,
        feature_names, sample_pairs_for_optimization, cv_folds, 
        max_combinations, epochs, batch_size, verbose
    )
    
    if verbose > 0:
        print("\nStep 2: Evaluating all feature pairs with fixed architecture...")
    
    # Step 2: Evaluate all pairs with fixed architecture
    evaluation_results = optimizer.evaluate_all_pairs_with_fixed_model(
        train_features, train_labels, test_features, test_labels,
        feature_names, global_optimization['best_params'],
        cv_folds, epochs, batch_size, verbose
    )
    
    # Save results as numpy arrays
    if save_results:
        np.save('feature_pair_names.npy', evaluation_results['pair_names'])
        np.save('feature_pair_cv_accuracies.npy', evaluation_results['cv_accuracies'])
        np.save('feature_pair_cv_stds.npy', evaluation_results['cv_stds'])
        np.save('feature_pair_test_accuracies.npy', evaluation_results['test_accuracies'])
        np.save('feature_pair_test_precisions.npy', evaluation_results['test_precisions'])
        np.save('feature_pair_confusion_matrices.npy', evaluation_results['confusion_matrices'])
        
        if verbose > 0:
            print("\n💾 Saved results as numpy arrays:")
            print("   - feature_pair_names.npy")
            print("   - feature_pair_cv_accuracies.npy")
            print("   - feature_pair_cv_stds.npy")
            print("   - feature_pair_test_accuracies.npy")
            print("   - feature_pair_test_precisions.npy")
            print("   - feature_pair_confusion_matrices.npy")
    
    # Combine results
    final_results = {
        'global_optimization': global_optimization,
        'evaluation_results': evaluation_results,
        'best_global_params': global_optimization['best_params'],
        'numpy_arrays': {
            'pair_names': evaluation_results['pair_names'],
            'cv_accuracies': evaluation_results['cv_accuracies'],
            'cv_stds': evaluation_results['cv_stds'],
            'test_accuracies': evaluation_results['test_accuracies'],
            'test_precisions': evaluation_results['test_precisions'],
            'confusion_matrices': evaluation_results['confusion_matrices']
        }
    }
    
    if verbose > 0:
        print(f"\n🎯 SUMMARY:")
        print(f"   Total feature pairs evaluated: {len(evaluation_results['pair_names'])}")
        print(f"   Best CV accuracy: {np.nanmax(evaluation_results['cv_accuracies']):.4f}")
        print(f"   Best test accuracy: {np.nanmax(evaluation_results['test_accuracies']):.4f}")
        best_idx = np.nanargmax(evaluation_results['test_accuracies'])
        print(f"   Best feature pair: {evaluation_results['pair_names'][best_idx]}")
        print(f"   Architecture used: {global_optimization['best_params']}")
    
    return final_results


def run_hyperparameter_optimization(train_features: np.ndarray,
                                   train_labels: np.ndarray,
                                   test_features: np.ndarray,
                                   test_labels: np.ndarray,
                                   feature_names: List[str],
                                   cv_folds: int = 5,
                                   max_combinations: int = 30,
                                   epochs: int = 50,
                                   batch_size: int = 32,
                                   random_state: int = 42,
                                   verbose: int = 1) -> Dict[str, Any]:
    """
    Legacy function - use run_fair_feature_comparison instead for fair comparison.
    """
    return run_fair_feature_comparison(
        train_features, train_labels, test_features, test_labels,
        feature_names, 15, cv_folds, max_combinations, epochs, 
        batch_size, random_state, True, verbose
    )
