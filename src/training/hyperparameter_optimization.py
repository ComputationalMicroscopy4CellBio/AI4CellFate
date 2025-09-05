import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
        
    def create_mlp_model(self, 
                        input_dim: int,
                        hidden_layers: List[int],
                        dropout_rate: float = 0.3,
                        l2_reg: float = 1e-4,
                        activation: str = 'relu',
                        batch_norm: bool = True,
                        learning_rate: float = 0.001):
        """
        Create MLP model with specified architecture.
        """
        def build_model():
            model = Sequential()
            model.add(layers.Input(shape=(input_dim,)))
            
            # batch normalization at input
            if batch_norm:
                model.add(layers.BatchNormalization())
            
            # Hidden layers
            for i, units in enumerate(hidden_layers):
                model.add(layers.Dense(
                    units, 
                    activation=activation,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
                ))
                model.add(layers.Dropout(dropout_rate))
                
            
            # Output layer
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
                [64],           # Single layer - large
                [16, 8],        # Two layers - small
                [32, 16],       # Two layers - medium
                [64, 32],       # Two layers - large
                [32, 16, 8],    # Three layers - medium
                [64, 32, 16],   # Three layers - large
                [128, 64, 32],  # Three layers - extra large
                [16, 16],       # Equal layers - small
                [32, 32],       # Equal layers - medium
                [64, 64],       # Equal layers - large
            ],
            
            # Regularization
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            'l2_reg': [1e-5, 1e-4, 1e-3, 1e-2],
            
            # Training parameters
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            
            # Architecture options
            'activation': ['relu', 'tanh', 'swish'],
            'batch_norm': [True, False],
        }
    
    def optimize_single_feature_pair(self, 
                                   X_train: np.ndarray, 
                                   y_train: np.ndarray,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   feature_names: List[str],
                                   cv_folds: int = 5,
                                   max_combinations: int = 50,
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
                    # Split data
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    # Clear session
                    tf.keras.backend.clear_session()
                    
                    # Create model
                    model_builder = self.create_mlp_model(
                        input_dim=X_train.shape[1],
                        **param_dict
                    )
                    model = model_builder()
                    
                    # Train model
                    history = model.fit(
                        X_fold_train, y_fold_train,
                        validation_data=(X_fold_val, y_fold_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        class_weight=class_weight_dict,
                        verbose=0
                    )
                    
                    # Evaluate on validation fold
                    y_pred_proba = model.predict(X_fold_val, verbose=0)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_fold_val, y_pred)
                    fold_scores.append(accuracy)
                    fold_predictions.extend(y_pred)
                    fold_true_labels.extend(y_fold_val)
                
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
                
                # Calculate test metrics
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_test_pred)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
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
    Convenience function to run hyperparameter optimization.
    
    Args:
        train_features: Training feature matrix (n_samples, n_features)
        train_labels: Training labels (n_samples,)
        test_features: Test feature matrix (n_samples, n_features)  
        test_labels: Test labels (n_samples,)
        feature_names: List of feature names
        cv_folds: Number of cross-validation folds
        max_combinations: Maximum hyperparameter combinations to try per feature pair
        epochs: Training epochs per model
        batch_size: Training batch size
        random_state: Random seed
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
    
    Returns:
        Dictionary containing optimization results
    """
    
    optimizer = MLPHyperparameterOptimizer(random_state=random_state)
    
    results = optimizer.optimize_all_feature_pairs(
        train_features, train_labels, test_features, test_labels,
        feature_names, cv_folds, max_combinations, epochs, batch_size, verbose
    )
    
    if verbose > 0:
        optimizer.print_summary(results)
    
    return results
