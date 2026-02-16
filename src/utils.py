from matplotlib import pyplot as plt
import os
import numpy as np
import tensorflow as tf
import argparse

def convert_namespace_to_dict(config): # TEMPORARY: Helper function to convert Namespace to dictionary
    if isinstance(config, argparse.Namespace):
        # Convert Namespace to a dictionary
        return {key: getattr(config, key) for key in vars(config)}
    return config # If it's already a dictionary, return as is

def configure_gpu():
    """Configure TensorFlow to use GPU if available."""
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU detected: {len(gpus)} GPU(s) available")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
            return True, len(gpus)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False, 0
    else:
        print("⚠ No GPU detected by TensorFlow.")
        print("  Diagnostic information:")
        
        # Check if CUDA is available in the system
        import subprocess
        try:
            nvidia_smi = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode('utf-8')
            print("  ✓ nvidia-smi found - CUDA driver is installed")
            # Extract GPU info from nvidia-smi
            if 'NVIDIA-SMI' in nvidia_smi:
                print("  ⚠ GPU hardware detected but TensorFlow can't see it.")
                print("  Possible solutions:")
                print("    1. Install CUDA toolkit and cuDNN:")
                print("       - For TensorFlow 2.17.0, you need CUDA 12.x and cuDNN 9.x")
                print("    2. Install tensorflow with GPU support:")
                print("       pip install tensorflow[and-cuda]")
                print("    3. Or use conda/mamba:")
                print("       conda install -c conda-forge tensorflow-gpu")
                print("    4. Check CUDA environment variables:")
                print("       echo $CUDA_HOME")
                print("       echo $LD_LIBRARY_PATH")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("  ✗ nvidia-smi not found - CUDA driver may not be installed")
            print("  Install NVIDIA drivers first: https://www.nvidia.com/drivers")
        
        # Check TensorFlow build info
        try:
            build_info = tf.sysconfig.get_build_info()
            print(f"  TensorFlow built with CUDA: {build_info.get('cuda_version', 'Unknown')}")
            print(f"  TensorFlow built with cuDNN: {build_info.get('cudnn_version', 'Unknown')}")
        except:
            pass
        
        print("  Training will use CPU (this will be slow).")
        return False, 0

def get_distribution_strategy(num_gpus=None):
    """
    Get TensorFlow distribution strategy for multi-GPU training.
    
    Args:
        num_gpus: Number of GPUs to use. If None, uses all available GPUs.
    
    Returns:
        Distribution strategy object
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPUs found, using default strategy (CPU or single GPU)")
        return tf.distribute.get_strategy()
    
    if num_gpus is None:
        num_gpus = len(gpus)
    
    if num_gpus == 1:
        print("Using single GPU")
        return tf.distribute.get_strategy()
    elif num_gpus > 1:
        print(f"Using MirroredStrategy with {num_gpus} GPUs")
        return tf.distribute.MirroredStrategy(devices=[f'/GPU:{i}' for i in range(num_gpus)])
    else:
        return tf.distribute.get_strategy()

def set_seed(seed):
    """
    Set the random seed for full reproducibility across CPU and GPU.
    
    This function configures:
    - Python's random module
    - NumPy random state
    - TensorFlow random operations
    - CUDA/cuDNN deterministic behavior
    
    Args:
        seed: Integer seed value for reproducibility
    """
    import os
    import random
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set TensorFlow random seed
    tf.random.set_seed(seed)
    
    # Configure TensorFlow for deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Disable TensorFlow's use of multithreading for determinism
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Enable deterministic operations in TensorFlow (TF 2.8+)
    try:
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        # Fallback for older TensorFlow versions
        pass
    
    print(f"✓ Seed set to {seed} with deterministic GPU operations enabled")

def save_model_weights_to_disk(encoder, decoder, discriminator, output_dir):
    """Save model weights to disk."""
    os.makedirs(output_dir, exist_ok=True)
    encoder.save_weights(os.path.join(output_dir, "encoder.weights.h5"))
    decoder.save_weights(os.path.join(output_dir, "decoder.weights.h5"))
    discriminator.save_weights(os.path.join(output_dir, "discriminator.weights.h5"))

def load_model_weights_from_disk(encoder, decoder, discriminator, output_dir):
    """Load model weights from disk."""
    encoder.load_weights(os.path.join(output_dir, "encoder.weights.h5"))
    decoder.load_weights(os.path.join(output_dir, "decoder.weights.h5"))
    discriminator.load_weights(os.path.join(output_dir, "discriminator.weights.h5"))
    return encoder, decoder, discriminator


def save_loss_plots_autoencoder(reconstruction_losses, adversarial_losses, 
                               val_reconstruction_losses=None, val_adversarial_losses=None,
                               output_dir="./results/loss_plots"):
    """
    Save loss plots for autoencoder training with optional validation losses.
    
    Args:
        reconstruction_losses: Training reconstruction losses
        adversarial_losses: Training adversarial losses
        val_reconstruction_losses: Validation reconstruction losses (optional)
        val_adversarial_losses: Validation adversarial losses (optional)
        output_dir: Directory to save plots
    """
    # Create subplots for better visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors and styles
    train_style = {'linestyle': '-', 'linewidth': 2, 'alpha': 0.8}
    val_style = {'linestyle': '--', 'linewidth': 2, 'alpha': 0.7}
    
    # Plot 1: Reconstruction Loss
    axes[0].plot(reconstruction_losses, label='Train Reconstruction', color='blue', **train_style)
    if val_reconstruction_losses is not None:
        axes[0].plot(val_reconstruction_losses, label='Val Reconstruction', color='blue', **val_style)
    axes[0].set_title('Reconstruction Loss', fontsize=14)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Adversarial Loss
    axes[1].plot(adversarial_losses, label='Train Adversarial', color='red', **train_style)
    if val_adversarial_losses is not None:
        axes[1].plot(val_adversarial_losses, label='Val Adversarial', color='red', **val_style)
    axes[1].set_title('Adversarial Loss', fontsize=14)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plots
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/autoencoder_loss_plots.eps", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/autoencoder_loss_plots.png", dpi=300, bbox_inches='tight')
    
    # Create a combined plot for overview
    plt.figure(figsize=(12, 8))
    
    # Plot training losses
    plt.plot(reconstruction_losses, label='Train Reconstruction', color='blue', linestyle='-', linewidth=2)
    plt.plot(adversarial_losses, label='Train Adversarial', color='red', linestyle='-', linewidth=2)
    
    # Plot validation losses if provided
    if val_reconstruction_losses is not None:
        plt.plot(val_reconstruction_losses, label='Val Reconstruction', color='blue', linestyle='--', linewidth=2, alpha=0.7)
        plt.plot(val_adversarial_losses, label='Val Adversarial', color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Title and labels
    plt.title("Autoencoder Training and Validation Losses", fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)
    
    # Save combined plot
    plt.savefig(f"{output_dir}/autoencoder_loss_combined.eps", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/autoencoder_loss_combined.png", dpi=300, bbox_inches='tight')
    
    # Close plots to avoid memory issues
    plt.close('all')
    
    print(f"Autoencoder loss plots saved to {output_dir}")


def save_loss_plots_cov(reconstruction_losses, adversarial_losses, cov_losses, contra_losses, 
                        val_reconstruction_losses=None, val_adversarial_losses=None, val_cov_losses=None, val_contra_losses=None,
                        output_dir="./results/loss_plots"):
    """
    Save loss plots for AI4CellFate training with optional validation losses.
    
    Args:
        reconstruction_losses: Training reconstruction losses
        adversarial_losses: Training adversarial losses
        cov_losses: Training covariance losses
        contra_losses: Training contrastive losses
        val_reconstruction_losses: Validation reconstruction losses (optional)
        val_adversarial_losses: Validation adversarial losses (optional)
        val_cov_losses: Validation covariance losses (optional)
        val_contra_losses: Validation contrastive losses (optional)
        output_dir: Directory to save plots
    """
    # Create subplots for better visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Define colors and styles
    train_style = {'linestyle': '-', 'linewidth': 2, 'alpha': 0.8}
    val_style = {'linestyle': '--', 'linewidth': 2, 'alpha': 0.7}
    
    # Plot 1: Reconstruction Loss
    axes[0, 0].plot(reconstruction_losses, label='Train Reconstruction', color='blue', **train_style)
    if val_reconstruction_losses is not None:
        axes[0, 0].plot(val_reconstruction_losses, label='Val Reconstruction', color='blue', **val_style)
    axes[0, 0].set_title('Reconstruction Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epochs', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Adversarial Loss
    axes[0, 1].plot(adversarial_losses, label='Train Adversarial', color='red', **train_style)
    if val_adversarial_losses is not None:
        axes[0, 1].plot(val_adversarial_losses, label='Val Adversarial', color='red', **val_style)
    axes[0, 1].set_title('Adversarial Loss', fontsize=14)
    axes[0, 1].set_xlabel('Epochs', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Covariance Loss
    axes[1, 0].plot(cov_losses, label='Train Covariance', color='purple', **train_style)
    if val_cov_losses is not None:
        axes[1, 0].plot(val_cov_losses, label='Val Covariance', color='purple', **val_style)
    axes[1, 0].set_title('Covariance Loss', fontsize=14)
    axes[1, 0].set_xlabel('Epochs', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Contrastive Loss
    axes[1, 1].plot(contra_losses, label='Train Contrastive', color='green', **train_style)
    if val_contra_losses is not None:
        axes[1, 1].plot(val_contra_losses, label='Val Contrastive', color='green', **val_style)
    axes[1, 1].set_title('Contrastive Loss', fontsize=14)
    axes[1, 1].set_xlabel('Epochs', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plots
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/loss_plots_detailed.eps", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/loss_plots_detailed.png", dpi=300, bbox_inches='tight')
    
    # Create a combined plot for overview
    plt.figure(figsize=(12, 8))
    
    # Plot training losses
    plt.plot(reconstruction_losses, label='Train Reconstruction', color='blue', linestyle='-', linewidth=2)
    plt.plot(adversarial_losses, label='Train Adversarial', color='red', linestyle='-', linewidth=2)
    plt.plot(cov_losses, label='Train Covariance', color='purple', linestyle='-', linewidth=2)
    plt.plot(contra_losses, label='Train Contrastive', color='green', linestyle='-', linewidth=2)
    
    # Plot validation losses if provided
    if val_reconstruction_losses is not None:
        plt.plot(val_reconstruction_losses, label='Val Reconstruction', color='blue', linestyle='--', linewidth=2, alpha=0.7)
        plt.plot(val_adversarial_losses, label='Val Adversarial', color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.plot(val_cov_losses, label='Val Covariance', color='purple', linestyle='--', linewidth=2, alpha=0.7)
        plt.plot(val_contra_losses, label='Val Contrastive', color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    # Title and labels
    plt.title("AI4CellFate Training and Validation Losses", fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc='upper right', fontsize=12, ncol=2)
    
    # Save combined plot
    plt.savefig(f"{output_dir}/loss_plot_combined.eps", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/loss_plot_combined.png", dpi=300, bbox_inches='tight')
    
    # Close plots to avoid memory issues
    plt.close('all')
    
    print(f"Loss plots saved to {output_dir}")
