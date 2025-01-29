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

def set_seed(seed):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)

def save_model_weights_to_disk(encoder, decoder, discriminator, output_dir):
    """Save model weights to disk."""
    os.makedirs(output_dir, exist_ok=True)
    encoder.save_weights(os.path.join(output_dir, "encoder.weights.h5"))
    decoder.save_weights(os.path.join(output_dir, "decoder.weights.h5"))
    discriminator.save_weights(os.path.join(output_dir, "discriminator.weights.h5"))

def save_loss_plots_autoencoder(reconstruction_losses, adversarial_losses, output_dir):
    # Create loss plot and save it under the 'results' directory
    plt.figure(figsize=(10, 5))

    # Plot both reconstruction and adversarial losses with different colors
    print(reconstruction_losses)
    print("len:",len(reconstruction_losses))
    plt.plot(reconstruction_losses, label='Reconstruction Loss', color='blue', linestyle='-', linewidth=2)
    plt.plot(adversarial_losses, label='Adversarial Loss', color='red', linestyle='--', linewidth=2)

    # Title and labels
    plt.title(f"Training Losses", fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # Add a grid and legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=12)
    
    # Save the plot with a dpi of 300
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/loss_plot.png", dpi=300)

    # Close the plot to avoid memory issues
    plt.close()


def save_loss_plots_cov(reconstruction_losses, adversarial_losses, cov_losses, output_dir):
    # Create loss plot and save it under the 'results' directory
    plt.figure(figsize=(10, 5))

    # Plot both reconstruction and adversarial losses with different colors
    plt.plot(reconstruction_losses, label='Reconstruction Loss', color='blue', linestyle='-', linewidth=2)
    plt.plot(adversarial_losses, label='Adversarial Loss', color='red', linestyle='--', linewidth=2)
    plt.plot(cov_losses, label='Covariance Loss', color='purple', linestyle='-.', linewidth=2)

    # Title and labels
    plt.title(f"Training Losses", fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # Add a grid and legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=12)
    
    # Save the plot with a dpi of 300
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/loss_plot.png", dpi=300)

    # Close the plot to avoid memory issues
    plt.close()


def save_loss_plots_clf(reconstruction_losses, adversarial_losses, cov_losses, classification_losses, output_dir):
    # Create loss plot and save it under the 'results' directory
    plt.figure(figsize=(10, 5))

    # Plot both reconstruction and adversarial losses with different colors
    plt.plot(reconstruction_losses, label='Reconstruction Loss', color='blue', linestyle='-', linewidth=2)
    plt.plot(adversarial_losses, label='Adversarial Loss', color='red', linestyle='--', linewidth=2)
    plt.plot(cov_losses, label='Covariance Loss', color='purple', linestyle='-.', linewidth=2)
    plt.plot(classification_losses, label='Classification Loss', color='green', linestyle=':', linewidth=2)

    # Title and labels
    plt.title(f"Training Losses", fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # Add a grid and legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=12)
    
    # Save the plot with a dpi of 300
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/loss_plot.png", dpi=300)

    # Close the plot to avoid memory issues
    plt.close()