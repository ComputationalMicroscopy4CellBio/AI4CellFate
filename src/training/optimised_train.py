import argparse
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import numpy as np
from ..config import CONFIG
from ..models import Encoder, Decoder, mlp_classifier, Discriminator
from .loss_functions import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--batch_size', type=int, default=CONFIG.get('batch_size'))
    parser.add_argument('--epochs', type=int, default=CONFIG.get('epochs'))
    parser.add_argument('--learning_rate', type=float, default=CONFIG.get('learning_rate'))
    parser.add_argument('--seed', type=int, default=CONFIG.get('seed'))
    parser.add_argument('--latent_dim', type=int, default=CONFIG.get('latent_dim'))
    parser.add_argument('--GaussianNoise_std', type=float, default=CONFIG.get('GaussianNoise_std'))
    parser.add_argument('--lambda_recon', type=float, default=CONFIG.get('lambda_recon'))
    parser.add_argument('--lambda_adv', type=float, default=CONFIG.get('lambda_adv'))
    parser.add_argument('--lambda_clf', type=float, default=CONFIG.get('lambda_clf'))
    parser.add_argument('--lambda_cov', type=float, default=CONFIG.get('lambda_cov'))
    return parser.parse_args()

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

def train_model(config, x_train, save_loss_plot=True, save_model_weights=True):
    # Set random seeds for reproducibility
    config = convert_namespace_to_dict(config)
    set_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])

    print(f"Training with batch size: {config['batch_size']}, epochs: {config['epochs']}, "
          f"learning rate: {config['learning_rate']}, seed: {config['seed']}, latent dim: {config['latent_dim']}")

    img_shape = (x_train.shape[1], x_train.shape[2], 1) # Assuming grayscale images

    # Create model instances
    encoder = Encoder(img_shape=img_shape, latent_dim=config['latent_dim'], num_classes=2, gaussian_noise_std=config['GaussianNoise_std']).model
    decoder = Decoder(latent_dim=config['latent_dim'], img_shape=img_shape, gaussian_noise_std=config['GaussianNoise_std']).model
    discriminator = Discriminator(latent_dim=config['latent_dim']).model

    # Optimizers
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)

    # Placeholder for storing losses
    reconstruction_losses = []
    adversarial_losses = []
    total_loss = []

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    for epoch in range(config['epochs']):
        epoch_reconstruction_losses, epoch_adversarial_losses = [], []

        for n_batch in range(len(x_train) // config['batch_size']):
            #idx = np.random.randint(0, x_train.shape[0], config['batch_size'])
            idx = rng.integers(0, x_train.shape[0], config['batch_size'])
            #idx = np.random.default_rng(config['seed']).integers(0, x_train.shape[0], config['batch_size'])
            image_batch = x_train[idx]

            with tf.GradientTape() as tape:
                # Forward pass through encoder and decoder
                z_imgs, z_score = encoder(image_batch, training=True)
                recon_imgs = decoder(z_imgs, training=True)[:, :, :, 0]

                # Reconstruction loss
                recon_loss = mse_loss(image_batch, recon_imgs)

                # Adversarial loss for discriminator
                z_discriminator_out = discriminator(z_imgs, training=True)
                adv_loss = bce_loss(real_y, z_discriminator_out)

                # Total autoencoder loss
                ae_loss = config['lambda_recon'] * recon_loss + config['lambda_adv'] * adv_loss
                total_loss.append(ae_loss)

            # Backpropagation for autoencoder (encoder + decoder)
            trainable_variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(ae_loss, trainable_variables)
            ae_optimizer.apply_gradients(zip(gradients, trainable_variables))

            # Train the discriminator 
            rand_vecs = tf.random.stateless_normal(
                shape=(config['batch_size'], config['latent_dim']),
                seed=(config['seed'], epoch + n_batch)
            )
            # which rand_vecs to use?
            #rand_vecs = tf.random.normal(shape=(config['batch_size'], config['latent_dim']))

            with tf.GradientTape() as tape:
                z_discriminator_out = discriminator(z_imgs, training=True)
                rand_discriminator_out = discriminator(rand_vecs, training=True)

                discriminator_loss = 0.5 * bce_loss(real_y, rand_discriminator_out) + \
                                     0.5 * bce_loss(fake_y, z_discriminator_out)

            # Update discriminator weights
            disc_gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            # Track individual losses for adjustment
            epoch_reconstruction_losses.append(config['lambda_recon'] * recon_loss)
            epoch_adversarial_losses.append(config['lambda_adv'] * adv_loss)

        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)

        reconstruction_losses.append(avg_recon_loss)
        adversarial_losses.append(avg_adv_loss)

        # Print and save results at the end of each epoch
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}")

    if save_loss_plot:
        print("Saving loss plot...")
        save_loss_plots(reconstruction_losses, adversarial_losses, config['epochs'])
    
    if save_model_weights:
        print("Saving model weights...")
        output_dir = "./results/models"
        os.makedirs(output_dir, exist_ok=True)

        encoder_weights_path = os.path.join(output_dir, "encoder.weights.h5")
        decoder_weights_path = os.path.join(output_dir, "decoder.weights.h5")
        discriminator_weights_path = os.path.join(output_dir, "discriminator.weights.h5")

        encoder.save_weights(encoder_weights_path)
        decoder.save_weights(decoder_weights_path)
        discriminator.save_weights(discriminator_weights_path)

        print(f"Encoder weights saved to {encoder_weights_path}")
        print(f"Decoder weights saved to {decoder_weights_path}")
        print(f"Discriminator weights saved to {discriminator_weights_path}")

    return {
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'reconstruction_losses': reconstruction_losses,
        'adversarial_losses': adversarial_losses
    }

def L2_norm(grad_list):
    """
    Compute the global L2 norm of a list of gradients.
    """
    squares = [tf.reduce_sum(g**2) for g in grad_list if g is not None]
    if len(squares) == 0:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.sqrt(tf.add_n(squares))

def train_cov(config, x_train, y_train, save_loss_plot=True, save_model_weights=True):
    """Train the full CellFate model - the autoencoder and the classifier (MLP), 
    with latent space disentanglement for interpretability."""

    config = convert_namespace_to_dict(config)
    set_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])

    print(f"Training with batch size: {config['batch_size']}, epochs: {config['epochs']}, "
          f"learning rate: {config['learning_rate']}, seed: {config['seed']}, latent dim: {config['latent_dim']}")
    
    img_shape = (x_train.shape[1], x_train.shape[2], 1) # Assuming grayscale images

    # Create model instances
    encoder = Encoder(img_shape=img_shape, latent_dim=config['latent_dim'], num_classes=2, gaussian_noise_std=config['GaussianNoise_std']).model
    decoder = Decoder(latent_dim=config['latent_dim'], img_shape=img_shape, gaussian_noise_std=config['GaussianNoise_std']).model
    discriminator = Discriminator(latent_dim=config['latent_dim']).model

    # Optimizers
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)

    lambda_recon = tf.Variable(config["lambda_recon"], trainable=False, dtype=tf.float32)
    lambda_cov = tf.Variable(config["lambda_cov"], trainable=False, dtype=tf.float32)

    total_lambda = tf.Variable(lambda_recon + lambda_cov, trainable=False, dtype=tf.float32)
    lambda_recon.assign(lambda_recon / total_lambda)
    lambda_cov.assign(lambda_cov / total_lambda)

    # MOO: Initialise empty first losses - one for each loss
    L_0_recon = None
    L_0_cov = None
    L_0_recon_list = []
    L_0_cov_list = []

    # MOO: Initialise epsilon for safe division - could go in config
    epsilon = 1e-8

    # MOO: Initialise beta parameter for GradNorm - could go in config
    beta = 0.5

    # Placeholder for storing losses
    reconstruction_losses = []
    adversarial_losses = []
    cov_losses = []
    total_loss = []

    # MOO: Placeholder for storing weights
    lambda_recon_vals = []
    lambda_cov_vals = []

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    for epoch in range(config['epochs']): 
        epoch_reconstruction_losses, epoch_adversarial_losses, epoch_cov_losses  = [], [], []

        recon_loss_norms = []
        cov_loss_norms = []

        for n_batch in range(len(x_train) // config['batch_size']):
            idx = rng.integers(0, x_train.shape[0], config['batch_size'])
            #idx = np.random.randint(0, x_train.shape[0], config['batch_size'])
            image_batch = x_train[idx]

            # MOO: Measure baseline losses - do this for all loss functions
            if L_0_recon is None or L_0_cov is None:
                with tf.GradientTape() as tape_init:
                    # Forward pass through encoder and decoder
                    z_imgs, z_score = encoder(image_batch, training=True)
                    recon_imgs = decoder(z_imgs, training=True)[:, :, :, 0]

                    # Reconstruction loss
                    L_recon_init = lambda_recon * mse_loss(image_batch, recon_imgs)

                    # Covariance loss
                    cov, z_std_loss, diag_cov_mean, off_diag_loss = cov_loss_terms(z_imgs)
                    L_cov_init = lambda_cov * (0.5 * diag_cov_mean + 0.5 * z_std_loss)

                # Append L_0 init values to L_0 lists
                L_0_recon_list.append(float(L_recon_init))
                L_0_cov_list.append(float(L_cov_init))
                del tape_init

            with tf.GradientTape(persistent = True) as tape:
                # Forward pass through encoder and decoder
                z_imgs, z_score = encoder(image_batch, training=True)
                recon_imgs = decoder(z_imgs, training=True)[:, :, :, 0]

                # Reconstruction loss
                recon_loss = mse_loss(image_batch, recon_imgs)

                # Adversarial loss for discriminator
                z_discriminator_out = discriminator(z_imgs, training=True)
                adv_loss = bce_loss(real_y, z_discriminator_out)

                # Covariance loss TODO: CHECK THIS LOSS
                cov, z_std_loss, diag_cov_mean, off_diag_loss = cov_loss_terms(z_imgs)
                cov_loss = 0.5 * diag_cov_mean + 0.5 * z_std_loss

                # Total autoencoder loss
                ae_loss = config['lambda_recon'] * recon_loss + config['lambda_adv'] * adv_loss + config['lambda_cov'] * cov_loss
                total_loss.append(ae_loss)

            # MOO: Compute gradients for encoder and decoder - gradients for each loss
            trainable_variables = encoder.trainable_variables + decoder.trainable_variables
            recon_gradients = tape.gradient(recon_loss, trainable_variables)
            cov_gradients = tape.gradient(cov_loss, trainable_variables)
 
            recon_gradients = [g * lambda_recon if g is not None else None for g in recon_gradients]
            cov_gradients = [g * lambda_cov if g is not None else None for g in cov_gradients]

            # MOO: Compute the L2 norm of all loss gradients
            recon_loss_norm = L2_norm(recon_gradients)
            cov_loss_norm = L2_norm(cov_gradients)
            recon_loss_norms.append(float(recon_loss_norm))
            cov_loss_norms.append(float(cov_loss_norm))
            
            gradients = tape.gradient(ae_loss, trainable_variables)
            ae_optimizer.apply_gradients(zip(gradients, trainable_variables))
            del tape

            # Train the discriminator
            rand_vecs = tf.random.stateless_normal(
                shape=(config['batch_size'], config['latent_dim']),
                seed=(config['seed'], epoch + n_batch)
            )
            #rand_vecs = tf.random.normal(shape=(config['batch_size'], config['latent_dim']))

            with tf.GradientTape() as tape:
                z_discriminator_out = discriminator(z_imgs, training=True)
                rand_discriminator_out = discriminator(rand_vecs, training=True)

                discriminator_loss = 0.5 * bce_loss(real_y, rand_discriminator_out) + \
                                     0.5 * bce_loss(fake_y, z_discriminator_out)

            # Update discriminator weights
            disc_gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            # MOO: Track individual losses for adjustment - add in lambda values here (not config)
            epoch_reconstruction_losses.append(lambda_recon * recon_loss)
            epoch_cov_losses.append(lambda_cov * cov_loss)
            epoch_adversarial_losses.append(config['lambda_adv'] * adv_loss)
        
        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)
        avg_cov_loss = np.mean(epoch_cov_losses)

        reconstruction_losses.append(avg_recon_loss)
        adversarial_losses.append(avg_adv_loss)
        cov_losses.append(avg_cov_loss)

        #MOO
        lambda_recon_vals.append(lambda_recon)
        lambda_cov_vals.append(lambda_cov)

        # Print and save results at the end of each epoch
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, "
              f"Covariance loss: {avg_cov_loss:.4f}")
    
         # MOO: Get L_0 values if first epoch (average of lists)
        if L_0_recon is None or L_0_cov is None:
            L_0_recon = np.mean(L_0_recon_list)
            L_0_cov = np.mean(L_0_cov_list)

        # MOO: Update weights for the losses
        if L_0_recon is not None and L_0_cov is not None:
            r_recon = (avg_recon_loss / (L_0_recon + epsilon))
            r_cov = (avg_cov_loss / (L_0_cov + epsilon))
                
            # MOO: Get average gradient norms
            recon_loss_norm = np.mean(recon_loss_norms)
            cov_loss_norm = np.mean(cov_loss_norms)
            grad_avg = (recon_loss_norm + cov_loss_norm) / 2.0

            factor_recon = grad_avg * r_recon ** beta / (recon_loss_norm + epsilon)
            factor_cov = grad_avg * r_cov ** beta / (cov_loss_norm + epsilon)
            factor_recon = tf.clip_by_value(factor_recon, 1e-2, 1e2)
            factor_cov = tf.clip_by_value(factor_cov,   1e-2, 1e2)
            lambda_recon.assign(lambda_recon * tf.cast(factor_recon, tf.float32))
            lambda_cov.assign(lambda_cov * tf.cast(factor_cov, tf.float32))
        
        # MOO: Normalise lambda values by sum of their totals
        total_lambda.assign(lambda_recon + lambda_cov)
        lambda_recon.assign(lambda_recon / total_lambda)
        lambda_cov.assign(lambda_cov / total_lambda)

    if save_loss_plot:
        print("Saving loss plot...")
        save_loss_plots_cov(reconstruction_losses, adversarial_losses, cov_losses, config['epochs'])
    
    if save_model_weights:
        print("Saving model weights...")
        output_dir = "./results/models"
        os.makedirs(output_dir, exist_ok=True)

        encoder_weights_path = os.path.join(output_dir, "encoder.weights.h5")
        decoder_weights_path = os.path.join(output_dir, "decoder.weights.h5")
        discriminator_weights_path = os.path.join(output_dir, "discriminator.weights.h5")

        encoder.save_weights(encoder_weights_path)
        decoder.save_weights(decoder_weights_path)
        discriminator.save_weights(discriminator_weights_path)

        print(f"Encoder weights saved to {encoder_weights_path}")
        print(f"Decoder weights saved to {decoder_weights_path}")
        print(f"Discriminator weights saved to {discriminator_weights_path}")

    return {
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'reconstruction_losses': reconstruction_losses,
        'adversarial_losses': adversarial_losses,
    }


def save_loss_plots(reconstruction_losses, adversarial_losses, epochs):
    # Create loss plot and save it under the 'results' directory
    plt.figure(figsize=(10, 5))

    # Plot both reconstruction and adversarial losses with different colors
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
    results_dir = './results/loss_plots'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/loss_plot.png", dpi=300)

    # Close the plot to avoid memory issues
    plt.close()


def save_loss_plots_cov(reconstruction_losses, adversarial_losses, cov_losses, epochs):
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
    results_dir = './results/loss_plots'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/loss_plot.png", dpi=300)

    # Close the plot to avoid memory issues
    plt.close()


def main():
    try:
        # Try parsing arguments if the script is run from the terminal
        args = parse_arguments()
    except SystemExit:
        args = {
            'batch_size': 30,
            'epochs': 50,
            'learning_rate': 0.001,
            'seed': 42,
            'latent_dim': 20,
            'GaussianNoise_std': 0.003
        }
    x_train = np.random.rand(100, 20, 20)  # Random input data
    y_train = np.random.randint(0, 2, 100)  # Random labels
    x_test = np.random.rand(20, 20, 20)  # Random input data
    y_test = np.random.randint(0, 2, 20)  # Random labels
    train_cov(args, x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()

