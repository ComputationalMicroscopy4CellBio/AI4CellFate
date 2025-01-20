import argparse
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import numpy as np
from ..config import CONFIG
from ..models import Encoder, Decoder, mlp_classifier, Discriminator
from .loss_functions import *
from .optimisation_functions import *

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
    return parser.parse_args()

def convert_namespace_to_dict(config): # TEMPORARY: Helper function to convert Namespace to dictionary
    if isinstance(config, argparse.Namespace):
        # Convert Namespace to a dictionary
        return {key: getattr(config, key) for key in vars(config)}
    return config # If it's already a dictionary, return as is

def train_model(config, x_train, save_loss_plot=True, save_model_weights=True):
    # Set random seeds for reproducibility
    config = convert_namespace_to_dict(config)
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])

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
    
    # MOO: Initialise weights for the losses - one for each loss
    lambda_recon = tf.Variable(config["lambda_recon"], trainable=False, dtype=tf.float32)
    lambda_adv = tf.Variable(config["lambda_adv"], trainable=False, dtype=tf.float32)

    # MOO: Normalise lambda values by sum of their totals
    total_lambda = tf.Variable(lambda_recon + lambda_adv, trainable=False, dtype=tf.float32)
    lambda_recon.assign(lambda_recon / total_lambda)
    lambda_adv.assign(lambda_adv / total_lambda)

    # MOO: Initialise empty first losses - one for each loss
    L_0_recon = None
    L_0_adv = None
    L_0_recon_list = []
    L_0_adv_list = []

    # MOO: Initialise epsilon for safe division - could go in config
    epsilon = 1e-8

    # MOO: Initialise beta parameter for GradNorm - could go in config
    beta = 0.5

    # Placeholder for storing losses
    reconstruction_losses = []
    adversarial_losses = []
    total_loss = []

    # MOO: Placeholder for storing weights
    lambda_recon_vals = []
    lambda_adv_vals = []

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    for epoch in range(config['epochs']):
        epoch_reconstruction_losses, epoch_adversarial_losses = [], []
        
        # MOO: Initialise empty average gradient norms list
        recon_loss_norms = []
        adv_loss_norms = []

        for n_batch in range(len(x_train) // config['batch_size']):
            idx = np.random.randint(0, x_train.shape[0], config['batch_size'])
            image_batch = x_train[idx]

            # MOO: Measure baseline losses - do this for all loss functions
            if L_0_recon is None or L_0_adv is None:
                with tf.GradientTape() as tape_init:
                    # Forward pass through encoder and decoder
                    z_imgs, z_score = encoder(image_batch, training=True)
                    recon_imgs = decoder(z_imgs, training=True)[:, :, :, 0]

                    # Reconstruction loss
                    L_recon_init = lambda_recon * mse_loss(image_batch, recon_imgs)

                    # Adversarial loss for discriminator
                    z_discriminator_out = discriminator(z_imgs, training=True)
                    L_adv_init = lambda_recon * bce_loss(real_y, z_discriminator_out)

                # Append L_0 init values to L_0 lists
                L_0_recon_list.append(float(L_recon_init))
                L_0_adv_list.append(float(L_adv_init))
                del tape_init

            # Determine appropriate values for weights
            with tf.GradientTape(persistent=True) as tape:
                # Forward pass through encoder and decoder
                z_imgs, z_score = encoder(image_batch, training=True)
                recon_imgs = decoder(z_imgs, training=True)[:, :, :, 0]

                # Reconstruction loss
                recon_loss = mse_loss(image_batch, recon_imgs)

                # Adversarial loss for discriminator
                z_discriminator_out = discriminator(z_imgs, training=True)
                adv_loss = bce_loss(real_y, z_discriminator_out)

                # Total autoencoder loss
                ae_loss = lambda_recon * recon_loss + lambda_adv * adv_loss
                total_loss.append(ae_loss)

            # MOO: Compute gradients for encoder and decoder - gradients for each loss
            trainable_variables = encoder.trainable_variables + decoder.trainable_variables
            recon_gradients = tape.gradient(recon_loss, trainable_variables)
            adv_gradients = tape.gradient(adv_loss, trainable_variables)
 
            recon_gradients = [g * lambda_recon if g is not None else None for g in recon_gradients]
            adv_gradients = [g * lambda_adv if g is not None else None for g in adv_gradients]

            # MOO: Compute the L2 norm of all loss gradients
            recon_loss_norm = L2_norm(recon_gradients)
            adv_loss_norm = L2_norm(adv_gradients)
            recon_loss_norms.append(float(recon_loss_norm))
            adv_loss_norms.append(float(adv_loss_norm))

            gradients = tape.gradient(ae_loss, trainable_variables)
            ae_optimizer.apply_gradients(zip(gradients, trainable_variables))
            del tape

            # Train the discriminator
            rand_vecs = tf.random.normal(shape=(config['batch_size'], config['latent_dim']))

            with tf.GradientTape() as tape:
                z_discriminator_out = discriminator(z_imgs, training=True)
                rand_discriminator_out = discriminator(rand_vecs, training=True)

                discriminator_loss = 0.5 * bce_loss(real_y, rand_discriminator_out) + \
                                     0.5 * bce_loss(fake_y, z_discriminator_out)

            # Calculate gradients for discriminator
            disc_gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            # MOO: Track individual losses for adjustment - add in lambda values here (not config)
            epoch_reconstruction_losses.append(lambda_recon * recon_loss)
            epoch_adversarial_losses.append(lambda_adv * adv_loss)

        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)

        reconstruction_losses.append(avg_recon_loss)
        adversarial_losses.append(avg_adv_loss)

        # MOO: Store weights for the epoch
        lambda_recon_vals.append(lambda_recon)
        lambda_adv_vals.append(lambda_adv)

        # Print and save results at the end of each epoch
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, "
              f"Reconstruction lambda: {lambda_recon.numpy():.4f}, "
              f"Adversarial lambda: {lambda_adv.numpy():.4f}")
        
        # MOO: Get L_0 values if first epoch (average of lists)
        if L_0_recon is None or L_0_adv is None:
            L_0_recon = np.mean(L_0_recon_list)
            L_0_adv = np.mean(L_0_adv_list)

        # MOO: Update weights for the losses
        if L_0_recon is not None and L_0_adv is not None:
            r_recon = (avg_recon_loss / (L_0_recon + epsilon))
            r_adv = (avg_adv_loss / (L_0_adv + epsilon))
                
            # MOO: Get average gradient norms
            recon_loss_norm = np.mean(recon_loss_norms)
            adv_loss_norm = np.mean(adv_loss_norms)
            grad_avg = (recon_loss_norm + adv_loss_norm) / 2.0

            factor_recon = grad_avg * r_recon ** beta / (recon_loss_norm + epsilon)
            factor_adv = grad_avg * r_adv ** beta / (adv_loss_norm + epsilon)
            factor_recon = tf.clip_by_value(factor_recon, 1e-2, 1e2)
            factor_adv = tf.clip_by_value(factor_adv,   1e-2, 1e2)
            lambda_recon.assign(lambda_recon * tf.cast(factor_recon, tf.float32))
            lambda_adv.assign(lambda_adv * tf.cast(factor_adv, tf.float32))
        
        # MOO: Normalise lambda values by sum of their totals
        total_lambda.assign(lambda_recon + lambda_adv)
        lambda_recon.assign(lambda_recon / total_lambda)
        lambda_adv.assign(lambda_adv / total_lambda)

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
        'reconstruction_losses': reconstruction_losses,
        'adversarial_losses': adversarial_losses,
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator
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
    # Load the dataset
    x_train = np.load('data/stretched_x_train.npy')
    train_model(args, x_train)

if __name__ == '__main__':
    main()

