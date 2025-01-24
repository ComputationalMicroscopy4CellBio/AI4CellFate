import argparse
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import numpy as np
from ..config import CONFIG
from ..models import Encoder, Decoder, mlp_classifier, Discriminator
from .loss_functions import *
from ..evaluation.evaluate import Evaluation

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

def train_lambdas_autoencoder(config, x_train, epochs=5):
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

    # Initial losses
    lambda_recon = 1
    lambda_adv = 1

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    for epoch in range(epochs):
        epoch_reconstruction_losses, epoch_adversarial_losses = [], []

        for n_batch in range(len(x_train) // config['batch_size']):
            idx = rng.integers(0, x_train.shape[0], config['batch_size'])
            image_batch = x_train[idx]

            with tf.GradientTape() as tape:
                # Forward pass through encoder and decoder
                z_imgs, z_score = encoder(image_batch, training=True)
                recon_imgs = decoder(z_imgs, training=True)#[:, :, :, 0]

                # Reconstruction loss
                recon_loss = ms_ssim_loss(tf.expand_dims(image_batch, axis=-1), recon_imgs)

                # Adversarial loss for discriminator
                z_discriminator_out = discriminator(z_imgs, training=True)
                adv_loss = bce_loss(real_y, z_discriminator_out)

                # Total autoencoder loss
                ae_loss = lambda_recon * recon_loss + lambda_adv * adv_loss
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

            with tf.GradientTape() as tape:
                z_discriminator_out = discriminator(z_imgs, training=True)
                rand_discriminator_out = discriminator(rand_vecs, training=True)

                discriminator_loss = 0.5 * bce_loss(real_y, rand_discriminator_out) + \
                                     0.5 * bce_loss(fake_y, z_discriminator_out)

            # Update discriminator weights
            disc_gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            # Track individual losses for adjustment
            epoch_reconstruction_losses.append(lambda_recon * recon_loss)
            epoch_adversarial_losses.append(lambda_adv * adv_loss)
        
        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)

        reconstruction_losses.append(avg_recon_loss)
        adversarial_losses.append(avg_adv_loss)

        # Print and save results at the end of each epoch
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, lambda recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}")

    return {
        'recon_loss': reconstruction_losses,
        'adv_loss': adversarial_losses,
    }

def train_lambdas_cov(config, encoder, decoder, discriminator, x_train, epochs=20):
    config = convert_namespace_to_dict(config)
    set_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])

    print(f"Training with batch size: {config['batch_size']}, epochs: {config['epochs']}, "
          f"learning rate: {config['learning_rate']}, seed: {config['seed']}, latent dim: {config['latent_dim']}")

    # Optimizers
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)

    # Initial losses
    lambda_recon = 1
    lambda_adv = 1
    lambda_cov = 1

    # Placeholder for storing losses
    reconstruction_losses = []
    adversarial_losses = []
    cov_losses = []
    total_loss = []

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    for epoch in range(epochs): 
        epoch_reconstruction_losses, epoch_adversarial_losses, epoch_cov_losses = [], [], []

        for n_batch in range(len(x_train) // config['batch_size']):
            idx = rng.integers(0, x_train.shape[0], config['batch_size'])
            image_batch = x_train[idx]

            with tf.GradientTape() as tape:
                # Forward pass through encoder and decoder
                z_imgs, z_score = encoder(image_batch, training=True)
                recon_imgs = decoder(z_imgs, training=True)

                # Reconstruction loss
                recon_loss = ms_ssim_loss(tf.expand_dims(image_batch, axis=-1), recon_imgs)

                # Adversarial loss for discriminator
                z_discriminator_out = discriminator(z_imgs, training=True)
                adv_loss = bce_loss(real_y, z_discriminator_out)

                # Covariance loss
                cov, z_std_loss, diag_cov_mean, off_diag_loss = cov_loss_terms(z_imgs)
                cov_loss = 0.5 * diag_cov_mean + 0.5 * z_std_loss

                # Total autoencoder loss
                ae_loss = lambda_recon * recon_loss + lambda_adv * adv_loss + lambda_cov * cov_loss
                total_loss.append(ae_loss)

            # Backpropagation for autoencoder
            trainable_variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(ae_loss, trainable_variables)
            ae_optimizer.apply_gradients(zip(gradients, trainable_variables))

            # Train the discriminator
            rand_vecs = tf.random.stateless_normal(
                shape=(config['batch_size'], config['latent_dim']),
                seed=(config['seed'], epoch + n_batch)
            )
            with tf.GradientTape() as tape:
                z_discriminator_out = discriminator(z_imgs, training=True)
                rand_discriminator_out = discriminator(rand_vecs, training=True)

                discriminator_loss = 0.5 * bce_loss(real_y, rand_discriminator_out) + \
                                     0.5 * bce_loss(fake_y, z_discriminator_out)

            # Update discriminator weights
            disc_gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            # Track individual losses for adjustment
            epoch_reconstruction_losses.append(lambda_recon * recon_loss)
            epoch_adversarial_losses.append(lambda_adv * adv_loss)
            epoch_cov_losses.append(lambda_cov * cov_loss)
        
        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)
        avg_cov_loss = np.mean(epoch_cov_losses)

        reconstruction_losses.append(avg_recon_loss)
        adversarial_losses.append(avg_adv_loss)
        cov_losses.append(avg_cov_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, "
              f"Covariance loss: {avg_cov_loss:.4f}, lamdba recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}, lambda cov: {lambda_cov:.4f}")

    return {
        'recon_loss': reconstruction_losses,
        'adv_loss': adversarial_losses,
        'cov_losses': cov_losses
    } 

def train_autoencoder_scaled(config, x_train, save_loss_plot=True, save_model_weights=True):
    # Set random seeds for reproducibility
    config = convert_namespace_to_dict(config)
    set_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])

    print(f"Training with batch size: {config['batch_size']}, epochs: {config['epochs']}, "
          f"learning rate: {config['learning_rate']}, seed: {config['seed']}, latent dim: {config['latent_dim']}")

    img_shape = (x_train.shape[1], x_train.shape[2], 1) # Assuming grayscale images

    # Optimizers
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)

    # Placeholder for storing losses
    reconstruction_losses = []
    adversarial_losses = []
    total_loss = []

    # Adjust the lambdas based on previous 5 epoch training
    losses = train_lambdas_autoencoder(config, x_train, epochs=5)
    lambda_recon = 1/losses['recon_loss'][-1]
    lambda_adv = 1/losses['adv_loss'][-1]
    total = lambda_recon + lambda_adv
    lambda_recon = lambda_recon / total
    lambda_adv = lambda_adv / total
    print(f"Initial lambda recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}")

    # Create model instances
    encoder = Encoder(img_shape=img_shape, latent_dim=config['latent_dim'], num_classes=2, gaussian_noise_std=config['GaussianNoise_std']).model
    decoder = Decoder(latent_dim=config['latent_dim'], img_shape=img_shape, gaussian_noise_std=config['GaussianNoise_std']).model
    discriminator = Discriminator(latent_dim=config['latent_dim']).model

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    for epoch in range(config['epochs']):
        epoch_reconstruction_losses, epoch_adversarial_losses = [], []

        for n_batch in range(len(x_train) // config['batch_size']):
            idx = rng.integers(0, x_train.shape[0], config['batch_size'])
            image_batch = x_train[idx]

            with tf.GradientTape() as tape:
                # Forward pass through encoder and decoder
                z_imgs, z_score = encoder(image_batch, training=True)
                recon_imgs = decoder(z_imgs, training=True)#[:, :, :, 0]

                # Reconstruction loss
                recon_loss = ms_ssim_loss(tf.expand_dims(image_batch, axis=-1), recon_imgs)

                # Adversarial loss for discriminator
                z_discriminator_out = discriminator(z_imgs, training=True)
                adv_loss = bce_loss(real_y, z_discriminator_out)

                # Total autoencoder loss
                ae_loss = lambda_recon * recon_loss + lambda_adv * adv_loss
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

            with tf.GradientTape() as tape:
                z_discriminator_out = discriminator(z_imgs, training=True)
                rand_discriminator_out = discriminator(rand_vecs, training=True)

                discriminator_loss = 0.5 * bce_loss(real_y, rand_discriminator_out) + \
                                     0.5 * bce_loss(fake_y, z_discriminator_out)

            # Update discriminator weights
            disc_gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            # Track individual losses for adjustment
            epoch_reconstruction_losses.append(lambda_recon * recon_loss)
            epoch_adversarial_losses.append(lambda_adv * adv_loss)
        
        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)

        # if epoch == 4:
            
        #     lambda_recon = (1/avg_recon_loss)
        #     lambda_adv = (1/avg_adv_loss)
        #     total = lambda_recon + lambda_adv
        #     lambda_recon = lambda_recon / total
        #     lambda_adv = lambda_adv / total

        reconstruction_losses.append(avg_recon_loss)
        adversarial_losses.append(avg_adv_loss)

        # Print and save results at the end of each epoch
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, lambda recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}")

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

def train_cov_scaled(config, encoder, decoder, discriminator, x_train, y_train, save_loss_plot=True, save_model_weights=True, save_every_epoch=True):
    config = convert_namespace_to_dict(config)
    set_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])

    print(f"Training with batch size: {config['batch_size']}, epochs: {config['epochs']}, "
          f"learning rate: {config['learning_rate']}, seed: {config['seed']}, latent dim: {config['latent_dim']}")

    # Optimizers
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)

    # Adjust the lambdas based on previous 5 epoch training
    previous_encoder = encoder
    previous_decoder = decoder
    previous_discriminator = discriminator
    losses = train_lambdas_cov(config, previous_encoder, previous_decoder, previous_discriminator, x_train, epochs=20)

    lambda_recon = 1/losses['recon_loss'][-1]
    lambda_adv = 1/losses['adv_loss'][-1]
    lambda_cov = 1/losses['cov_losses'][-1]
    total = lambda_recon + lambda_adv + lambda_cov
    lambda_recon = lambda_recon / total
    lambda_adv = lambda_adv / total
    lambda_cov = lambda_cov / total
    print(f"Initial lambda recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}, lambda cov: {lambda_cov:.4f}")

    # Placeholder for storing losses
    reconstruction_losses = []
    adversarial_losses = []
    cov_losses = []
    total_loss = []

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    # Initialize the evaluation class
    evaluation = Evaluation(output_dir="./results/evaluation")

    for epoch in range(config['epochs']): 
        epoch_reconstruction_losses, epoch_adversarial_losses, epoch_cov_losses = [], [], []

        for n_batch in range(len(x_train) // config['batch_size']):
            idx = rng.integers(0, x_train.shape[0], config['batch_size'])
            image_batch = x_train[idx]

            with tf.GradientTape() as tape:
                # Forward pass through encoder and decoder
                z_imgs, z_score = encoder(image_batch, training=True)
                recon_imgs = decoder(z_imgs, training=True)

                # Reconstruction loss
                recon_loss = ms_ssim_loss(tf.expand_dims(image_batch, axis=-1), recon_imgs)

                # Adversarial loss for discriminator
                z_discriminator_out = discriminator(z_imgs, training=True)
                adv_loss = bce_loss(real_y, z_discriminator_out)

                # Covariance loss
                cov, z_std_loss, diag_cov_mean, off_diag_loss = cov_loss_terms(z_imgs)
                cov_loss = 0.5 * diag_cov_mean + 0.5 * z_std_loss

                # Total autoencoder loss
                ae_loss = lambda_recon * recon_loss + lambda_adv * adv_loss + lambda_cov * cov_loss
                total_loss.append(ae_loss)

            # Backpropagation for autoencoder
            trainable_variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(ae_loss, trainable_variables)
            ae_optimizer.apply_gradients(zip(gradients, trainable_variables))

            # Train the discriminator
            rand_vecs = tf.random.stateless_normal(
                shape=(config['batch_size'], config['latent_dim']),
                seed=(config['seed'], epoch + n_batch)
            )
            with tf.GradientTape() as tape:
                z_discriminator_out = discriminator(z_imgs, training=True)
                rand_discriminator_out = discriminator(rand_vecs, training=True)

                discriminator_loss = 0.5 * bce_loss(real_y, rand_discriminator_out) + \
                                     0.5 * bce_loss(fake_y, z_discriminator_out)

            # Update discriminator weights
            disc_gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            # Track individual losses for adjustment
            epoch_reconstruction_losses.append(lambda_recon * recon_loss)
            epoch_adversarial_losses.append(lambda_adv * adv_loss)
            epoch_cov_losses.append(lambda_cov * cov_loss)
        
        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)
        avg_cov_loss = np.mean(epoch_cov_losses)

        reconstruction_losses.append(avg_recon_loss)
        adversarial_losses.append(avg_adv_loss)
        cov_losses.append(avg_cov_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, "
              f"Covariance loss: {avg_cov_loss:.4f}, lamdba recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}, lambda cov: {lambda_cov:.4f}")

        # Save visualizations every 10 epochs
        if save_every_epoch and (epoch + 1) % 10 == 0:
            epoch_dir = os.path.join(evaluation.output_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)

            all_z_imgs = encoder.predict(x_train)[0]

            # Save reconstruction images
            evaluation.reconstruction_images(image_batch, recon_imgs, epoch + 1, n=10)
            
            # Save covariance matrix
            evaluation.plot_cov_matrix(cov_loss_terms(all_z_imgs)[0], epoch + 1)
            
            # Save latent space visualization
            evaluation.visualize_latent_space(all_z_imgs, y_train, epoch + 1)

    # Save final loss plot
    if save_loss_plot:
        print("Saving loss plot...")
        save_loss_plots_cov(reconstruction_losses, adversarial_losses, cov_losses, config['epochs'])
    
    # Save model weights
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
    # x_train = np.random.rand(100, 20, 20)  # Random input data
    # y_train = np.random.randint(0, 2, 100)  # Random labels
    # x_test = np.random.rand(20, 20, 20)  # Random input data
    # y_test = np.random.randint(0, 2, 20)  # Random labels
    # train_cov(args, x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()

