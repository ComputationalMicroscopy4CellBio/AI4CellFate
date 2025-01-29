import argparse
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import numpy as np
from ..config import CONFIG
from ..utils import *
from ..models import Encoder, Decoder, mlp_classifier, Discriminator
from .loss_functions import *


def train_lambdas_autoencoder(config, x_train, encoder=None, decoder=None, discriminator=None, epochs=5):
    # Set random seeds for reproducibility
    config = convert_namespace_to_dict(config)
    set_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])

    print(f"Training with batch size: {config['batch_size']}, epochs: {config['epochs']}, "
          f"learning rate: {config['learning_rate']}, seed: {config['seed']}, latent dim: {config['latent_dim']}")
    
    img_shape = (x_train.shape[1], x_train.shape[2], 1) # Assuming grayscale images

    # Create model instances
    if encoder is None:
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
            idx = tf.convert_to_tensor(idx, dtype=tf.int32)
            #image_batch = x_train[idx]
            image_batch = tf.gather(x_train, idx)

            with tf.GradientTape() as tape:
                # Forward pass through encoder and decoder
                z_imgs = encoder(image_batch, training=True)
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
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'recon_loss': reconstruction_losses,
        'adv_loss': adversarial_losses,
    }


def train_autoencoder_scaled(config, x_train, reconstruction_losses=None, adversarial_losses=None, encoder=None, decoder=None, discriminator=None, save_loss_plot=True, save_model_weights=True):
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
    reconstruction_losses_total = []
    adversarial_losses_total = []
    total_loss = []

    reconstruction_losses_total.extend(reconstruction_losses)
    adversarial_losses_total.extend(adversarial_losses)

    lambda_recon = 1/reconstruction_losses[-1]
    lambda_adv = 1/adversarial_losses[-1]
    total = lambda_recon + lambda_adv
    lambda_recon = lambda_recon / total
    lambda_adv = lambda_adv / total
    print(f"Initial lambda recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}")

    # Create model instances
    if encoder is None:
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
                z_imgs = encoder(image_batch, training=True)
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

        reconstruction_losses_total.append(avg_recon_loss)
        adversarial_losses_total.append(avg_adv_loss)

        # Print and save results at the end of each epoch
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, lambda recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}")

    if save_loss_plot:
        print("Saving loss plot...")
        save_loss_plots_autoencoder(reconstruction_losses_total, adversarial_losses_total, output_dir="./results/loss_plots/autoencoder")
    
    if save_model_weights:
        print("Saving model weights...")
        save_model_weights_to_disk(encoder, decoder, discriminator, output_dir="./results/models/autoencoder")

    return {
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'reconstruction_losses': reconstruction_losses_total,
        'adversarial_losses': adversarial_losses_total,
    }

