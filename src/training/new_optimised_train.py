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

    # Initial losses
    lambda_recon = 5
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
        'lambda_recon': lambda_recon,
        'lambda_adv': lambda_adv
    }


def train_lambdas_cov(config, encoder, decoder, discriminator, x_train, y_train, lambda_recon=6, lambda_adv=4, epochs=20):
    config = convert_namespace_to_dict(config)
    set_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])

    print(f"Training with batch size: {config['batch_size']}, epochs: {config['epochs']}, "
          f"learning rate: {config['learning_rate']}, seed: {config['seed']}, latent dim: {config['latent_dim']}")

    # Optimizers
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)

    if encoder is None:
        img_shape = (x_train.shape[1], x_train.shape[2], 1) # Assuming grayscale images
        encoder = Encoder(img_shape=img_shape, latent_dim=config['latent_dim'], num_classes=2, gaussian_noise_std=config['GaussianNoise_std']).model
        decoder = Decoder(latent_dim=config['latent_dim'], img_shape=img_shape, gaussian_noise_std=config['GaussianNoise_std']).model
        discriminator = Discriminator(latent_dim=config['latent_dim']).model

    # Initial losses
    lambda_cov = 0
    lambda_contra = 8
    save_loss_plot = True

    # Placeholder for storing losses
    reconstruction_losses = []
    adversarial_losses = []
    cov_losses = []
    contra_losses = []
    total_loss = []

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    for epoch in range(epochs): 
        epoch_reconstruction_losses, epoch_adversarial_losses, epoch_cov_losses, epoch_contra_losses = [], [], [], []

        for n_batch in range(len(x_train) // config['batch_size']):
            idx = rng.integers(0, x_train.shape[0], config['batch_size'])
            image_batch = x_train[idx]

            with tf.GradientTape() as tape:
                # Forward pass through encoder and decoder
                z_imgs = encoder(image_batch, training=True)
                recon_imgs = decoder(z_imgs, training=True)

                # Reconstruction loss
                recon_loss = ms_ssim_loss(tf.expand_dims(image_batch, axis=-1), recon_imgs)

                # Adversarial loss for discriminator
                z_discriminator_out = discriminator(z_imgs, training=True)
                adv_loss = bce_loss(real_y, z_discriminator_out)

                # Covariance loss
                # cov, z_std_loss, diag_cov_mean, off_diag_loss = cov_loss_terms(z_imgs)
                # cov_loss = 0.5 * diag_cov_mean + 0.5 * z_std_loss #off_diag_loss
                cov_loss = unified_regularization_loss(z_imgs)[1]

                # Contrastive loss
                contra_loss = contrastive_loss(z_imgs, np.eye(2)[y_train[idx]], tau=0.5)
                #contra_loss = max_margin_contrastive_loss(z_imgs, y_train[idx])

                # Total autoencoder loss
                ae_loss = lambda_recon * recon_loss + lambda_adv * adv_loss + lambda_cov * cov_loss + lambda_contra * contra_loss
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
            epoch_contra_losses.append(lambda_contra * contra_loss)
        
        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)
        avg_cov_loss = np.mean(epoch_cov_losses)
        avg_contra_loss = np.mean(epoch_contra_losses)

        reconstruction_losses.append(avg_recon_loss)
        adversarial_losses.append(avg_adv_loss)
        cov_losses.append(avg_cov_loss)
        contra_losses.append(avg_contra_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, "
              f"Contrastive loss: {avg_contra_loss:.4f}, "
              f"Covariance loss: {avg_cov_loss:.4f}, lamdba recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}, lambda cov: {lambda_cov:.4f}, lambda contra: {lambda_contra:.4f}")
    
    if save_loss_plot:
        save_loss_plots_cov(reconstruction_losses, adversarial_losses, cov_losses, contra_losses, output_dir="./results/loss_plots/autoencoder_cov")

    return {
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'recon_loss': reconstruction_losses,
        'adv_loss': adversarial_losses,
        'cov_loss': cov_losses,
        'contra_loss': contra_losses
    }

def train_cov_scaled(config, x_train, y_train, reconstruction_losses=None, adversarial_losses=None, cov_losses=None, contra_losses=None, encoder=None, decoder=None, discriminator=None, save_loss_plot=True, save_model_weights=True, save_every_epoch=False):
    config = convert_namespace_to_dict(config)
    set_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])

    print(f"Training with batch size: {config['batch_size']}, epochs: {config['epochs']}, "
          f"learning rate: {config['learning_rate']}, seed: {config['seed']}, latent dim: {config['latent_dim']}")

    # Optimizers
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)

    # Placeholder for storing losses
    reconstruction_losses_total = []
    adversarial_losses_total = []
    cov_losses_total = []
    contra_losses_total = []

    reconstruction_losses_total.extend(reconstruction_losses)
    adversarial_losses_total.extend(adversarial_losses)
    cov_losses_total.extend(cov_losses)
    contra_losses_total.extend(contra_losses)

    lambda_recon = 1/reconstruction_losses[-1]
    lambda_adv = 1/adversarial_losses[-1]
    lambda_cov = 0
    lambda_contra = 1/contra_losses[-1]
    total = lambda_recon + lambda_adv + lambda_cov + lambda_contra
    lambda_recon = lambda_recon / total
    lambda_adv = lambda_adv / total
    lambda_cov = lambda_cov / total
    lambda_contra = lambda_contra / total

    print(f"Initial lambda recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}, lambda cov: {lambda_cov:.4f}, lambda contra: {lambda_contra:.4f}")

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    #initial_weights = encoder.get_weights()

    for epoch in range(config['epochs']): 
        epoch_reconstruction_losses, epoch_adversarial_losses, epoch_cov_losses, epoch_contra_losses = [], [], [], []

        # final_weights = encoder.get_weights()
        # assert all(np.array_equal(i, f) for i, f in zip(initial_weights, final_weights)), "Weights changed!"
    
        for n_batch in range(len(x_train) // config['batch_size']):
            idx = rng.integers(0, x_train.shape[0], config['batch_size'])
            image_batch = x_train[idx]

            with tf.GradientTape() as tape:
                # Forward pass through encoder and decoder
                z_imgs = encoder(image_batch, training=True)
                recon_imgs = decoder(z_imgs, training=True)

                # Reconstruction loss
                recon_loss = ms_ssim_loss(tf.expand_dims(image_batch, axis=-1), recon_imgs) 

                # Adversarial loss for discriminator
                z_discriminator_out = discriminator(z_imgs, training=True)
                adv_loss = bce_loss(real_y, z_discriminator_out)

                # Covariance loss
                cov, z_std_loss, diag_cov_mean, off_diag_loss = cov_loss_terms(z_imgs)
                cov_loss = off_diag_loss#0.5 * diag_cov_mean + 0.5 * z_std_loss

                # Contrastive loss
                contra_loss = contrastive_loss(z_imgs, np.eye(2)[y_train[idx]], tau=0.5)
                #contra_loss = 0
                # Total autoencoder loss
                ae_loss = lambda_recon * recon_loss + lambda_adv * adv_loss + 5 * lambda_cov * cov_loss + lambda_contra * contra_loss

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
            epoch_contra_losses.append(lambda_contra * contra_loss)
        
        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)
        avg_cov_loss = np.mean(epoch_cov_losses)
        avg_contra_loss = np.mean(epoch_contra_losses)

        reconstruction_losses_total.append(avg_recon_loss)
        adversarial_losses_total.append(avg_adv_loss)
        cov_losses_total.append(avg_cov_loss)
        contra_losses_total.append(avg_contra_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, "
              f"Contrastive loss: {avg_contra_loss:.4f}, "
              f"Covariance loss: {avg_cov_loss:.4f}, lamdba recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}, lambda cov: {lambda_cov:.4f}")

    # Save final loss plot
    if save_loss_plot:
        print("Saving loss plot...")
        save_loss_plots_cov(reconstruction_losses_total, adversarial_losses_total, cov_losses_total, contra_losses_total, output_dir="./results/loss_plots/autoencoder_cov")
    
    # Save model weights
    if save_model_weights:
        print("Saving model weights...")
        save_model_weights_to_disk(encoder, decoder, discriminator, output_dir="./results/models/autoencoder_cov")

    return {
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'reconstruction_losses': reconstruction_losses_total,
        'adversarial_losses': adversarial_losses_total,
        'cov_losses': cov_losses_total
    }


def train_lambdas_clf(config, encoder, decoder, discriminator, x_train, y_train, lambda_recon=1, lambda_adv=1, epochs=20):
    config = convert_namespace_to_dict(config)
    set_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])

    print(f"Training with batch size: {config['batch_size']}, epochs: {config['epochs']}, "
          f"learning rate: {config['learning_rate']}, seed: {config['seed']}, latent dim: {config['latent_dim']}")

    # Optimizers
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)

    # Create model instance
    classifier = mlp_classifier(latent_dim=config['latent_dim'])

    if encoder is None:
        img_shape = (x_train.shape[1], x_train.shape[2], 1) # Assuming grayscale images
        encoder = Encoder(img_shape=img_shape, latent_dim=config['latent_dim'], num_classes=2, gaussian_noise_std=config['GaussianNoise_std']).model
        decoder = Decoder(latent_dim=config['latent_dim'], img_shape=img_shape, gaussian_noise_std=config['GaussianNoise_std']).model
        discriminator = Discriminator(latent_dim=config['latent_dim']).model

    # Initial losses
    lambda_cov = 1
    lambda_clf = 1
    lambda_mi = 0.1

    # Placeholder for storing losses
    reconstruction_losses = []
    adversarial_losses = []
    cov_losses = []
    clf_losses = []

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    for epoch in range(epochs): 
        epoch_reconstruction_losses, epoch_adversarial_losses, epoch_cov_losses, epoch_clf_losses, epoch_mi_losses = [], [], [], [], []

        for n_batch in range(len(x_train) // config['batch_size']):
            idx = rng.integers(0, x_train.shape[0], config['batch_size'])
            image_batch = x_train[idx]

            with tf.GradientTape() as tape:
                # Forward pass through encoder and decoder
                z_imgs = encoder(image_batch, training=True)
                recon_imgs = decoder(z_imgs, training=True)

                # Reconstruction loss
                recon_loss = ms_ssim_loss(tf.expand_dims(image_batch, axis=-1), recon_imgs)

                # Adversarial loss for discriminator
                z_discriminator_out = discriminator(z_imgs, training=True)
                adv_loss = bce_loss(real_y, z_discriminator_out)

                # Mutual info loss
                #mi_loss = mutual_information_loss(z_imgs, np.eye(2)[y_train[idx]], classifier)
                mi_loss = 0
                # Covariance loss
                cov, z_std_loss, diag_cov_mean, off_diag_loss = cov_loss_terms(z_imgs)
                cov_loss = off_diag_loss #0.5 * diag_cov_mean + 0.5 * z_std_loss

                # Classification loss
                mlp_predictions = classifier(z_imgs, training=True)
                clf_loss = bce_loss(np.eye(2)[y_train[idx]], mlp_predictions)

                # Total autoencoder loss
                ae_loss = lambda_recon * recon_loss + lambda_adv * adv_loss + lambda_cov * cov_loss + lambda_clf * clf_loss + lambda_mi * mi_loss

            # Backpropagation for autoencoder
            trainable_variables = encoder.trainable_variables + decoder.trainable_variables + classifier.trainable_variables
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
            epoch_clf_losses.append(lambda_clf * clf_loss)
            epoch_mi_losses.append(lambda_mi * mi_loss)
        
        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)
        avg_cov_loss = np.mean(epoch_cov_losses)
        avg_clf_loss = np.mean(epoch_clf_losses)
        avg_mi_loss = np.mean(epoch_mi_losses)

        reconstruction_losses.append(avg_recon_loss)
        adversarial_losses.append(avg_adv_loss)
        cov_losses.append(avg_cov_loss)
        clf_losses.append(avg_clf_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, "
              f"Covariance loss: {avg_cov_loss:.4f}, "
              f"MI loss: {avg_mi_loss:.4f}, "
                f"Classification loss: {avg_clf_loss:.4f}, lamdba recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}, lambda cov: {lambda_cov:.4f}, lambda clf: {lambda_clf:.4f}")

    return {
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'recon_loss': reconstruction_losses,
        'adv_loss': adversarial_losses,
        'cov_loss': cov_losses,
        'clf_loss': clf_losses
    }

def train_clf_scaled(config, x_train, y_train, x_test, y_test, reconstruction_losses=None, adversarial_losses=None, cov_losses=None, clf_losses=None, encoder=None, decoder=None, discriminator=None, save_loss_plot=True, save_model_weights=True, save_every_epoch=False):
    config = convert_namespace_to_dict(config)
    set_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])

    print(f"Training with batch size: {config['batch_size']}, epochs: {config['epochs']}, "
          f"learning rate: {config['learning_rate']}, seed: {config['seed']}, latent dim: {config['latent_dim']}")

    # Optimizers
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)

    # Create model instance
    classifier = mlp_classifier(latent_dim=config['latent_dim'])

    # Placeholder for storing losses
    reconstruction_losses_total = []
    adversarial_losses_total = []
    cov_losses_total = []
    clf_losses_total = []
    validation_classification_losses = []

    reconstruction_losses_total.extend(reconstruction_losses)
    adversarial_losses_total.extend(adversarial_losses)
    cov_losses_total.extend(cov_losses)
    clf_losses_total.extend(clf_losses)

    lambda_recon = 1/reconstruction_losses[-1]
    lambda_adv = 1/adversarial_losses[-1]
    lambda_cov = 1/cov_losses[-1]
    lambda_clf = 1/clf_losses[-1]
    total = lambda_recon + lambda_adv + lambda_cov + lambda_clf
    lambda_recon = lambda_recon / total
    lambda_adv = lambda_adv / total
    lambda_cov = lambda_cov / total
    lambda_clf = lambda_clf / total

    print(f"Initial lambda recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}, lambda cov: {lambda_cov:.4f}, lambda clf: {lambda_clf:.4f}")

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    lambda_mi = 0.1

    #initial_weights = encoder.get_weights()

    for epoch in range(config['epochs']): 
        epoch_reconstruction_losses, epoch_adversarial_losses, epoch_cov_losses, epoch_clf_losses, epoch_mi_loss = [], [], [], [], []

        # final_weights = encoder.get_weights()
        # assert all(np.array_equal(i, f) for i, f in zip(initial_weights, final_weights)), "Weights changed!"
    
        for n_batch in range(len(x_train) // config['batch_size']):
            idx = rng.integers(0, x_train.shape[0], config['batch_size'])
            image_batch = x_train[idx]

            with tf.GradientTape() as tape:
                # Forward pass through encoder and decoder
                z_imgs = encoder(image_batch, training=True)
                recon_imgs = decoder(z_imgs, training=True)

                # Reconstruction loss
                recon_loss = ms_ssim_loss(tf.expand_dims(image_batch, axis=-1), recon_imgs) 

                # Mutual info loss
                #mi_loss = mutual_information_loss(z_imgs, np.eye(2)[y_train[idx]], classifier)
                mi_loss = 0
                # Adversarial loss for discriminator
                z_discriminator_out = discriminator(z_imgs, training=True)
                adv_loss = bce_loss(real_y, z_discriminator_out)

                # Covariance loss
                cov, z_std_loss, diag_cov_mean, off_diag_loss = cov_loss_terms(z_imgs)
                cov_loss = off_diag_loss#0.5 * diag_cov_mean + 0.5 * z_std_loss

                # Classification loss
                mlp_predictions = classifier(z_imgs, training=True)
                clf_loss = bce_loss(np.eye(2)[y_train[idx]], mlp_predictions)

                # Total autoencoder loss
                ae_loss = lambda_recon * recon_loss + lambda_adv * adv_loss + lambda_cov * cov_loss + lambda_clf * clf_loss + lambda_mi * mi_loss

            # Backpropagation for autoencoder
            trainable_variables = encoder.trainable_variables + decoder.trainable_variables + classifier.trainable_variables
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
            epoch_clf_losses.append(lambda_clf * clf_loss)
            epoch_mi_loss.append(lambda_mi * mi_loss)

        # Validation loss of the MLP classifier
        epoch_val_clf_loss = []
        for n_batch in range(len(x_test) // config['batch_size']):
            idx = np.random.randint(0, x_test.shape[0], config['batch_size'])
            test_image_batch = x_test[idx]

            # Use the encoder to get the latent space representation
            test_z_imgs = encoder(test_image_batch, training=False)

            # Get the predictions from the classifier
            mlp_predictions_val = classifier(test_z_imgs, training=False)
            validation_classification_loss = bce_loss(np.eye(2)[y_test[idx]], mlp_predictions_val) # One-hot encoding
            
            epoch_val_clf_loss.append(validation_classification_loss)
        
        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)
        avg_cov_loss = np.mean(epoch_cov_losses)
        avg_clf_loss = np.mean(epoch_clf_losses)
        avg_mi_loss = np.mean(epoch_mi_loss)
        avg_clf_val_loss = np.mean(epoch_val_clf_loss)

        reconstruction_losses_total.append(avg_recon_loss)
        adversarial_losses_total.append(avg_adv_loss)
        cov_losses_total.append(avg_cov_loss)
        clf_losses_total.append(avg_clf_loss)
        validation_classification_losses.append(avg_clf_val_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, "
              f"Covariance loss: {avg_cov_loss:.4f}, "
              f"MI loss: {avg_mi_loss:.4f}, "
                f"Classification loss: {avg_clf_loss:.4f}, lamdba recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}, lambda cov: {lambda_cov:.4f}, lambda clf: {lambda_clf:.4f}")

        print(f"Validation Classification Loss: {avg_clf_val_loss:.4f}")

    # Save final loss plot
    if save_loss_plot:
        print("Saving loss plot...")
        save_loss_plots_clf(reconstruction_losses_total, adversarial_losses_total, cov_losses_total, clf_losses_total, output_dir="./results/loss_plots/autoencoder_clf")
    
    # Save model weights
    if save_model_weights:
        print("Saving model weights...")
        save_full_model_weights_to_disk(encoder, decoder, discriminator, classifier, output_dir="./results/models/autoencoder_clf")

    return {
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'classifier': classifier,
        'reconstruction_losses': reconstruction_losses_total,
        'adversarial_losses': adversarial_losses_total,
        'cov_losses': cov_losses_total,
        'clf_losses': clf_losses_total
    }