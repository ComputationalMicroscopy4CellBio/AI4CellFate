import argparse
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import numpy as np
from ..config import CONFIG
from ..utils import *
from ..evaluation.evaluate import calculate_kl_divergence
from ..models import Encoder, Decoder, mlp_classifier, Discriminator
from .loss_functions import *
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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


def train_lambdas_cov(config, encoder, decoder, discriminator, x_train, y_train, x_test, y_test, lambda_recon=6, lambda_adv=4, epochs=20):
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
    lambda_cov = 0#0.0001
    lambda_contra = 0#8
    save_loss_plot = True
    good_conditions_stop = []

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
                cov, z_std_loss, diag_cov_mean, off_diag_loss = cov_loss_terms(z_imgs)
                cov_loss = 0.5 * diag_cov_mean + 0.5 * z_std_loss #off_diag_loss
                #cov_loss = 0.5 * unified_regularization_loss(z_imgs)[1]

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
        
        # Compute centroids
        z_imgs_train = encoder.predict(x_train)
        z_imgs_test = encoder.predict(x_test)
        centroid_class_0 = np.mean(z_imgs_train[y_train == 0], axis=0) #[y_test == 0]
        centroid_class_1 = np.mean(z_imgs_train[y_train == 1], axis=0)

        # Compute Euclidean distance between centroids
        distance = euclidean(centroid_class_0, centroid_class_1)
        kl_divergence = calculate_kl_divergence(z_imgs_train)

        if kl_divergence[0] < 0.2 and kl_divergence[1] < 0.2: 
            print("Latent Space is Gaussian-distributed!")
            print("Eucledian distance:", distance)

            # Compute CLASSIFICATION ACCURACY
            classifier = mlp_classifier(latent_dim=config['latent_dim']) #[:, [3, 8]] 

            # Train the classifier
            classifier.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(learning_rate=config['learning_rate']), metrics=['accuracy'])
            classifier.summary()

            x_val, x_test_, y_val, y_test_ = train_test_split(z_imgs_test, y_test, test_size=0.5, random_state=42) # 42 random state

            history = classifier.fit(z_imgs_train, y_train, batch_size=config['batch_size'], epochs=50, validation_data=(x_val, y_val)) # 

            num_classes = len(np.unique(y_train))
            y_pred = classifier.predict(x_test_)
            # y_pred_classes = np.argmax(y_pred, axis=1)

            threshold = 0.5
            y_pred_classes = np.zeros_like(y_pred[:, 1])  # Initialize as class 0
            y_pred_classes[y_pred[:, 1] > threshold] = 1  

            # Calculate confusion matrix
            cm = confusion_matrix(y_test_, y_pred_classes)

            class_sums = cm.sum(axis=1, keepdims=True)
            conf_matrix_normalized = cm / class_sums
            mean_diagonal = np.mean(np.diag(conf_matrix_normalized))
            precison = conf_matrix_normalized[0,0] / (conf_matrix_normalized[0,0] + conf_matrix_normalized[1,0])
            print(f"Mean diagonal: {mean_diagonal:.4f}, Precision: {precison:.4f}")

            if mean_diagonal > 0.65 and precison >= 0.7 and distance > 0.9:
                print("Classification accuracy is good! :)")
                good_conditions_stop.append(epoch)
                if epoch > 50: #epoch >= 25
                    break

        # if distance > 1.3:
        #     print("Classes are well separated! :)")
        #         if epoch >= 94:
        #             break
            
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
        'contra_loss': contra_losses,
        'good_conditions_stop': good_conditions_stop
    }

#### ADD DOCUMENTATION : FIRST TRAIN JUST AUTOENCODER, THEN TRAIN CELLFATE: AUTOENCODER + COVARIANCE + CONTRASTIVE ####
def train_autoencoder(config, x_train, encoder=None, decoder=None, discriminator=None):
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
    lambda_recon = config['lambda_recon'] # 5
    lambda_adv = config['lambda_adv'] # 1

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    for epoch in range(config['epochs']):
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
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, lambda recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}")
    
    save_loss_plots_autoencoder(reconstruction_losses, adversarial_losses, output_dir="./results/loss_plots/autoencoder")

    return {
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'recon_loss': reconstruction_losses,
        'adv_loss': adversarial_losses,
    }


def train_cellfate(config, encoder, decoder, discriminator, x_train, y_train, x_test, y_test):
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
    lambda_recon = config['lambda_recon'] # 6
    lambda_adv = config['lambda_adv'] # 4
    lambda_cov = config['lambda_cov'] #0.0001
    lambda_contra = config['lambda_contra'] #8
    save_loss_plot = True
    good_conditions_stop = []

    # Placeholder for storing losses
    reconstruction_losses = []
    adversarial_losses = []
    cov_losses = []
    contra_losses = []
    total_loss = []

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    for epoch in range(config['epochs']): 
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
                cov, z_std_loss, diag_cov_mean, off_diag_loss = cov_loss_terms(z_imgs)
                cov_loss = 0.5 * diag_cov_mean + 0.5 * z_std_loss #off_diag_loss
                #cov_loss = 0.5 * unified_regularization_loss(z_imgs)[1]

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
        
        # Compute centroids
        z_imgs_train = encoder.predict(x_train)
        z_imgs_test = encoder.predict(x_test)
        centroid_class_0 = np.mean(z_imgs_train[y_train == 0], axis=0) #[y_test == 0]
        centroid_class_1 = np.mean(z_imgs_train[y_train == 1], axis=0)

        # Compute Euclidean distance between centroids
        distance = euclidean(centroid_class_0, centroid_class_1)
        kl_divergence = calculate_kl_divergence(z_imgs_train)
        print("kl_divergence[0]:", kl_divergence[0], "kl_divergence[1]:", kl_divergence[1])
        if kl_divergence[0] < 0.2 and kl_divergence[1] < 0.2: 
            print("Latent Space is Gaussian-distributed!")
            print("Eucledian distance:", distance)

            # Compute CLASSIFICATION ACCURACY
            classifier = mlp_classifier(latent_dim=config['latent_dim']) #[:, [3, 8]] 

            # Train the classifier
            classifier.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(learning_rate=config['learning_rate']), metrics=['accuracy'])
            classifier.summary()

            x_val, x_test_, y_val, y_test_ = train_test_split(z_imgs_test, y_test, test_size=0.5, random_state=42) # 42 random state

            history = classifier.fit(z_imgs_train, y_train, batch_size=config['batch_size'], epochs=50, validation_data=(x_val, y_val)) # 

            num_classes = len(np.unique(y_train))
            y_pred = classifier.predict(x_test_)
            # y_pred_classes = np.argmax(y_pred, axis=1)

            threshold = 0.5
            y_pred_classes = np.zeros_like(y_pred[:, 1])  # Initialize as class 0
            y_pred_classes[y_pred[:, 1] > threshold] = 1  

            # Calculate confusion matrix
            cm = confusion_matrix(y_test_, y_pred_classes)

            class_sums = cm.sum(axis=1, keepdims=True)
            conf_matrix_normalized = cm / class_sums
            mean_diagonal = np.mean(np.diag(conf_matrix_normalized))
            precison = conf_matrix_normalized[0,0] / (conf_matrix_normalized[0,0] + conf_matrix_normalized[1,0])
            print(f"Mean diagonal: {mean_diagonal:.4f}, Precision: {precison:.4f}")

            if mean_diagonal > 0.65 and precison >= 0.7 and distance > 0.9:
                print("Classification accuracy is good! :)")
                good_conditions_stop.append(epoch)
                if epoch > 50: #epoch >= 25
                    break

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
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
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
        'contra_loss': contra_losses,
        'good_conditions_stop': good_conditions_stop
    }