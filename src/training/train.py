import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from ..utils import *
from ..evaluation.evaluate import calculate_kl_divergence, save_interpretations, save_confusion_matrix
from ..models import Encoder, Decoder, mlp_classifier, Discriminator
from .loss_functions import *
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# STAGE 1: Train Autoencoder (To wait for the reconstruction losses to converge before training the AI4CellFate model)
def train_autoencoder(config, x_train, x_val=None, encoder=None, decoder=None, discriminator=None, output_dir="./results"):
    """Train the adversarial autoencoder with optional validation monitoring."""
    
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
    
    # Validation losses
    val_reconstruction_losses = []
    val_adversarial_losses = []

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
                recon_imgs = decoder(z_imgs, training=True)

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
        
        # Compute validation losses if validation data is provided
        if x_val is not None:
            val_batch_size = min(config['batch_size'], len(x_val))
            val_recon_losses_epoch = []
            val_adv_losses_epoch = []
            
            for val_batch_start in range(0, len(x_val), val_batch_size):
                val_batch_end = min(val_batch_start + val_batch_size, len(x_val))
                val_image_batch = x_val[val_batch_start:val_batch_end]
                
                # Forward pass for validation (no training)
                val_image_batch_expanded = tf.expand_dims(val_image_batch, axis=-1)
                val_z_imgs = encoder(val_image_batch_expanded, training=False)
                val_recon_imgs = decoder(val_z_imgs, training=False)
                
                # Validation reconstruction loss
                val_recon_loss = ms_ssim_loss(val_image_batch_expanded, val_recon_imgs)
                
                # Validation adversarial loss
                val_real_y = 0.9 * np.ones((val_batch_end - val_batch_start, 1))
                val_z_discriminator_out = discriminator(val_z_imgs, training=False)
                val_adv_loss = bce_loss(val_real_y, val_z_discriminator_out)
                
                val_recon_losses_epoch.append(lambda_recon * val_recon_loss)
                val_adv_losses_epoch.append(lambda_adv * val_adv_loss)
            
            # Store validation losses for the epoch
            val_reconstruction_losses.append(np.mean(val_recon_losses_epoch))
            val_adversarial_losses.append(np.mean(val_adv_losses_epoch))
        
        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)

        reconstruction_losses.append(avg_recon_loss)
        adversarial_losses.append(avg_adv_loss)

        # Print and save results at the end of each epoch
        if x_val is not None:
            print(f"Epoch {epoch + 1}/{config['epochs']}: "
                  f"Train - Recon: {avg_recon_loss:.4f}, Adv: {avg_adv_loss:.4f} | "
                  f"Val - Recon: {val_reconstruction_losses[-1]:.4f}, Adv: {val_adversarial_losses[-1]:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{config['epochs']}: "
                  f"Reconstruction loss: {avg_recon_loss:.4f}, "
                  f"Adversarial loss: {avg_adv_loss:.4f}, lambda recon: {lambda_recon:.4f}, lambda adv: {lambda_adv:.4f}")
    
    # Save loss plots with validation losses if available
    if x_val is not None:
        save_loss_plots_autoencoder(reconstruction_losses, adversarial_losses, 
                                   val_reconstruction_losses, val_adversarial_losses,
                                   output_dir=f"{output_dir}/loss_plots_stage1")
    else:
        save_loss_plots_autoencoder(reconstruction_losses, adversarial_losses, 
                                   output_dir=f"{output_dir}/loss_plots_stage1")

    return {
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'recon_loss': reconstruction_losses,
        'adv_loss': adversarial_losses,
        'val_recon_loss': val_reconstruction_losses if x_val is not None else None,
        'val_adv_loss': val_adversarial_losses if x_val is not None else None
    }


# STAGE 2: Train AI4CellFate: Autoencoder + Covariance + Contrastive (Engineered Latent Space)
def train_cellfate(config, encoder, decoder, discriminator, x_train, y_train, x_val, y_val, x_test, y_test, output_dir="./results"):
    """Train the AI4CellFate model with the autoencoder, covariance, and contrastive loss."""

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
    
    # Validation losses
    val_reconstruction_losses = []
    val_adversarial_losses = []
    val_cov_losses = []
    val_contra_losses = []
    val_total_losses = []

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
                # cov_loss = covariance_loss_new(z_imgs)
                z_std_loss, diag_cov_mean = covariance_loss(z_imgs)
                cov_loss = 0.5 * diag_cov_mean + 0.5 * z_std_loss 

                # Contrastive loss
                contra_loss = contrastive_loss(z_imgs, np.eye(2)[y_train[idx]], tau=0.5)

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
        
        # Compute validation losses
        val_batch_size = min(config['batch_size'], len(x_val))
        val_recon_losses_epoch = []
        val_adv_losses_epoch = []
        val_cov_losses_epoch = []
        val_contra_losses_epoch = []
        
        for val_batch_start in range(0, len(x_val), val_batch_size):
            val_batch_end = min(val_batch_start + val_batch_size, len(x_val))
            val_image_batch = x_val[val_batch_start:val_batch_end]
            val_labels_batch = y_val[val_batch_start:val_batch_end]
            
            # Forward pass for validation (no training)
            val_image_batch_expanded = tf.expand_dims(val_image_batch, axis=-1)
            val_z_imgs = encoder(val_image_batch, training=False)
            val_recon_imgs = decoder(val_z_imgs, training=False)
            
            # Validation reconstruction loss
            val_recon_loss = ms_ssim_loss(val_image_batch_expanded, val_recon_imgs)
            
            # Validation adversarial loss
            val_real_y = 0.9 * np.ones((val_batch_end - val_batch_start, 1))
            val_z_discriminator_out = discriminator(val_z_imgs, training=False)
            val_adv_loss = bce_loss(val_real_y, val_z_discriminator_out)
            
            # Validation covariance loss
            val_z_std_loss, val_diag_cov_mean = covariance_loss(val_z_imgs)
            val_cov_loss = 0.5 * val_diag_cov_mean + 0.5 * val_z_std_loss
            
            # Validation contrastive loss
            val_contra_loss = contrastive_loss(val_z_imgs, np.eye(2)[val_labels_batch], tau=0.5)
            
            val_recon_losses_epoch.append(lambda_recon * val_recon_loss)
            val_adv_losses_epoch.append(lambda_adv * val_adv_loss)
            val_cov_losses_epoch.append(lambda_cov * val_cov_loss)
            val_contra_losses_epoch.append(lambda_contra * val_contra_loss)
        
        # Store validation losses for the epoch
        val_reconstruction_losses.append(np.mean(val_recon_losses_epoch))
        val_adversarial_losses.append(np.mean(val_adv_losses_epoch))
        val_cov_losses.append(np.mean(val_cov_losses_epoch))
        val_contra_losses.append(np.mean(val_contra_losses_epoch))
        val_total_losses.append(val_reconstruction_losses[-1] + val_adversarial_losses[-1] + 
                               val_cov_losses[-1] + val_contra_losses[-1])
        
        # Compute centroids
        z_imgs_train = encoder.predict(x_train)
        z_imgs_val = encoder.predict(x_val)
        z_imgs_test = encoder.predict(x_test)
        centroid_class_0 = np.mean(z_imgs_train[y_train == 0], axis=0) 
        centroid_class_1 = np.mean(z_imgs_train[y_train == 1], axis=0)

        # Compute Euclidean distance between centroids
        distance = euclidean(centroid_class_0, centroid_class_1)
        kl_divergence = calculate_kl_divergence(z_imgs_train)
        print("kl_divergence[0]:", kl_divergence[0], "kl_divergence[1]:", kl_divergence[1])

        if kl_divergence[0] < 1 and kl_divergence[1] < 1: 
            print("Latent Space is Gaussian-distributed!")
            print("Eucledian distance:", distance)

            # Compute classification accuracy and use it as a stopping criterion
            try:
                # Clear any existing TensorFlow session state
                tf.keras.backend.clear_session()
                
                # Train the classifier with explicit memory cleanup
                classifier = mlp_classifier(latent_dim=config['latent_dim'])
                classifier.compile(
                    loss='sparse_categorical_crossentropy', 
                    optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']), 
                    metrics=['accuracy']
                )
                
                # Prepare data
                x_val_ = z_imgs_val
                y_val_ = y_val
                x_test_ = z_imgs_test
                y_test_ = y_test
                
                # Train with original batch size (restored for consistency)
                history = classifier.fit(
                    z_imgs_train, y_train, 
                    batch_size=config['batch_size'], 
                    epochs=50, 
                    validation_data=(x_val_, y_val_),
                    verbose=0  # Reduce output
                )

                y_pred = classifier.predict(x_test_, verbose=0)
                threshold = 0.5
                y_pred_classes = np.zeros_like(y_pred[:, 1])
                y_pred_classes[y_pred[:, 1] > threshold] = 1  

                # Calculate confusion matrix
                cm = confusion_matrix(y_test_, y_pred_classes)
                print("cm:", cm)
                class_sums = cm.sum(axis=1, keepdims=True)
                conf_matrix_normalized = cm / class_sums
                mean_diagonal = np.mean(np.diag(conf_matrix_normalized))
                precison = conf_matrix_normalized[0,0] / (conf_matrix_normalized[0,0] + conf_matrix_normalized[1,0])
                recall_class_1 = conf_matrix_normalized[1,1] / (conf_matrix_normalized[1,0] + conf_matrix_normalized[1,1])
                f1_score = 2 * (precison * recall_class_1) / (precison + recall_class_1)
                print(f"Mean diagonal: {mean_diagonal:.4f}, Precision: {precison:.4f}, Recall: {recall_class_1:.4f}, F1 score: {f1_score:.4f}")

                # Clean up classifier to prevent memory leaks
                del classifier
                tf.keras.backend.clear_session()
                
                if mean_diagonal > 0.65 and precison >= 0.7: # and distance > 0.9 
                    print("Classification accuracy is good! :)")
                    good_conditions_stop.append(epoch)

                    if epoch > 10: 
                        # Save confusion matrix
                        save_confusion_matrix(conf_matrix_normalized, output_dir, epoch)
                        print("kl_divergence[0]:", kl_divergence[0], "kl_divergence[1]:", kl_divergence[1])
                        break
                            
            except Exception as e:
                print(f"Classification failed at epoch {epoch}: {e}")
                print("Continuing training without classification check...")
                # Clean up in case of error
                tf.keras.backend.clear_session()
                continue

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
              f"Train - Recon: {avg_recon_loss:.4f}, Adv: {avg_adv_loss:.4f}, Contr: {avg_contra_loss:.4f}, Cov: {avg_cov_loss:.4f} | "
              f"Val - Recon: {val_reconstruction_losses[-1]:.4f}, Adv: {val_adversarial_losses[-1]:.4f}, Contr: {val_contra_losses[-1]:.4f}, Cov: {val_cov_losses[-1]:.4f}")
    
    if save_loss_plot:
        save_loss_plots_cov(reconstruction_losses, adversarial_losses, cov_losses, contra_losses, 
                           val_reconstruction_losses, val_adversarial_losses, val_cov_losses, val_contra_losses,
                           output_dir=f"{output_dir}/loss_plots_stage2")

    # Generate and save latent feature interpretations
    print("Generating latent feature interpretations...")
    z_train_final = encoder.predict(x_train, verbose=0)
    save_interpretations(decoder, z_train_final, output_dir=f"{output_dir}/interpretations")

    return {
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'recon_loss': reconstruction_losses,
        'adv_loss': adversarial_losses,
        'cov_loss': cov_losses,
        'contra_loss': contra_losses,
        'val_recon_loss': val_reconstruction_losses,
        'val_adv_loss': val_adversarial_losses,
        'val_cov_loss': val_cov_losses,
        'val_contra_loss': val_contra_losses,
        'val_total_loss': val_total_losses,
        'good_conditions_stop': good_conditions_stop
    }