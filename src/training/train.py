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
    return parser.parse_args()

def convert_namespace_to_dict(config): # TEMPORARY: Helper function to convert Namespace to dictionary
    if isinstance(config, argparse.Namespace):
        # Convert Namespace to a dictionary
        return {key: getattr(config, key) for key in vars(config)}
    return config # If it's already a dictionary, return as is

def train_model(config, x_train, save_loss_plot=True, save_model_weights=True):
    # Set random seeds for reproducibility
    config = convert_namespace_to_dict(config)
    print("hello")
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

    # Placeholder for storing losses
    reconstruction_losses = []
    adversarial_losses = []
    total_loss = []

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    for epoch in range(config['epochs']):
        epoch_reconstruction_losses, epoch_adversarial_losses = [], []

        for n_batch in range(len(x_train) // config['batch_size']):
            idx = np.random.randint(0, x_train.shape[0], config['batch_size'])
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
            rand_vecs = tf.random.normal(shape=(config['batch_size'], config['latent_dim']))

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

def train_cellfate(config, x_train, y_train, x_test, y_test, save_loss_plot=True, save_model_weights=True):
    """Train the full CellFate model - the autoencoder and the classifier (MLP), 
    with latent space disentanglement for interpretability."""

    config = convert_namespace_to_dict(config)
    print("hello")
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])

    print(f"Training with batch size: {config['batch_size']}, epochs: {config['epochs']}, "
          f"learning rate: {config['learning_rate']}, seed: {config['seed']}, latent dim: {config['latent_dim']}")

    img_shape = (x_train.shape[1], x_train.shape[2], 1) # Assuming grayscale images

    # Create model instances
    encoder = Encoder(img_shape=img_shape, latent_dim=config['latent_dim'], num_classes=2, gaussian_noise_std=config['GaussianNoise_std']).model
    decoder = Decoder(latent_dim=config['latent_dim'], img_shape=img_shape, gaussian_noise_std=config['GaussianNoise_std']).model
    discriminator = Discriminator(latent_dim=config['latent_dim']).model
    classifier = mlp_classifier(latent_dim=config['latent_dim'])

    # Optimizers
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=0.0, beta_2=0.9)

    # Placeholder for storing losses
    reconstruction_losses = []
    adversarial_losses = []
    classification_losses = []
    cov_losses = []
    total_loss = []
    validation_classification_losses = []

    real_y = 0.9 * np.ones((config['batch_size'], 1))
    fake_y = 0.1 * np.ones((config['batch_size'], 1))

    for epoch in range(config['epochs']): 
        epoch_reconstruction_losses, epoch_adversarial_losses, epoch_classification_losses, epoch_cov_losses  = [], [], [], []

        for n_batch in range(len(x_train) // config['batch_size']):
            idx = np.random.randint(0, x_train.shape[0], config['batch_size'])
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

                # Classification loss
                mlp_predictions = classifier(z_imgs, training=True)
                classification_loss = bce_loss(np.eye(2)[y_train[idx]], mlp_predictions) # One-hot encoding

                # Total autoencoder loss
                ae_loss = config['lambda_recon'] * recon_loss + config['lambda_adv'] * adv_loss + config['lambda_clf'] * classification_loss
                total_loss.append(ae_loss)

            # Backpropagation for autoencoder (encoder + decoder)
            trainable_variables = encoder.trainable_variables + decoder.trainable_variables + classifier.trainable_variables
            gradients = tape.gradient(ae_loss, trainable_variables)
            ae_optimizer.apply_gradients(zip(gradients, trainable_variables))

            # Train the discriminator
            rand_vecs = tf.random.normal(shape=(config['batch_size'], config['latent_dim']))

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
            epoch_classification_losses.append(config['lambda_clf'] * classification_loss)
        
        # Validation loss of the MLP classifier
        epoch_val_clf_loss = []
        for n_batch in range(len(x_test) // config['batch_size']):
            idx = np.random.randint(0, x_test.shape[0], config['batch_size'])
            test_image_batch = x_test[idx]

            # Use the encoder to get the latent space representation
            test_z_imgs, _ = encoder(test_image_batch, training=False)

            # Get the predictions from the classifier
            mlp_predictions_val = classifier(test_z_imgs, training=False)
            validation_classification_loss = bce_loss(np.eye(2)[y_test[idx]], mlp_predictions_val) # One-hot encoding
            
            epoch_val_clf_loss.append(validation_classification_loss)

        # Store average losses for the epoch
        avg_recon_loss = np.mean(epoch_reconstruction_losses)
        avg_adv_loss = np.mean(epoch_adversarial_losses)
        avg_clf_loss = np.mean(epoch_classification_losses)
        avg_clf_val_loss = np.mean(epoch_val_clf_loss)

        reconstruction_losses.append(avg_recon_loss)
        adversarial_losses.append(avg_adv_loss)
        classification_losses.append(avg_clf_loss)
        validation_classification_losses.append(avg_clf_val_loss)

        # Print and save results at the end of each epoch
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Reconstruction loss: {avg_recon_loss:.4f}, "
              f"Adversarial loss: {avg_adv_loss:.4f}, "
              f"Classification loss: {avg_clf_loss:.4f}")
        
        # Print validation loss
        print(f"Validation Classification Loss: {avg_clf_val_loss:.4f}")

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
        classifier_weights_path = os.path.join(output_dir, "classifier.weights.h5")

        encoder.save_weights(encoder_weights_path)
        decoder.save_weights(decoder_weights_path)
        discriminator.save_weights(discriminator_weights_path)
        classifier.save_weights(classifier_weights_path)

        print(f"Encoder weights saved to {encoder_weights_path}")
        print(f"Decoder weights saved to {decoder_weights_path}")
        print(f"Discriminator weights saved to {discriminator_weights_path}")
        print(f"Classifier weights saved to {classifier_weights_path}")

    return {
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'classifier': classifier,
        'reconstruction_losses': reconstruction_losses,
        'adversarial_losses': adversarial_losses,
        'classification_losses': classification_losses,
        'validation_classification_losses': validation_classification_losses,
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


def save_loss_plots_full(reconstruction_losses, adversarial_losses, classification_losses, validation_classification_losses, cov_losses, epochs):
    # Create directory for saving plots
    results_dir = './results/loss_plots'
    os.makedirs(results_dir, exist_ok=True)

    # Create the main training losses plot
    plt.figure(figsize=(10, 5))

    # Plot reconstruction and adversarial losses
    plt.plot(reconstruction_losses, label='Reconstruction Loss', color='blue', linestyle='-', linewidth=2)
    plt.plot(adversarial_losses, label='Adversarial Loss', color='red', linestyle='--', linewidth=2)
    plt.plot(classification_losses, label='Classification Loss', color='green', linestyle='-.', linewidth=2)
    plt.plot(cov_losses, label='Covariance Loss', color='purple', linestyle=':', linewidth=2)

    # Title and labels
    plt.title("Training Losses", fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # Add grid and legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=12)
    
    # Save the plot
    plt.savefig(f"{results_dir}/loss_plot.png", dpi=300)
    plt.close()  # Close the plot to avoid memory issues

    # Create validation classification loss plot
    plt.figure(figsize=(10, 5))

    # Plot validation classification loss
    plt.plot(validation_classification_losses, label='Validation Classification Loss', color='orange', linestyle='-', linewidth=2)

    # Title and labels
    plt.title("MLP Classification Validation", fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    # Add grid and legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=12)

    # Save the validation classification loss plot
    plt.savefig(f"{results_dir}/validation_loss_plot.png", dpi=300)
    plt.close()  # Close the plot to avoid memory issues


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
    train_cellfate(args, x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()

