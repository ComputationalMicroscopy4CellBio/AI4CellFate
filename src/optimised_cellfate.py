import numpy as np
from src.training.new_optimised_train import *
from src.evaluation.evaluate import Evaluation
from src.training.train import *
from src.utils import *

# Function to load data
def load_data():
    """Load training and testing data."""
    #x_train = np.load('./data/stretched_x_train.npy')  # TODO: replace with data loader later
    #x_test = np.load('./data/stretched_x_test.npy')
    x_train = np.load('./data/centered_x_train.npy')
    x_test = np.load('./data/centered_x_test.npy')
    y_train = np.load('./data/train_labels.npy')
    y_test = np.load('./data/test_labels.npy')
    return x_train, x_test, y_train, y_test

def evaluate_model(encoder, decoder, x_train, y_train, output_dir, classifier=None, x_test=None, y_test=None, full_evaluation=False):
    """Evaluate the trained model."""
    evaluator = Evaluation(output_dir)
    
    print("HERE")
    z_imgs = encoder.predict(x_train)
    recon_imgs = decoder.predict(z_imgs)
 
    evaluator.reconstruction_images(x_train, recon_imgs[:,:,:,0], epoch=0)
    
    # Visualize latent space
    evaluator.visualize_latent_space(z_imgs, y_train, epoch=0)

    # Covariance matrix
    cov_matrix = cov_loss_terms(z_imgs)[0]
    evaluator.plot_cov_matrix(cov_matrix, epoch=0)
    
    # KL divergence
    print("KL Divergences in each dimension: ", evaluator.calculate_kl_divergence(z_imgs))

    if full_evaluation: # try with only x_test (no x_val)
        # Predict labels and plot confusion matrix
        test_latent_space = encoder.predict(x_test)
        y_pred = classifier.predict(test_latent_space) # [:,:2]
        evaluator.plot_confusion_matrix(y_test, y_pred, num_classes=2)

# Main function
def main():
    """Main function with the full workflow of the CellFate project."""
    
    # Load data
    x_train, x_test, y_train, y_test = load_data()

    # Config for training
    config = {
        'batch_size': 30,
        'epochs': 10,
        'learning_rate': 0.001,
        'seed': 42,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
    }

   #Train the lambda optimisation autoencoder
    lambda_autoencoder_results = train_lambdas_autoencoder(config, x_train, epochs=15)
    encoder = lambda_autoencoder_results['encoder']
    decoder = lambda_autoencoder_results['decoder']
    discriminator = lambda_autoencoder_results['discriminator']
    reconstruction_losses = lambda_autoencoder_results['recon_loss']
    adversarial_losses = lambda_autoencoder_results['adv_loss']

    #Train the autoencoder starting from the optimal lambdas
    # scaled_autoencoder_results = train_autoencoder_scaled(config, x_train, reconstruction_losses, adversarial_losses, encoder, decoder, discriminator)
    # encoder = scaled_autoencoder_results['encoder']
    # decoder = scaled_autoencoder_results['decoder']
    # discriminator = scaled_autoencoder_results['discriminator']

    #Evaluate the autoencoder
    evaluate_model(encoder, decoder, x_train, y_train, output_dir="./results/optimisation/autoencoder", full_evaluation=False)
    #save_model_weights_to_disk(encoder, decoder, discriminator, output_dir="./results/models/autoencoder")

    # img_shape = (x_train.shape[1], x_train.shape[2], 1)
    # encoder = Encoder(img_shape=img_shape, latent_dim=config['latent_dim'], num_classes=2, gaussian_noise_std=config['GaussianNoise_std']).model
    # decoder = Decoder(latent_dim=config['latent_dim'], img_shape=img_shape, gaussian_noise_std=config['GaussianNoise_std']).model
    # discriminator = Discriminator(latent_dim=config['latent_dim']).model

    # encoder.load_weights("/Users/inescunha/Documents/GitHub/CellFate/results/models/autoencoder/encoder.weights.h5")
    # decoder.load_weights("/Users/inescunha/Documents/GitHub/CellFate/results/models/autoencoder/decoder.weights.h5")
    # discriminator.load_weights("/Users/inescunha/Documents/GitHub/CellFate/results/models/autoencoder/discriminator.weights.h5")

    config = {
        'batch_size': 30,
        'epochs': 30,
        'learning_rate': 0.001,
        'seed': 42,
        'latent_dim': 2,
        'GaussianNoise_std': 0.003,
    }
 
    #Train the lambda optimisation autoencoder + cov
    lambda_ae_cov_results = train_lambdas_cov(config, encoder, decoder, discriminator, x_train, y_train, x_test, y_test, epochs=150) #lambda_recon=scaled_autoencoder_results['lambda_recon'], lambda_adv=scaled_autoencoder_results['lambda_adv']
    encoder = lambda_ae_cov_results['encoder']
    decoder = lambda_ae_cov_results['decoder']
    discriminator = lambda_ae_cov_results['discriminator']
    reconstruction_losses = lambda_ae_cov_results['recon_loss']
    adversarial_losses = lambda_ae_cov_results['adv_loss']
    cov_losses = lambda_ae_cov_results['cov_loss']
    contra_losses = lambda_ae_cov_results['contra_loss']

    save_model_weights_to_disk(encoder, decoder, discriminator, output_dir="./results/models/autoencoder_cov")
    # Train the autoencoder starting from the optimal lambdas
    #scaled_ae_cov_results = train_cov_scaled(config, x_train, y_train, reconstruction_losses, adversarial_losses, cov_losses, contra_losses, encoder, decoder, discriminator)
    #scaled_ae_cov_results = train_cov(config, encoder, decoder, discriminator, x_train, y_train)
    # Evaluate the autoencoder + cov
    evaluate_model(lambda_ae_cov_results['encoder'], lambda_ae_cov_results['decoder'], x_train, y_train, output_dir="./results/optimisation/autoencoder_cov", full_evaluation=False)
    
    # config = {
    #     'batch_size': 30,
    #     'epochs': 20,
    #     'learning_rate': 0.001,
    #     'seed': 42,
    #     'lambda_recon': 0.4401, 
    #     'lambda_adv': 0.1290,
    #     'lambda_cov': 0.3269,
    #     'latent_dim': 10,
    #     'GaussianNoise_std': 0.003,   
    # }

    # Train the lambda optimisation autoencoder + cov + classifier
    # lambda_ae_clf_results = train_lambdas_clf(config, encoder, decoder, discriminator, x_train, y_train, lambda_recon=scaled_autoencoder_results['lambda_recon'], lambda_adv=scaled_autoencoder_results['lambda_adv'])
    # encoder = lambda_ae_clf_results['encoder']
    # decoder = lambda_ae_clf_results['decoder']
    # discriminator = lambda_ae_clf_results['discriminator']
    # reconstruction_losses = lambda_ae_clf_results['recon_loss']
    # adversarial_losses = lambda_ae_clf_results['adv_loss']
    # cov_losses = lambda_ae_clf_results['cov_loss']
    # clf_losses = lambda_ae_clf_results['clf_loss']

    # config = {
    #     'batch_size': 30,
    #     'epochs': 20,
    #     'learning_rate': 0.001,
    #     'seed': 42,
    #     'latent_dim': 10,
    #     'GaussianNoise_std': 0.003,
    # }

    # # Train the autoencoder + cov + clf starting from the optimal lambdas
    # scaled_ae_clf_results = train_clf_scaled(config, x_train, y_train, x_test, y_test, reconstruction_losses, adversarial_losses, cov_losses, clf_losses, encoder, decoder, discriminator)

    # final_encoder = scaled_ae_clf_results['encoder']
    # final_decoder = scaled_ae_clf_results['decoder']
    # final_discriminator = scaled_ae_clf_results['discriminator']
    # final_classifier = scaled_ae_clf_results['classifier']
                                                       
    # # Evaluate the autoencoder + cov + clf
    # evaluate_model(final_encoder, final_decoder, x_train, y_train, output_dir="./results/optimisation/autoencoder_clf", classifier=final_classifier, x_test=x_test, y_test=y_test, full_evaluation=True)
    


if __name__ == '__main__':
    main()