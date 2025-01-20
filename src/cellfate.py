import numpy as np
from src.training.optimised_train import train_model
from src.evaluation.evaluate import Evaluation

def evaluate_model(encoder, decoder, x_train, y_train, full_evaluation=False, output_dir="./results/evaluation"):

    """Evaluate the trained model."""
    evaluator = Evaluation(output_dir)
    z_imgs, _ = encoder.predict(x_train)
    recon_imgs = decoder.predict(z_imgs)
    #print(recon_imgs)
    evaluator.reconstruction_images(x_train, recon_imgs[:,:,:,0])
    evaluator.visualize_latent_space(z_imgs, y_train)


def main():
    """Main function with the full workflow of the CellFate project."""
    # Load data
    print("Loading data...")
    x_train = np.load("data/stretched_x_train.npy")
    y_train = np.load("data/train_labels.npy")
    # Config for training
    config = {
        'batch_size': 30,
        'epochs': 10,
        'learning_rate': 0.001,
        'seed': 69,
        'latent_dim': 10,
        'GaussianNoise_std': 0.003,
        'lambda_recon': 5,
        'lambda_adv': 0.05,
        'lambda_clf': 0.05,
        'lambda_cov': 0.1,
    }
    # Train the autoencoder model
    autoencoder_results = train_model(config, x_train)
    encoder = autoencoder_results['encoder']
    decoder = autoencoder_results['decoder']
    discriminator = autoencoder_results['discriminator']
    evaluate_model(encoder, decoder, x_train, y_train)

if __name__ == "__main__":
    main()