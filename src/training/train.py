import argparse
import tensorflow as tf
import numpy as np
from ..config import CONFIG
from ..models import Encoder, Decoder, mlp_classifier, Discriminator

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--batch_size', type=int, default=CONFIG.get('batch_size'))
    parser.add_argument('--epochs', type=int, default=CONFIG.get('epochs'))
    parser.add_argument('--learning_rate', type=float, default=CONFIG.get('learning_rate'))
    parser.add_argument('--seed', type=int, default=CONFIG.get('seed'))
    parser.add_argument('--latent_dim', type=int, default=CONFIG.get('latent_dim'))
    return parser.parse_args()

#TODO: add a different function for training the autoencoder only vs training the full model
#TODO: add validation losses to all training functions
def train_model(args): 
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print(f"Training with batch size: {args.batch_size}, epochs: {args.epochs}, "
          f"learning rate: {args.learning_rate}, seed: {args.seed}, latent dim: {args.latent_dim}")

    for epoch in range(args.epochs):
        # TODO: Implement training loop
        print(f"Epoch {epoch + 1}/{args.epochs} - Placeholder for training logic.")
        pass

# This allows running the script directly or importing it elsewhere
if __name__ == '__main__':
    args = parse_arguments()  # Parse arguments if running from command line
    train_model(args)  # Call the training function with parsed arguments
