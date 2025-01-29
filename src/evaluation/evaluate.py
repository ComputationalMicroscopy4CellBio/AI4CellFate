import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import norm, entropy

class Evaluation:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_epoch_dir(self, epoch):
        """Create and return a subdirectory for a specific epoch."""
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        return self.output_dir

    def reconstruction_images(self, image_batch, recon_imgs, epoch, n=10):
        """Visualize and save original and reconstructed images for a specific epoch."""
        epoch_dir = self._get_epoch_dir(epoch)
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(image_batch[i][:, :], cmap="gray", vmin=0.0, vmax=image_batch.max())
            plt.title("Original")
            plt.axis("off")
            plt.colorbar()

            # Display reconstruction
            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(recon_imgs[i][:, :], cmap="gray", vmin=0.0, vmax=image_batch.max())
            plt.title("Reconstructed")
            plt.axis("off")
            plt.colorbar()
        plt.tight_layout()
        output_path = os.path.join(epoch_dir, "reconstruction_images.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Reconstruction images saved to {output_path}")

    def plot_confusion_matrix(self, y_test, y_pred, num_classes):
        """Plot and save the confusion matrix."""
        cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
        class_sums = cm.sum(axis=1, keepdims=True)
        conf_matrix_normalized = cm / class_sums

        # Add actual values to the confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Annotate the matrix
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, f"{conf_matrix_normalized[i, j]}", ha="center", va="center", color="black")

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Confusion matrix saved to {output_path}")

    def plot_cov_matrix(self, cov_matrix, epoch):
        """Visualize and save the covariance matrix of latent variables for a specific epoch."""
        epoch_dir = self._get_epoch_dir(epoch)
        normalized_cov_matrix = (cov_matrix - np.min(cov_matrix)) / (np.max(cov_matrix) - np.min(cov_matrix))
        plt.imshow(normalized_cov_matrix, cmap="viridis", vmin=0.1, vmax=1.0)
        plt.colorbar()
        plt.title("Normalized Covariance Matrix")
        output_path = os.path.join(epoch_dir, "cov_matrix.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Covariance matrix saved to {output_path}")

    def visualize_latent_space(self, latent_space, y_train, epoch):
        """Visualize and save latent space features for a specific epoch."""
        epoch_dir = self._get_epoch_dir(epoch)
        cor_vals = [np.corrcoef(np.eye(2)[y_train][:, 0], latent_space[:, i])[0, 1] for i in range(latent_space.shape[1])]
        cor_vals = np.array(cor_vals)
        feat_0, feat_1 = np.argsort(np.abs(cor_vals))[-2:]  # Find top 2 correlated features

        print(f"Top correlated features: {feat_0}, {feat_1}")

        # Scatter plot
        scatter = plt.scatter(latent_space[:, feat_0], latent_space[:, feat_1], c=y_train, cmap='viridis', alpha=0.7)
        plt.xlabel(f"Latent Variable {feat_0}")
        plt.ylabel(f"Latent Variable {feat_1}")
        plt.title("Latent Space")
        plt.grid(True)

        # Add legend
        handles, _ = scatter.legend_elements()
        plt.legend(handles, ['Fate 0', 'Fate 1'], title="Classes", loc="lower right")

        # Save the plot
        output_path = os.path.join(epoch_dir, "latent_space.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Latent space visualization saved to {output_path}")

    def calculate_kl_divergence(self, latent_samples, num_bins=100):
        """
        Calculate the KL divergence between the empirical distribution of latent samples
        and a standard Gaussian distribution.

        Args:
            latent_samples (numpy.ndarray): Latent samples of shape (n_samples, latent_dim).
            num_bins (int): Number of bins to use for histogram estimation.

        Returns:
            list: KL divergence for each latent dimension.
        """
        kl_divergences = []

        for dim in range(latent_samples.shape[1]):
            # Get the samples for the current dimension
            samples = latent_samples[:, dim]

            # Compute histogram for the empirical distribution
            hist, bin_edges = np.histogram(samples, bins=num_bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Compute the Gaussian PDF for the bin centers
            gaussian_pdf = norm.pdf(bin_centers)

            # Normalize both distributions
            hist += 1e-10  # Avoid division by zero
            hist /= np.sum(hist)  # Normalize histogram to make it a valid probability distribution
            gaussian_pdf += 1e-10
            gaussian_pdf /= np.sum(gaussian_pdf)

            # Compute KL divergence
            kl_div = entropy(hist, gaussian_pdf)
            kl_divergences.append(kl_div)

        return kl_divergences
