import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

class Evaluation:
    def __init__(self, output_dir="./results/evaluation"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def reconstruction_images(self, image_batch, recon_imgs, n=10):
        """Visualize original and reconstructed images."""
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
        output_path = os.path.join(self.output_dir, "reconstruction_images.png")
        plt.savefig(output_path, dpi=300)
        plt.show()
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
                plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(output_path, dpi=300)
        plt.show()
        print(f"Confusion matrix saved to {output_path}")

    def plot_cov_matrix(self, cov_matrix):
        """Visualize and save the covariance matrix of latent variables."""
        normalized_cov_matrix = (cov_matrix - np.min(cov_matrix)) / (np.max(cov_matrix) - np.min(cov_matrix))
        plt.imshow(normalized_cov_matrix, cmap="viridis", vmin=0.1, vmax=1.0)
        plt.colorbar()
        plt.title("Normalized Covariance Matrix")
        output_path = os.path.join(self.output_dir, "cov_matrix.png")
        plt.savefig(output_path, dpi=300)
        plt.show()
        print(f"Covariance matrix saved to {output_path}")

    def visualize_latent_space(self, latent_space, y_train):
        """Visualize latent space features and their correlation with labels."""
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
        output_path = os.path.join(self.output_dir, "latent_space.png")
        plt.savefig(output_path, dpi=300)
        plt.show()
        print(f"Latent space visualization saved to {output_path}")
