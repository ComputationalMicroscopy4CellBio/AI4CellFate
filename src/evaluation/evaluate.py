import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import norm, entropy, shapiro, kstest, jarque_bera, anderson
from scipy import stats
import seaborn as sns

def calculate_kl_divergence(latent_samples, num_bins=100):
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

def shapiro_wilk_test(latent_samples):
    """
    Perform Shapiro-Wilk test for normality on each latent dimension.
    
    Args:
        latent_samples (numpy.ndarray): Latent samples of shape (n_samples, latent_dim).
    
    Returns:
        dict: Dictionary with statistics and p-values for each dimension.
    """
    results = {}
    
    for dim in range(latent_samples.shape[1]):
        samples = latent_samples[:, dim]
        # Shapiro-Wilk works best for samples <= 5000
        if len(samples) > 5000:
            samples = np.random.choice(samples, 5000, replace=False)
        
        statistic, p_value = shapiro(samples)
        results[f'dim_{dim}'] = {
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05  # Standard significance level
        }
    
    return results

def plot_qq_plots(latent_samples, save_path=None):
    """
    Create Q-Q plots for each latent dimension against normal distribution.
    
    Args:
        latent_samples (numpy.ndarray): Latent samples of shape (n_samples, latent_dim).
        save_path (str, optional): Path to save the plot.
    """
    latent_dim = latent_samples.shape[1]
    
    # Create subplots
    fig, axes = plt.subplots(1, latent_dim, figsize=(5*latent_dim, 5))
    if latent_dim == 1:
        axes = [axes]
    
    for dim in range(latent_dim):
        samples = latent_samples[:, dim]
        
        # Create Q-Q plot
        stats.probplot(samples, dist="norm", plot=axes[dim])
        axes[dim].set_title(f'Q-Q Plot - Dimension {dim}')
        axes[dim].grid(True, alpha=0.3)
        
        # Add R-squared value
        _, (slope, intercept, r) = stats.probplot(samples, dist="norm")
        axes[dim].text(0.05, 0.95, f'R² = {r**2:.3f}', transform=axes[dim].transAxes, 
                      bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Q-Q plots saved to {save_path}")
    else:
        plt.show()

def save_confusion_matrix(conf_matrix_normalized, output_dir, epoch):
    """
    Save confusion matrix as both plot and numpy array.
    
    Args:
        conf_matrix_normalized (numpy.ndarray): Normalized confusion matrix
        output_dir (str): Directory to save files
        epoch (int): Current epoch number
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save confusion matrix plot (clean, no annotations)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Fate 0', 'Fate 1'])
    plt.yticks(tick_marks, ['Fate 0', 'Fate 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Save plot
    conf_matrix_path = os.path.join(output_dir, f"confusion_matrix_epoch_{epoch}.png")
    plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {conf_matrix_path}")
    
    # Save confusion matrix values as .npy
    conf_matrix_values_path = os.path.join(output_dir, f"confusion_matrix_values_epoch_{epoch}.npy")
    np.save(conf_matrix_values_path, conf_matrix_normalized)
    print(f"Confusion matrix values saved to {conf_matrix_values_path}")


def reconstruction_images(image_batch, recon_imgs, n=10):
        """Visualize and save original and reconstructed images for a specific epoch."""
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
        plt.show()

def plot_confusion_matrix(y_test, y_pred, num_classes):
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
    plt.show()

def plot_cov_matrix(cov_matrix):
    """Visualize and save the covariance matrix of latent variables for a specific epoch."""
    normalized_cov_matrix = (cov_matrix - np.min(cov_matrix)) / (np.max(cov_matrix) - np.min(cov_matrix))
    plt.imshow(normalized_cov_matrix, cmap="viridis", vmin=0.1, vmax=1.0)
    plt.colorbar()
    plt.title("Normalized Covariance Matrix")
    plt.show()

def visualize_latent_space(latent_space, y_train):
    """Visualize and save latent space features for a specific epoch."""
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

    plt.show()


class Evaluation:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_epoch_dir(self, epoch):
        """Create and return a subdirectory for a specific epoch."""
        # epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        # os.makedirs(epoch_dir, exist_ok=True)
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
        #scatter = plt.scatter(latent_space[:, feat_0], latent_space[:, feat_1], c=y_train, cmap='viridis', alpha=0.7)
        plt.figure(figsize=(8, 6), dpi=300)
        
        plt.scatter(latent_space[y_train == 0][:, feat_0], latent_space[y_train == 0][:, feat_1], 
            color='#648fff', label="Fate 0", alpha=1, edgecolors='k', linewidth=0.5, rasterized=True)  
        plt.scatter(latent_space[y_train == 1][:, feat_0], latent_space[y_train == 1][:, feat_1], 
            color='#dc267f', label="Fate 1", alpha=1, edgecolors='k', linewidth=0.5, rasterized=True)  

        plt.xlabel("Latent Feature 0 (z0)", fontsize=18, fontname="Arial")
        plt.ylabel("Latent Feature 1 (z1)", fontsize=18, fontname="Arial")
        plt.title("Latent Space", fontsize=20, fontname="Arial")

        # Legend and grid
        plt.legend(fontsize=14)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Save the plot
        output_path = os.path.join(epoch_dir, "latent_space.eps")
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        output_path_png = os.path.join(epoch_dir, "latent_space.png")
        plt.savefig(output_path_png, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Latent space visualization saved to {output_path}")

    def calculate_kl_divergence(self, latent_samples, num_bins=100):
        """Use the standalone calculate_kl_divergence function."""
        return calculate_kl_divergence(latent_samples, num_bins)

def save_interpretations(decoder, latent_space, output_dir, num_steps=7):
    """
    Generate and save synthetic cells by perturbing latent features for interpretability.
    
    Args:
        decoder: Trained decoder model (Keras)
        latent_space (np.ndarray): Latent representations from training data (num_samples, latent_dim)
        output_dir (str): Directory to save the interpretation plots
        num_steps (int): Number of perturbation steps for each latent dimension
    """
    print("Generating synthetic cells through latent feature perturbation...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get latent space statistics from training data
    latent_min = latent_space.min(axis=0)
    latent_max = latent_space.max(axis=0)
    latent_dim = latent_space.shape[1]
    
    # Create baseline latent vector (mean of training data)
    baseline_latent = latent_space.mean(axis=0)
    
    # Create figure
    fig, axes = plt.subplots(latent_dim, num_steps, figsize=(num_steps*2, latent_dim*2))
    
    # Handle case where latent_dim = 1 (axes would be 1D)
    if latent_dim == 1:
        axes = axes.reshape(1, -1)
    
    # Store all reconstructions to find global vmin/vmax
    all_reconstructions = []
    
    for dim in range(latent_dim):
        # Create perturbation values from min to max of this dimension
        perturbations = np.linspace(latent_min[dim], latent_max[dim], num_steps)
        
        for i, value in enumerate(perturbations):
            # Create perturbed latent vector
            perturbed_latent = baseline_latent.copy()
            perturbed_latent[dim] = value
            
            # Reshape for model input (add batch dimension)
            perturbed_latent_batch = perturbed_latent.reshape(1, -1)
            
            # Generate synthetic image
            synthetic_image = decoder.predict(perturbed_latent_batch, verbose=0)
            all_reconstructions.append(synthetic_image[0])  # Remove batch dimension
    
    # Find global min/max for consistent colorbar
    all_reconstructions = np.array(all_reconstructions)
    vmin = 0.25
    vmax = all_reconstructions.max()
    
    # Plot with consistent colorbar
    recon_idx = 0
    for dim in range(latent_dim):
        perturbations = np.linspace(latent_min[dim], latent_max[dim], num_steps)
        
        for i, value in enumerate(perturbations):
            # Get the reconstruction
            img_np = all_reconstructions[recon_idx]
            
            # Show image (squeeze to remove channel dimension if needed)
            if len(img_np.shape) == 3 and img_np.shape[2] == 1:
                img_np = img_np[:, :, 0]
            
            im = axes[dim, i].imshow(img_np, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[dim, i].set_title(f'{value:.2f}', fontsize=8)
            axes[dim, i].axis('off')
            
            # Add dimension label to first image of each row
            if i == 0:
                axes[dim, i].set_ylabel(f'Dim {dim}', rotation=90, fontsize=10)
            
            recon_idx += 1
    
    # Add a single colorbar for the entire figure
    fig.colorbar(im, ax=axes, shrink=0.6, aspect=30)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, "synthetic_cells_latent_traversal.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Synthetic cell traversals saved to '{output_path}'")

def evaluate_model(encoder, decoder, x_train, y_train, output_dir):
    """Evaluate the trained model."""
    evaluator = Evaluation(output_dir)
    
    z_imgs = encoder.predict(x_train)
    recon_imgs = decoder.predict(z_imgs)
    
    # Get reconstruction images
    evaluator.reconstruction_images(x_train, recon_imgs[:,:,:,0], epoch=0)

    # Visualize latent space
    evaluator.visualize_latent_space(z_imgs, y_train, epoch=0)

    # KL divergence
    print("KL Divergences in each dimension: ", evaluator.calculate_kl_divergence(z_imgs))
    
    # Generate latent feature interpretations
    save_interpretations(decoder, z_imgs, output_dir)
