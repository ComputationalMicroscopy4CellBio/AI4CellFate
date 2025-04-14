import tensorflow as tf
import numpy as np

def mse_loss(y_true, y_pred):
    """
    Mean squared error loss function.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Mean squared error between true and predicted values
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))

def bce_loss(y_true, y_pred):
    """
    Binary cross entropy loss function.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        
    Returns:
        Binary cross entropy loss between true and predicted values
    """
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred) # should I add label_smoothing=0.001?

def cce_loss(y_true, y_pred):
    """
    Categorical cross entropy loss function.
    
    Args:
        y_true: Ground truth one-hot encoded labels
        y_pred: Predicted class probabilities
        
    Returns:
        Categorical cross entropy loss between true and predicted values
    """
    return tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true, y_pred)

#### Contrastive loss ####

def contrastive_loss(z, y_true, tau=0.5):
    """
    Supervised contrastive loss function that pulls together embeddings of samples from same class
    while pushing apart embeddings from different classes.
    
    Args:
        z: Batch of embeddings
        y_true: Ground truth labels in one-hot format
        tau: Temperature scaling parameter
        
    Returns:
        Average contrastive loss over the batch
    """
    z = tf.math.l2_normalize(z, axis=1)  # Normalize latent vectors to unit norm
    
    sim_matrix = tf.matmul(z, z, transpose_b=True)  # Compute cosine similarity
    sim_matrix /= tau  # Apply temperature scaling

    # Convert one-hot labels to class indices
    y_true = tf.argmax(y_true, axis=1)  # Shape: (batch_size,)

    # Compute mask for positive pairs
    mask = tf.cast(tf.equal(y_true[:, None], y_true[None, :]), dtype=tf.float32)  # (batch, batch)

    # Exponentiate similarities (NT-Xent loss operates in log-space)
    exp_sim_matrix = tf.exp(sim_matrix)  # Apply exponentiation
    
    # Mask out self-comparisons (we don't want to compare a sample to itself)
    exp_sim_matrix *= 1.0 - tf.eye(tf.shape(z)[0])  # Remove diagonal entries

    # Compute denominators (sum over all similarities including positives)
    denom = tf.reduce_sum(exp_sim_matrix, axis=1, keepdims=True) + 1e-8  # Avoid div by zero

    # Compute positive pair similarities
    sim_pos = tf.reduce_sum(mask * exp_sim_matrix, axis=1, keepdims=True) + 1e-8  # Avoid div by zero

    # Contrastive loss (NT-Xent formulation)
    loss = -tf.math.log(sim_pos / denom)  # Compute log-ratio of positive to all similarities
    return tf.reduce_mean(loss)  # Average over batch



##### Covariance loss #####
def get_off_diag_values(x):
    """
    Extract off-diagonal values from a square matrix.
    
    Args:
        x: Square matrix tensor
        
    Returns:
        Flattened tensor of off-diagonal values
    """
    x_flat = tf.reshape(x,[-1])[:-1]
    x_reshape = tf.reshape(x_flat,[x.shape[0]-1, x.shape[0]+1])[:, 1:]
    off_diag_values = tf.reshape(x_reshape,[-1])
    return off_diag_values

def covariance_loss(z_batch):
    """
    Compute covariance-based regularization losses to encourage disentangled representations.
    
    Args:
        z_batch: Batch of latent vectors
        
    Returns:
        Tuple containing:
        - Covariance matrix
        - Standard deviation loss encouraging unit variance
        - Mean of diagonal covariance terms
        - Mean of off-diagonal covariance terms
    """
    z_batch = z_batch - tf.reduce_mean(z_batch, axis=0)
    z_std = tf.sqrt(tf.math.reduce_variance(z_batch, axis=0) + 0.0001)
    z_std_loss = tf.reduce_mean( tf.nn.relu(1 - z_std) )
    cov = (tf.transpose(z_batch) @ z_batch) / (z_batch.shape[0] ) 
    diag_cov = tf.linalg.diag_part(cov) 
    diag_cov_mean = tf.reduce_mean(tf.abs(diag_cov))    
    off_diag_loss = tf.reduce_mean(tf.abs(get_off_diag_values(cov)))
    return cov, z_std_loss, diag_cov_mean , off_diag_loss
    

##### SSIM loss #####
def gaussian_filter(kernel_size=5, sigma=1.0):
    """
    Create a 2D Gaussian filter kernel.
    
    Args:
        kernel_size: Size of the kernel
        sigma: Standard deviation of Gaussian
        
    Returns:
        2D Gaussian filter kernel as a tensor
    """
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
    g = tf.exp(-0.5 * (x / sigma)**2)
    g = g / tf.reduce_sum(g)
    kernel = tf.tensordot(g, g, axes=0)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
    return tf.cast(kernel, tf.float32)

def ssim_single_scale(img1, img2, k1=0.01, k2=0.03, filter_size=3, filter_sigma=1.0, L=1.0):
    """
    Compute Structural Similarity Index (SSIM) between two images at a single scale.
    
    Args:
        img1, img2: Input images to compare
        k1, k2: Constants for numerical stability
        filter_size: Size of Gaussian filter
        filter_sigma: Standard deviation of Gaussian filter
        L: Dynamic range of pixel values
        
    Returns:
        SSIM score between 0 and 1
    """
    kernel = gaussian_filter(kernel_size=filter_size, sigma=filter_sigma)
    c1 = tf.constant((k1 * L)**2, dtype=tf.float32)
    c2 = tf.constant((k2 * L)**2, dtype=tf.float32)

    # Mean values
    mu1 = tf.nn.conv2d(img1, kernel, strides=[1, 1, 1, 1], padding="SAME")
    mu2 = tf.nn.conv2d(img2, kernel, strides=[1, 1, 1, 1], padding="SAME")

    # Variances and covariances
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1**2, kernel, strides=[1, 1, 1, 1], padding="SAME") - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2**2, kernel, strides=[1, 1, 1, 1], padding="SAME") - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, kernel, strides=[1, 1, 1, 1], padding="SAME") - mu1_mu2

    # SSIM calculation
    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return tf.clip_by_value(ssim, 0, 1)


def ms_ssim(img1, img2, power_factors=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], filter_size=3, filter_sigma=1.0, L=1.0):
    """
    Multi-Scale Structural Similarity Index (MS-SSIM) between two images.
    
    Args:
        img1, img2: Input images to compare
        power_factors: Weights for different scales
        filter_size: Size of Gaussian filter
        filter_sigma: Standard deviation of Gaussian filter
        L: Dynamic range of pixel values
        
    Returns:
        MS-SSIM score between 0 and 1
    """
    img1 = tf.cast(img1, tf.float32)  # Ensure float32
    img2 = tf.cast(img2, tf.float32)  # Ensure float32

    msssim = []
    weights = tf.constant(power_factors, dtype=tf.float32)

    for weight in weights[:-1]:  # Loop over all but the last scale
        ssim_map = ssim_single_scale(img1, img2, filter_size=filter_size, filter_sigma=filter_sigma, L=L)
        msssim.append(weight * tf.reduce_mean(ssim_map))

        # Downsample the images
        img1 = tf.nn.avg_pool2d(img1, ksize=2, strides=2, padding="VALID")
        img2 = tf.nn.avg_pool2d(img2, ksize=2, strides=2, padding="VALID")

    # Compute SSIM for the final scale
    final_ssim = ssim_single_scale(img1, img2, filter_size=filter_size, filter_sigma=filter_sigma, L=L)
    msssim.append(weights[-1] * tf.reduce_mean(final_ssim))

    return tf.reduce_sum(msssim)


def ms_ssim_loss(y_true, y_pred):
    """
    Multi-Scale SSIM-based reconstruction loss.
    
    Args:
        y_true: Ground truth images
        y_pred: Predicted/reconstructed images
        
    Returns:
        1 - MS-SSIM score, to convert similarity metric to a loss
    """
    return 1 - ms_ssim(y_true, y_pred)