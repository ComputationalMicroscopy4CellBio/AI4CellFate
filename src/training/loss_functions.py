import tensorflow as tf

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def bce_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred) # should I add label_smoothing=0.001?

def cce_loss(y_true, y_pred):
    return tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true, y_pred)

def mutual_information_loss(z, y_true, classifier):
    # Predict class probabilities from latent space
    y_pred = classifier(z, training=True)
    
    # Calculate log p(y|z)
    log_p_y_given_z = tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-8), axis=1)
    
    # Calculate q(y) - the empirical distribution
    q_y = tf.reduce_mean(y_true, axis=0)
    log_q_y = tf.math.log(q_y + 1e-8)
    
    # Extra to reshape
    log_q_y = tf.reduce_sum(y_true * log_q_y, axis=1)

    log_p_y_given_z = tf.cast(log_p_y_given_z, tf.float32)
    log_q_y = tf.cast(log_q_y, tf.float32)
    
    # Calculate the mutual information loss
    mi_loss = tf.reduce_mean(log_p_y_given_z - log_q_y)
    return -(mi_loss - 1.0) # Negative since we want to maximize MI - added constant to be positive

# def contrastive_loss(z, y_true, tau=0.5):
#     """Contrastive loss (NT-Xent) to enforce class separation in latent space."""
#     z = tf.math.l2_normalize(z, axis=1)  # Normalize latent vectors (prevents NaNs)
    
#     sim_matrix = tf.matmul(z, z, transpose_b=True)  # Compute cosine similarity
#     sim_matrix /= tau  # Temperature scaling

#     # Convert one-hot labels to class indices
#     y_true = tf.argmax(y_true, axis=1)  # Shape: (batch_size,)

#     # Compute the mask for positive pairs
#     mask = tf.cast(tf.equal(y_true[:, None], y_true[None, :]), dtype=tf.float32)  # Shape: (batch, batch)

#     # Compute positive similarities
#     sim_pos = tf.reduce_sum(mask * sim_matrix, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-8)  # Avoid division by zero
    
#     # Compute negative similarities (all pairs except self and positives)
#     exp_sim_matrix = tf.exp(sim_matrix)  # Exponentiate similarity scores
#     exp_sim_matrix = exp_sim_matrix * (1.0 - tf.eye(tf.shape(z)[0]))  # Remove self-comparisons
#     neg_sum = tf.reduce_sum(exp_sim_matrix, axis=1) + 1e-8  # Avoid division by zero
    
#     # Contrastive loss (NT-Xent)
#     loss = -tf.math.log(tf.exp(sim_pos) / neg_sum + 1e-8)  # Avoid log(0)
#     return tf.reduce_mean(loss)  # Average loss over batch

def contrastive_loss(z, y_true, tau=0.5):
    """NT-Xent contrastive loss to enforce class separation in latent space."""
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

# def contrastive_loss(z, y_true, tau=0.5):
#     """Contrastive loss (NT-Xent) to enforce class separation in latent space."""
#     z = tf.math.l2_normalize(z, axis=1)  # Normalize latent vectors
#     sim_matrix = tf.matmul(z, z, transpose_b=True)  # Cosine similarity (shape: [batch, batch])

#     # Convert one-hot to class labels
#     y_true = tf.argmax(y_true, axis=1)

#     # Create a mask where entries are 1 if same class, 0 otherwise
#     mask = tf.cast(tf.equal(y_true[:, None], y_true[None, :]), dtype=tf.float32)

#     # Sum of similarities for positive pairs
#     sim_pos = tf.reduce_sum(mask * sim_matrix, axis=1)  # Sum similarities within class

#     # Contrastive loss (negative log probability of same-class pairs)
#     loss = -tf.reduce_mean(tf.math.log(sim_pos + 1e-8) / tau)
    
#     return loss

##### COVARIANCE LOSS #####
def get_off_diag_values(x):
    x_flat = tf.reshape(x,[-1])[:-1]
    x_reshape = tf.reshape(x_flat,[x.shape[0]-1, x.shape[0]+1])[:, 1:]
    off_diag_values = tf.reshape(x_reshape,[-1])
    return off_diag_values

def cov_loss_terms(z_batch):
    z_batch = z_batch - tf.reduce_mean(z_batch, axis=0)
    z_std = tf.sqrt(tf.math.reduce_variance(z_batch, axis=0) + 0.0001)
    z_std_loss = tf.reduce_mean( tf.nn.relu(1 - z_std) )
    cov = (tf.transpose(z_batch) @ z_batch) / (z_batch.shape[0] ) 
    diag_cov = tf.linalg.diag_part(cov) 
    diag_cov_mean = tf.reduce_mean(tf.abs(diag_cov))    
    off_diag_loss = tf.reduce_mean(tf.abs(get_off_diag_values(cov)))
    return cov, z_std_loss, diag_cov_mean , off_diag_loss

def get_off_diag_values(x):
    x_flat = tf.reshape(x,[-1])[:-1]
    x_reshape = tf.reshape(x_flat,[x.shape[0]-1, x.shape[0]+1])[:, 1:]
    off_diag_values = tf.reshape(x_reshape,[-1])
    return off_diag_values
def unified_regularization_loss(z_batch):
    # Center the batch
    z_centered = z_batch - tf.reduce_mean(z_batch, axis=0)
    # Variance term
    std = tf.sqrt(tf.math.reduce_variance(z_batch, axis=0) + 0.0001)
    variance_loss = tf.reduce_mean(tf.nn.relu(1 - std))
    # Covariance terms
    cov = (tf.transpose(z_centered) @ z_centered) / (z_batch.shape[0] - 1)
    diag_cov = tf.linalg.diag_part(cov)
    diag_cov_mean = tf.reduce_mean(tf.abs(diag_cov))
    # Off-diagonal losses
    off_diag_loss = tf.reduce_mean(tf.abs(get_off_diag_values(cov)))
    off_diag_mean_10 = tf.reduce_mean(tf.abs(get_off_diag_values(cov[0:10,0:10])))
    
    return variance_loss, diag_cov_mean, off_diag_loss, off_diag_mean_10
    


def gaussian_filter(kernel_size=5, sigma=1.0):  # Use smaller kernel size
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
    g = tf.exp(-0.5 * (x / sigma)**2)
    g = g / tf.reduce_sum(g)
    kernel = tf.tensordot(g, g, axes=0)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
    return tf.cast(kernel, tf.float32)

def ssim_single_scale(img1, img2, k1=0.01, k2=0.03, filter_size=3, filter_sigma=1.0, L=1.0):
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
    MS-SSIM-based reconstruction loss.
    """
    return 1 - ms_ssim(y_true, y_pred)