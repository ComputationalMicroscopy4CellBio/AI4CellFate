import tensorflow as tf
import numpy as np

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

#### Contrastive loss ####

### Adapted from paper "Supervised Contrastive Learning" ###
def pdist_euclidean(A):
    # Euclidean pdist
    # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    r = tf.reduce_sum(A*A, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    return tf.sqrt(D)

def square_to_vec(D):
    '''Convert a squared form pdist matrix to vector form.
    '''
    n = D.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    d_vec = tf.gather_nd(D, list(zip(triu_idx[0], triu_idx[1])))
    return d_vec

def get_contrast_batch_labels(y):
    '''
    Make contrast labels by taking all the pairwise in y
    y: tensor with shape: (batch_size, )
    returns:   
        tensor with shape: (batch_size * (batch_size-1) // 2, )
    '''
    y_col_vec = tf.reshape(tf.cast(y, tf.float32), [-1, 1])
    D_y = pdist_euclidean(y_col_vec)
    d_y = square_to_vec(D_y)
    y_contrasts = tf.cast(d_y == 0, tf.int32)
    return y_contrasts

def tfa_contrastive_loss(y_true, y_pred, margin = 1.0) -> tf.Tensor:
    r"""Computes the contrastive loss between `y_true` and `y_pred`.

    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.

    The euclidean distances `y_pred` between two embedding matrices
    `a` and `b` with shape `[batch_size, hidden_size]` can be computed
    as follows:

    >>> a = tf.constant([[1, 2],
    ...                 [3, 4],
    ...                 [5, 6]], dtype=tf.float16)
    >>> b = tf.constant([[5, 9],
    ...                 [3, 6],
    ...                 [1, 8]], dtype=tf.float16)
    >>> y_pred = tf.linalg.norm(a - b, axis=1)
    >>> y_pred
    <tf.Tensor: shape=(3,), dtype=float16, numpy=array([8.06 , 2.   , 4.473],
    dtype=float16)>

    <... Note: constants a & b have been used purely for
    example purposes and have no significant value ...>

    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Args:
      y_true: 1-D integer `Tensor` with shape `[batch_size]` of
        binary labels indicating positive vs negative pair.
      y_pred: 1-D float `Tensor` with shape `[batch_size]` of
        distances between two embedding matrices.
      margin: margin term in the loss definition.

    Returns:
      contrastive_loss: 1-D float `Tensor` with shape `[batch_size]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    return y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(
        tf.math.maximum(margin - y_pred, 0.0)
    )

def max_margin_contrastive_loss(z, y, margin=1.0, metric='euclidean'):
    '''
    Wrapper for the maximum margin contrastive loss (Hadsell et al. 2006)
    `tfa.losses.contrastive_loss`
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
        metric: one of ('euclidean', 'cosine')
    '''
    # compute pair-wise distance matrix
    if metric == 'euclidean':
        D = pdist_euclidean(z)
    elif metric == 'cosine':
        D = 1 - tf.matmul(z, z, transpose_a=False, transpose_b=True)
    # convert squareform matrix to vector form
    d_vec = square_to_vec(D)
    # make contrastive labels
    y_contrasts = get_contrast_batch_labels(y)
    loss = tfa_contrastive_loss(y_contrasts, d_vec, margin=margin)
    # exploding/varnishing gradients on large batch?
    return tf.reduce_mean(loss)

def supervised_nt_xent_loss(z, y, temperature=0.5, base_temperature=0.07):
    '''
    Supervised normalized temperature-scaled cross entropy loss. 
    A variant of Multi-class N-pair Loss from (Sohn 2016)
    Later used in SimCLR (Chen et al. 2020, Khosla et al. 2020).
    Implementation modified from: 
        - https://github.com/google-research/simclr/blob/master/objective.py
        - https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
    '''
    batch_size = tf.shape(z)[0]
    contrast_count = 1
    anchor_count = contrast_count
    y = tf.expand_dims(y, -1)

    # mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
    #     has the same class as sample i. Can be asymmetric.
    mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32)
    anchor_dot_contrast = tf.divide(
        tf.matmul(z, tf.transpose(z)),
        temperature
    )
    # # for numerical stability
    logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
    logits = anchor_dot_contrast - logits_max
    # # tile mask
    logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - \
        tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))

    # compute mean of log-likelihood over positive
    # this may introduce NaNs due to zero division,
    # when a class only has one example in the batch
    mask_sum = tf.reduce_sum(mask, axis=1)
    mean_log_prob_pos = tf.reduce_sum(
        mask * log_prob, axis=1)[mask_sum > 0] / mask_sum[mask_sum > 0]

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    # loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))
    loss = tf.reduce_mean(loss)
    return loss


# According to the formula
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



##### Covariance loss #####
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