import tensorflow as tf

def L2_norm(grad_list):
    """
    Compute the global L2 norm of a list of gradient tensors.
    """
    squares = [tf.reduce_sum(g**2) for g in grad_list if g is not None]
    if len(squares) == 0:
        return 0.0
    return tf.sqrt(tf.reduce_sum(squares))

def scale_gradients(grad_list, norm, lambd = 1, epsilon = 1e-8):
    """
    Scale a list of gradients by a given norm.
    """
    return [(g * lambd) / (norm + epsilon) for g in grad_list if g is not None]