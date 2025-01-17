import tensorflow as tf

def L2_norm(grad_list):
    """
    Compute the global L2 norm of a list of gradient tensors.
    """
    squares = [tf.reduce_sum(g**2) for g in grad_list if g is not None]
    if len(squares) == 0:
        return 0.0
    return tf.sqrt(tf.reduce_sum(squares))

def scale_gradients(grad_list, norm, lam=1, eps=1e-8):
    scaled = []
    for g in grad_list:
        if g is None:
            scaled.append(None)
        else:
            # Here: g is shape (same as variable shape),
            # norm is scalar, lam is scalar
            scaled.append(g * lam / (norm + eps))
    return scaled