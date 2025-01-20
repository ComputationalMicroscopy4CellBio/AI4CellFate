import tensorflow as tf

def L2_norm(grad_list):
    """
    Compute the global L2 norm of a list of gradients.
    """
    squares = [tf.reduce_sum(g**2) for g in grad_list if g is not None]
    if len(squares) == 0:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.sqrt(tf.add_n(squares))