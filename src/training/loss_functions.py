import tensorflow as tf

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def bce_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred) # should I add label_smoothing=0.001?

def cce_loss(y_true, y_pred):
    return tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true, y_pred)


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

