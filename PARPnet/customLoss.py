import tensorflow as tf
import keras.backend as K

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def paprLoss(y_true, y_pred):
    yPower = K.sqrt(K.sum(K.square(y_pred), axis=1))
    yMax = K.max(yPower, axis=-1)
    yMean = K.mean(yPower, axis=-1)
    yPAPR = 10 * log10(yMax/yMean)
    return yPAPR
