import tensorflow as tf

def swish(x):
    return (tf.keras.activations.sigmoid(x) * x)