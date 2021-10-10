import tensorflow as tf
from functools import partial
from itertools import product
import numpy as np


def swish(x):
    """
    Swish activation function, `swish(x) = x / (1 + exp(-x))`.
    Applies the Swish activation function.

    :param x: Input tensor.
    :return: Tensor with the sigmoid activation: `x / (1 + exp(-x))`.
    """
    return tf.keras.activations.sigmoid(x) * x


w_array = np.ones((2, 2))
w_array[0, 0] = 1
w_array[1, 0] = 4
w_array[0, 1] = 1.5
w_array[1, 1] = 0.5

def w_categorical_crossentropy(y_true, y_pred, weights=w_array):
    """
    :param y_true: 2d array-like, or label indicator array
        Ground truth (correct) labels
    :param y_pred: 2d array-like, or label indicator array
        Predicted labels, as returned by a classifier.

    :param weights: array-like of shape (n_labels,n_labels)
    :return: Weighted loss float
    """
    nb_cl = len(weights)
    final_mask = tf.keras.backend.zeros_like(y_pred[:, 0])
    y_pred_max = tf.keras.backend.max(y_pred, axis=1)
    y_pred_max = tf.keras.backend.reshape(y_pred_max, (tf.keras.backend.shape(y_pred)[0], 1))
    y_pred_max_mat = tf.keras.backend.cast(tf.keras.backend.equal(y_pred, y_pred_max), tf.keras.backend.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred) * final_mask
