import tensorflow as tf
import functools
from functools import partial
from itertools import product 
import numpy as np

def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = tf.keras.backend.zeros_like(y_pred[:, 0])
    y_pred_max = tf.keras.backend.max(y_pred, axis=1)
    y_pred_max = tf.keras.backend.reshape(y_pred_max, (tf.keras.backend.shape(y_pred)[0], 1))
    y_pred_max_mat = tf.keras.backend.cast(tf.keras.backend.equal(y_pred, y_pred_max), tf.keras.backend.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits= y_pred) * final_mask

w_array = np.ones((2,2))
w_array[0, 0] =1
w_array[1, 0] = 4
w_array[0, 1] =1.5
w_array[1, 1] =0.5

        
ncce = partial(w_categorical_crossentropy, weights=w_array)
ncce.__name__ ='ncce'