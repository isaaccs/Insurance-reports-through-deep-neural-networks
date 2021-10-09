from unittest import TestCase
from utils.model_utils import swish, w_categorical_crossentropy
import numpy as np
import tensorflow as tf


class TestModelUtils(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestModelUtils, self).__init__(*args, **kwargs)
        self.swish_output = np.array([-4.1223075e-08, -2.6894143e-01,
                                      0.0000000e+00,  7.3105860e-01, 2.0000000e+01], dtype=np.float32)

    def test_fit(self):
        a = tf.constant([-20, -1.0, 0.0, 1.0, 20], dtype=tf.float32)
        x = swish(a).numpy()
        self.assertTrue((x == self.swish_output).all())
        y_true = np.array([[1., 0.], [0., 1.]])
        y_pred = np.array([[1., 0.], [0., 1.]])
        a=w_categorical_crossentropy(y_true, y_pred)
        print(a)
        self.assertTrue(ncce(y_true, y_pred) == 0)
