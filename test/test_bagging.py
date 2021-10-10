from unittest import TestCase
from utils.bagging import generate_data
import numpy as np


class TestBagging(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBagging, self).__init__(*args, **kwargs)
        self.x = np.vstack((np.zeros((1000, 10)), np.ones((100, 10))))
        self.y = np.hstack((np.array(['a']*1000), np.array(['b']*100)))
        self.x_output = np.vstack((np.zeros((90, 10)), np.ones((90, 10))))
        self.y_output = np.hstack((np.array(['a']*90), np.array(['b']*90)))

    def test_fit(self):
        x, y = generate_data(self.x, self.y, methode='balanced', threshold=0.9)
        self.assertEqual(x.shape[0], self.x_output.shape[0])
        self.assertTrue((x == self.x_output).all())
        self.assertTrue((y == self.y_output).all())

        x, y = generate_data(self.x, self.y, methode=[0.9, 0.9], threshold=0.9)
        expected_x = np.vstack((np.zeros((900, 10)), np.ones((90, 10))))
        expected_y = np.hstack((np.array(['a'] * 900), np.array(['b'] * 90)))
        self.assertEqual(x.shape[0], expected_x.shape[0])
        self.assertTrue((x == expected_x).all())
        self.assertTrue((y == expected_y).all())
