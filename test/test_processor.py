from unittest import TestCase

import numpy as np

from utils.processor import PreprocessingTemplate, one_hot, identity


class TestPreprocessingTemplate(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPreprocessingTemplate, self).__init__(*args, **kwargs)
        self.logs = np.array(["e1", "e2", "e3", "e2", "e2", "e4", "e1", "e3", "e4"])
        self.config_dict = {"window_size": 2,
                            "min_n_gram": 1,
                            "max_n_gram": 2,
                            "encoding": "embedding",
                            "embedding_dim": 100,
                            "window_context": 2,
                            "min_word_count": 2,
                            "sample": 1e-3,
                            "sg": 0}
        self.feature_extraction_folder = ''

    def test_fit(self):
        feature_extractor = PreprocessingTemplate(self.config_dict, self.feature_extraction_folder)
        _, _, embedding = feature_extractor.fit_transform(self.logs)

        x, y = feature_extractor.create_xy(self.logs)

        expected_x = np.array([np.array(['e1', 'e2'], dtype='<U2'),
                               np.array(['e2', 'e3'], dtype='<U2'),
                               np.array(['e3', 'e2'], dtype='<U2'),
                               np.array(['e2', 'e2'], dtype='<U2'),
                               np.array(['e2', 'e4'], dtype='<U2'),
                               np.array(['e4', 'e1'], dtype='<U2'),
                               np.array(['e1', 'e3'], dtype='<U2')])
        expected_y = np.array(['e3', 'e2', 'e2', 'e4', 'e1', 'e3', 'e4'])

        self.assertTrue((x == expected_x).all())
        self.assertTrue((y == expected_y).all())

        x = feature_extractor.analyser(['e1', 'e2'])
        expected_x = np.array(['e1', 'e2', 'e1 e2'])
        self.assertTrue((x == expected_x).all())

        expected_vocabulary = ['e1', 'e2', 'e3', 'e4', 'e1 e2', 'e2 e3', 'e3 e2',
                               'e2 e2', 'e2 e4', 'e4 e1', 'e1 e3', 'e3 e4']
        expected_vocabulary.sort()
        vocabulary = list(feature_extractor.vocabulary.keys())
        vocabulary.sort()
        self.assertTrue(vocabulary == expected_vocabulary)

        expected_x = np.eye(3)
        x = one_hot(np.array([0, 1, 2]), 3)
        self.assertTrue((x == expected_x).all())

        self.assertTrue(embedding.shape[0] == len(expected_vocabulary))
        self.assertTrue(embedding.shape[1] == self.config_dict["embedding_dim"])
        a = feature_extractor.transform(['e3'])
