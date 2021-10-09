from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from six.moves import range
import six
from gensim.models.fasttext import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.general_utils import load_object, save_object
import tensorflow as tf


class PreprocessingTemplate:
    def __init__(self, config_dict, feature_extraction_folder):
        self.config_dict = config_dict
        self.feature_extraction_folder = feature_extraction_folder
        self.embedding_save_path = os.path.join(self.feature_extraction_folder, 'embedding.csv')
        self.analyser = None
        self.vocabulary = None
        self.embedding_matrix = None
        self.tokenizer = None
        self.load_saved()
        self.maxlen=10

    def load_saved(self):
        try:
            self.analyser = load_object(os.path.join(self.feature_extraction_folder, "analyzer.pickle"))
        except FileNotFoundError as e:
            pass

        try:
            self.vocabulary = load_object(os.path.join(self.feature_extraction_folder, "vocabulary.pickle"))
        except FileNotFoundError as e:
            pass

        try:
            self.embedding_matrix = np.genfromtxt(self.embedding_save_path, delimiter=',')
        except OSError as e:
            pass

        try:
            self.maxlen = load_object(os.path.join(self.feature_extraction_folder, "maxlen.pickle"))
        except FileNotFoundError as e:
            pass

    def feature_engineering(self, templates):
        """
        Apply a function to handle preprocessing, tokenization and n-grams generation.
        :param templates: list of sentence
        :return: x preprocess
        """

        x = [self.analyser(seq) for seq in templates]
        return x

    def instantiate_embedding_matrix(self, templates):
        """
        Create and fit the Embedding matrix based on the Fasttext algorithms
        :param templates: list of template
        :return: embedding matrix
        """

        fasttext_model = FastText(templates, min_n=0, max_n=4
                                  , size=self.config_dict["embedding_dim"], window=self.config_dict["window_context"],
                                  min_count=10, sample=self.config_dict["sample"],
                                  sg=self.config_dict["sg"],
                                  iter=100)
        fasttext_save_path = os.path.join(self.feature_extraction_folder, "ft_model.model")
        fasttext_model.save(fasttext_save_path)

        word_index = self.vocabulary
        vocabulary_size = len(word_index)
        embedding_matrix = np.zeros((vocabulary_size, self.config_dict["embedding_dim"]))
        words_not_found = []
        for word, i in word_index.items():
            if i >= vocabulary_size:
                continue
            embedding_vector = fasttext_model.wv.get_vector(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                words_not_found.append(word)

        np.savetxt(self.embedding_save_path, embedding_matrix, delimiter=',')

        return embedding_matrix

    def fit(self, templates):
        if self.vocabulary and self.analyser:
            pass
        else:
            vectorizer = TfidfVectorizer(
                ngram_range=(self.config_dict["min_n_gram"], self.config_dict["max_n_gram"]), lowercase=True,
                stop_words=None, min_df=1)
            vectorizer.fit(templates)
            self.analyser = vectorizer.build_analyzer()
            self.vocabulary = vectorizer.vocabulary_
            save_object(os.path.join(self.feature_extraction_folder, "analyzer.pickle"), self.analyser)
            save_object(os.path.join(self.feature_extraction_folder, "vocabulary.pickle"), self.vocabulary)
            inputs = self.feature_engineering(templates)
            self.maxlen = max(max(len(x) for x in inputs), self.maxlen)

    def transform(self, templates):
        vocab_size = len(self.vocabulary)
        inputs = self.feature_engineering(templates)
        #inputs = [[self.vocabulary[x] for x in l] for l in inputs]
        inputs = [[self.vocabulary[x] for x in l if x in self.vocabulary] for l in inputs]
        inputs = pad_sequences(inputs, maxlen=self.maxlen)

        if self.config_dict["encoding"] == "one_hot":
            inputs = [one_hot(np.array(x), vocab_size) for x in inputs]

        elif self.config_dict["encoding"] == "normalize":
            inputs = [np.array(x) / vocab_size for x in inputs]

        return np.array(inputs)

    def fit_transform(self, templates):
        if self.analyser is None:
            self.fit(templates)
        x = self.transform(templates)
        save_object(os.path.join(self.feature_extraction_folder, "maxlen.pickle"), self.maxlen)
        if self.config_dict["encoding"] == "embedding":
            if self.embedding_matrix is None:
                self.embedding_matrix = self.instantiate_embedding_matrix(templates)
            return x, self.embedding_matrix
        else:
            return x


def one_hot(x, vocabulary_size):
    """
    :param x: vector of label
    :type x: numpy ndarray
    :param vocabulary_size: size of vocabulary
    :type vocabulary_size: int
    :return: One hot encoded matrix of size (x.shape[0], vocabulary_size)
    :rtype: numpy array
    """

    b = np.zeros((x.size, vocabulary_size))
    b[np.arange(x.size), x] = 1
    return b


class OneHotEncoding:
    def __init__(self, feature_extraction_folder):
        self.label_to_int = {}
        self.feature_extraction_folder = feature_extraction_folder
        self.load_saved()

    def load_saved(self):
        try:
            self.label_to_int = load_object(os.path.join(self.feature_extraction_folder, "label_to_int.pickle"))
        except FileNotFoundError as e:
            pass

    def fit(self, label):
        label_to_int = {}
        for j in range(len(label)):
            label_to_int[label[j]] = j
        self.label_to_int = label_to_int
        save_object(os.path.join(self.feature_extraction_folder, "label_to_int.pickle"), self.label_to_int)

    def transform(self, y):
        y = np.vectorize(self.label_to_int.get)(y)
        b = np.zeros((y.size, len(self.label_to_int)))
        b[np.arange(y.size), y] = 1
        return b

    def fit_transform(self,y,label):
        self.fit(self,label)
        return self.transform(self,y)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input)-n+1):
        output.append(input[i:i+n])
    return output


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the beginning or the end
    if padding='post.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
