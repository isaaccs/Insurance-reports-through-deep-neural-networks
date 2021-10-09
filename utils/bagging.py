import numpy as np
import random
np.random.seed(7)


def randomly(percent):
    """
    :param percent: list of percentage to draw for each class
    :return: new list of percentage to draw for each class
    """
    new_percent = []
    for i in range(len(percent)):
        value = random.uniform(percent[i]-0.05, percent[i]+0.05)
        new_percent.append(value)
    return new_percent


def generate_data(x, y, methode, threshold):
    """

    :param x: The input samples, array-like of shape (n_samples, max_len)
    :param y: The target values, ndarray of shape (n_samples)
    :param methode: methode of balanced methode, str or list
    :param threshold: percent of desired observations in the under-represented class
    :return: x_new ndarray array of shape (new_samples_size, max_len), y_new  ndarray array of shape (new_samples_size)
    """

    if methode == "classical":
        return x, y
    label, indexes, counts_elements = np.unique(y, return_counts=True, return_index=True)

    if type(methode) is list:
        percent = methode
    else:
        min_p = np.min(counts_elements)
        percent = [threshold * min_p / x for x in counts_elements]
        if methode == "randomly":
            percent = randomly(percent)

    sample_size = [a * b for a, b in zip(percent, counts_elements)]
    res = {label[i]: round(sample_size[i]) for i in range(len(label))}

    x_train = np.empty((0, x.shape[1]))
    y_train = np.empty((0,))
    for key, value in res.items():
        msk = y == key
        idx = np.random.choice(sum(msk), int(value), replace=False)
        x_train = np.concatenate([x_train, x[msk, :][idx, :]])
        y_train = np.hstack((y_train, np.repeat(key, len(idx))))

    indexes = None
    counts_elements = None
    percent = None
    sample_size = None
    label = None
    res = None

    return x_train, y_train
