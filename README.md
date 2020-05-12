# Automatic analysis of insurance reports through deep neural networks to identify severe claims.

## Abstract
In this paper, we develop a methodology to automatically classify claims using the information contained in text reports (redacted at their opening). From this automatic analysis, the aim is to predict if a claim is expected to be particularly severe or not. The difficulty is the rarity of such extreme claims in the database, and hence the difficulty, for classical prediction techniques like logistic regression to accurately predict the outcome. Since data is unbalanced (too few observations are associated with a positive label), we propose different rebalance algorithm to deal with this issue. We discuss the use of different embedding methodologies used to process text data, and the role of the architectures of the networks.


## Dependencies

* [Tensorflow](https://www.tensorflow.org)
* [NumPy](https://numpy.org)
* [scikit-learn](https://scikit-learn.org/stable/)
* [Matplotlib](https://matplotlib.org)
* [seaborn](https://seaborn.pydata.org)
* [pandas](https://pandas.pydata.org)
* [gensim](https://radimrehurek.com/gensim/)
* [itertools](https://docs.python.org/2/library/itertools.html)
* [functools](https://docs.python.org/3/library/functools.html)





