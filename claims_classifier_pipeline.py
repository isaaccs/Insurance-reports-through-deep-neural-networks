import numpy as np
from utils.metrics import F1
from utils.models import Model
from utils.processor import PreprocessingTemplate, OneHotEncoding
import os
from sklearn.metrics import f1_score, precision_score, classification_report, precision_recall_curve, confusion_matrix
from utils.bagging import generate_data
import tensorflow as tf
from utils.activation import swish
from utils.loss import ncce
from utils.general_utils import load_object, save_object, make_directory
import sys
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('../')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
ARTIFACTS = os.path.join(ROOT_DIR, "results", "artifacts")
make_directory(ARTIFACTS)

input_dir = 'data/'


class ClaimsClassifier:
    def __init__(self,
                 methode,
                 nb_dataset=2,
                 threshold=0.9,
                 type_model='LSTM',
                 w_model='f1_min',
                 cutoff='F1',
                 config_dict={
                   "min_n_gram": 1,
                   "max_n_gram": 3,
                   "encoding": "embedding",
                   "embedding_dim": 100,
                   "window_context": 3,
                   "min_word_count": 1,
                   "sample": 1e-3,
                   "sg": 0}):
        self.type_model = type_model
        self.nb_dataset = nb_dataset
        self.methode = methode
        self.threshold = threshold
        self.w_model = w_model
        self.cutoff = cutoff
        self.embedding_matrix = None
        self.config_dict = config_dict
        self.encoder = None
        self.n_out = 0
        self.min_label = None
        self.list_model = []
        self.list_weight = []
        self.label = []
        self.name=''
        self.feature_extraction = None
        self.model_folder = os.path.join(ARTIFACTS, "model_folder")
        self.feature_extraction_folder = os.path.join(ARTIFACTS, "feature_extraction")
        self.graphic_folder = os.path.join(ARTIFACTS, "graphic")
        make_directory(self.feature_extraction_folder)
        make_directory(self.model_folder)
        make_directory(self.graphic_folder)
        self.load_saved()
        return

    def load_saved(self):
        self.feature_extraction = PreprocessingTemplate(self.config_dict, self.feature_extraction_folder)
        self.encoder = OneHotEncoding(self.feature_extraction_folder)
        self.load_models()

    def new_model(self):
        try:
            for f in os.listdir(self.model_folder):
                os.remove(os.path.join(self.model_folder, f))
        except:
            pass
        self.list_model = []
        self.list_weight = []

    def fit(self,
            x_train,
            y_train,
            x_test,
            y_test,
            trainable=True,
            custom=True,
            batch_size=32,
            epochs=5
            ):

        self.create_name_fig(trainable, custom)
        self.new_model()
        label, indexes, counts_elements = np.unique(y_train, return_counts=True, return_index=True)
        self.min_label = label[np.argmin(counts_elements)]
        self.label = [y_train[index] for index in sorted(indexes)]
        save_object(os.path.join(self.model_folder, "label.pickle"), self.label)
        save_object(os.path.join(self.model_folder, "min_label.pickle"), self.min_label)

        self.feature_extraction = PreprocessingTemplate(self.config_dict, self.feature_extraction_folder)
        corpus = np.hstack((x_train, x_test))
        if self.config_dict["encoding"] == "embedding" and custom:
            _, self.embedding_matrix = self.feature_extraction.fit_transform(x_train)

        else:
            _ = self.feature_extraction.fit_transform(corpus)
        x_train = self.feature_extraction.transform(x_train)
        x_test = self.feature_extraction.transform(x_test)

        self.encoder = OneHotEncoding(self.feature_extraction_folder)
        self.encoder.fit(self.label)
        y_test = self.encoder.transform(y_test)

        self.n_out = len(label)
        if self.type_model == 'LSTM':
            params = {'units_size1': 256, 'units_sizes': [128], 'dense_size1': 500, 'dense_size2': [300],
                      'dropout': 0.004}
        else:
            params = {'filter_sizes': [1, 2], 'nb_filter': 1024, 'dense_size1': 400, 'dense_size2': [250],
                      'dropout': 0.001}
        model = Model(self.type_model,
                      vocabulary_size=len(self.feature_extraction.vocabulary),
                      embedding_dim=self.config_dict['embedding_dim'],
                      embedding_matrix=self.embedding_matrix,
                      trainable=trainable,
                      params=params,
                      filename=self.model_folder)

        for i in range(self.nb_dataset):
            x_temp, y_temp = generate_data(x_train, y_train, self.methode, self.threshold)
            y_temp = self.encoder.transform(y_temp)
            model.fit(x_train=x_temp, y_train=y_temp, x_test=x_test, y_test=y_test,
                      batch_size=batch_size, epochs=epochs, i=i)
        self.calculate_weight(x_test=x_test, y_test=y_test)
        if self.cutoff == 'F1':
            self.find_optimal_cutoff(x_test=x_test, y_test=y_test)
        return

    def calculate_weight(self, x_test, y_test):
        self.load_models()
        y_test = np.argmax(y_test, axis=1)
        p = 0
        for i in range(len(self.list_model)):
            y_pred = self.list_model[i].predict(x_test)
            y_pred = np.argmax(y_pred, axis=1)
            if self.w_model == 'f1':
                p = f1_score(y_test, y_pred, average='weighted')
            if self.w_model == 'precision':
                p = precision_score(y_test, y_pred, average='weighted')
            if self.w_model == 'f1_min':
                s = classification_report(y_test, y_pred, target_names=self.label, output_dict=True)
                p = float(s[self.min_label]['f1-score'])
            if self.w_model == 'precision_min':
                s = classification_report(y_test, y_pred, target_names=self.label, output_dict=True)
                p = float(s[self.min_label]['precision'])
            if self.nb_dataset == 1 or self.w_model is None:
                p = 1
            if p == 0:
                p = 0.00000000001
            self.list_weight.append(p)
        save_object(os.path.join(self.model_folder, "list_weight.pickle"), self.list_weight)
        return

    def predict(self, x_dev):
        y_pred = self.predict_proba(x_dev)
        if self.cutoff:
            return np.where(y_pred[:, 1] >= self.cutoff, 1, 0)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, x_dev, return_interval=False):
        #x_dev = self.feature_extraction.transform(x_dev)
        self.load_models()
        y_pred = []
        for i in range(len(self.list_model)):
            y_pred.append(self.list_model[i].predict(x_dev))
        if return_interval:
            return y_pred
        else:
            return sum([a * b for a, b in zip(y_pred, self.list_weight)])/sum(self.list_weight)

    def load_models(self):
        try:
            if not self.list_model:
                for i in range(self.nb_dataset):
                    self.list_model.append(tf.keras.models.load_model(os.path.join(self.model_folder,
                                                                                   'weights.best.{}.hdf5'.format(i)),
                                                                      custom_objects={'swish': swish, 'ncce': ncce}))
        except OSError as e:
            pass

        try:
            self.list_weight = load_object(os.path.join(self.model_folder, "list_weight.pickle"))
        except FileNotFoundError as e:
            pass

        try:
            self.cutoff = load_object(os.path.join(self.model_folder, "cutoff.pickle"))
        except FileNotFoundError as e:
            pass

        try:
            self.label = load_object(os.path.join(self.model_folder, "label.pickle"))
        except FileNotFoundError as e:
            pass

        try:
            self.min_label = load_object(os.path.join(self.model_folder, "min_label.pickle"))
        except FileNotFoundError as e:
            pass

    def find_optimal_cutoff(self, x_test, y_test):
        self.load_models()
        y_pred = []
        for i in range(len(self.list_model)):
            y_pred.append(self.list_model[i].predict(x_test))
        y_pred = sum([a * b for a, b in zip(y_pred, self.list_weight)]) / sum(self.list_weight)
        y_test = np.argmax(y_test, axis=1)
        self.cutoff = F1(y_test, y_pred[:, 1])
        save_object(os.path.join(self.model_folder, "cutoff.pickle"), self.cutoff)

    def score(self, x_dev, y_dev, return_interval=True):
        y_dev = self.encoder.transform(y_dev)
        x_dev = self.feature_extraction.transform(x_dev)
        self.find_optimal_cutoff(x_test=x_dev, y_test=y_dev)
        y_dev = np.argmax(y_dev, axis=1)
        y_pred = self.predict_proba(x_dev, return_interval=return_interval)
        lower, upper = self.calculate_interval(y_dev, y_pred, alpha=0.05)
        y_pred = sum([a * b for a, b in zip(y_pred, self.list_weight)]) / sum(self.list_weight)


        plot_pr_curve(y_dev, y_pred[:,1], self.name, self.graphic_folder)
        if self.cutoff:
            y_pred = np.where(y_pred[:, 1] >= self.cutoff, 1, 0)
        else:
            y_pred = np.argmax(y_pred, axis=1)
        TP, FP, FN, TN = get_confusion_matrix_values(y_dev, y_pred)

        print('lower bound')
        print(lower)
        print('upper bound')
        print(upper)
        print('precision')
        print((beta.ppf(0.025, TP + 1, FP + 1), beta.ppf(0.975, TP + 1, FP + 1)))  # precision
        print('recall')
        print((beta.ppf(0.025, TP + 1, FN + 1), beta.ppf(0.975, TP + 1, FN + 1)))  # recall

        print(classification_report(y_dev, y_pred, target_names=self.label))

    def create_name_fig(self, trainable, custom):
        name_fig = self.type_model+'_'
        if trainable and custom:
            name_fig += 'non_static_'
        elif custom:
            name_fig += 'static_'
        else:
            name_fig += 'random_'

        if isinstance(self.methode, str):
            name_fig += self.methode
        else:
            name_fig += 'lightly'

        if self.config_dict["max_n_gram"] == 1:
            name_fig += '_words'
        else:
            name_fig += '_n_grams'

        self.name = name_fig

    def calculate_interval(self, y_dev, y_pred, alpha=0.05):
        metrics = []
        for i in range(len(y_pred)):
            y_temp = np.where(y_pred[i][:, 1]>=self.cutoff,1, 0)
            s = classification_report(y_dev, y_temp, target_names=self.label, output_dict=True)
            f1_grave = float(s[self.min_label]['f1-score'])
            precision_grave = float(s[self.min_label]['precision'])
            recall_grave = float(s[self.min_label]['recall'])
            metrics += [[precision_grave, recall_grave, f1_grave]]

        lower_bound = np.percentile(metrics, 100 * (alpha / 2), axis=0)
        upper_bound = np.percentile(metrics, 100 * (1 - alpha / 2), axis=0)
        #estimate, stderr = weighted_avg_and_std(metrics, self.list_weight)

        return lower_bound, upper_bound

def plot_pr_curve(y_test, y_pred,name,path):
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_pred)
    plt.plot(lr_recall, lr_precision, marker='.')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(path, "{}.png".format(name)), dpi=300)
    plt.close("all")


def get_confusion_matrix_values(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0][0], cm[0][1], cm[1][0], cm[1][1]


if __name__ == '__main__':
    train = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(input_dir, 'test.csv'))
    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))

    x_train = train.text
    y_train = train.indic_s1_dgl_new
    x_test = test.text
    y_test = test.indic_s1_dgl_new
    x_dev = dev.text
    y_dev = dev.indic_s1_dgl_new

    cc = ClaimsClassifier(methode="balanced")

    print('******static*******')
    cc.fit(x_train, y_train, x_test, y_test, trainable=False,
    custom=True)
    cc.score(x_dev, y_dev)
    print('******non static*******')
    cc.fit(x_train, y_train, x_test, y_test, trainable=True,
    custom=True)
    cc.score(x_dev, y_dev)

    print('******rand*******')
    cc = ClaimsClassifier(methode="randomly")
    cc.fit(x_train, y_train, x_test, y_test, trainable=True,
    custom=False)
    cc.score(x_dev, y_dev)
    print('******static*******')
    cc.fit(x_train, y_train, x_test, y_test, trainable=False,
    custom=True)
    cc.score(x_dev, y_dev)
    print('******non static*******')
    cc.fit(x_train, y_train, x_test, y_test, trainable=True,
    custom=True)
    cc.score(x_dev, y_dev)


