import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, classification_report, confusion_matrix,\
    roc_curve,auc,precision_recall_curve,roc_curve
import pandas as pd
import itertools


def top_n_accuracy(preds, truths, n):
    best_n = np.argsort(preds, axis=1)[:,-n:]
    ts = np.argmax(truths, axis=1)
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n[i,:]:
            successes += 1
    return float(successes)/ts.shape[0]
 

def prediction_evaluation(ytest, ypred, directory) :
    metrix = ['Top1-Accuracy',
              'Top2-Accuracy',
              'Top3-Accuracy',
              'Top4-Accuracy',
              'Top5-Accuracy'
             ]
    scores = [top_n_accuracy(ypred, ytest, 1),
              top_n_accuracy(ypred, ytest, 2),
              top_n_accuracy(ypred, ytest, 3),
              top_n_accuracy(ypred, ytest, 4),
              top_n_accuracy(ypred, ytest, 5)
             ]
    scores = np.round(scores,3)

    score = pd.DataFrame({'Metric': metrix, 'Score': scores})

    fig = plt.figure(figsize=(8,8))
    plt.plot(score.Metric, score.Score, marker='o', markersize=10,
             linestyle='--', color='darkgreen', label='Accuracy')
    plt.plot(score.Metric, 1-score.Score, marker='^', markersize=10,
             linestyle='--', color='darkred', label='Error')
    plt.title("Performance along the output ranking")
    plt.xlabel("TopN-Accuracies")
    plt.ylabel("Accuracies")
    plt.legend()
    fig.savefig(str(directory+'/TopN-Accuracies-Predictions.png'), dpi=300)
    return score, fig


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def F1(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    from sklearn.metrics import f1_score
    fpr, tpr, threshold = roc_curve(target, predicted)
    threshold = threshold[1:]
    threshold = threshold[threshold != 0]
    threshold = threshold[threshold != 1]
    MC = []
    taille = len(threshold)
    for i in range(taille):
        pred=predicted.copy()
        pred[pred< threshold[i]]=0
        pred[pred >= threshold[i]]=1
        indic = f1_score(target, pred)
        MC.append(indic)
    importance = {"seuil": threshold, "score": MC}
    importance = pd.DataFrame(importance)
    importance = importance.sort_values("score", ascending=False)
    importance.index = range(0, len(importance))
    roc_t = importance[:1]
    return list(roc_t['seuil'])

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=0)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights,axis=0)
    return average, np.sqrt(variance)