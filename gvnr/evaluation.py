
import pkg_resources
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import warnings
import logging
logger = logging.getLogger()

C = 10e6
tol = 10e-4

def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = np.squeeze(np.asarray(np.sum(y, axis=1, dtype=np.int)))
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    y_pred = np.zeros(y.shape, dtype=np.int)
    for i in range(y.shape[0]):
        for j in range(num_label[i]):
            y_pred[i, y_sort[i, j]] = 1
    return y_pred

def predict_cv(X, y, train_ratio=0.2, n_splits=10, random_state=0):
    micro, macro = [], []
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio,random_state=random_state)
    for train_index, test_index in shuffle.split(X):
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X[train_index], X[test_index]
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    tol=tol,
                    solver="liblinear",
                    max_iter=1000,
                    multi_class="ovr"),
                n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_pred = construct_indicator(y_score, y_test)
        mi = f1_score(y_test, y_pred, average="micro")
        ma = f1_score(y_test, y_pred, average="macro")
        micro.append(mi)
        macro.append(ma)
    return np.mean(micro), np.mean(macro)

def get_score(vectors, labels, multilabels=False, proportions=[0.1, 0.2, 0.3, 0.4, 0.5], n_trials = 10):
    n_trials = 10
    np.random.seed(1) # for reproducibility
    scores = {
        'f1_micro':np.zeros(len(proportions)),
        'f1_macro':np.zeros(len(proportions))
    }
    for i,p in enumerate(proportions):
        mean_accuracy = {
            'f1_micro':0,
            'f1_macro':0
        }
        if multilabels is False:
            for _ in range(n_trials):
                train_ids, test_ids = random_train_test(len(vectors), p)
                while len(set(labels[train_ids])) != len(set(labels)):
                    train_ids, test_ids = random_train_test(len(vectors), p)
                sco = train_and_predict(vectors[train_ids], vectors[test_ids], labels[train_ids], labels[test_ids])
                for k in sco:
                    mean_accuracy[k] += sco[k]/n_trials
            for k in scores:
                scores[k][i] = mean_accuracy[k]
        else:
            scores["f1_micro"][i], scores["f1_macro"][i] = predict_cv(vectors, labels, train_ratio=p, n_splits=n_trials, random_state=0)
    return scores


def random_train_test(size, proportion):
    ids = np.arange(size)
    np.random.shuffle(ids)
    limit = int(size * proportion)
    return ids[:limit], ids[limit:]

def train_and_predict(train_data, test_data, train_labels, test_labels):
    clf = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            tol=tol,
            solver="liblinear",
            max_iter=1000,
            multi_class="ovr"),
        n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(train_data, train_labels)
    predicted_labels = clf.predict(test_data)
    scores = {
    'f1_micro': f1_score(test_labels, predicted_labels, average='micro'),
    'f1_macro': f1_score(test_labels, predicted_labels, average='macro')
    }
    return scores