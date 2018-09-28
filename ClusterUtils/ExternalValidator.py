import pandas as pd
import numpy as np
import math
import time

"""
Ref.:https://github.com/scikit-learn/scikit-learn/blob/f0ab589f/sklearn/metrics/cluster/supervised.py
"""


def entropy(labels):
    """
    Calculates the entropy for a labeling.
    Ref.: https://en.wikipedia.org/wiki/Entropy#Information_theory
    """
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    # number of occurrence
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = pi.sum()
    return -np.sum((pi / pi_sum) * (np.log(pi) - math.log(pi_sum)))


def get_contingency_matrix(true_labels, pred_labels):
    """
    :param true_labels: array of integer labels
    :param pred_labels: array of integer labels
    :return:
    """
    return np.array([list(zip(true_labels, pred_labels)).count((pred, true)) for pred in set(pred_labels) for true in
                     set(true_labels)]).reshape((len(set(pred_labels)), len(set(true_labels))))


def convert_labels(labels):
    # convert labels from pandas data frame index into np array
    list_labels = list(set(labels))
    return np.array([list_labels.index(each) for each in labels])


def find_norm_MI(true_labels, pred_labels):
    # Return a number corresponding to the NMI of the two sets of labels.

    # convert labels into np array
    # into format like [0000111122222]....
    true_labels = convert_labels(true_labels)

    # First, implement MI
    contingency = get_contingency_matrix(true_labels, pred_labels)
    contingency_sum = contingency.sum()

    def pi(x):
        return contingency.sum(axis=1)[x] / contingency_sum

    def pj(y):
        return contingency.sum(axis=0)[y] / contingency_sum

    def pij(x, y):
        return contingency[x][y] / contingency_sum

    mi_sum = 0
    for x in range(contingency.shape[0]):
        for y in range(contingency.shape[1]):
            _pi = pi(x)
            _pj = pj(y)
            _pij = pij(x, y)

            if _pi == 0 or _pj == 0 or _pij == 0:
                continue

            mi_sum += _pij * math.log(_pij / (_pi * _pj))

    # Second, implement NMI
    # Calculate the expected value for the mutual information
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(true_labels), entropy(pred_labels)
    nmi = mi_sum / max(np.sqrt(h_true * h_pred), 1e-10)

    return nmi


def find_norm_rand(true_labels, pred_labels):
    def comb(N, k):
        """
        N takes k, number of combination
        https://github.com/scipy/scipy/blob/master/scipy/special/_comb.pyx
        """
        # Fallback
        N = int(N)
        k = int(k)

        if k > N or N < 0 or k < 0:
            return 0

        M = N + 1
        nterms = min(k, N - k)

        numerator = 1
        denominator = 1
        for i in range(1, nterms + 1):
            numerator *= M - i
            denominator *= i

        return numerator // denominator

    def comb2(n):
        return comb(n, 2)

    # convert labels into np array
    # into format like [0000111122222]....
    true_labels = convert_labels(true_labels)

    # Return a number corresponding to the NRI of the two sets of labels.
    contingency = get_contingency_matrix(true_labels, pred_labels)
    x, y = contingency.shape

    n_samples = len(true_labels)

    sum_comb_c = sum(comb2(n_c) for n_c in np.ravel(contingency.sum(axis=1)))
    sum_comb_k = sum(comb2(n_k) for n_k in np.ravel(contingency.sum(axis=0)))

    sum_tmp = 0
    for i in range(x):
        for j in range(y):
            sum_tmp += comb2(contingency[i][j])

    sum_comb = sum_tmp

    prod_comb = (sum_comb_c * sum_comb_k) / comb2(n_samples)
    mean_comb = (sum_comb_k + sum_comb_c) / 2.
    return (sum_comb - prod_comb) / (mean_comb - prod_comb)


def find_accuracy(true_labels, pred_labels):
    # Implement.
    # Return a number corresponding to the accuracy of the two sets of labels.

    # convert labels into np array
    # into format like [0000111122222]....
    true_labels = convert_labels(true_labels)
    matrix = get_contingency_matrix(true_labels, pred_labels)

    # for each col, find the largest num and sum it
    # then divide by total number of data
    return sum([max(matrix[:, i_col]) for i_col in range(matrix.shape[1])]) / sum(sum(matrix))


# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class ExternalValidator:
    """
    Parameters
    ----------
    df : pandas DataFrame, optional
        A DataFrame produced by running one of your algorithms.
        The relevant labels are automatically extracted.
    true_labels : list or array-type object, mandatory
        A list of strings or integers corresponding to the true labels of
        each sample
    pred_labels: list or array-type object, optional
        A list of integers corresponding to the predicted cluster index for each
        sample
    """

    def __init__(self, df=None, true_labels=None, pred_labels=None):
        df = df.drop('CENTROID', axis=0)  # IMPORTANT -- Drop centroid rows before processing
        self.DF = df
        self.true_labels = true_labels
        self.pred_labels = pred_labels

        if df is not None:
            self.extract_labels()
        elif true_labels is None or pred_labels is None:
            print('Warning: No data provided')

    def extract_labels(self):
        self.true_labels = self.DF.index
        self.pred_labels = self.DF['CLUSTER']

    def normalized_mutual_info(self):
        start_time = time.time()
        nmi = find_norm_MI(self.true_labels, self.pred_labels)
        print("NMI finished in  %s seconds" % (time.time() - start_time))
        return nmi

    def normalized_rand_index(self):
        start_time = time.time()
        nri = find_norm_rand(self.true_labels, self.pred_labels)
        print("NRI finished in  %s seconds" % (time.time() - start_time))
        return nri

    def accuracy(self):
        start_time = time.time()
        a = find_accuracy(self.true_labels, self.pred_labels)
        print("Accuracy finished in  %s seconds" % (time.time() - start_time))
        return a
