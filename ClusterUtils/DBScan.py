import pandas as pd
import numpy as np
import time
from ClusterUtils.SuperCluster import SuperCluster
from ClusterUtils.ClusterPlotter import _plot_generic_


def dist_func(a, b):
    return np.linalg.norm(a - b)


def find_neighbors(X, i_core, eps):
    return set(filter(lambda i_point: False if i_core == i_point else dist_func(X[i_core], X[i_point]) <= eps, range(len(X))))


def dbscan(X, eps=1, min_points=10, verbose=False):
    """
    Implementation of dbscan
    :param X: np.darray of samples
    :param eps: minimal distance
    :param min_points:  minimal number of points to form a cluster
    :param verbose: toggle log or not
    :return: a array or list-type object corresponding to the predicted cluster
    """
    # number of cluster
    C = 0

    # Labels for each data point
    # if its int, it represents the cluster it belongs to
    # if its string, if its "noise", its noise
    NOISE = "NOISE"
    labels = [None] * len(X)

    # for each and every point in the dataset
    for i_point in range(len(X)):
        # this point has been visited, skip
        if labels[i_point] is not None:
            continue

        neighbors = find_neighbors(X, i_point, eps)
        # because the neighbors doesn't contain the original point, we add 1 to compensate that
        # no enough neighbor, this is a NOISE point
        if len(neighbors) + 1 < min_points:
            labels[i_point] = NOISE
            continue

        # assign this point to cluster C
        C += 1
        labels[i_point] = C

        # make it a queue so its safe to add items when looping
        queue_neighbor = list(neighbors)
        while len(queue_neighbor) != 0:
            cursor = queue_neighbor.pop(0)

            # add it to this cluster
            if labels[cursor] == NOISE:
                labels[cursor] = C

            # this has been visited, skip
            if labels[cursor] is not None:
                continue

            # add this point to this cluster
            labels[cursor] = C
            # find the neighbors of current point
            tmp_neighbors = find_neighbors(X, cursor, eps)

            if len(tmp_neighbors) >= min_points:
                # add the result into current list
                queue_neighbor += list(filter(lambda x: x not in queue_neighbor, tmp_neighbors))

    # assign data point with label NOISE into cluster 0
    return list(map(lambda x: 0 if x == NOISE else x, labels))


# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class DBScan(SuperCluster):
    """
    Perform DBSCAN clustering from vector array.
    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    csv_path : str, default: None
        Path to file for dataset csv
    keep_dataframe : bool, default: True
        Hold on the results pandas DataFrame generated after each run.
        Also determines whether to use pandas DataFrame as primary internal data state
    keep_X : bool, default: True
        Hold on the results generated after each run in a more generic array-type format
        Use these values if keep_dataframe is False
    verbose: bool, default: False
        Optional log level
    """

    def __init__(self, eps=1, min_points=10, csv_path=None, keep_dataframe=True,
                 keep_X=True, verbose=False):
        self.eps = eps
        self.min_points = min_points
        self.verbose = verbose
        self.csv_path = csv_path
        self.keep_dataframe = keep_dataframe
        self.keep_X = keep_X

    # X is an array of shape (n_samples, n_features)
    def fit(self, X):
        if self.keep_X:
            self.X = X
        start_time = time.time()
        self.labels = dbscan(X, eps=self.eps, min_points=self.min_points, verbose=self.verbose)
        print("DBSCAN finished in  %s seconds" % (time.time() - start_time))
        return self

    def show_plot(self):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_generic_(df=self.DF)
        elif self.keep_X:
            _plot_generic_(X=self.X, labels=self.labels)
        else:
            print('No data to plot.')

    def save_plot(self, name):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_generic_(df=self.DF, save=True, n=name)
        elif self.keep_X:
            _plot_generic_(X=self.X, labels=self.labels, save=True, n=name)
        else:
            print('No data to plot.')
