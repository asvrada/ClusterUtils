import pandas as pd
import numpy as np
import random
import time
from ClusterUtils.SuperCluster import SuperCluster
from ClusterUtils.ClusterPlotter import _plot_kmeans_

THRESHOLD = 1e-9


def init_random(data, n_clusters):
    return [data[i] for i in np.random.permutation(len(data))[:n_clusters]]


def init_kmeanpp(data, n_clusters):
    pass


def init_global(data, n_clusters):
    pass


def objective_function(X, assignment, centroids):
    """
    Calculate the cost of current assignment by SSE (Square Sum Error)
    :param assignment: a np.array of integer that represents id of clusters
    :param centroids: a list of np.array that represents centroids for each cluster
    :return: float
    """

    # Determine distance by the square of euclidean distance
    def dist(x):
        """
        :param x: np.array, for example: [3.0, 4.0]
        :return: for example: 25.0
        """
        return sum(each ** 2 for each in x)

    # first lets calculate the obj func for each cluster
    datasets = [X[np.where(assignment == cluster)[0]] for cluster in range(len(centroids))]
    sse_sum = 0
    for i in range(len(datasets)):
        sse_sum += sum(dist(row - centroids[i]) for row in datasets[i])

    return sse_sum


def cluster_lloyds(X, n_clusters=3, init='random', n_init=1, max_iter=300, verbose=False):
    init_methods = {
        "random": init_random,
        "k-mean++": init_kmeanpp,
        "global": init_global
    }

    best_centroids, best_assignment, best_inertia = None, None, None

    # for each iteration of this algorithm
    for _run in range(n_init):
        # init centroids
        centroids = init_methods[init](X, n_clusters)
        assignment = None
        inertia = 0
        for _iter in range(max_iter):
            assignment = np.array([np.argmin([np.linalg.norm(centroid - row) for centroid in centroids]) for row in X])
            centroids = [np.average(X[np.where(assignment == cluster)[0]], axis=0) for cluster in range(n_clusters)]

            # when obj func diff <= LIMIT, break
            sse = objective_function(X, assignment, centroids)
            if abs(sse - inertia) <= THRESHOLD:
                if best_inertia is not None and sse >= best_inertia:
                    # skip this run
                    break
                # that's it
                best_centroids = centroids
                best_assignment = assignment
                best_inertia = sse

                print("Stop K-mean after iteration {} at run {}".format(_iter, _run))
                break

            # not good enough, go no iteration
            inertia = sse

    return best_centroids, best_assignment, best_inertia


def cluster_hartigans(X, n_clusters=3, init='random', n_init=1, max_iter=300, verbose=False):
    return 0, 0, 0


def k_means(X, n_clusters=3, init='random', algorithm='lloyds', n_init=1, max_iter=300, verbose=False):
    cluster_algorithm = {
        "lloyds": cluster_lloyds,
        "hartigans": cluster_hartigans
    }

    # Implement.
    # Input: np.darray of samples
    best_centroids, best_assignment, best_inertia = cluster_algorithm[algorithm](X, n_clusters, init, n_init, max_iter, verbose)

    # Return the following:
    #
    # 1. labels: An array or list-type object corresponding to the predicted
    #  cluster numbers,e.g., [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # 2. centroids: An array or list-type object corresponding to the vectors
    # of the centroids, e.g., [[0.5, 0.5], [-1, -1], [3, 3]]
    # 3. inertia: A number corresponding to some measure of fitness,
    # generally the best of the results from executing the algorithm n_init times.
    # You will want to return the 'best' labels and centroids by this measure.

    return best_assignment, best_centroids, best_inertia


# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class KMeans(SuperCluster):
    """
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init : {'random', 'k-means++', 'global'}
        Method for initialization, defaults to 'random'.
    algorithm : {'lloyds', 'hartigans'}
        Method for determing algorithm, defaults to 'lloyds'.
    n_init : int, default: 1
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
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

    def __init__(self, n_clusters=3, init='random', algorithm='lloyds', n_init=1, max_iter=300,
                 csv_path=None, keep_dataframe=True, keep_X=True, verbose=False):
        self.n_clusters = n_clusters
        self.init = init
        self.algorithm = algorithm
        self.n_init = n_init
        self.max_iter = max_iter
        self.csv_path = csv_path
        self.keep_dataframe = keep_dataframe
        self.keep_X = keep_X
        self.verbose = verbose

    # X is an array of shape (n_samples, n_features)
    def fit(self, X):
        if self.keep_X:
            self.X = X
        start_time = time.time()
        self.labels, self.centroids, self.inertia = \
            k_means(X, n_clusters=self.n_clusters, init=self.init,
                    n_init=self.n_init, max_iter=self.max_iter, verbose=self.verbose)
        print(self.init + " k-means finished in  %s seconds" % (time.time() - start_time))
        return self

    def show_plot(self):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_kmeans_(df=self.DF)
        elif self.keep_X:
            _plot_kmeans_(X=self.X, labels=self.labels, centroids=self.centroids)
        else:
            print('No data to plot.')

    def save_plot(self, name='kmeans_plot'):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_kmeans_(df=self.DF, save=True, n=name)
        elif self.keep_X:
            _plot_kmeans_(X=self.X, labels=self.labels,
                          centroids=self.centroids, save=True, n=name)
        else:
            print('No data to plot.')
