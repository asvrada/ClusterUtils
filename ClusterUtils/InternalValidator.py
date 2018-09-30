import pandas as pd
import numpy as np
import math
import time
from ClusterUtils.ClusterPlotter import _plot_cvnn_
from ClusterUtils.ClusterPlotter import _plot_silhouette_


def silhouette(dataset, cluster_num):
    """
    :param dataset: DataFrame contains clustering results, n rows
    :param cluster_num:
    :return:
    """

    def d(x, y):
        """
        Distance between x and y
        :param x: data sample
        :param y: data sample
        :return: distance between x and y
        """
        return np.linalg.norm(x - y)

    def a(cluster, i):
        """
        let a(i) be the average distance between i and all other data within the same cluster
        :param cluster: index of cluster the data point belongs to
        :param i: the data point
        :type i: np.array
        :return: float
        """
        divider = len(dataset_converted[cluster]) - 1
        if divider == 0:
            # there is only 1 element in this cluster
            return 0
        return sum(d(i, data) for data in dataset_converted[cluster]) / divider

    def b(cluster, i):
        """
        Let b(i) be the smallest average distance of i to all points in any other cluster, of which i is not a member
        :param cluster: index of cluster
        :type cluster: int
        :param i: the data point
        :type i: np.array
        :return: float
        """
        ave_dist_clusters = []
        for i_cluster in range(cluster_num):
            if i_cluster == cluster:
                continue
            # distance to all data points in another cluster
            if len(dataset_converted[i_cluster]) == 0:
                continue
            dist = sum(d(i, data) for data in dataset_converted[i_cluster]) / len(dataset_converted[i_cluster])
            ave_dist_clusters.append(dist)

        return min(ave_dist_clusters)

    def s(cluster, i):
        """
        silhouette value for single data point i
        :param cluster: index of cluster
        :type cluster: int
        :param i: data point
        :type i: np.array
        :return:
        """
        ret_a = a(cluster, i)
        ret_b = b(cluster, i)

        if abs(ret_a - ret_b) <= 1e-10:
            return 0

        return (ret_b - ret_a) / max(ret_a, ret_b)

    # convert dataset into better format
    # [[x,y,cluster]] -> [[(x,y),(x,y)], [(x,y)]]
    dataset_converted = dict()
    for i in range(cluster_num):
        dataset_converted[i] = []

    for row in dataset.values:
        cluster = int(row[-1])
        row = row[:-1]
        dataset_converted[cluster].append(np.array(row))

    tmp_sum = 0
    tmp_n = 0
    for i_cluster in range(cluster_num):
        tmp_sum += sum(s(i_cluster, i) for i in dataset_converted[i_cluster])
        tmp_n += len(dataset_converted[i_cluster])

    # the average i over all data of the entire dataset
    return tmp_sum / tmp_n


def tabulate_silhouette(datasets, cluster_nums):
    """
    https://github.com/scikit-learn/scikit-learn/blob/f0ab589f/sklearn/metrics/cluster/unsupervised.py#L22
    """

    # Inputs:
    # datasets: Your provided list of clustering results.
    # cluster_nums: A list of integers corresponding to the number of clusters
    # in each of the datasets, e.g.,
    # datasets = [np.darray, np.darray, np.darray]
    # cluster_nums = [2, 3, 4]

    # Return a pandas DataFrame corresponding to the results.
    # x = num of cluster
    # y = silhouette index of that number of cluster
    y = [silhouette(dataset, cluster_num) for dataset, cluster_num in zip(datasets, cluster_nums)]

    return pd.DataFrame({"CLUSTERS": cluster_nums, "SILHOUETTE_IDX": y})


def tabulate_cvnn(datasets, cluster_nums, k_vals):
    # Implement.

    # Inputs:
    # datasets: DataFrames Your provided list of clustering results.
    # cluster_nums: A list of integers corresponding to the number of clusters
    # in each of the datasets, e.g.,
    # datasets = [np.darray, np.darray, np.darray]
    # cluster_nums = [2, 3, 4]

    # Return a pandas DataFrame corresponding to the results.

    return None


# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class InternalValidator:
    """
    Parameters
    ----------
    datasets : list or array-type object, mandatory
        A list of datasets. The final column should cointain predicted labeled results
        (By default, the datasets generated are pandas DataFrames, and the final
        column is named 'CLUSTER')
    cluster_nums : list or array-type object, mandatory
        A list of integers corresponding to the number of clusters used (or found).
        Should be the same length as datasets.
    k_vals: list or array-type object, optional
        A list of integers corresponding to the desired values of k for CVNN
        """

    def __init__(self, datasets, cluster_nums, k_vals=[1, 5, 10, 20]):
        # WHY STRIPPED?
        self.datasets = list(map(lambda df: df.drop('CENTROID', axis=0), datasets))
        self.cluster_nums = cluster_nums
        self.k_vals = k_vals

    def make_cvnn_table(self):
        start_time = time.time()
        self.cvnn_table = tabulate_cvnn(self.datasets, self.cluster_nums, self.k_vals)
        print("CVNN finished in  %s seconds" % (time.time() - start_time))

    def show_cvnn_plot(self):
        _plot_cvnn_(self.cvnn_table)

    def save_cvnn_plot(self, name='cvnn_plot'):
        _plot_cvnn_(self.cvnn_table, save=True, n=name)

    def make_silhouette_table(self):
        start_time = time.time()
        self.silhouette_table = tabulate_silhouette(self.datasets, self.cluster_nums)
        print("Silhouette Index finished in  %s seconds" % (time.time() - start_time))

    def show_silhouette_plot(self):
        _plot_silhouette_(self.silhouette_table)

    def save_silhouette_plot(self, name='silhouette_plot'):
        _plot_silhouette_(self.cvnn_table, save=True, n=name)

    def save_csv(self, cvnn=False, silhouette=False, name='internal_validator'):
        if cvnn is False and silhouette is False:
            print('Please pass either cvnn=True or silhouette=True or both')
        if cvnn is not False:
            filename = name + '_cvnn_' + (str(round(time.time()))) + '.csv'
            self.cvnn_table.to_csv(filename)
            print('Dataset saved as: ' + filename)
        if silhouette is not False:
            filename = name + '_silhouette_' + (str(round(time.time()))) + '.csv'
            self.silhouette_table.to_csv(filename)
            print('Dataset saved as: ' + filename)
        if cvnn is False and silhouette is False:
            print('No data to save.')
