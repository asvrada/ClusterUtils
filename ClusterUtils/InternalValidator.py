import pandas as pd
import numpy as np
import collections
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


def compute_kNN(dataset, k):
    """
    Compute the k-nearest neighbors for each data point in the given dataset
    Not include points within the same cluster

    :param dataset:
    :type dataset: pandas.DataFrame
    :param k: number of nearest neighbors to find
    :type k: int
    :return: a 2D array, for each data point, compute a list of points that are in other clusters that is it's kNN
    :rtype: numpy.ndarray
    """
    # first lets compute the proximity matrix
    dataset_without_cluster = dataset[dataset.columns.drop("CLUSTER")].values
    proximity_matrix = np.array([[np.linalg.norm(row - other) for other in dataset_without_cluster] for row in dataset_without_cluster])

    # then get the kNN for each data point, remove it self
    kNN = [np.argsort(arr)[1:k + 1] for arr in proximity_matrix]

    # then for each point we remove its neighbors that are in the same cluster
    assignment = list(map(int, dataset["CLUSTER"].values))

    kNN = np.array(list(map(lambda i_data: list(filter(lambda idx: assignment[idx] != assignment[i_data], kNN[i_data])), range(len(kNN)))))

    return kNN


def cvnn_seperation(dataset, cluster_num, k):
    """
    Calculate the degree of seperation of current clustering
    1. for each object in each cluster, find out whether at least one of its kNNs is in other clusters
    2. for objects with positive answers, assign a weight to each of them
    3. alculate the average weight of objects in the same cluster
    4. take the maximum average weight among all clusters as the intercluster separation

    :param dataset:
    :type dataset: pandas.DataFrame
    :param cluster_num: number of cluster
    :type cluster_num: int
    :param k: the k value for kNN
    :type k: int
    :return: the seperation index of this clustering
    :rtype: float
    """
    kNN = compute_kNN(dataset, k)
    # do it for a single cluster
    assignment = np.array(list(map(int, dataset["CLUSTER"].values)))
    counter = collections.Counter(assignment)

    ret = max(1 / counter[i_cluster] * sum(map(lambda x: len(x) / k, kNN[np.where(assignment == i_cluster)[0]])) for i_cluster in range(cluster_num))
    return ret


def cvnn_compactness(dataset, cluster_num):
    """
    Compute the compactness of the given clustering result

    :param dataset:
    :type dataset:
    :param cluster_num:
    :type cluster_num:
    :return: the lower, the better
    :rtype: float
    """
    dataset_without_cluster = dataset[dataset.columns.drop("CLUSTER")].values
    assignment = np.array(list(map(int, dataset["CLUSTER"].values)))
    counter = collections.Counter(assignment)
    proximity_matrix = np.array([[np.linalg.norm(row - other) for other in dataset_without_cluster] for row in dataset_without_cluster])

    sum_ans = 0
    for i_cluster in range(cluster_num):
        scalar = 2 / counter[i_cluster] * (counter[i_cluster] - 1)
        idx = np.where(assignment == i_cluster)[0]

        sum_matrix = proximity_matrix[idx][:, idx].sum() / 2
        sum_ans += scalar * sum_matrix

    return sum_ans


def cvnn(datasets, cluster_nums, k):
    """
    Clustering Validation Index based on Nearest Neighbors
    Compute the CVNN for a given dataset among a range of cluster nums

    :param datasets: the dataset, each row is the data + cluster
    :type datasets: list[pandas.DataFrame]
    :param cluster_nums: number of cluster in this dataset
    :type cluster_nums: list[int]
    :param k: number of nearest neighbors
    :type k: int
    :return: The CVNN index, the lower the better
    :rtype: float
    """
    print("CVNN, ", cluster_nums, k)

    list_sep = []
    list_com = []

    for i_cn in range(len(cluster_nums)):
        print("Loop, cluster ", cluster_nums[i_cn])
        dataset = datasets[i_cn]
        cluster_num = cluster_nums[i_cn]
        sep = cvnn_seperation(dataset, cluster_num, k)
        com = cvnn_compactness(dataset, cluster_num)

        print("Sep: {}, Com: {}".format(sep, com))

        list_sep.append(sep)
        list_com.append(com)

    max_sep = max(list_sep)
    max_com = max(list_com)

    return [((sep / max_sep) if max_sep != 0 else 0) + com / max_com for sep, com in zip(list_sep, list_com)]


def tabulate_cvnn(datasets, cluster_nums, k_vals):
    """
    CVNN on multiple datasets
    :param datasets: list of dataset
    :type datasets: list[pandas.DataFrame]
    :param cluster_nums: list of numbers of cluster for each dataset
    :type cluster_nums: list[int]
    :param k_vals: list of k for CVNN
    :type k_vals: list[int]
    :return: A DataFrame corresponding to the results
    :rtype: pandas.DataFrame
    """
    # Implement.

    # Inputs:
    # datasets: DataFrames Your provided list of clustering results.
    # cluster_nums: A list of integers corresponding to the number of clusters
    # in each of the datasets, e.g.,
    # datasets = [np.darray, np.darray, np.darray]
    # cluster_nums = [2, 3, 4]

    # Return a pandas DataFrame corresponding to the results.

    y = [cvnn(datasets, cluster_nums, k) for k in k_vals]

    k = np.ravel([[each] * len(cluster_nums) for each in k_vals])
    cluster_nums = cluster_nums * len(k_vals)
    y = np.ravel(y)
    return pd.DataFrame({"CLUSTERS": cluster_nums, "CVNN": y, "K": k})


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
        if 'CENTROID' in datasets[0].index:
            self.datasets = list(map(lambda df: df.drop('CENTROID', axis=0), datasets))
        else:
            self.datasets = datasets
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
        _plot_silhouette_(self.silhouette_table, save=True, n=name)

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
