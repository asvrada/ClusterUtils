"""
Code to solve Part 2 questions
"""

from ClusterUtils import InternalValidator
from ClusterUtils import ExternalValidator
from ClusterUtils import KMeans

import numpy as np


def problem2(dataset_path):
    """
    3. Use the internal measurements in Part 1 to find the proper cluster number for the image segmentation.csv dataset.
    """

    # to achieve this, we use Silhouette Index
    km = KMeans(init="k-mean++", csv_path=dataset_path, n_init=5)

    dfs = []
    cs = []
    for i in range(2, 9):
        cs.append(i)
        km.n_clusters = i
        dfs.append(km.fit_predict_from_csv())

    iv = InternalValidator(dfs, cluster_nums=cs)
    iv.make_silhouette_table()
    iv.show_silhouette_plot()

    iv.make_cvnn_table()
    iv.show_cvnn_plot()


def problem3(dataset_path, n_cluster):
    """
    4. Given the true cluster number, run your Lloyd’s K-means algorithm on the image segmentation.csv dataset, and evaluate the results in terms of the external measurements completed in Part I.
    """

    km = KMeans(init="k-mean++", algorithm="lloyds", csv_path=dataset_path, n_clusters=n_cluster, n_init=3, verbose=False)
    data = km.fit_predict_from_csv()
    # km.show_plot()

    ev = ExternalValidator(data)
    nmi = ev.normalized_mutual_info()
    nri = ev.normalized_rand_index()
    a = ev.accuracy()

    print([nmi, nri, a])
    return [nmi, nri, a]


if __name__ == '__main__':
    # dataset = "./Mini_Datasets/three_globs_mini.csv"
    dataset = "./Datasets/image_segmentation.csv"
    # dataset = "./Datasets/well_separated.csv"
    # problem2(dataset)
    n_cluster = 7

    ret = []
    for _ in range(10):
        ret.append(problem3(dataset, n_cluster))

    print(ret)
