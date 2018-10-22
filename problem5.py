import numpy as np
import pandas as pd
import collections


def silhouette(proximity_matrix, cluster_labels):
    def d(x, y):
        return proximity_matrix[x][y]

    def a(cluster, i):
        divider = len(dataset[cluster]) - 1
        if divider == 0:
            # there is only 1 element in this cluster
            return 0
        # todo
        return sum(d(i, i_point) for i_point in dataset[cluster]) / divider

    def b(cluster, i):
        ave_dist_clusters = []
        # loop through clusters
        for lop_cluster in dataset:
            if lop_cluster == cluster:
                continue
            # if there is no data point in a cluster, skip
            if len(dataset[lop_cluster]) == 0:
                continue

            # distance to all data points in another cluster
            dist = sum(d(i, i_point) for i_point in dataset[lop_cluster]) / len(dataset[lop_cluster])
            ave_dist_clusters.append(dist)

        return min(ave_dist_clusters)

    def s(cluster, i):
        ret_a = a(cluster, i)
        ret_b = b(cluster, i)

        # if 0
        if abs(ret_a - ret_b) <= 1e-10:
            return 0

        return (ret_b - ret_a) / max(ret_a, ret_b)

    dataset = collections.defaultdict(list)

    for i_label in range(len(cluster_labels)):
        dataset[cluster_labels[i_label]].append(i_label)

    each_points = [s(cluster, i) for i, cluster in enumerate(cluster_labels)]
    print(each_points)

    tmp_sum = 0
    for cluster in dataset:
        # for each cluster, sum the s index of all data points in this cluster, then divided by length of this cluster
        tmp_sum += sum(s(cluster, i) for i in dataset[cluster]) / len(dataset[cluster])

    print("All data: ", tmp_sum / len(dataset))
    return tmp_sum / len(dataset)


if __name__ == '__main__':
    data = [
        1, 0.92, 0.33, 0.61, 0.82
        , 0.92, 1, 0.43, 0.01, 0.22
        , 0.33, 0.43, 1, 0.75, 0.11
        , 0.61, 0.01, 0.75, 1, 0.17
        , 0.82, 0.22, 0.11, 0.17, 1
    ]

    data = np.array(data).reshape((5, 5))

    labels = [0, 0, 1, 1, 2]

    # turn the similarity matrix into proximity one
    # 1. set diagonal as 0
    np.fill_diagonal(data, 0)
    # 2. revert sign of elements
    data *= -1
    # 3. find the smallest and substract it from each element
    data -= np.min(data)
    # 4. set diagonal as 0
    np.fill_diagonal(data, 0)

    silhouette(data, labels)
