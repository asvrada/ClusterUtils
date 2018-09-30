import numpy as np
import collections


def silhouette(matrix, cluster_label):
    def d(x, y):
        return matrix[x][y]

    def a(cluster, i):
        divider = len(matrix_cluster[cluster]) - 1
        if divider == 0:
            # there is only 1 element in this cluster
            return 0
        return sum(d(i, data) for data in matrix_cluster[cluster]) / divider

    def b(cluster, i):
        ave_dist_clusters = []
        for i_cluster in set(cluster_label):
            if i_cluster == cluster:
                continue
            # distance to all data points in another cluster
            if len(matrix_cluster[i_cluster]) == 0:
                continue
            dist = sum(d(i, data) for data in matrix_cluster[i_cluster]) / len(matrix_cluster[i_cluster])
            ave_dist_clusters.append(dist)

        return min(ave_dist_clusters)

    def s(cluster, i):
        ret_a = a(cluster, i)
        ret_b = b(cluster, i)

        if abs(ret_a - ret_b) <= 1e-10:
            return 0

        return (ret_b - ret_a) / max(ret_a, ret_b)

    # build a tmp matrix
    # point index begin with 0
    matrix_cluster = collections.defaultdict(list)
    for i in range(len(cluster_label)):
        label = cluster_label[i]
        matrix_cluster[label].append(i)

    tmp_sum = 0
    tmp_n = 0
    N = len(cluster_label)
    size_x, size_y = matrix.shape
    for i_point in range(N):
        tmp = s(cluster_label[i_point], i_point)
        print(tmp)
        pass
    # for i_cluster in range(cluster_num):
    #     tmp_sum += sum(s(i_cluster, i) for i in dataset_converted[i_cluster])
    #     tmp_n += len(dataset_converted[i_cluster])

    # the average i over all data of the entire dataset
    # return tmp_sum / tmp_n


if __name__ == '__main__':
    data = [
        1, 0.92, 0.33, 0.61, 0.82
        , 0.92, 1, 0.43, 0.01, 0.22
        , 0.33, 0.43, 1, 0.75, 0.11
        , 0.61, 0.01, 0.75, 1, 0.17
        , 0.82, 0.22, 0.11, 0.17, 1
    ]

    data = np.array(data).reshape((5, 5))

    labels = [1, 1, 2, 2, 3]

    silhouette(data, labels)
