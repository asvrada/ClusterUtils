import numpy as np
import math

from ClusterUtils.ExternalValidator import entropy_1d


def purity_mat(matrix):
    return sum([max(matrix[:, i_col]) for i_col in range(matrix.shape[1])]) / sum(sum(matrix))


def purity_row(row):
    return max(row) / sum(row)


def entropy_2d(contingency):
    size_x, size_y = contingency.shape
    contingency_sum = contingency.sum()

    def pi(x):
        return contingency.sum(axis=1)[x] / contingency_sum

    def pij(x, y):
        return contingency[x][y] / contingency_sum

    sum_outer = 0
    for x in range(size_x):
        # inner loop

        sum_inner = 0
        for y in range(size_y):
            if contingency[x][y] == 0:
                continue
            sum_inner += pij(x, y) / pi(x) * math.log((pij(x, y) / pi(x)), 2)

        sum_outer += pi(x) * sum_inner

    return -sum_outer


def F_measure(contigency, i, j):
    contigency_sum = contigency.sum()

    def precision(i, j):
        return contigency[i][j] / contigency_sum

    def recall(i, j):
        return contigency[i][j] / contigency.sum(axis=0)[j]

    return (2 * precision(i, j) * recall(i, j)) / (precision(i, j) + recall(i, j))


if __name__ == '__main__':
    data = [8, 22, 0, 0, 767, 4, 45, 22,
            654, 34, 89, 123, 12, 76, 13, 2,
            6, 301, 2, 3, 98, 23, 31, 1001,
            4, 21, 34, 2, 3, 543, 112, 0]
    # data_la = [3, 5, 40, 506, 96, 27,
    #            4, 7, 280, 29, 39, 2,
    #            1, 1, 1, 7, 4, 671,
    #            10, 162, 3, 119, 73, 2,
    #            331, 22, 5, 70, 13, 23,
    #            5, 358, 12, 212, 48, 13]

    data = np.array(data).reshape((4, 8))
    # data_la = np.array(data_la).reshape((6, 6))

    # For question (a) and (b)
    # Entropy
    for cluster in range(data.shape[0]):
        print("{:.2f} & {:.2f}".format(entropy_1d(data[cluster]), purity_row(data[cluster])))

    print(entropy_2d(np.array(data)))
    # print(purity_mat(data))

    # For question (c)
    # print(F_measure(data, 2, 7))
    pass
