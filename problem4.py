import numpy as np
import math


def purity_mat(matrix):
    return sum([max(matrix[:, i_col]) for i_col in range(matrix.shape[1])]) / sum(sum(matrix))


def purity_row(row):
    return max(row) / sum(row)


def entropy_1d(array):
    size_x = len(array)
    array_sum = sum(array)

    def pi(x):
        return array[x] / array_sum

    sum_outer = 0
    for x in range(size_x):
        if array[x] == 0:
            continue
        sum_outer += pi(x) * math.log(pi(x), 2)

    return -sum_outer


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


if __name__ == '__main__':
    data = [8, 22, 0, 0, 767, 4, 45, 22,
            654, 34, 89, 123, 12, 76, 13, 2,
            6, 301, 2, 3, 98, 23, 31, 1001,
            4, 21, 34, 2, 3, 543, 112, 0]

    data = np.array(data).reshape((4, 8))
    # print(type(data[0]))

    # Entropy
    for cluster in range(data.shape[0]):
        print("{:.2f} & {:.2f}".format(entropy_1d(data[cluster]), purity_row(data[cluster])))

    print(entropy_2d(np.array(data)))
    print(purity_mat(data))
