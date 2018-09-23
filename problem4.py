import numpy as np
from ClusterUtils.ExternalValidator import entropy, find_accuracy


def purity_mat(matrix):
    return sum([max(matrix[:, i_col]) for i_col in range(matrix.shape[1])]) / sum(sum(matrix))


def purity_row(row):
    return max(row) / sum(row)


if __name__ == '__main__':
    data = [8, 22, 0, 0, 767, 4, 45, 22,
            654, 34, 89, 123, 12, 76, 13, 2,
            6, 301, 2, 3, 98, 23, 31, 1001,
            4, 21, 34, 2, 3, 543, 112, 0]

    data = np.array(data).reshape((4, 8))
    print(type(data[0]))

    # Entropy
    # for cluster in range(data.shape[0]):
    #     print("{:.2f} & {:.2f}".format(entropy(data[cluster]), purity_row(data[cluster])))

    print(entropy(data))
    print(purity_mat(data))
