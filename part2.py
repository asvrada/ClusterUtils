"""
Code to solve Part 2 questions
"""

from ClusterUtils import InternalValidator
from ClusterUtils import KMeans

# 3. Use the internal measurements in Part 1 to find the proper cluster number for the image segmentation.csv dataset.

# to achieve this, we use Silhouette Index

km = KMeans(init="global", csv_path="./Datasets/image_segmentation.csv", n_init=3, verbose=True)

dfs = []
cs = []
for i in range(2, 9):
    cs.append(i)
    km.n_clusters = i
    dfs.append(km.fit_predict_from_csv())

iv = InternalValidator(dfs, cluster_nums=cs)
iv.make_silhouette_table()
iv.show_silhouette_plot()

print("The best number of cluster is 2")
