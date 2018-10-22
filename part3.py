import numpy as np
import pandas as pd

from ClusterUtils import DBScan
from ClusterUtils import ExternalValidator

if __name__ == '__main__':
    db = DBScan(eps=0.3, min_points=10, csv_path='./Datasets/rockets.csv')
    data = db.fit_predict_from_csv()

    ev = ExternalValidator(data)
    nmi = ev.normalized_mutual_info()
    nri = ev.normalized_rand_index()
    a = ev.accuracy()

    print([nmi, nri, a])
