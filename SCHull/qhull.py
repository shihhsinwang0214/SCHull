from abc import ABCMeta

import numpy as np
from scipy.spatial import ConvexHull

class Qhull(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_chull_graph(self, data, rank, n, *args, **kwargs):
        if n==1:
            return np.array([])
        elif n==2:
            return np.array([[0,1]])
        elif n==3:
            return np.array([[0,1],[1,2],[2,0]])
        else:
            hull =  ConvexHull(data,qhull_options='QJ')
            if hull.simplices[0].shape[0]==3:
                edges = set()
                for simplex in hull.simplices:
                    for i in range(3):
                        edge = sorted([simplex[i], simplex[(i + 1) % 3]])
                        edges.add(tuple(edge))
                return np.array(list(edges))
            else:
                return hull.simplices
