from sklearn.datasets import make_classification
import numpy as np
from knn import KNN
import matplotlib.pyplot as plt

def draw_graph():
    dist = []
    dist2 = []
    win = []
    win2 = []
    data1 = []
    data0 = []
    tab = {}
    k = 4

    X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)

    tab['data'] = X1
    tab['target'] = Y1

    knn = KNN(tab)
    nearest = knn.findKNearest(np.zeros(2), k)

    for i, e in enumerate(tab['data']):
        dist.append([((e[0]**2 + e[1]**2) ** 0.5), tab['target'][i], i])

    dist = sorted(dist, key=lambda x: x[0])

    for i, e in enumerate(dist):
        if e[1]:
            data1.append(e[2])
        else:
            data0.append(e[2])

    for i, e in enumerate(dist[:k]):
        if e[1]:
            win.append(e[2])
        else:
            win2.append(e[2])
        dist2.append(e[0:2])

    plt.plot(tab['data'][data0, 0], tab['data'][data0, 1], 'ro', mew=1.5, ms=1.5) #ms, mw length and width
    plt.plot(tab['data'][data1, 0], tab['data'][data1, 1], 'go', mew=1.5, ms=1.5)
    plt.plot(tab['data'][[win], 0], tab['data'][[win], 1], 'bo', mew=1.5, ms=1.5)
    plt.plot(tab['data'][[win2], 0], tab['data'][[win2], 1], 'bo', mew=1.5, ms=1.5)
    circle = plt.Circle((0, 0), dist[k-1][0], color='black', fill=False, linewidth=1)
    ax = plt.gca()
    ax.add_patch(circle)
    plt.ylim(-3, 3) #hardcode y length
    plt.xlim(-3, 3)
    plt.show()

draw_graph()