from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

def mean_distance_filter(keypoints,plot_filter=False,min_distance=2):
    if plot_filter:
        fig, ax = plt.subplots()
    keys = keypoints.copy()
    dist_f = cdist(keys, keys)  # init
    MAX= 2147483647
    np.fill_diagonal(dist_f,MAX)

    while dist_f.min() <= min_distance:
        key_1, key_2 = np.argwhere(dist_f == np.min(dist_f))[0]
        x_1, y_1 = keys[key_1]
        x_2, y_2 = keys[key_2]
        new_point = [(x_1+x_2)/2,(y_1+y_2)/2]
        if plot_filter:
            ax.plot([int(keys[key_1][0]), new_point[0]], [int(keys[key_1][1]), new_point[1]], marker='o',
                    linestyle=':', color="red")
            ax.plot([int(keys[key_2][0]), new_point[0]], [int(keys[key_2][1]), new_point[1]], marker='o',
                    linestyle=':', color="red")
        keys.append(new_point)
        keys.remove(keys[key_1])
        keys.remove(keys[key_2])
        dist_f = cdist(keys, keys)  # init
        np.fill_diagonal(dist_f, MAX)

    if plot_filter:
        print("corrected: ",len(keypoints)-len(keys),' from ',len(keypoints),' keypoints')
        x, y = zip(*keys)
        x_o, y_o = zip(*keypoints)
        ax.scatter(x_o, y_o,color='green')
        ax.scatter(x, y,alpha=0.5,color='red')
        plt.show()
    return keys