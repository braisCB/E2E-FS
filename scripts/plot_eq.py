import json
import numpy as np
import os
import glob
from matplotlib import pyplot as plt, cm
from mpl_toolkits import mplot3d


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../')
    image_directory = os.path.dirname(os.path.realpath(__file__)) + '/images/'

    x = np.linspace(0, 1, 100)
    y = x.copy()

    xx, yy = np.meshgrid(x, y)

    r1 = xx * (1. - xx) + yy * (1. - yy)
    r2 = r1 + np.abs(xx + yy - 1)
    r3 = r2 + np.square(xx - 1.)

    plt.figure()
    ax = plt.axes(projection='3d')
    # ax.view_init(20, 0)
    ax.contour3D(xx, yy, r1, 50)
    ax.autoscale_view('tight')

    # markers = ['o', 'v', '^', 'p', '*', '+', 'x', '<', '>']
    # plt.figure()
    # for i, key in enumerate(keys):
    #     plt.plot(n_features, BA_means[key], marker=markers[i % len(markers)], linewidth=2)
    # plt.legend(keys, loc='best', fontsize=12)
    # plt.title(dataset_name.upper(), fontsize=14)
    # plt.xlabel('# features')
    # plt.ylabel('time (s)')
    # if not os.path.isdir(image_directory):
    #     os.makedirs(image_directory)
    plt.savefig(image_directory + 'convex.png')
    # plt.show()


if __name__ == '__main__':
    main()

