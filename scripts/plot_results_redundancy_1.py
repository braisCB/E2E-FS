import json
import numpy as np
import os
import glob
from matplotlib import pyplot as plt


def main(dataset):
    dataset_name = dataset.split('/')[-1]
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../')

    directory = os.path.dirname(os.path.realpath(__file__)) + '/' + dataset + '/info/'
    image_directory = os.path.dirname(os.path.realpath(__file__)) + '/images/'
    files = glob.glob(directory + '*.json')

    BA_means = {}

    print(os.getcwd())
    for file in files:
        fs_class = file.split('.')[-2].split('_')[-1]
        with open(file, 'r') as outfile:
            stats = json.load(outfile)
        n_features = np.asarray(stats['classification']['n_features'])
        for key in ['real_feats']:
            if key not in stats['classification']:
                continue
            BA = np.asarray(stats['classification'][key])
            if BA.ndim == 2 and BA.shape[1] == 1:
                BA = BA.flatten()
            BA_mean = BA.mean(axis=-1)
            if isinstance(BA_mean, float):
                BA_mean = np.asarray([BA_mean] * 5)
            BA_means[fs_class] = BA_mean
            print('method : ', fs_class)
            print('score', key, ' : ', BA_mean)

    unsrt_keys = list(sorted(BA_means.keys()))
    keys = []
    for k in unsrt_keys:
        if 'E2E' not in k:
            keys.append(k)
    for k in unsrt_keys:
        if 'E2E' in k:
            keys.append(k)

    x = 0 # np.arange(len(n_features))
    # the label locations
    width = 1. # / (len(keys) + 1.)  # the width of the bars

    fig, ax = plt.subplots()
    rects = []
    for i, key in enumerate(keys):
        rects.append(ax.bar(i, BA_means[key][0], width))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('% real features', fontsize=18)
    ax.set_title('Redundancy test', fontsize=18)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, fontsize=14)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    ax.set_aspect('auto')
    # ax.legend(fontsize=18)
    ax.figure.set_size_inches(15,10)
    plt.xticks(rotation=90)

    # def autolabel(rects):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')
    #
    # autolabel(rects1)
    # autolabel(rects2)

    fig.tight_layout()

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
    plt.savefig(image_directory + dataset_name + '_redundancy_test.png')
    # plt.show()


if __name__ == '__main__':
    dataset = 'redundancy/redundancy_1'
    main(dataset)

