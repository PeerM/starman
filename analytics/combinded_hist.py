from itertools import combinations

import pandas
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn
import os


stats_file_random = "results/random/7/openaigym.episode_batch.0.8865.stats.json"
stats_file_rainbow = "results/rainbow/7/val_monitor/2/openaigym.episode_batch.0.22641.stats.json"
stats_file_unbiased_random = "results/unbiased_random/1/openaigym.episode_batch.0.5398.stats.json"


def load_reward_series(stats_file):
    stats = json.load(open(stats_file))
    reward_list = stats["episode_rewards"]
    return pandas.Series(reward_list)


# def multihist(a, b):
#     fig, ax = plt.subplots()
#
#     start_bins = min(min(a), min(b))
#     end_bins = max(max(a), max(b))
#     range_bins = [start_bins, end_bins]
#     heights = []
#     bins = []
#     a_heights, a_bins = np.histogram(a, range=range_bins, bins=13, normed=True)
#     b_heights, b_bins = np.histogram(b, bins=a_bins, normed=True)
#
#     width = (a_bins[1] - a_bins[0]) / 2.5
#
#     ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
#     ax.bar(b_bins[:-1] + width, b_heights, width=width, facecolor='seagreen')
#     seaborn.despine(ax=ax, offset=10)


def multihist_three(a_tuple, b_tuple, c_tuple=None):
    fig, ax = plt.subplots()
    a = a_tuple[1]
    b = b_tuple[1]
    c_tuple_available = c_tuple is not None
    if c_tuple_available:
        c = c_tuple[1]

    start_bins = min(min(a), min(b), min(c) if c_tuple_available else 100000)
    end_bins = max(max(a), max(b), max(c) if c_tuple_available else 0)
    range_bins = [start_bins, end_bins]
    heights = []
    bins = []
    a_heights, a_bins = np.histogram(a, range=range_bins, bins=8, density=True)
    # a_heights, a_bins = np.histogram(a, range=range_bins, bins=20, normed=True)
    b_heights, b_bins = np.histogram(b, bins=a_bins, density=True)

    if c_tuple_available:
        spacing = 3.5
    else:
        spacing = 2.5
    width = (a_bins[1] - a_bins[0]) / spacing

    ax.bar(a_bins[:-1], a_heights, width=width, facecolor=a_tuple[2])
    ax.bar(b_bins[:-1] + width, b_heights, width=width, facecolor=b_tuple[2])
    if c_tuple_available:
        c_heights, c_bins = np.histogram(c, bins=a_bins, density=True)
        ax.bar(c_bins[:-1] + width * 2, c_heights, width=width, facecolor=c_tuple[2])
    seaborn.despine(ax=ax, offset=10)


reward_series_random = ("biased", load_reward_series(stats_file_random), "cornflowerblue")
reward_series_rainbow = ("rainbow", load_reward_series(stats_file_rainbow), 'seagreen')
reward_series_unbiased_random = ("unbiased", load_reward_series(stats_file_unbiased_random), "sandybrown")
series = [reward_series_rainbow, reward_series_random, reward_series_unbiased_random]
os.chdir("analytics/figures/reward_hist")
for a_tuple, b_tuple in combinations(series, 2):
    multihist_three(a_tuple, b_tuple)
    plt.savefig("{}_{}.pdf".format(a_tuple[0], b_tuple[0]))
    plt.clf()


multihist_three(reward_series_random, reward_series_rainbow, reward_series_unbiased_random)
# reward_series_random.hist(normed=True, color="green")
# reward_series_rainbow.hist(normed=True, color="blue")
# plt.show()
plt.savefig("all_hists.pdf")
