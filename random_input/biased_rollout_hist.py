import pandas
import matplotlib.pyplot as plt
import json
episode_stats =json.load(open("results/random/7/openaigym.episode_batch.0.8865.stats.json"))
reward_list = episode_stats["episode_rewards"]
reward_series = pandas.Series(reward_list)
reward_series.hist()
plt.savefig("biased_random_input_rewards.pdf")