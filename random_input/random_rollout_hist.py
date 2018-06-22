import pandas
import matplotlib.pyplot as plt
reward_list =[349.0, 353.0, 90.0, 0.0, 99.0, 0.0, 150.0, 103.0, 172.0, 286.0, 20.0, 0.0, 62.0, 386.0, 0.0, 66.0, 284.0, 81.0, 0.0, 0.0, 34.0, 355.0, 0.0, 31.0, 0.0, 243.0, 5.0, 5.0, 0.0, 19.0, 0.0, 66.0, 0.0, 56.0, 354.0, 175.0, 168.0, 226.0, 0.0, 0.0]
reward_series = pandas.Series(reward_list)
reward_series.hist()
plt.savefig("random_input_rewards.pdf")