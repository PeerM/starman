# coding: utf-8

# In[59]:


from forked_lib.parse_fm2 import parse_fm2
import pandas
import matplotlib.pyplot as plt
from operator import getitem
from functools import partial


# In[2]:

def movie_to_distribution(path, filename, reduce_to=7):
    action_readable_count = movie_to_action_count(path, filename)

    action_readable_count_reduced = action_readable_count[:reduce_to] + [("%i Others" % (len(action_readable_count) - reduce_to),
                                                                          sum(list(zip(*action_readable_count[reduce_to:]))[1]))]

    action_readable_count_df = pandas.DataFrame(action_readable_count_reduced)

    plot = plt.pie(action_readable_count_df[1], labels=action_readable_count_df[0])
    plt.savefig("dist_" + filename[:-3] + "pdf", bbox_inches='tight')
    return plot


# In[3]:

def movie_to_action_count(path, filename):
    with open(path + filename) as movie_file:
        movie_inputs = pandas.DataFrame.from_dict(parse_fm2(movie_file))

    button_order = list(movie_inputs.iloc[1].index)

    button_weightings = [2 ** i for i in range(8)]
    inputs_encoded = movie_inputs.apply(axis=1, func=lambda series: sum(series.values * button_weightings))

    def decode_action(action):
        readable = []
        for i in range(8):
            if action & 2 ** i:
                readable.append(button_order[i])
        return readable

    action_readable_count = [("+".join(decode_action(encoded_action)), count) for encoded_action, count in inputs_encoded.value_counts().iteritems()]
    action_readable_count = [(action_str if action_str != "" else "NOOP", count) for action_str, count in action_readable_count]
    return action_readable_count


# In[4]:
movie_to_distribution("/home/peer/PycharmProjects/starman/movies/", "HL_SMBwarpless_V4.fm2", 7)
plt.clf()
# In[4]:
movie_to_distribution("/home/peer/PycharmProjects/starman/movies/", "1-1_basic_5.fm2", 6)
plt.clf()
# In[4]:
movie_to_distribution("/home/peer/PycharmProjects/starman/movies/", "happylee-supermariobros,warped.fm2", 5)
plt.clf()
# In[5]:
movie_to_action_count("/home/peer/PycharmProjects/starman/movies/", "11_1-1_small_walking.fm2")
