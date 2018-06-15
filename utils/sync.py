import retro
import matplotlib

from forked_lib.parse_fm2 import parse_fm2
from small_evo.wrappers import AutoRenderer

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from reward_plot import RewardPlotter, make_axes

movie_name = 'movies/6_1-1_with_shortcut.fm2'
with open(movie_name) as movie_file:
    inputs = list(parse_fm2(movie_file))

env = retro.make("SuperMarioBros-Nes", retro.State.DEFAULT, use_restricted_actions=retro.Actions.ALL)
env.reset()
env.reset()

rewards = []
buttons = [name.upper() if name is not None else None for name in env.unwrapped.buttons]
search_start = 280
search_end = 350
for offset in range(search_start, search_end, 1):
    env.reset()
    episode_reward = 0
    for frame_counter, input_dict in enumerate(inputs[offset:]):
        keys = [input_dict[name] if name is not None else False for name in buttons]
        # for i in range(env.unwrapped.num_buttons):
        #     keys.append(movie.get_key(i, 0))
        _obs, rew, _done, _info = env.step(keys)
        episode_reward += rew
    rewards.append(episode_reward)
    print("episode offset: {}; reward: {}".format(offset, episode_reward))
    # pandas.Series(rewards).plot()

print(rewards)
env.close()
# plt.savefig(movie_name[:-3] + "pdf")
with open(movie_name[:-3] + "info", "a+") as info_file:
    info_file.write("default state offset: {}; at: {}\n".format(rewards.index(max(rewards)) + search_start, max(rewards)))
plt.close()
