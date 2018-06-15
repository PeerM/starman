import retro
import matplotlib

from forked_lib.parse_fm2 import parse_fm2
from small_evo.wrappers import AutoRenderer

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from reward_plot import RewardPlotter, make_axes

movie_name = '/home/peer/mario/1-1_shortcut_2.fm2'
playback_offset = 0
retro_state = retro.State.NONE
with open(movie_name) as movie_file:
    inputs = list(parse_fm2(movie_file))

env = retro.make("SuperMarioBros-Nes", retro_state, use_restricted_actions=retro.Actions.ALL)
env = AutoRenderer(env, 1)
env.reset()
env.reset()

axes = make_axes()
plotter = RewardPlotter(axes, save_period=6, render_period=20, max_entries=3000, sum_reward_up=True)

rewards = []
buttons = [name.upper() if name is not None else None for name in env.unwrapped.buttons]
for frame_counter, input_dict in enumerate(inputs[playback_offset:]):
    keys = [input_dict[name] if name is not None else False for name in buttons]
    # for i in range(env.unwrapped.num_buttons):
    #     keys.append(movie.get_key(i, 0))
    _obs, rew, _done, _info = env.step(keys)
    rewards.append(rew)
    plotter.update(rew, frame_counter)
print("total reward: {}".format(sum(rewards)))
# pandas.Series(rewards).plot()

env.render("close")
env.close()
plt.savefig(movie_name[:-3] + "pdf")
with open(movie_name[:-3] + "info", "a+") as info_file:
    info_file.write("total reward: {}\n".format(sum(rewards)))
plt.close()
