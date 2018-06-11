import retro
import matplotlib

from small_evo.wrappers import AutoRenderer

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from reward_plot import RewardPlotter, make_axes

movie = retro.Movie('/home/peer/mario/1-1_basic_5.fm2')

env = retro.make("SuperMarioBros-Nes", None, use_restricted_actions=retro.Actions.ALL)
env.initial_state = movie.get_state()
env.reset()
env = AutoRenderer(env, 1)

axes = make_axes()
plotter = RewardPlotter(axes, save_period=4, render_period=20, max_entries=800, sum_reward_up=True)

frame_counter = 0
while movie.step():
    keys = []
    for i in range(env.unwrapped.num_buttons):
        keys.append(movie.get_key(i, 0))
    _obs, _rew, _done, _info = env.step(keys)
    plotter.update(_rew, frame_counter)
    # if frame_counter % 2 == 0:
    #     env.render()
    frame_counter += 1

env.close()
plt.close()
