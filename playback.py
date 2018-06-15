import retro
import matplotlib

from small_evo.wrappers import AutoRenderer

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from reward_plot import RewardPlotter, make_axes

movie_name = 'movies/1-1_basic_5.fm2'
movie = retro.Movie(movie_name)

env = retro.make("SuperMarioBros-Nes", retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
env.reset()
env.reset()
env = AutoRenderer(env, 1)

axes = make_axes()
plotter = RewardPlotter(axes, save_period=4, render_period=20, max_entries=3000, sum_reward_up=True)

frame_counter = 0
rewards = []
while movie.step():
    keys = []
    for i in range(env.unwrapped.num_buttons):
        keys.append(movie.get_key(i, 0))
    _obs, rew, _done, _info = env.step(keys)
    rewards.append(rew)
    plotter.update(rew, frame_counter)
    # if frame_counter % 2 == 0:
    # env.render()
    frame_counter += 1
print("total reward: {}".format(sum(rewards)))
# pandas.Series(rewards).plot()

env.render("close")
env.close()
plt.savefig(movie_name[:-3] + "pdf")
plt.close()
