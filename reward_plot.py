from forked_lib.realtime_plot import RealtimePlot
import matplotlib.pyplot as plt


def make_axes():
    fig, axes = plt.subplots()
    return axes


class RewardPlotter(object):
    def __init__(self, axes, save_period=10, render_period=60, max_entries=100, sum_reward_up=False):
        self.render_period = render_period
        self.save_period = save_period
        self.step = 0
        self.axes = axes
        self.display = RealtimePlot(axes, max_entries=max_entries)
        self.sum_reward = 0
        self.sum_reward_up = sum_reward_up

    def update(self, reward, step=None):
        if step is not None:
            self.step = step
        else:
            step = self.step
        if self.sum_reward_up:
            self.sum_reward += reward
            reward = self.sum_reward
        if step % self.save_period == 0:
            self.display.add(step, reward)
        if step % self.render_period == 0:
            plt.pause(0.00001)

    def render(self):
        plt.pause(2)

    def clear(self):
        plt.clf()

    def close(self):
        plt.close()
