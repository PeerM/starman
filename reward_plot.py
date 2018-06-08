from forked_lib.realtime_plot import RealtimePlot
import matplotlib.pyplot as plt


def make_axes():
    fig, axes = plt.subplots()
    return axes


class RewardPlotter(object):
    def __init__(self, axes, save_period=10, render_period=60, max_entries=100):
        self.render_period = render_period
        self.save_period = save_period
        self.step = 0
        self.axes = axes
        self.display = RealtimePlot(axes, max_entries=max_entries)

    def update(self, total_reward, step=None):
        if step is not None:
            self.step = step
        else:
            step = self.step
        if step % self.save_period == 0:
            self.display.add(step, total_reward)
        if step % self.render_period == 0:
            plt.pause(0.00001)

    def render(self):
        plt.pause(0.1)
