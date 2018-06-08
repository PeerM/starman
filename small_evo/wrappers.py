import gym
import numpy as np


class ForrestAndFlatten(gym.Wrapper):
    def __init__(self, env, embedder):
        """"""
        gym.Wrapper.__init__(self, env)
        self.embedder = embedder

    def _step(self, action):
        """"""
        obs, reward, done, info = self.env.step(action)
        embedded = self.embedder.transform(np.array(obs).flatten().reshape(1, -1))[0].todense().T
        return embedded, reward, done, info


class AutoRenderer(gym.Wrapper):
    def __init__(self, env, auto_render_period=60):
        """"""
        gym.Wrapper.__init__(self, env)
        self.auto_render_period = auto_render_period
        self.counter = 1

    def step(self, action):
        """"""
        if self.counter % self.auto_render_period == 0:
            self.counter = 1
            self.env.render()
        else:
            self.counter += 1
        return self.env.step(action)

    def reset(self):
        self.counter = 1
        return self.env.reset()
