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
        embedded = self.embedder.transform(np.array(obs).flatten().reshape(1, -1))[0]
        return embedded, reward, done, info
