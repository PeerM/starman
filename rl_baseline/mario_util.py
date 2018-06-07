"""
Environments and wrappers for Sonic training.
Fork of openai retro-baselines
"""

import gym
import numpy as np
import retro
from gym.wrappers import Monitor, TimeLimit

from atari_wrappers import WarpFrame, FrameStack
from small_evo.wrappers import AutoRenderer


def make_env(stack=True, scale_rew=True, render=None, monitor=None, timelimit=False):
    """
    Create an environment with some standard wrappers.
    """
    env = retro.make("SuperMarioBros-Nes")
    env = MarioDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    if timelimit:
        env = TimeLimit(env, max_episode_steps=200 * 60)
    if monitor is not None:
        env = Monitor(env, monitor, write_upon_reset=True)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    if render is not None:
        env = AutoRenderer(env, auto_render_period=render)
    return env


class MarioDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env):
        super(MarioDiscretizer, self).__init__(env)
        buttons = env.buttons
        # buttons = ["B", "A", "START", "UP", "DOWN", "LEFT", "RIGHT"]
        actions = [['A'], ['B'], ['LEFT'], ['RIGHT'], ['DOWN'],
                   ['LEFT', 'A'], ['LEFT', 'B'],
                   ['RIGHT', 'A'], ['RIGHT', 'B'],
                   ['DOWN', 'A'], ['DOWN', 'B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):  # pylint: disable=W0221
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """

    def reward(self, reward):
        return reward * 0.01


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """

    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):  # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info
