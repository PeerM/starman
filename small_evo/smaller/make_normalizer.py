import pickle

import retro
import random

from gym.wrappers import Monitor
from sklearn import preprocessing

from atari_wrappers import FrameStack, EpisodicLifeEnv
from rl_baseline.mario_util import make_env

# raw_env = retro.make("SuperMarioBros-Nes")
from small_evo.wrappers import AutoRenderer

monitor = None
action_repeat = True
episodic_life = True
render = 2
env = retro.make("SuperMarioBros-Nes")
if monitor is not None:
    env = Monitor(env, monitor)
if render is not None:
    env = AutoRenderer(env, auto_render_period=render)
if action_repeat:
    env = FrameStack(env, 4)
if episodic_life:
    env = EpisodicLifeEnv(env, [0] * 9)
raw_env = env.unwrapped
# env = AllowBacktracking(make_env(stack=False, scale_rew=False))
first_obs = env.reset()
order = ["coins", "levelHi", "levelLo", "lives", "score", "scrolling", "time", "xscrollHi", "xscrollLo"]

index_right = raw_env.buttons.index("RIGHT")
index_a = raw_env.buttons.index("A")
index_b = raw_env.buttons.index("B")
infos = []
for episode_i in range(3):
    env.reset()
    for i in range(5000):
        action = [0] * 9
        action[index_right] = 1
        action[index_b] = 1
        # action[index_a] = 0
        action[index_a] = 1 if random.random() > 0.1 else 0
        obs, reward, done, aditional_information = env.step(action)
        infos.append(aditional_information)
        if done:
            print("episode done")
            env.reset()

scaler = preprocessing.StandardScaler().fit(list(map(lambda info: [info[name] for name in order], infos)))

raw_env.render(close=True)

with open("scaler.pickle", "wb") as pickle_out_file:
    pickle.dump(scaler, pickle_out_file)
