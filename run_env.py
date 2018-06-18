import retro
import random

from gym.wrappers import Monitor

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
            break
raw_env.render(close=True)
