import retro
import random
from atari_wrappers import FrameStack

raw_env = retro.make("SuperMarioBros-Nes")
env = FrameStack(raw_env, 4)
first_obs = env.reset()
index_right = raw_env.buttons.index("RIGHT")
index_a = raw_env.buttons.index("A")
index_b = raw_env.buttons.index("B")
infos = []
for i in range(5000):
    action = [0] * 9
    action[index_right] = 1
    action[index_b] = 1
    action[index_a] = 1 if random.random() > 0.1 else 0
    obs, reward, end_of_episode, aditional_information = env.step(action)
    infos.append(aditional_information)
    if end_of_episode:
        print("dead")
        break
    if i % 15 == 0:
        env.render()
raw_env.render(close=True)
