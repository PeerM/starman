from scipy.optimize import differential_evolution

from agent import Agent
from embedding import load
import numpy as np
import retro

from atari_wrappers import FrameStack
from small_evo.wrappers import ForrestAndFlatten

raw_env = retro.make("SuperMarioBros-Nes")
embedder = load()
env = ForrestAndFlatten(FrameStack(raw_env, 4), embedder)

index_right = raw_env.buttons.index("RIGHT")
index_a = raw_env.buttons.index("A")
index_b = raw_env.buttons.index("B")
first_obs = env.reset()
first_obs, _, _, _ = env.step([0] * 9)
env.reset()
# plt.imshow(first_obs)
# plt.show()
default_agent = Agent(4, first_obs.shape[0], deterministic=True)
weight_shape = default_agent.weight_shape()
nr_weights = weight_shape[0] * weight_shape[1]
print(nr_weights)


def do_run(seq_weights, *args):
    # print("next")
    weights = np.reshape(seq_weights, weight_shape)
    agent = Agent(4, first_obs.shape[0], deterministic=True)
    agent.change_weights(weights)
    obs = env.reset()  # TODO reset needs to return embedding
    obs, _, _, _ = env.step([0] * 9)
    reward_sum = 0
    for i in range(2000):
        action_numeric = agent.get_action(obs)
        action = [0] * 9
        action[index_right] = 1 if action_numeric & 0b10 > 0 else 0
        action[index_b] = 1
        action[index_a] = 1 if action_numeric & 0b01 > 0 else 0
        obs, reward, end_of_ep, something = env.step(action)
        reward_sum += reward
        if i % 200 == 0:
            env.render()
        # plt.imshow(obs)
        # plt.show()
        if end_of_ep:
            break
    print(reward_sum)
    return 6000 - reward_sum


def log_status(xk, val):
    print("{},{}".format(xk, val))


# differential_evolution(do_run, np.tile([-1, 1], (nr_weights, 1)))
result = differential_evolution(do_run, [(-1, 1)] * nr_weights, disp=True, callback=log_status)
print("fun:{}".format(result.fun))
np.save("evolution_result.npy", result.x)
