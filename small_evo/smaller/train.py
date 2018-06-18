import pickle
import random
from itertools import count

import retro
import numpy as np
from gym.wrappers import Monitor, TimeLimit
from tqdm import tqdm

from multiprocessing import Pool
from atari_wrappers import FrameStack, EpisodicLifeEnv
from small_evo.agent import Agent
from small_evo.wrappers import AutoRenderer
from es import OpenES

order = ["coins", "levelHi", "levelLo", "lives", "score", "scrolling", "time", "xscrollHi", "xscrollLo"]


class Evaluator(object):
    def __init__(self, render=None, max_episode_steps=2000):
        monitor = None
        action_repeat = True
        episodic_life = True
        env = retro.make("SuperMarioBros-Nes")
        if monitor is not None:
            env = Monitor(env, monitor)
        if render is not None:
            env = AutoRenderer(env, auto_render_period=render)
        if action_repeat:
            env = FrameStack(env, 8)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        if episodic_life:
            env = EpisodicLifeEnv(env, [0] * 9)

        env.reset()
        _, _, _, first_info = env.step([0] * 9)  # TODO the order of the info dict is random
        self.first_info = first_info
        env.reset()
        self.env = env
        raw_env = env.unwrapped
        self.index_right = raw_env.buttons.index("RIGHT")
        self.index_a = raw_env.buttons.index("A")
        self.index_b = raw_env.buttons.index("B")
        self.obs_shape = len(first_info.values())
        self.agent = Agent(2, self.obs_shape, deterministic=True, embed=True)
        self.weight_shape = self.agent.weight_shape()
        self.n_weights = self.weight_shape[0] * self.agent.weight_shape()[1]
        with open("scaler.pickle", "rb") as pickle_out_file:
            self.scaler = pickle.load(pickle_out_file)

    def evaluate(self, seq_weights):
        weights = np.reshape(seq_weights, self.weight_shape)
        self.agent.change_weights(weights)
        done = False
        last_info = self.first_info
        reward_sum = 0
        while True:
            action = [0] * 9
            action[self.index_right] = 1
            action[self.index_b] = 1
            actual_obs = np.array([last_info[name] for name in order])  # TODO move this into a wrapper
            scaled_obs = self.scaler.transform([actual_obs])[0]
            action[self.index_a] = self.agent.get_action(scaled_obs)
            obs, reward, done, info = self.env.step(action)
            reward_sum += reward
            last_info = info
            if done:
                break
        self.env.reset()
        return reward_sum


def init_worker():
    global evaler
    evaler = Evaluator()


def worker(weights):
    global evaler
    return evaler.evaluate(weights)


MY_REQUIRED_REWARD = 800


def evolve():
    evaler = Evaluator()
    solver = OpenES(evaler.n_weights, popsize=100)
    del evaler
    # for generation in tqdm(count(), unit="generation"):
    pool = Pool(4, init_worker)
    best_solution_so_far = None
    try:
        for generation in count():

            # ask the ES to give us a set of candidate solutions
            solutions = solver.ask()

            # create an array to hold the solutions.
            # solver.popsize = population size
            # rewards = np.zeros(solver.popsize)

            # calculate the reward for each given solution
            # using your own evaluate() method
            # for i in range(solver.popsize):
            #     rewards[i] = evaler.evaluate(solutions[i])
            rewards = pool.map(worker, solutions, 10)

            # give rewards back to ES
            solver.tell(rewards)

            # get best parameter, reward from ES
            reward_vector = solver.result()

            generation_max_reward = max(rewards)
            print("gen: {},max:{},vector:{}".format(generation, generation_max_reward, reward_vector[1]))
            best_solution_so_far = reward_vector[0]
            if generation_max_reward > MY_REQUIRED_REWARD:
                return reward_vector[0]
    except KeyboardInterrupt:
        return best_solution_so_far


if __name__ == '__main__':
    solutions = evolve()
    print(solutions)

# some result i guess 200 reward
# [ 0.09923301 -0.20626588  0.19587401 -0.14792714 -0.18294119  0.1533143
#  -0.00447617 -0.26682253  0.0278062  -0.20441508  0.18823991  0.03357062
#   0.23752833 -0.21154231 -0.00536211  0.01096144 -0.08628416  0.12551947]

# should be 600
# [ 0.02562316  0.37133697  0.00982241 -0.02487976 -0.19998196 -0.03655382
#  -0.20460871  0.06029608 -0.03905059 -0.09734652 -0.04051968  0.11789757
#   0.10608996  0.06954718  0.04162831  0.06654395  0.15691736 -0.03615978]
