import random
from itertools import count

import retro
import numpy as np
from gym.wrappers import Monitor, TimeLimit
from tqdm import tqdm

from atari_wrappers import FrameStack, EpisodicLifeEnv
from small_evo.agent import Agent
from small_evo.wrappers import AutoRenderer
from es import OpenES

order = ["coins", "levelHi", "levelLo", "lives", "score", "scrolling", "time", "xscrollHi", "xscrollLo"]


class Evaluator(object):
    def __init__(self):
        monitor = None
        action_repeat = True
        episodic_life = True
        render = 5
        env = retro.make("SuperMarioBros-Nes")
        if monitor is not None:
            env = Monitor(env, monitor)
        if render is not None:
            env = AutoRenderer(env, auto_render_period=render)
        if action_repeat:
            env = FrameStack(env, 8)
        env = TimeLimit(env, max_episode_steps=1000)
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
            action[self.index_a] = self.agent.get_action(actual_obs)
            obs, reward, done, info = self.env.step(action)
            reward_sum += reward
            last_info = info
            if done:
                break
        self.env.reset()
        return reward_sum


MY_REQUIRED_REWARD = 800


def evolve():
    evaler = Evaluator()
    solver = OpenES(evaler.n_weights, popsize=25)

    # for generation in tqdm(count(), unit="generation"):

    best_solution_so_far = None
    try:
        for generation in count():

            # ask the ES to give us a set of candidate solutions
            solutions = solver.ask()

            # create an array to hold the solutions.
            # solver.popsize = population size
            rewards = np.zeros(solver.popsize)

            # calculate the reward for each given solution
            # using your own evaluate() method
            for i in range(solver.popsize):
                rewards[i] = evaler.evaluate(solutions[i])

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
