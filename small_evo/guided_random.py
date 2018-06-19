import random

import retro
from anyrl.models import Model, RandomAgent
from anyrl.rollouts import BasicPlayer, BasicRoller
from gym import Wrapper, ActionWrapper
from gym.spaces import Discrete, MultiBinary
from gym.wrappers import Monitor
from anyrl.rollouts.rollout import Rollout
from atari_wrappers import FrameStack, EpisodicLifeEnv
from small_evo.wrappers import AutoRenderer


class MarioRestrictedDiscretizer(ActionWrapper):

    def __init__(self, env, run="B", jump="A"):
        super().__init__(env)
        # if isinstance(self.actions_space, Discrete):
        #     raise Exception("Mario With discrete not supported yet")
        # elif isinstance(self.env.actions_space, MultiBinary):
        self.multi_binary = True
        self.index_a = self.env.unwrapped.buttons.index(jump)
        self.index_b = self.env.unwrapped.buttons.index(run)
        self.index_right = self.env.unwrapped.buttons.index("RIGHT")
        self.action_space = Discrete(2)

    def action(self, action):
        new_action = [0] * 9
        new_action[self.index_right] = 1
        new_action[self.index_b] = 1
        new_action[self.index_a] = action
        return new_action


# class WeightedRandomAgent(Model):
#
#     def __init__(self, chance_of_zero=0.1) -> None:
#         super().__init__()
#         self.chance_of_zero = chance_of_zero
#
#     @property
#     def stateful(self):
#         return False
#
#     def start_state(self, batch_size):
#         return 1
#
#     def step(self, observations, states):
#         return {"actions": [1 if random.random() > self.chance_of_zero else 0 for _ in observations],
#                 "states": [1 for _ in observations]}


if __name__ == '__main__':
    monitor = "results/random/5"
    action_repeat = True
    episodic_life = True
    render = 60
    env = retro.make("SuperMarioBros-Nes")
    if episodic_life:
        env = EpisodicLifeEnv(env, [0] * 9)
    if monitor is not None:
        env = Monitor(env, monitor, video_callable=lambda i: True)
    if render is not None:
        env = AutoRenderer(env, auto_render_period=render)
    if action_repeat:
        env = FrameStack(env, 4)
    env = MarioRestrictedDiscretizer(env)
    # model = WeightedRandomAgent()
    model = RandomAgent(lambda: 1 if random.random() > 0.1 else 0)
    player = BasicRoller(env, model, min_episodes=4)
    for i in range(10):
        rollouts = player.rollouts()
        print([roll.total_reward for roll in rollouts])
