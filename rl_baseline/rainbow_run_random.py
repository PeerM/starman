#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
Fork of openai retro-baselines
"""

import tensorflow as tf
from tqdm import tqdm

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer

from rl_baseline.mario_util import AllowBacktracking, make_env


def main():
    """Run DQN until the environment throws an exception."""
    # "results/rainbow/2/videos/6"
    env = make_env(stack=False, scale_rew=False, render=20, monitor=None, timelimit=False)
    # env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    # TODO we might not want to allow backtracking, it kinda hurts in mario
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        # TODO rebuild the online_net form the saved model
        # type <anyrl.models.dqn_dist.NatureDistQNetwork object at ???>
        # important methods
        #
        model = dqn.online_net
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)

        with tf.device("/cpu"):
            # sess.run(tf.global_variables_initializer())

            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            try:
                for i in tqdm(range(100000)):
                    trajectories = player.play()
                    for trajectori in trajectories:
                        trajectori
                        pass
            except KeyboardInterrupt:
                env.close()


if __name__ == '__main__':
    main()
