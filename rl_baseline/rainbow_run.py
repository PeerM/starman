#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
Fork of openai retro-baselines
"""
from itertools import count

import tensorflow as tf
from tqdm import tqdm

from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.rollouts import BatchedPlayer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer

from rl_baseline.loaded import LoadedNetwork
from rl_baseline.mario_util import AllowBacktracking, make_env


def main():
    """Run DQN until the environment throws an exception."""
    # "results/rainbow/2/videos/6"
    env = make_env(stack=False, scale_rew=False, render=None, monitor="results/rainbow/2/videos/4", timelimit=False)
    # env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()

    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph("results/rainbow/2/rainbow.ckpt.meta")
        # saver.restore(sess, tf.train.latest_checkpoint('results/rainbow/2'))
        saver.restore(sess, 'results/rainbow/2/rainbow.ckpt')
        model = LoadedNetwork(sess, gym_space_vectorizer(env.observation_space))
        # rebuild the online_net form the saved model
        # type <anyrl.models.dqn_dist.NatureDistQNetwork object at ???>
        player = NStepPlayer(BatchedPlayer(env, model), 3)

        with tf.device("/cpu"):
            # sess.run(tf.global_variables_initializer())

            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            try:
                for episode_index in tqdm(range(20), unit="episode"):
                    for i in count():
                        trajectories = player.play()
                        end_of_episode = False
                        for trajectory in trajectories:
                            if trajectory["is_last"]:
                                end_of_episode = True
                        if end_of_episode:
                            break
            except KeyboardInterrupt:
                env.close()


if __name__ == '__main__':
    main()
