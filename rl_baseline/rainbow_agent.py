#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
Fork of openai retro-baselines
"""

import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer

from rl_baseline.mario_util import AllowBacktracking, make_env


def handle_ep(steps, reward):
    print("steps: {}\t, reward: {}".format(steps, reward))


def main():
    """Run DQN until the environment throws an exception."""
    env = make_env(stack=False, scale_rew=False, render=None, monitor="results/rainbow/3/train_monitor",
                   episodic_life=True)
    # I think the env itself allows Backtracking
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n, gym_space_vectorizer(env.observation_space),
                                  min_val=-200, max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        saver = tf.train.Saver(name="rainbow", keep_checkpoint_every_n_hours=1)
        sess.run(tf.global_variables_initializer())
        saver.save(sess, "results/rainbow/3/training", global_step=0)
        dqn.train(num_steps=1_000_000,  # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000,
                  handle_ep=handle_ep)  # in seconds
        saver.save(sess, "results/rainbow/3/final", global_step=1_000_000)


if __name__ == '__main__':
    main()
