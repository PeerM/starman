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
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from reward_plot import RewardPlotter, make_axes

from rl_baseline.loaded import LoadedNetwork
from rl_baseline.mario_util import AllowBacktracking, make_env


def main():
    """Run DQN until the environment throws an exception."""
    # "results/rainbow/2/videos/6"
    save_dir = "results/rainbow/7/val_monitor/1"
    env = make_env(stack=False, scale_rew=False, render=3, monitor=None,
                   timelimit=False, episodic_life=True, video=lambda id: True)
    # env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()

    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph("results/rainbow/7/final-4000000.meta", clear_devices=True)
        # saver.restore(sess, tf.train.latest_checkpoint('results/rainbow/2'))
        saver.restore(sess, 'results/rainbow/7/final-4000000')
        model = LoadedNetwork(sess, gym_space_vectorizer(env.observation_space))
        # rebuild the online_net form the saved model
        # type <anyrl.models.dqn_dist.NatureDistQNetwork object at ???>
        player = NStepPlayer(BatchedPlayer(env, model), 3)

        with tf.device("/cpu"):
            # sess.run(tf.global_variables_initializer())
            try:
                for episode_index in tqdm(range(6), unit="episode"):
                    axes = make_axes()
                    plotter = RewardPlotter(axes, save_period=20, render_period=300, max_entries=600)
                    for i in count():
                        trajectories = player.play()
                        end_of_episode = False
                        current_total_reward = None
                        for trajectory in trajectories:
                            current_total_reward = trajectory["total_reward"]
                            if trajectory["is_last"]:
                                end_of_episode = True
                        plotter.update(current_total_reward, step=i)
                        if end_of_episode:
                            # plt.show()
                            plotter.render()
                            # plotter.save_file("{}/e{}.pdf".format(save_dir, episode_index))
                            plotter.close()
                            break
            except KeyboardInterrupt:
                env.close()
                plt.close()


if __name__ == '__main__':
    main()
