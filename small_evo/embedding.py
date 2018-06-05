from sklearn.ensemble import RandomTreesEmbedding

from atari_wrappers import FrameStack
import pickle
import random
import numpy as np


def train():
    embedder = RandomTreesEmbedding(n_estimators=10, random_state=1, max_leaf_nodes=30)
    import retro

    raw_env = retro.make("SuperMarioBros-Nes")
    env = FrameStack(raw_env, 4)
    first_obs = env.reset()
    index_right = raw_env.buttons.index("RIGHT")
    index_a = raw_env.buttons.index("A")
    index_b = raw_env.buttons.index("B")
    all_observations = []
    for i in range(100):
        action = [0] * 9
        action[index_right] = 1
        action[index_b] = 1
        action[index_a] = 1 if random.random() > 0.1 else 0
        obs, reward, end_of_episode, aditional_information = env.step(action)
        flat = np.array(obs).flatten()
        all_observations.append(flat)
        if end_of_episode:
            print("dead")
            env.reset()
        if i % 300 == 0:
            env.render()
    raw_env.render(close=True)
    embedder.fit(all_observations)
    print("done")
    with open("data/embedder2.pickle", "wb") as pickle_outfile:
        pickle.dump(embedder, pickle_outfile)


def load(filename="data/embedder1.pickle") -> RandomTreesEmbedding:
    with open(filename, "rb") as pickle_infile:
        embedder = pickle.load(pickle_infile)
    return embedder


if __name__ == '__main__':
    train()
