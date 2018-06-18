from sklearn.linear_model import LinearRegression
import numpy as np


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class.
    Columns are predictions (samples).
    """
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)


# weights are classes times elements

class Agent:
    def __init__(self, num_actions, input_shape, deterministic=False, embed=False):
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.deterministic = deterministic
        self.weights = None
        self.model = LinearRegression()
        self.embed = embed
        if embed:
            random_state = np.random.RandomState(2)
            self.embed_weight = random_state.normal(0, 1, (input_shape, input_shape))

    def weight_shape(self):
        return (self.num_actions, self.input_shape)

    def change_weights(self, new_weights):
        self.weights = new_weights

    def get_action(self, obs):
        # Assume X_test is [3073 x 10000], Y_test [10000 x 1]
        if self.embed:
            obs = self.embed_weight.dot(obs)
            obs = 1 / (1 + np.exp(-obs))  # sigmoid
            # obs = np.maximum(obs, 0)  # relu
        scores = self.weights.dot(obs)  # 10 x 10000, the class scores for all test examples
        # find the index with max score in each column (the predicted class)
        if self.deterministic:
            prediction = np.argmax(scores)
        else:
            weighted = softmax(scores)
            prediction = np.argmax(np.random.multinomial(1, weighted))
        return prediction
