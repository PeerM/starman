from anyrl.models import Model
import tensorflow as tf
import numpy as np


class LoadedNetwork(Model):

    def __init__(self, session, obs_vectorizer) -> None:
        super().__init__()
        self.obs_vectorizer = obs_vectorizer
        self.session = session
        graph = tf.get_default_graph()
        # <class 'tuple'>: (<tf.Tensor 'online/Sum:0' shape=(?, 11) dtype=float32>, <tf.Tensor 'online/Reshape_3:0' shape=(?, 11, 51) dtype=float32>)        w1 = graph.get_tensor_by_name("w1:0")
        sum = graph.get_tensor_by_name("online/Sum:0")
        reshape_3 = graph.get_tensor_by_name("online/Reshape_3:0")
        self.step_outs = (sum, reshape_3)
        # Tensor("online/Placeholder:0", shape=(?, 84, 84, 4), dtype=uint8)
        self.step_obs_ph = graph.get_tensor_by_name("online/Placeholder:0")

    @property
    def stateful(self):
        return False

    def start_state(self, batch_size):
        return None

    def step(self, observations, states):
        feed = self.step_feed_dict(observations, states)
        values, dists = self.session.run(self.step_outs, feed_dict=feed)
        return {
            'actions': np.argmax(values, axis=1),
            'states': None,
            'action_values': values,
            'action_dists': dists
        }

    def step_feed_dict(self, observations, states):
        """Produce a feed_dict for taking a step."""
        return {self.step_obs_ph: self.obs_vectorizer.to_vecs(observations)}
