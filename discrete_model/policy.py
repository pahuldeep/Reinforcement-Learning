import tensorflow as tf
import numpy as np

class DiscretePolicy(object):
    def __init__(self, num_actions):
        self.action_dim = num_actions

    def sample(self, action_logits):
        action_probs = tf.nn.softmax(action_logits)
        action = np.random.choice(self.action_dim, p=action_probs.numpy())
        return action

    def get_action(self, action_logits):
        action = self.sample(action_logits)
        return action

    def entropy(self, action_probabilities):
        return -tf.reduce_sum(action_probabilities * tf.math.log(action_probabilities), axis=-1)