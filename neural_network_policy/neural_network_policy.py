from tensorflow import keras
from keras import layers

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

"""Here is some policy working"""
# num_actions = 5
# policy = DiscretePolicy(num_actions)

# logits = tf.random.uniform((num_actions,), minval=-1, maxval=1)
# action_probs = tf.nn.softmax(logits)

# action = policy.get_action(logits)
# entropy = policy.entropy(action_probs)

# print(f"Action: {action}")
# print(f"Entropy: {entropy.numpy()}")

class Brain(keras.Model):
    
    def __init__(self, action_dim = 5, input_shape = (1, 8 * 8)):
        super(Brain, self).__init__()
        self.dense1 = layers.Dense(32, input_shape = input_shape, activation="relu")
        self.logits = layers.Dense(action_dim)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        if len(x.shape) >= 2 and x.shape[0] != 1:
            x = tf.reshape(x, (1, -1))
        return self.logits(self.dense1(x))
    
    def process(self, observations):
        action_logits = self.predict_on_batch(observations)
        return action_logits

class Agent(object):

    def __init__(self, action_dim = 5, input_dim = (1, 8 * 8)):
        self.brain = Brain(action_dim, input_dim)
        self.policy = DiscretePolicy(action_dim)


    def get_action(self, observations):
        action_logits = self.brain.process(observations)
        action = self.policy.get_action(np.squeeze(action_logits, 0))
        return action




