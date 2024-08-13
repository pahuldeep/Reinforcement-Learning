import numpy as np

import tensorflow as tf
from tensorflow import keras

class Brain(keras.Model):
    
    def __init__(self, action_dim = 5, input_shape = (1, 8 * 8)):
        super(Brain, self).__init__()
        self.dense1 = keras.layers.Dense(32, input_shape = input_shape, activation="relu")
        self.logits = keras.layers.Dense(action_dim)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        if len(x.shape) >= 2 and x.shape[0] != 1:
            x = tf.reshape(x, (1, -1))
        logits = self.logits(self.dense1(x))
        return logits
    
    def process(self, observations):
        action_logits = self.predict_on_batch(observations)
        return action_logits

