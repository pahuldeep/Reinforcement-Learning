import tensorflow as tf
class DiscretePolicy(object):
    def __init__(self, num_actions):
        self.action_dim = num_actions

    def sample(self, action_logits):
        action = tf.random.categorical(logits=[action_logits], num_samples=1)
        return action

    def get_action(self, action_logits):
        action = self.sample(action_logits)
        return action

    def entropy(self, action_logits):
        action_probs = tf.nn.softmax(action_logits)
        return -tf.reduce_sum(action_probs * tf.math.log(action_probs), axis=-1)