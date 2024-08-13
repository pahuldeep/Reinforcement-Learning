import tensorflow as tf
import numpy as np
    
class ContinuousPolicy(object):
    def __init__(self, num_actions):
        self.action_dim = num_actions
    
    def sample(self, mu, var):
        sigma = np.sqrt(var)
        normal_distribution = tf.random.normal(mean=mu, stddev=sigma)
        return normal_distribution.sample(1)

    def get_action(self, mu, var):
        action = self.sample(mu, var)
        return action
    
class Continuous_Multi_Policy(object):
    def __init__(self, num_actions):
        self.action_dim = num_actions
    
    def sample(self, mu, covariance_diag):
        noise = tf.random.normal(shape=(self.action_dim,), mean=0.0, stddev=1.0)
        # Scale and shift the noise to match the mean and covariance
        action = mu + noise * tf.sqrt(covariance_diag)
        return action
    
    def get_action(self, mu, covariance_diag):
        action = self.sample(mu, covariance_diag)
        return action