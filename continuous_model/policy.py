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
        mu = tf.convert_to_tensor(mu, dtype=tf.float32)
        covariance_diag = tf.convert_to_tensor(covariance_diag, dtype=tf.float32)
        sigma = tf.sqrt(covariance_diag)

        mu_np = mu.numpy()
        sigma_np = sigma.numpy()

        samples = np.random.normal(mu_np, sigma_np, size=(1, self.action_dim))
        return samples
    
    def get_action(self, mu, covariance_diag):
        action = self.sample(mu, covariance_diag)
        return action