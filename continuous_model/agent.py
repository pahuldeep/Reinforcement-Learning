import numpy as np

from brain import Brain
from policy import Continuous_Multi_Policy

class ContinuousAgent(object):
    
    def __init__(self, action_dim=5, input_dim=(8, 8)):
        self.brain = Brain(action_dim, input_dim)
        self.policy = Continuous_Multi_Policy(action_dim)

    def get_action(self, observations):
        action_logits = self.brain.process(observations)
        action_logits = np.squeeze(action_logits, axis=0)  
        action = self.policy.get_action(action_logits, np.ones_like(action_logits))  # Assuming unit covariance for simplicity
        return action
    


