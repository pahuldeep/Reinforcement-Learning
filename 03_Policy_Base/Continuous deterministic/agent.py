import numpy as np
from brain import Brain
from policy import ContinuousMultiPolicy

class ContinuousAgent:
    def __init__(self, action_dim, input_dim):
        self.brain = Brain(action_dim, input_dim)
        self.policy = ContinuousMultiPolicy(action_dim)

    def get_action(self, observations):
        
        action_logits = self.brain.process(observations)
        action_logits = np.squeeze(action_logits, axis=0)    # Remove batch dimension 
        covariance_diag = np.ones_like(action_logits) * 0.1  # Example: Fixed variance

        # Get sampled action from the policy
        action = self.policy.get_action(action_logits, covariance_diag)
        return action
