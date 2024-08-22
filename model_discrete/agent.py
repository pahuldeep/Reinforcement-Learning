import numpy as np
from .brain import Brain
from .policy import DiscretePolicy

class DiscreteAgent(object):
    def __init__(self, action_dim, input_dim):
        self.brain = Brain(action_dim, input_dim)
        self.policy = DiscretePolicy(action_dim)

    def get_action(self, observations):
        action_logits = self.brain.process(observations)
        action = self.policy.get_action(action_logits[0])
        return action.numpy()[0][0]