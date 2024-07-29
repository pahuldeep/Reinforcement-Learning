""" uncomment to test basics understanding for the distributions """ 
"""------------------------------------------------"""
from scipy.stats import bernoulli, multinomial
import numpy as np

# THIS IS FOR BINARY POLICY DISTRIBUTION
binary_policy = bernoulli(0.5)

# Sample actions from the distribution
for i in range(5):
    action = binary_policy.rvs()
    print("Single-Action:", action)

# THIS IS FOR DISCRETE POLICY DISTRIBUTION
action_dim = 4  
action_probabilities = [0.25, 0.25, 0.25, 0.25]

# Define the Multinomial distribution with a total count of 1
discrete_policy = multinomial(n=1, p=action_probabilities)

# Sample actions from the distribution
for i in range(5):
    action = discrete_policy.rvs()
    # Get the index of the action
    action_index = np.argmax(action)
    print("Multi-Action: ", action_index, action)

