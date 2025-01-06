from scipy.stats import bernoulli, multinomial, norm, multivariate_normal
import numpy as np

# THIS IS FOR BINARY POLICY DISTRIBUTION
def binary_policy_distribution(p=0.5, n=5):
    binary_policy = bernoulli(p)
    actions = []
    for i in range(n):
        action = binary_policy.rvs()
        actions.append(action)
    return actions

# THIS IS FOR DISCRETE POLICY DISTRIBUTION
def discrete_policy_distribution(action_dim=4, action_probabilities=None, n=5):
    if action_probabilities is None:
        action_probabilities = [1.0 / action_dim] * action_dim
    discrete_policy = multinomial(n=1, p=action_probabilities)
    actions = []
    for i in range(n):
        action = discrete_policy.rvs()
        action_index = np.argmax(action)
        actions.append((action_index, action))
    return actions

# THIS IS FOR CONTINUOUS POLICY DISTRIBUTION (NORMAL)
def normal_distribution(mean=0.0, sigma=1.0, n=5):
    continuous_policy = norm(loc=mean, scale=sigma)
    actions = []
    for _ in range(n):
        action = continuous_policy.rvs(1)
        actions.append(action)
    return actions

# THIS IS FOR CONTINUOUS MULTIVARIATE NORMAL DISTRIBUTION
def multivariate_normal_distribution(mu=None, covariance_diag=None, n=5):
    if mu is None:
        mu = [0.0, 0.0]
    if covariance_diag is None:
        covariance_diag = [3.0, 3.0]
    covariance_matrix = np.diag(covariance_diag)
    continuous_multi_policy = multivariate_normal(mean=mu, cov=covariance_matrix)
    actions = []
    for _ in range(n):
        action = continuous_multi_policy.rvs(1)
        actions.append(action)
    return actions



if __name__ == "__main__":

    
    print("Binary Policy Distribution:")
    for action in binary_policy_distribution(p=0.5, n=5):
        print("Single-Action:", action)

    
    print("\nDiscrete Policy Distribution:")
    for action in discrete_policy_distribution(action_dim=4, action_probabilities=[0.25, 0.25, 0.25, 0.25], n=5):
        print("Multi-Action: ", action[0], action[1])

    
    print("\nContinuous Policy Distribution (Normal):")
    for action in normal_distribution(mean=0.0, sigma=1.0, n=5):
        print('Single: ', action)

    print("\nContinuous Multi-Policy Distribution (Multivariate Normal):")
    for action in multivariate_normal_distribution(mu = [0.0, 0.0], covariance_diag = [3.0, 3.0], n=5):
        print('Multi: ', action)

