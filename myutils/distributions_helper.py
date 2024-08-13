# from scipy.stats import bernoulli, multinomial, norm, multivariate_normal
# import numpy as np

# def binary_policy_distribution(p=0.5, n=5):
#     binary_policy = bernoulli(p)
#     actions = []
#     for i in range(n):
#         action = binary_policy.rvs()
#         actions.append(action)
#     return actions

# def discrete_policy_distribution(action_dim=4, action_probabilities=None, n=5):
#     if action_probabilities is None:
#         action_probabilities = [1.0 / action_dim] * action_dim
#     discrete_policy = multinomial(n=1, p=action_probabilities)
#     actions = []
#     for i in range(n):
#         action = discrete_policy.rvs()
#         action_index = np.argmax(action)
#         actions.append((action_index, action))
#     return actions

# def normal_distribution(mean=0.0, sigma=1.0, n=5):
#     continuous_policy = norm(loc=mean, scale=sigma)
#     actions = []
#     for _ in range(n):
#         action = continuous_policy.rvs(1)
#         actions.append(action)
#     return actions

# def multivariate_normal_distribution(mu=None, covariance_diag=None, n=5):
#     if mu is None:
#         mu = [0.0, 0.0]
#     if covariance_diag is None:
#         covariance_diag = [3.0, 3.0]
#     covariance_matrix = np.diag(covariance_diag)
#     continuous_multi_policy = multivariate_normal(mean=mu, cov=covariance_matrix)
#     actions = []
#     for _ in range(n):
#         action = continuous_multi_policy.rvs(1)
#         actions.append(action)
#     return actions

# # Test the functions
# print("Binary Policy Distribution:")
# print(binary_policy_distribution())

# print("\nDiscrete Policy Distribution:")
# print(discrete_policy_distribution())

# print("\nContinuous Policy Distribution (Normal):")
# print(normal_distribution())

# print("\nContinuous Multi-Policy Distribution (Multivariate Normal):")
# print(multivariate_normal_distribution())


from scipy.stats import bernoulli, multinomial, norm, multivariate_normal
import numpy as np

# THIS IS FOR BINARY POLICY DISTRIBUTION
binary_policy = bernoulli(0.5)

print("\nBinary Policy Distribution:")
for i in range(5):
    action = binary_policy.rvs()
    print("Single-Action:", action)
print()

# THIS IS FOR DISCRETE POLICY DISTRIBUTION
action_dim = 4  
action_probabilities = [0.25, 0.25, 0.25, 0.25]

discrete_policy = multinomial(n=1, p=action_probabilities)

print("Discrete Policy Distribution:")
for i in range(5):
    action = discrete_policy.rvs()
    # Get the index of the action
    action_index = np.argmax(action)
    print("Multi-Action: ", action_index, action)
print()

# THIS IS FOR CONTINUOUS POLICY DISTRIBUTION (NORMAL)
mu = 0.0
sigma = 1.0

continuous_policy = norm(loc=mu, scale=sigma)

print("Continuous Policy Distribution (Normal):")
for _ in range(5):
    action = continuous_policy.rvs(1)
    print('Single: ', action)
print()

# THIS IS FOR CONTINUOUS MULTIVARIATE NORMAL DISTRIBUTION
mu = [0.0, 0.0]
covariance_diag = [3.0, 3.0]
covariance_matrix = np.diag(covariance_diag)

continuous_multi_policy = multivariate_normal(mean=mu, cov=covariance_matrix)

print("Continuous Multi-Policy Distribution (Multivariate Normal):")
for _ in range(5):
    action = continuous_multi_policy.rvs(1)
    print('Multi: ', action)



