import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import clear_output

def random_exploration(env, num_step):

    observation, info = env.reset(seed=2024)
    rewards = []

    for _ in range(num_step):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        # If the episode is terminated, reset the environment to the start cell
        if terminated:
            observation, info = env.reset()
        
        # Display the current state of the environment
        clear_output(wait=True)
        plt.imshow(env.render())
        plt.show()
        
    return rewards


env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="rgb_array")
rewards = random_exploration(env, 10)
env.close()

print(f"Total reward: {np.round(np.sum(rewards), 2)}")