from neural_network_policy import Agent
import gymnasium as gym

import matplotlib.pyplot as plt
from IPython.display import clear_output

def evaluate(agent, env):
    observation, info = env.reset()  # Update reset method to return observation and info
    episode_reward = 0.0
    done = False
    step_num = 0

    while not done:
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)  # Update step method
        done = terminated or truncated

        episode_reward += reward
        step_num += 1

        # Display the current state of the environment
        clear_output(wait=True)
        plt.imshow(env.render())
        plt.show()
            
    return step_num, episode_reward, done, info

env = gym.make('FrozenLake-v1')
agent = Agent(env.action_space.n, (env.observation_space.n,))  

steps, reward, done, info = evaluate(agent, env)
print(f"steps: {steps} reward: {reward} done: {done} info: {info}")
env.close()
