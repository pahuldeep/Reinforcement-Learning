import sys
import os

module_environment = os.path.join(os.getcwd(), "custom_env")
sys.path.append(module_environment)

from customGrid import GridworldEnv
from neural_network_policy import Agent

# Agent evaluation function
def evaluate(agent, env, render=True):
    observation = env.reset()
    episode_reward = 0.0
    done = False
    step_num = 0

    while not done:
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)

        episode_reward += reward
        step_num += 1

        if render:
            env.render()
            
        print(f"steps: {step_num} reward: {episode_reward} done: {done} info: {info}")
    return step_num, episode_reward, done, info

env = GridworldEnv()
agent = Agent(env.action_space.n, env.observation_space.shape)

steps, reward, done, info = evaluate(agent, env)
# print(f"steps: {steps} reward: {reward} done: {done} info: {info}")
env.close()
