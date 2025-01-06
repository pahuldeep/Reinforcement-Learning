import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_continuous.agent import ContinuousAgent
from myutils.eval import custom_evaluate
import gymnasium as gym

env = gym.make("MountainCarContinuous-v0", render_mode="human")

action_dim = 2 * env.action_space.shape[0]
state_dim = env.observation_space.shape

agent = ContinuousAgent(action_dim, (1, state_dim))

steps, reward, done, info = custom_evaluate(agent, env)
print(f"steps: {steps} reward: {reward} done: {done} info: {info} ")
