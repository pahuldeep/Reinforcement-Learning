from continuous_model.agent import ContinuousAgent
from myutils.eval import evaluate
import gymnasium as gym

env = gym.make("MountainCarContinuous-v0")
action_dim = 2 * env.action_space.shape[0]

agent = ContinuousAgent(action_dim, env.observation_space.shape)

steps, reward, done, info = evaluate(agent, env)
print(f"steps: {steps} reward: {reward} done: {done} info: {info} ")
