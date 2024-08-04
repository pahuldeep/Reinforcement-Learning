import os
import sys

module_utils = os.path.join(os.getcwd(), "myutils")
module_model = os.path.join(os.getcwd(), "discrete_model")
module_environment = os.path.join(os.getcwd(), "custom_env")
sys.path.append(module_utils)
sys.path.append(module_model)
sys.path.append(module_environment)

from customGrid import GridworldEnv
from discrete_model.agent import DiscreteAgent
from myutils.eval import evaluate

env = GridworldEnv()
agent = DiscreteAgent(env.action_space.n, env.observation_space.shape)

steps, reward, done, info = evaluate(agent, env)
print(f"steps: {steps} reward: {reward} done: {done} info: {info}")
env.close()
