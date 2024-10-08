import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Environment.Custom.customGrid import GridworldEnv
from model_discrete.agent import DiscreteAgent
from myutils.eval import custom_evaluate

env = GridworldEnv()

# here action = discrete, state = discrete
print(env.action_space.n, env.observation_space.shape) 

agent = DiscreteAgent(env.action_space.n, env.observation_space.shape)

steps, reward, done, info = custom_evaluate(agent, env)
env.close()
