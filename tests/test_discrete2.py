import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
# import tensorflow as tf

from discrete_model.policy import DiscretePolicy
from discrete_model.agent import DiscreteAgent
from myutils.eval import evaluate_agent

env = gym.make('FrozenLake-v1', render_mode="human")
action = env.action_space.n
states = env.observation_space.n 

# trained_brain = tf.keras.models.load_model('trained_brain.h5')

agent = DiscreteAgent(action, input_dim=(1, states))
# agent.brain = trained_brain  # Use the trained brain

evaluate_agent(agent, env, num_episodes=10)
