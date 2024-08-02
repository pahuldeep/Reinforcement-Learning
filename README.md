# Reinforcement Learning

This repository is dedicated to exploring various reinforcement learning (RL) environments using `gymnasium`. 

The primary objective is to gain familiarity with different RL environments, understand their dynamics, and development of RL algorithms.

## Setup

### Prerequisites

Ensure you have dependencies installed. You can download it from source sites.

### Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/pahuldeep/Reinforcement-Learning.git
    cd Reinforcement-Learning
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install gymnasium
    ```

## Basic RL Workflow

Here is an example workflow using the `PPO` algorithm with the `CartPole-v1` environment:

```python
import gymnasium as gym
from stable_baselines3 import PPO

# Create the environment
env = gym.make('CartPole-v1')

# Create the RL agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the agent
model.save("ppo_cartpole")

# Load the trained agent
model = PPO.load("ppo_cartpole")

# Evaluate the agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()
```

## Current Work

Currently, this repository is under active development. Stay tuned for updates on additional environments and advanced RL algorithms.
