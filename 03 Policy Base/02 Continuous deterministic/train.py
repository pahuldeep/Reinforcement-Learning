import os
import torch
from agent import ContinuousAgent
import gymnasium as gym

# Define paths for saving and loading checkpoints
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "agent_checkpoint.pth")
SAVE_INTERVAL = 1000  # Save checkpoint every 1000 steps


def save_checkpoint(agent, filepath, step=None):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    checkpoint_data = {
        "model_state_dict": agent.brain.state_dict(),
        "step": step
    }
    torch.save(checkpoint_data, filepath)
    print(f"Checkpoint saved at {filepath}")


def load_checkpoint(agent, filepath):
   
    if os.path.exists(filepath):
        checkpoint_data = torch.load(filepath)
        agent.brain.load_state_dict(checkpoint_data["model_state_dict"])
        print(f"Checkpoint loaded from {filepath}. Step: {checkpoint_data.get('step', 'N/A')}")
        return checkpoint_data.get("step", 0)
    else:
        print(f"No checkpoint found at {filepath}. Starting from scratch.")
        return 0


def train(agent, env, max_steps=5000, render=False):
    
    observation, info = env.reset()
    total_reward = 0.0
    step_num = 0

    # Load checkpoint if it exists
    starting_step = load_checkpoint(agent, CHECKPOINT_FILE)

    while step_num < max_steps:
        # Get action from the agent
        action = agent.get_action(observation)
        observation, reward, done, truncated, info = env.step(action)

        total_reward += reward
        step_num += 1

        if render:
            env.render()

        # Reset environment if episode is done
        if done or truncated:
            observation, info = env.reset()

        # Save checkpoint periodically
        if step_num % SAVE_INTERVAL == 0:
            save_checkpoint(agent, CHECKPOINT_FILE, step=step_num)

    print(f"Training completed. Total steps: {step_num}, Total reward: {total_reward}")


if __name__ == "__main__":

    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    action_dim = 2 * env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    agent = ContinuousAgent(action_dim, state_dim)

    # Train the agent with periodic checkpoint saving
    train(agent, env, max_steps=5000, render=True)
