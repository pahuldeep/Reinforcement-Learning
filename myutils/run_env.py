import gymnasium as gym
import argparse

def run_environment(env_name, num_steps):
    
    env = gym.make(env_name, render_mode="human")
    env.reset()

    for _ in range(num_steps):
        env.render()
        env.step(env.action_space.sample())

    env.close()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run a Gymnasium environment.")
    parser.add_argument("env_name", type=str, help="Name of the Gym environment (e.g., 'FrozenLake-v1').")
    parser.add_argument("num_steps", type=int, help="Number of steps to run the environment.")

    args = parser.parse_args()

    # Run the environment with the provided arguments
    run_environment(args.env_name, args.num_steps)
