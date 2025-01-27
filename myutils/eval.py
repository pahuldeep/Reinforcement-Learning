import numpy as np

def evaluate_agent(agent, env, num_episodes=10):
    total_rewards = []
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            observation_one_hot = np.eye(env.observation_space.n)[observation].reshape(1, -1)
            action = agent.get_action(observation_one_hot)
            next_observation, reward, done, _, _ = env.step(action)
            
            total_reward += reward
            observation = next_observation
        
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward