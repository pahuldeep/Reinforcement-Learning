# book author: Maxim Lapan
import typing as tt 
from dataclasses import dataclass

import torch 
import torch.nn as nn 
import torch.optim as optim 

from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np
import gymnasium as gym


HIDDEN_SIZE = 128 
BATCH_SIZE = 16 

class Net(nn.Module):
    def __init__(self, observation_size, hidden_size, num_action):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout regularize
            nn.Linear(hidden_size, num_action),
        )
    def forward(self, x):
        return self.net(x)

# # Add Memory for Partial Observability
# class RNNNet(nn.Module):
#     def __init__(self, observation_size, hidden_size, num_action):
#         super(RNNNet, self).__init__()
#         self.lstm = nn.LSTM(observation_size, hidden_size, batch_first=True)
#         self.linear = nn.Linear(hidden_size, num_action)

#     def forward(self, x, hidden_state):
#         out, hidden_state = self.lstm(x, hidden_state)
#         action_scores = self.linear(out)
#         return action_scores, hidden_state

# # Scalable to continuous action
# class ContinuousNet(nn.Module):
#     def __init__(self, observation_size, hidden_size, action_size):
#         super(ContinuousNet, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(observation_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, action_size)
#         )
#         self.log_std = nn.Parameter(torch.zeros(action_size))  # Learnable std

#     def forward(self, x):
#         mean = self.net(x)
#         std = self.log_std.exp()
#         return mean, std

# # Sampling Actions in iteration batch
# mean, std = net(observe.unsqueeze(0))
# action = torch.normal(mean, std).detach().numpy()

    
@dataclass
class EpisodeSteps:
    observe: np.ndarray
    action: int

@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeSteps]

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()

def adaptive_percentile(iter_no, start=90, end=50, decay=0.01):
    return max(end, start - decay * iter_no * (start - end))
    
def shaped_reward(env, reward, is_done):
    progress_reward = env.unwrapped.state[2] * 10  
    return reward + progress_reward if not is_done else reward

def iterate_batch(env, net, size):
    batch = []
    episode_reward = 0
    episode_steps = []
    observation, _ = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        observe = torch.tensor(observation)

        action_probability = sm(net(observe.unsqueeze(0)))
        action_value = action_probability.data.numpy()[0]
        
        action = np.random.choice(len(action_value), p=action_value)
        next_observation, reward, is_done, is_trunc, _ = env.step(action) 
        
        reward = shaped_reward(env, reward, is_done) # Sparse Reward Sensitivity

        episode_reward += reward
        step = EpisodeSteps(observe=observation, action=action)
        episode_steps.append(step)

        if is_done or is_trunc:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []

            next_observation, _ = env.reset()
            if len(batch) == size:
                yield batch
                batch = []

            observation = next_observation

def filter_batch(batch, percent):
    rewards = list(map(lambda x: x.reward, batch))

    reward_bound = np.percentile(rewards, percent) 
    reward_mean = np.mean(rewards)

    train_observations = []
    train_actions = []
    for episode in batch:
        if episode.reward < reward_bound:
            continue

        train_observations.extend(map(lambda step: step.observe, episode.steps))
        train_actions.extend(map(lambda step: step.action, episode.steps))

    train_observation_value = torch.FloatTensor(np.vstack(train_observations))
    train_action_value = torch.LongTensor(train_actions)

    return train_observation_value, train_action_value, reward_bound, reward_mean

# Run one episode and record the video
def test_agent(env, net):
    env = gym.wrappers.HumanRendering(env)
    observation, _ = env.reset()
    sm = nn.Softmax(dim=1)

    total_reward = 0
    while True:
        env.render()  # Render the environment (optional for visualization)
        observe = torch.tensor(observation, dtype=torch.float32)
        action_probabilities = sm(net(observe.unsqueeze(0)))
        action = torch.argmax(action_probabilities).item()  # Choose the most probable action

        observation, reward, done, trunc, _ = env.step(action)
        total_reward += reward

        if done or trunc:
            break

    print(f"Total reward in test episode: {total_reward}")
    env.close()


if __name__ == "__main__":

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(env, video_folder="video_dir")

    obs_size = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    net = Net(obs_size, HIDDEN_SIZE, n_actions)

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # Hyperparameter Sensitivity

    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batch(env, net, BATCH_SIZE)):
        
        # Adaptive Percentile Threshold
        current_percentile = adaptive_percentile(iter_no)
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, current_percentile)
        # obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE) # fixed
        
        optimizer.zero_grad()
        action_scores_v = net(obs_v)

        # Exploration-Exploitation Tradeoff
        probs_v = nn.Softmax(dim=1)(action_scores_v)
        entropy = compute_entropy(probs_v)
        
        # Poor Sample Efficiency
        weights = torch.FloatTensor([episode.reward for episode in batch])
        weights = weights / weights.sum()
        loss_v = (objective(action_scores_v, acts_v) * weights).mean() - 0.001 * entropy # 0.01 is entropy weight

        # loss_v = objective(action_scores_v, acts_v)

        loss_v.backward()
        optimizer.step()
        scheduler.step() 

        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))

        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if reward_m > 38:
            print("Solved!")
            break

    writer.close()    

    test_agent(env, net)
         
