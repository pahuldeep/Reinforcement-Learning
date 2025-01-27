import torch
import torch.nn as nn
import torch.nn.functional as F

class Brain(nn.Module):
    def __init__(self, action_dim, input_dim):
        super(Brain, self).__init__()
        self.dense1 = nn.Linear(input_dim, 32)  
        self.logits = nn.Linear(32, action_dim)  # Output layer

    def forward(self, inputs):
        x = F.relu(self.dense1(inputs))  # Apply activation
        logits = self.logits(x)          # Compute
        return logits

    def process(self, observations):
        # Ensure no gradients are computed during inference
        with torch.no_grad():
            observations_tensor = torch.tensor(observations, dtype=torch.float32)

            if observations_tensor.dim() == 1: 
                observations_tensor = observations_tensor.unsqueeze(0)

            action_logits = self.forward(observations_tensor)
        return action_logits.numpy()  # Convert back to NumPy array for policy

