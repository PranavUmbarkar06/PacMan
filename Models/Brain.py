import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class Brain(nn.Module):
    def __init__(self, input_channels=5, num_actions=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 20, 20)
            flattened_size = self.features(dummy).view(1, -1).size(1)

        self.flatten = nn.Flatten()
        self.value_stream = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        features = self.flatten(self.features(x))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayMemory:
    def __init__(self, capacity=100_000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.memory.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def masked_max(q_values, masks):
    masked_q = q_values.masked_fill(masks <= 0, float("-inf"))
    max_q = masked_q.max(dim=1).values
    return torch.where(torch.isfinite(max_q), max_q, torch.zeros_like(max_q))


def huber_loss(prediction, target):
    return F.smooth_l1_loss(prediction, target)
