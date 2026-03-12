import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()
        # Input: 3 layers of 20x20
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # 32 channels * 20 * 20 = 12,800 features
        self.fc1 = nn.Linear(32 * 20 * 20, 128)
        self.fc2 = nn.Linear(128, 4) # 4 Actions: Up, Down, Left, Right

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayMemory:
    def __init__(self, capacity=20000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)