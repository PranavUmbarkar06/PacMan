import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Models import Brain, Game
 # Your class
from map import LAYOUT
import random
# Hardware Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Agents
p_brain = Brain.Brain().to(device)
g_brain = Brain.Brain().to(device)

p_opt = optim.Adam(p_brain.parameters(), lr=0.00025)
g_opt = optim.Adam(g_brain.parameters(), lr=0.00025)

p_mem = Brain.ReplayMemory()
g_mem = Brain.ReplayMemory()

# RL Hyperparameters
gamma = 0.99
epsilon = 1.0
eps_decay = 0.998
batch_size = 64

def get_action(brain, state, eps):
    if random.random() < eps:
        return random.randint(0, 3)
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    return brain(state_t).argmax().item()

def optimize(brain, optimizer, memory):
    if len(memory.memory) < batch_size: return
    
    batch = memory.sample(batch_size)
    s, a, r, ns, d = zip(*batch)

    s = torch.FloatTensor(np.array(s)).to(device)
    a = torch.LongTensor(a).to(device)
    r = torch.FloatTensor(r).to(device)
    ns = torch.FloatTensor(np.array(ns)).to(device)
    d = torch.FloatTensor(d).to(device)

    # Q-Value Calculation
    q_values = brain(s).gather(1, a.unsqueeze(1)).squeeze()
    next_q = brain(ns).max(1)[0].detach()
    expected_q = r + (gamma * next_q * (1 - d))

    loss = F.mse_loss(q_values, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main Loop
env = Game.PacMan(LAYOUT)
for episode in range(70):
    state = env.reset()
    done = False
    p_total, g_total = 0, 0
    
    # 10-Episode Toggle
    train_pacman = (episode // 10) % 2 == 0 

    while not done:
        # Both choose actions
        p_act = get_action(p_brain, state, epsilon)
        g_act = get_action(g_brain, state, epsilon)

        next_state, p_rew, g_rew, done = env.step(p_act, g_act)

        # Store experience
        p_mem.push(state, p_act, p_rew, next_state, done)
        g_mem.push(state, g_act, g_rew, next_state, done)

        # Alternating Update
        if train_pacman:
            optimize(p_brain, p_opt, p_mem)
        else:
            optimize(g_brain, g_opt, g_mem)

        state = next_state
        p_total += p_rew
        g_total += g_rew

    epsilon = max(0.1, epsilon * eps_decay)
    
    if episode % 10 == 0:
        role = "PACMAN" if train_pacman else "GHOST"
        print(f"Ep {episode} | Training: {role} | P-Rew: {p_total:.1f} | G-Rew: {g_total:.1f}")
        torch.save(p_brain.state_dict(), "p_brain.pth")
        torch.save(g_brain.state_dict(), "g_brain.pth")