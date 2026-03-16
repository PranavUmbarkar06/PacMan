import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Models import Brain, Game
from map import LAYOUT
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Only Pacman needs a brain now
p_brain = Brain.Brain().to(device)
p_opt = optim.Adam(p_brain.parameters(), lr=0.00025)
p_mem = Brain.ReplayMemory()

gamma, epsilon, eps_decay, batch_size = 0.99, 1.0, 0.998, 64

def get_action(brain, state, eps):
    if random.random() < eps: return random.randint(0, 3)
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    return brain(state_t).argmax().item()

def optimize():
    if len(p_mem.memory) < batch_size: return
    batch = p_mem.sample(batch_size)
    s, a, r, ns, d = zip(*batch)

    s = torch.FloatTensor(np.array(s)).to(device)
    a = torch.LongTensor(a).to(device)
    r = torch.FloatTensor(r).to(device)
    ns = torch.FloatTensor(np.array(ns)).to(device)
    d = torch.FloatTensor(d).to(device)

    q_values = p_brain(s).gather(1, a.unsqueeze(1)).squeeze()
    next_q = p_brain(ns).max(1)[0].detach()
    expected_q = r + (gamma * next_q * (1 - d))

    loss = F.mse_loss(q_values, expected_q)
    p_opt.zero_grad()
    loss.backward()
    p_opt.step()

env = Game.PacMan(LAYOUT)
total_episodes = 2500

for episode in range(total_episodes):
    state = env.reset() 
    done = False
    p_total = 0
    
    while not done:
        # 1. Pacman (AI) vs Ghost (A* Algorithm)
        p_act = get_action(p_brain, state, epsilon)
        g_act = Game.get_a_star_action(env) 

        next_state, p_rew, _, done = env.step(p_act, g_act)

        p_mem.push(state, p_act, p_rew, next_state, done)
        optimize()

        state = next_state
        p_total += p_rew

    epsilon = max(0.1, epsilon * eps_decay)
    
    if episode % 10 == 0:
        print(f"Ep {episode:03d} | P-Rew: {p_total:>7.1f} | Eps: {epsilon:.2f}")
        torch.save(p_brain.state_dict(), "p_brain.pth")