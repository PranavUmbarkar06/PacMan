import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from Models import Brain, Game
from Models.Game import get_a_star_action
from map import LAYOUT
import random
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Two Brains + Two Target Networks for stability
p_brain = Brain.Brain().to(device)
p_target = copy.deepcopy(p_brain)
g_brain = Brain.Brain().to(device)
g_target = copy.deepcopy(g_brain)

p_opt = optim.Adam(p_brain.parameters(), lr=0.00025)
g_opt = optim.Adam(g_brain.parameters(), lr=0.00025)

p_mem = Brain.ReplayMemory()
g_mem = Brain.ReplayMemory()

gamma, batch_size = 0.99, 64
target_update = 1000 # Increased for stability
UPDATE_EVERY = 4 # Speed Optimization: Update networks every 4 steps

def get_action(brain, state, env, pos, eps):
    valid = env.get_valid_actions(pos)
    if random.random() < eps:
        return random.choice(valid)
    
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = brain(state_t).cpu().numpy()[0]
    
    # Masking: set illegal moves to -infinity
    mask = np.full(4, -np.inf)
    for a in valid: mask[a] = q_values[a]
    return np.argmax(mask)

def optimize(brain, target, optimizer, memory):
    if len(memory.memory) < batch_size: return
    s, a, r, ns, d = zip(*memory.sample(batch_size))

    s = torch.FloatTensor(np.array(s)).to(device)
    a = torch.LongTensor(a).to(device)
    r = torch.FloatTensor(r).to(device)
    ns = torch.FloatTensor(np.array(ns)).to(device)
    d = torch.FloatTensor(d).to(device)

    current_q = brain(s).gather(1, a.unsqueeze(1)).squeeze()
    
    # --- Double DQN ---
    # 1. Use online network to select best action
    with torch.no_grad():
        best_next_actions = brain(ns).argmax(1).unsqueeze(1)
        # 2. Use target network to evaluate best action
        next_q = target(ns).gather(1, best_next_actions).squeeze()
        
    expected_q = r + (gamma * next_q * (1 - d))

    # Huber Loss is better for outliers than MSE
    loss = F.smooth_l1_loss(current_q, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

env = Game.PacMan(LAYOUT)
steps = 0

try:
    # ==========================================
    # STAGE 1: Train Pacman vs A* Ghost
    # ==========================================
    print("--- STAGE 1: Training Pacman vs A* Ghost ---")
    epsilon_p = 1.0
    eps_decay_p = 0.995 # Faster decay
    eps_min = 0.05
    
    for episode in range(800): # 800 episodes for Stage 1
        state = env.reset() 
        done, p_total = False, 0
        p_losses = []
        episode_steps = 0
        
        while not done:
            # Pacman uses Neural Net
            p_act = get_action(p_brain, state, env, env.pacman_pos, epsilon_p)
            
            # Ghost uses A* Rule (Half speed - only moves every 2 frames)
            if steps % 2 == 0:
                g_act = get_a_star_action(env, is_ghost=True)
            else:
                g_act = 99

            next_state, p_rew, g_rew, done = env.step(p_act, g_act)
            p_mem.push(state, p_act, p_rew, next_state, done)

            episode_steps += 1
            if episode_steps >= 1000: # Time limit
                done = True

            if steps % UPDATE_EVERY == 0:
                loss = optimize(p_brain, p_target, p_opt, p_mem)
                if loss: p_losses.append(loss)

            state = next_state
            p_total += p_rew; steps += 1

            if steps % target_update == 0:
                p_target.load_state_dict(p_brain.state_dict())

        epsilon_p = max(eps_min, epsilon_p * eps_decay_p)
        avg_loss = np.mean(p_losses) if p_losses else 0
        if episode % 10 == 0:
            print(f"S1 Ep {episode:3} | P-Rew: {p_total:6.1f} | Eps P: {epsilon_p:.2f} | Avg Loss: {avg_loss:.4f} | Steps: {episode_steps}")

    print("Stage 1 Complete.")


    # ==========================================
    # STAGE 2: Train Ghost vs Trained Pacman
    # ==========================================
    # p_brain weights are already in memory, switch to evaluation
    p_brain.eval() 
    print("\n--- STAGE 2: Training Ghost vs Trained Pacman ---")
    
    epsilon_g = 1.0
    eps_decay_g = 0.995
    eps_min = 0.05

    # Curriculum Learning: Start with noisy Pacman
    curriculum_eps_p = 0.3 
    curriculum_decay = 0.99 
    MAX_EPISODE_STEPS = 1000 # Force reset if the game lasts too long
    
    for episode in range(800): 
        state = env.reset() 
        done, g_total = False, 0
        g_losses = []
        episode_steps = 0
        
        while not done:
            p_act = get_action(p_brain, state, env, env.pacman_pos, curriculum_eps_p)
            
            if steps % 2 == 0:
                g_act = get_action(g_brain, state, env, env.ghost_pos, epsilon_g)
            else:
                g_act = 99

            next_state, p_rew, g_rew, done = env.step(p_act, g_act)
            
            episode_steps += 1
            if episode_steps >= MAX_EPISODE_STEPS: # Time limit
                done = True

            if g_act != 99:
                 g_mem.push(state, g_act, g_rew, next_state, done)

            if steps % UPDATE_EVERY == 0:
                loss = optimize(g_brain, g_target, g_opt, g_mem)
                if loss: g_losses.append(loss)

            state = next_state
            g_total += g_rew
            steps += 1

            if steps % target_update == 0:
                g_target.load_state_dict(g_brain.state_dict())

        epsilon_g = max(eps_min, epsilon_g * eps_decay_g)
        curriculum_eps_p = max(0.01, curriculum_eps_p * curriculum_decay)
        
        if episode % 10 == 0:
            avg_loss = np.mean(g_losses) if g_losses else 0
            print(f"S2 Ep {episode:3} | G-Rew: {g_total:6.1f} | Eps G: {epsilon_g:.2f} | Avg Loss: {avg_loss:.4f} | Steps: {episode_steps}")

except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving progress...")

finally:
    # This block runs whether the loop finishes naturally OR is interrupted
    print("Saving models...")
    torch.save(p_brain.state_dict(), "p_brain.pth")
    torch.save(g_brain.state_dict(), "g_brain.pth")
    print("Done. Models saved to p_brain.pth and g_brain.pth")