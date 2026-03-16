import numpy as np
import random
import heapq

class PacMan:
    def __init__(self, board):
        self.original_layout = np.array(board).astype(float) * 2.0
        self.moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        self.reset()

    def reset(self):
        self.board = np.copy(self.original_layout)
        self.items = (self.original_layout == 0).astype(float) 
        self.ghost_stunned_timer = 0
        
        empty_slots = np.argwhere(self.original_layout == 0).tolist()
        spots = random.sample(empty_slots, 4)
        
        self.pacman_pos = spots[0]
        self.ghost_pos = spots[1]
        self.stunner_pos = spots[2]
        self.bomb_pos = spots[3]
        
        self.items[self.stunner_pos[0], self.stunner_pos[1]] = -1.0
        self.items[self.bomb_pos[0], self.bomb_pos[1]] = -2.0
        
        return self.get_state()

    def get_state(self):
        state = np.zeros((3, 20, 20), dtype=np.float32)
        state[0] = self.board + self.items 
        state[1, self.pacman_pos[0], self.pacman_pos[1]] = 1.0
        state[2, self.ghost_pos[0], self.ghost_pos[1]] = 1.0
        return state

    def step(self, p_action, g_action):
        # 1. Helper for Distance tracking
        def get_manhattan(p1, p2):
            return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

        d_old = get_manhattan(self.pacman_pos, self.ghost_pos)
        d_bomb_old = get_manhattan(self.pacman_pos, self.bomb_pos)

        p_reward = -0.1 # Small step penalty to encourage speed
        g_reward = -0.1 # Ghost also wants to finish the job quickly
        done = False

        # --- Pacman Movement ---
        p_dr, p_dc = self.moves[p_action]
        p_nr, p_nc = self.pacman_pos[0] + p_dr, self.pacman_pos[1] + p_dc
        
        if self.board[p_nr][p_nc] != 2.0:
            self.pacman_pos = [p_nr, p_nc]
            val = self.items[p_nr, p_nc]
            if val == 1.0: 
                p_reward += 20
                self.items[p_nr, p_nc] = 0
            elif val == -1.0: 
                self.ghost_stunned_timer = 20
                p_reward += 60
                self.items[p_nr, p_nc] = 0
            elif val == -2.0: 
                p_reward += 250
                g_reward -= 250 # Ghost fails if Pacman hits the bomb
                done = True 

        # --- Ghost Movement (Speed Balanced) ---
        if self.ghost_stunned_timer > 0:
            self.ghost_stunned_timer -= 1
            # Ghost gets penalized for being stunned
            g_reward -= 0.5
        elif g_action == -1:
            # Ghost is skipping a frame (Wait logic)
            pass 
        else:
            g_dr, g_dc = self.moves[g_action]
            g_nr, g_nc = self.ghost_pos[0] + g_dr, self.ghost_pos[1] + g_dc
            if self.board[g_nr][g_nc] != 2.0:
                self.ghost_pos = [g_nr, g_nc]

        # --- 2. Multi-Agent Reward Shaping ---
        d_new = get_manhattan(self.pacman_pos, self.ghost_pos)
        
        # Pacman logic: Get away
        if d_new > d_old: p_reward += 0.05
        else: p_reward -= 0.05
        
        # Ghost logic: Get closer
        if self.ghost_stunned_timer == 0:
            if d_new < d_old: g_reward += 0.2  # Reward stalking
            else: g_reward -= 0.1             # Punish losing Pacman
        else:
            # While stunned, Ghost should be rewarded for INCREASING distance (running away)
            if d_new > d_old: g_reward += 0.2

        # --- 3. Terminal Conflicts (Zero-Sum) ---
        if self.pacman_pos == self.ghost_pos:
            if self.ghost_stunned_timer == 0:
                p_reward -= 250
                g_reward += 250 # Ghost Win!
                done = True
            else:
                # Pacman "eats" a stunned ghost
                p_reward += 100
                g_reward -= 100
                # Reset ghost to a random empty spot instead of ending game? 
                # For now, let's just count it as a Pacman advantage.

        if not np.any(self.items == 1.0):
            p_reward += 500
            g_reward -= 500 # Ghost failed to protect the pellets
            done = True

        return self.get_state(), p_reward, g_reward, done

def get_a_star_action(env):
    if env.ghost_stunned_timer > 0: 
        return random.randint(0, 3) 
    
    start, target = tuple(env.ghost_pos), tuple(env.pacman_pos)
    pq = [(0, start, [])]
    visited = {start: 0}

    while pq:
        (priority, current, path) = heapq.heappop(pq)
        if current == target: 
            return path[0] if path else random.randint(0, 3)

        for action, (dr, dc) in env.moves.items():
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < 20 and 0 <= neighbor[1] < 20 and env.board[neighbor[0], neighbor[1]] != 2.0:
                new_cost = len(path) + 1
                if neighbor not in visited or new_cost < visited[neighbor]:
                    visited[neighbor] = new_cost
                    # A* Heuristic: Cost so far + Manhattan distance to target
                    h = abs(neighbor[0]-target[0]) + abs(neighbor[1]-target[1])
                    heapq.heappush(pq, (new_cost + h, neighbor, path + [action]))
    
    return random.randint(0, 3)