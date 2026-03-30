import numpy as np
import random
import heapq

def get_a_star_action(env, is_ghost=True):
    start = env.ghost_pos if is_ghost else env.pacman_pos
    goal = env.pacman_pos if is_ghost else env.ghost_pos
    
    queue = [(0, start[0], start[1], [])]
    visited = set([(start[0], start[1])])
    
    while queue:
        dist, r, c, path = queue.pop(0)
        if [r, c] == goal:
            return path[0] if path else -1
            
        for a, (dr, dc) in env.moves.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < 20 and 0 <= nc < 20 and env.board[nr, nc] != 2.0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((dist + 1, nr, nc, path + [a]))
                
    valid = env.get_valid_actions(start)
    return random.choice(valid) if valid else 0

class PacMan:
    def __init__(self, board):
        self.original_layout = np.array(board).astype(float) * 2.0
        self.moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]} # Up, Down, Left, Right
        self.reset()

    def reset(self):
        self.board = np.copy(self.original_layout)
        self.items = (self.original_layout == 0).astype(float) 
        self.ghost_stunned_timer = 0
        
        # 1. Get all valid empty slots
        empty_slots = np.argwhere(self.original_layout == 0).tolist()
        
        # 2. Define the four absolute corners of a 20x20 grid
        corners = [[0, 0], [0, 19], [19, 0], [19, 19]]
        
        # 3. Find the closest valid empty slot for each corner
        # This prevents the game from crashing if a corner is a wall
        spawn_points = []
        for corner in corners:
            # Sort empty slots by distance to this specific corner
            closest_slot = min(empty_slots, key=lambda s: abs(s[0]-corner[0]) + abs(s[1]-corner[1]))
            spawn_points.append(closest_slot)
            # Remove it from list so we don't pick the same spot twice
            empty_slots.remove(closest_slot)

        # 4. Assign positions (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
        # Let's put Pacman and Ghost at opposite diagonals
        self.pacman_pos = spawn_points[0] # Top-Left
        self.ghost_pos  = spawn_points[3] # Bottom-Right
        self.stunner_pos = spawn_points[1] # Top-Right
        self.bomb_pos    = spawn_points[2] # Bottom-Left
        
        # 5. Place special items on the items map
        self.items[self.stunner_pos[0], self.stunner_pos[1]] = -1.0
        self.items[self.bomb_pos[0], self.bomb_pos[1]] = -2.0
        
        return self.get_state()

    def get_state(self):
        state = np.zeros((6, 20, 20), dtype=np.float32)
        # Channel 0: Walls
        state[0] = (self.board == 2.0).astype(float)
        # Channel 1: Pellets
        state[1] = (self.items == 1.0).astype(float)
        # Channel 2: Stunners
        state[2] = (self.items == -1.0).astype(float)
        # Channel 3: Bombs
        state[3] = (self.items == -2.0).astype(float)
        # Channel 4: Pacman
        state[4, self.pacman_pos[0], self.pacman_pos[1]] = 1.0
        # Channel 5: Ghost
        state[5, self.ghost_pos[0], self.ghost_pos[1]] = 1.0
        return state

    def get_a_star_distance(self, start, goal):
        # A simple BFS/A* to get exactly shortest path length considering walls
        queue = [(0, start[0], start[1])]
        visited = set([(start[0], start[1])])
        while queue:
            dist, r, c = queue.pop(0)
            if [r, c] == goal:
                return dist
            for dr, dc in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 20 and 0 <= nc < 20 and self.board[nr, nc] != 2.0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((dist + 1, nr, nc))
        return 999 # Fallback if unreachable


    def get_valid_actions(self, pos):
        """Hard Rule: Filter out any action that leads to a wall (2.0)"""
        valid = []
        for action, (dr, dc) in self.moves.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < 20 and 0 <= nc < 20 and self.board[nr, nc] != 2.0:
                valid.append(action)
        return valid if valid else [0] # Safety fallback

    def step(self, p_action, g_action):
        # Use A* distance for better reward shaping
        d_old = self.get_a_star_distance(self.pacman_pos, self.ghost_pos)
        p_reward, g_reward, done = -0.1, -0.1, False # Reduced step penalty to ease exploration

        # --- Pacman Movement ---
        if p_action in self.get_valid_actions(self.pacman_pos):
            p_dr, p_dc = self.moves[p_action]
            self.pacman_pos = [self.pacman_pos[0] + p_dr, self.pacman_pos[1] + p_dc]
            val = self.items[self.pacman_pos[0], self.pacman_pos[1]]
            if val == 1.0: 
                p_reward += 10.0
                self.items[self.pacman_pos[0], self.pacman_pos[1]] = 0
            elif val == -1.0:
                p_reward += 30.0
                self.ghost_stunned_timer = 20
                self.items[self.pacman_pos[0], self.pacman_pos[1]] = 0
            elif val == -2.0:
                p_reward += 50.0
                done = True 
        else:
             p_reward -= 1.0 # Penalty for hitting a wall

        # --- Ghost Movement ---
        if self.ghost_stunned_timer > 0:
            self.ghost_stunned_timer -= 1
        elif g_action in self.get_valid_actions(self.ghost_pos):
            g_dr, g_dc = self.moves[g_action]
            self.ghost_pos = [self.ghost_pos[0] + g_dr, self.ghost_pos[1] + g_dc]
        else:
             g_reward -= 1.0 # Penalty for hitting a wall

        # --- Vectorized Shaping ---
        d_new = self.get_a_star_distance(self.pacman_pos, self.ghost_pos)
        
        # Ghost Reward Shaping: Reward for getting closer to Pacman
        if self.ghost_stunned_timer == 0:
            if d_new < d_old:
                g_reward += 2.0
            elif d_new > d_old:
                g_reward -= 2.0


        # --- Conflicts ---
        if self.pacman_pos == self.ghost_pos:
            if self.ghost_stunned_timer == 0:
                p_reward -= 30; g_reward += 30; done = True
            else:
                p_reward += 50; g_reward -= 50
                empty_slots = np.argwhere(self.board == 0).tolist()
                self.ghost_pos = random.choice(empty_slots)

        if not np.any(self.items == 1.0):
            p_reward += 100; done = True

        return self.get_state(), p_reward, g_reward, done
    




