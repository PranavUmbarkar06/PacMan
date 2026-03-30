from collections import deque
import random

import numpy as np


ACTION_DELTAS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}


def get_rule_based_ghost_action(env):
    """Move along the shortest path to Pacman, or pick a valid fallback move."""
    path = env.shortest_path(env.ghost_pos, env.pacman_pos)
    if len(path) >= 2:
        return env.action_from_positions(path[0], path[1])

    valid_actions = env.get_valid_actions(env.ghost_pos)
    return random.choice(valid_actions) if valid_actions else None


class PacMan:
    def __init__(self, layout, ghost_move_interval=2, ghost_start_delay=0, max_steps=500):
        self.grid = np.array(layout, dtype=np.int8)
        self.rows, self.cols = self.grid.shape
        self.ghost_move_interval = ghost_move_interval
        self.ghost_start_delay = ghost_start_delay
        self.max_steps = max_steps
        self.num_actions = len(ACTION_DELTAS)
        self.reset()

    def configure_difficulty(self, ghost_move_interval=None, ghost_start_delay=None):
        if ghost_move_interval is not None:
            self.ghost_move_interval = max(1, int(ghost_move_interval))
        if ghost_start_delay is not None:
            self.ghost_start_delay = max(0, int(ghost_start_delay))

    def _random_open_cell(self, forbidden=None):
        forbidden = set() if forbidden is None else set(forbidden)
        candidates = [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if self.board[row, col] == 0 and (row, col) not in forbidden
        ]
        return random.choice(candidates)

    def reset(self, randomize_positions=False, min_spawn_distance=8):
        self.board = self.grid.copy()
        if randomize_positions:
            self.pacman_pos = self._random_open_cell()
            self.ghost_pos = self._random_open_cell(forbidden={self.pacman_pos})
            attempts = 0
            while (
                self.maze_distance(self.pacman_pos, self.ghost_pos) < min_spawn_distance
                and attempts < 50
            ):
                self.ghost_pos = self._random_open_cell(forbidden={self.pacman_pos})
                attempts += 1
        else:
            self.pacman_pos = self._closest_open_cell((1, 1))
            self.ghost_pos = self._closest_open_cell((self.rows - 2, self.cols - 2))
        self.pellets = {
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if self.board[row, col] == 0
        }
        self.pellets.discard(self.pacman_pos)
        self.pellets.discard(self.ghost_pos)
        self.total_pellets = len(self.pellets)
        self.steps = 0
        self.ghost_tick = 0
        self.no_pellet_steps = 0
        self.previous_pacman_pos = self.pacman_pos
        self.recent_positions = deque([self.pacman_pos], maxlen=8)
        self.last_outcome = "running"
        self._refresh_items()
        return self.get_state()

    def _closest_open_cell(self, target):
        open_cells = np.argwhere(self.board == 0)
        best = min(open_cells, key=lambda pos: abs(pos[0] - target[0]) + abs(pos[1] - target[1]))
        return (int(best[0]), int(best[1]))

    def _refresh_items(self):
        self.items = np.zeros_like(self.board, dtype=np.float32)
        for row, col in self.pellets:
            self.items[row, col] = 1.0

    def in_bounds(self, position):
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_wall(self, position):
        row, col = position
        return self.board[row, col] == 1

    def move(self, position, action_idx):
        delta = ACTION_DELTAS[action_idx]
        next_pos = (position[0] + delta[0], position[1] + delta[1])
        if not self.in_bounds(next_pos) or self.is_wall(next_pos):
            return position
        return next_pos

    def get_valid_actions(self, position):
        valid = []
        for action_idx in ACTION_DELTAS:
            next_pos = self.move(position, action_idx)
            if next_pos != position:
                valid.append(action_idx)
        return valid

    def legal_action_mask(self, position=None):
        position = self.pacman_pos if position is None else position
        mask = np.zeros(self.num_actions, dtype=np.float32)
        for action_idx in self.get_valid_actions(position):
            mask[action_idx] = 1.0
        return mask

    def action_from_positions(self, start, end):
        delta = (end[0] - start[0], end[1] - start[1])
        for action_idx, action_delta in ACTION_DELTAS.items():
            if action_delta == delta:
                return action_idx
        return None

    def shortest_path(self, start, goal):
        if start == goal:
            return [start]

        queue = deque([start])
        parents = {start: None}

        while queue:
            current = queue.popleft()
            for action_idx in ACTION_DELTAS:
                nxt = self.move(current, action_idx)
                if nxt == current or nxt in parents:
                    continue
                parents[nxt] = current
                if nxt == goal:
                    path = [goal]
                    node = goal
                    while parents[node] is not None:
                        node = parents[node]
                        path.append(node)
                    path.reverse()
                    return path
                queue.append(nxt)

        return [start]

    def maze_distance(self, start, goal):
        path = self.shortest_path(start, goal)
        if path[-1] != goal:
            return self.rows * self.cols
        return len(path) - 1

    def nearest_pellet_distance(self, position=None):
        position = self.pacman_pos if position is None else position
        if not self.pellets:
            return 0
        return min(self.maze_distance(position, pellet) for pellet in self.pellets)

    def ghost_distance(self):
        return self.maze_distance(self.pacman_pos, self.ghost_pos)

    def get_state(self):
        state = np.zeros((5, self.rows, self.cols), dtype=np.float32)
        state[0] = (self.board == 1).astype(np.float32)

        for row, col in self.pellets:
            state[1, row, col] = 1.0

        state[2, self.pacman_pos[0], self.pacman_pos[1]] = 1.0
        state[3, self.ghost_pos[0], self.ghost_pos[1]] = 1.0
        if self.ghost_tick % self.ghost_move_interval == self.ghost_move_interval - 1:
            state[4].fill(1.0)

        return state

    def step(self, pacman_action):
        self.steps += 1
        reward = 0.0
        done = False
        info = {
            "pellets_remaining": len(self.pellets),
            "ghost_moved": False,
            "outcome": "running",
        }

        valid_actions = self.get_valid_actions(self.pacman_pos)
        old_pellet_distance = self.nearest_pellet_distance()
        old_ghost_distance = self.ghost_distance()
        old_position = self.pacman_pos

        if pacman_action not in valid_actions:
            reward -= 0.2
        self.pacman_pos = self.move(self.pacman_pos, pacman_action)

        if self.pacman_pos in self.pellets:
            self.pellets.remove(self.pacman_pos)
            reward += 10.0
            self.no_pellet_steps = 0
        else:
            reward -= 0.1
            self.no_pellet_steps += 1

        new_pellet_distance = self.nearest_pellet_distance()
        if new_pellet_distance < old_pellet_distance:
            reward += 0.25 * (old_pellet_distance - new_pellet_distance)

        if self.pacman_pos == old_position:
            reward -= 0.2
        elif self.pacman_pos == self.previous_pacman_pos:
            reward -= 0.15
        elif self.pacman_pos in self.recent_positions:
            reward -= 0.05

        if self.pacman_pos == self.ghost_pos:
            reward -= 450.0
            done = True
            info["outcome"] = "caught"
        else:
            self.ghost_tick += 1
            if self.steps > self.ghost_start_delay and self.ghost_tick % self.ghost_move_interval == 0:
                ghost_action = get_rule_based_ghost_action(self)
                if ghost_action is not None:
                    self.ghost_pos = self.move(self.ghost_pos, ghost_action)
                    info["ghost_moved"] = True

            new_ghost_distance = self.ghost_distance()
            if new_ghost_distance > old_ghost_distance:
                reward += 0.3 * (new_ghost_distance - old_ghost_distance)
            if new_ghost_distance <= 2:
                reward += 0.1 * max(new_ghost_distance - old_ghost_distance, 0)

            if self.pacman_pos == self.ghost_pos:
                reward -= 450.0
                done = True
                info["outcome"] = "caught"

        if not done and not self.pellets:
            reward += 500.0
            done = True
            info["outcome"] = "cleared"

        if not done and self.steps >= self.max_steps:
            done = True
            info["outcome"] = "timeout"

        self.last_outcome = info["outcome"]
        self.previous_pacman_pos = old_position
        self.recent_positions.append(self.pacman_pos)
        info["pellets_remaining"] = len(self.pellets)
        self._refresh_items()

        return self.get_state(), reward, done, info
