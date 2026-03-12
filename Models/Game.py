import numpy as np
class PacMan:
    def __init__(self,board):
        self.board=np.array(board)
        self.reset()
        self.moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}


    def reset(self):
        self.pacman_pos=[1,1]
        self.ghost_pos=[18,18]
        self.pellets= (self.board==0).copy()
        return self.get_state()

    def get_state(self):
        state=np.zeros((3,20,20),dtype=np.float32)
        #0 layer is board,1 layer is pacman, 2 layer is ghost
        state[0]=np.array(self.board)
        state[1, self.pacman_pos[0], self.pacman_pos[1]] = 1.0
        
        
        state[2, self.ghost_pos[0], self.ghost_pos[1]] = 1.0

        return state
    
    def step_pac(self,action):
        dr,dc=self.moves[action]
        new_r = self.pacman_pos[0] + dr
        new_c = self.pacman_pos[1] + dc

        reward= -0.1

        done= False

        if self.board[new_r][new_c]==0:
            self.pacman_pos=[new_r,new_c]
            if self.pellets[new_r][new_c]:
                self.pellets[new_r][new_c] = False
                reward+=10
        
        if self.pacman_pos==self.ghost_pos:
            reward+= -200
            done = True
        return self.get_state(),reward,done
    
    def step(self, p_action, g_action):
   
        p_reward = -0.1  
        g_reward = -0.1 
        done = False

      
        p_dr, p_dc = self.moves[p_action]
        p_nr, p_nc = self.pacman_pos[0] + p_dr, self.pacman_pos[1] + p_dc
        if self.board[p_nr][p_nc] == 0:
            self.pacman_pos = [p_nr, p_nc]
            if self.pellets[p_nr][p_nc]:
                self.pellets[p_nr][p_nc] = False
                p_reward += 10

       
        g_dr, g_dc = self.moves[g_action]
        g_nr, g_nc = self.ghost_pos[0] + g_dr, self.ghost_pos[1] + g_dc
        if self.board[g_nr][g_nc] == 0:
            self.ghost_pos = [g_nr, g_nc]

       
        if self.pacman_pos == self.ghost_pos:
            p_reward -= 200  # PACMAN PENALTY
            g_reward += 200  # GHOST REWARD
            done = True
        
        # 4. Check if all pellets are gone (Pacman wins)
        if not np.any(self.pellets):
            p_reward += 500
            done = True

        return self.get_state(), p_reward, g_reward, done
    
    
