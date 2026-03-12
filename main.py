import pygame
import numpy as np
import torch
from Models import Brain, Game 
from map import LAYOUT
 # Ensure your Model is imported

# Constants
TILE_SIZE = 30
FPS = 15
WINDOW_TIME = 2000 # 2 seconds in milliseconds

def main():
    pygame.init()
    env = Game.PacMan(LAYOUT)
    screen = pygame.display.set_mode((20 * TILE_SIZE, 20 * TILE_SIZE))
    clock = pygame.time.Clock()
    
    # Load AI Brain (Optional if you just want to test logic first)
    p_brain = Brain.PacmanBrain()
    # p_brain.load_state_dict(torch.load("pacman_master.pth")) # Load if you have it
    p_brain.eval()

    is_user_turn = True
    last_switch_time = pygame.time.get_ticks()
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        p_action = None
        
        # 1. Switch Logic (The Round Robin)
        if current_time - last_switch_time > WINDOW_TIME:
            is_user_turn = not is_user_turn
            last_switch_time = current_time
            print(f"--- Switched to {'USER' if is_user_turn else 'AGENT'} Turn ---")

        # 2. Input Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # User only acts if it's their turn
            if is_user_turn and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:    p_action = 0
                if event.key == pygame.K_DOWN:  p_action = 1
                if event.key == pygame.K_LEFT:  p_action = 2
                if event.key == pygame.K_RIGHT: p_action = 3

        # 3. Agent Logic (If it's NOT user turn)
        if not is_user_turn:
            with torch.no_grad():
                state = env.get_state()
                state_t = torch.FloatTensor(state).unsqueeze(0)
                p_action = p_brain(state_t).argmax().item()
            # Add a small delay so the AI doesn't move 60 times a second
            pygame.time.delay(100) 

        # 4. Step the Environment
        if p_action is not None:
            g_action = np.random.randint(0, 4) # Ghost remains random/AI
            state, p_rew, g_rew, done = env.step(p_action, g_action)
            
            if done:
                env.reset()

        # 5. Visual Feedback (Show who is in control)
        screen.fill((0, 0, 0))
        # Draw Map... (Same as your previous code)
        for r in range(20):
            for c in range(20):
                rect = (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if env.board[r, c] == 1:
                    pygame.draw.rect(screen, (0, 0, 255), rect)
                elif env.pellets[r, c]:
                    pygame.draw.circle(screen, (255, 215, 0), (c*TILE_SIZE+15, r*TILE_SIZE+15), 3)

        # Draw a "Control Indicator" Border
        indicator_color = (0, 255, 0) if is_user_turn else (255, 165, 0)
        pygame.draw.rect(screen, indicator_color, (0, 0, 20*TILE_SIZE, 20*TILE_SIZE), 5)

        # Draw Agents
        pygame.draw.circle(screen, (255, 255, 0), (env.pacman_pos[1]*TILE_SIZE+15, env.pacman_pos[0]*TILE_SIZE+15), 12)
        pygame.draw.circle(screen, (255, 0, 0), (env.ghost_pos[1]*TILE_SIZE+15, env.ghost_pos[0]*TILE_SIZE+15), 12)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()



if __name__ == "__main__":
    main()