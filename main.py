import pygame
import torch
import numpy as np
import random
import os
from Models.Brain import Brain
from Models.Game import PacMan, get_a_star_action 
from map import LAYOUT

# Constants
TILE_SIZE = 30
FPS = 5 
GHOST_SPEED_DIVIDER = 2  # Ghost moves once every N frames. Increase to slow him down further.

def main():
    pygame.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = PacMan(LAYOUT)
    screen = pygame.display.set_mode((20 * TILE_SIZE, 20 * TILE_SIZE))
    pygame.display.set_caption("Neural Pacman vs Slow A* Ghost")
    clock = pygame.time.Clock()

    # --- 1. Load Images ---
    try:
        p_img_orig = pygame.image.load("pacman.png").convert_alpha()
        g_img_orig = pygame.image.load("ghost.png").convert_alpha()
        p_img_orig = pygame.transform.scale(p_img_orig, (26, 26))
        g_img_orig = pygame.transform.scale(g_img_orig, (26, 26))
        using_custom_imgs = True
    except pygame.error:
        using_custom_imgs = False

    # --- 2. Load Neural Brain ---
    p_brain = Brain().to(device)
    if os.path.exists("p_brain.pth"):
        p_brain.load_state_dict(torch.load("p_brain.pth", map_location=device, weights_only=True))
        print("Loaded Pacman Neural Network.")
    else:
        print("Warning: p_brain.pth not found. Pacman will act randomly.")
    
    p_brain.eval()

    frame_counter = 0
    running = True

    while running:
        frame_counter += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- 3. Decision Making ---
        state = env.get_state()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Pacman Logic: Always moves based on the Brain
            p_action = p_brain(state_t).argmax().item()
            
            # Ghost Logic: Only moves every Nth frame (controlled by GHOST_SPEED_DIVIDER)
            if frame_counter % GHOST_SPEED_DIVIDER == 0:
                g_action = get_a_star_action(env, is_ghost=True)
            else:
                g_action = -1 # Signal to the environment to stay still

        # --- 4. Step Environment ---
        _, p_rew, _, done = env.step(p_action, g_action)
        
        if done:
            print(f"Game Over! Pacman Reward: {p_rew:.2f}")
            pygame.time.delay(800) 
            env.reset()
            frame_counter = 0

        # --- 5. Rendering ---
        screen.fill((0, 0, 0))
        
        for r in range(20):
            for c in range(20):
                rect = (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if env.board[r, c] == 2.0: # Wall
                    pygame.draw.rect(screen, (33, 33, 255), rect, 2)
                elif env.items[r, c] == 1.0: # Pellet
                    pygame.draw.circle(screen, (255, 184, 151), (c*TILE_SIZE+15, r*TILE_SIZE+15), 2)
                elif env.items[r, c] == -1.0: # Stunner
                    pygame.draw.circle(screen, (0, 255, 0), (c*TILE_SIZE+15, r*TILE_SIZE+15), 6)
                elif env.items[r, c] == -2.0: # Bomb
                    pygame.draw.circle(screen, (255, 0, 0), (c*TILE_SIZE+15, r*TILE_SIZE+15), 6)

        # Draw Agents
        p_pos = (env.pacman_pos[1]*TILE_SIZE+2, env.pacman_pos[0]*TILE_SIZE+2)
        g_pos = (env.ghost_pos[1]*TILE_SIZE+2, env.ghost_pos[0]*TILE_SIZE+2)

        if using_custom_imgs:
            angles = [90, 270, 0, 180] # Up, Down, Left, Right
            rot_p = pygame.transform.rotate(p_img_orig, angles[p_action])
            screen.blit(rot_p, p_pos)
            
            # Only update ghost rotation if he actually moved
            g_draw_angle = angles[g_action] if g_action != -1 else 0
            rot_g = pygame.transform.rotate(g_img_orig, g_draw_angle)
            if env.ghost_stunned_timer > 0:
                rot_g.fill((100, 100, 255, 128), special_flags=pygame.BLEND_RGBA_MULT)
            screen.blit(rot_g, g_pos)
        else:
            pygame.draw.circle(screen, (255, 255, 0), (p_pos[0]+13, p_pos[1]+13), 12)
            g_color = (100, 100, 255) if env.ghost_stunned_timer > 0 else (255, 0, 0)
            pygame.draw.circle(screen, g_color, (g_pos[0]+13, g_pos[1]+13), 12)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()