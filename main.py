#Run this if on a new system

#import os

#os.system("pip install -r requirements.txt")
#Actual code

import pygame
import torch
import numpy as np
import random
from Models.Brain import Brain
from Models.Game import PacMan, get_a_star_action
from map import LAYOUT

# Constants
TILE_SIZE = 30
FPS = 15  # Adjust this to change visual game speed

def main():
    pygame.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = PacMan(LAYOUT)
    screen = pygame.display.set_mode((20 * TILE_SIZE, 20 * TILE_SIZE))
    pygame.display.set_caption("Neural Pacman vs Elite A* Ghost (Speed Balanced)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18, bold=True)

    # --- 1. Load and Scale Images ---
    try:
        p_img_orig = pygame.image.load("pacman.png").convert_alpha()
        g_img_orig = pygame.image.load("ghost.png").convert_alpha()
        p_img_orig = pygame.transform.scale(p_img_orig, (26, 26))
        g_img_orig = pygame.transform.scale(g_img_orig, (26, 26))
        using_custom_imgs = True
    except pygame.error:
        print("Images not found! Using fallback shapes.")
        using_custom_imgs = False

    # --- 2. Load Pacman Brain ---
    p_brain = Brain().to(device)
    try:
        p_brain.load_state_dict(torch.load("p_brain.pth", map_location=device, weights_only=True))
        print("Successfully loaded trained Pacman Brain.")
    except Exception as e:
        print(f"Loading error: {e}. Check if p_brain.pth exists.")
    
    p_brain.eval()

    # --- 3. Speed Control Variables ---
    frame_counter = 0
    running = True

    while running:
        frame_counter += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- 4. Decision Making ---
        state = env.get_state()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Pacman always moves
            p_action = p_brain(state_t).argmax().item()
            
            # GHOST SPEED LOGIC:
            # Ghost only moves on even frames (50% speed).
            # This allows Pacman to step around the ghost instead of deadlocking.
            if frame_counter % 2 == 0:
                g_action = get_a_star_action(env)
            else:
                g_action = -1  # Signal for "Stay Still"

        # --- 5. Step Environment ---
        _, p_rew, _, done = env.step(p_action, g_action)
        
        if done:
            pygame.time.delay(800) 
            env.reset()
            frame_counter = 0

        # --- 6. Rendering ---
        screen.fill((0, 0, 0))
        
        # Draw Walls and Items
        for r in range(20):
            for c in range(20):
                rect = (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                cell_val = env.board[r, c] + env.items[r, c]
                
                if cell_val == 2.0: # Wall
                    pygame.draw.rect(screen, (30, 30, 150), rect)
                elif cell_val == 1.0: # Pellet
                    pygame.draw.circle(screen, (255, 215, 0), (c*TILE_SIZE+15, r*TILE_SIZE+15), 3)
                elif cell_val == -1.0: # Stunner (Green)
                    pygame.draw.rect(screen, (0, 255, 0), (c*TILE_SIZE+7, r*TILE_SIZE+7, 16, 16))
                elif cell_val == -2.0: # Bomb (Red/White)
                    pygame.draw.circle(screen, (255, 255, 255), (c*TILE_SIZE+15, r*TILE_SIZE+15), 10)
                    pygame.draw.circle(screen, (200, 0, 0), (c*TILE_SIZE+15, r*TILE_SIZE+15), 5)

        # Draw Agents
        p_pos = (env.pacman_pos[1]*TILE_SIZE+2, env.pacman_pos[0]*TILE_SIZE+2)
        g_pos = (env.ghost_pos[1]*TILE_SIZE+2, env.ghost_pos[0]*TILE_SIZE+2)

        if using_custom_imgs:
            angles = [90, 270, 0, 180] # Up, Down, Left, Right
            rot_p = pygame.transform.rotate(p_img_orig, angles[p_action])
            screen.blit(rot_p, p_pos)
            
            if env.ghost_stunned_timer > 0:
                # Ghost Stunned Visual
                stun_overlay = pygame.Surface((26, 26))
                stun_overlay.set_alpha(160)
                stun_overlay.fill((0, 0, 255))
                screen.blit(g_img_orig, g_pos)
                screen.blit(stun_overlay, g_pos)
            else:
                # Normal Ghost
                # Use a default angle or look at Pacman if g_action is -1
                ghost_angle_idx = g_action if g_action != -1 else 0
                rot_g = pygame.transform.rotate(g_img_orig, angles[ghost_angle_idx])
                screen.blit(rot_g, g_pos)
        else:
            pygame.draw.circle(screen, (255, 255, 0), (p_pos[0]+13, p_pos[1]+13), 12)
            g_color = (0, 0, 255) if env.ghost_stunned_timer > 0 else (255, 0, 0)
            pygame.draw.circle(screen, g_color, (g_pos[0]+13, g_pos[1]+13), 12)

        # --- 7. UI Overlay ---
        status_text = "GHOST: STUNNED" if env.ghost_stunned_timer > 0 else "GHOST: HUNTING"
        status_color = (0, 255, 255) if env.ghost_stunned_timer > 0 else (255, 50, 50)
        
        screen.blit(font.render(status_text, True, status_color), (15, 15))
        screen.blit(font.render(f"P-Reward: {p_rew:.2f}", True, (255, 255, 255)), (15, 35))
        screen.blit(font.render(f"Speed: 100% vs 50%", True, (200, 200, 200)), (15, 55))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()