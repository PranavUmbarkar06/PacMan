import pygame
import numpy as np
import random
import os
# Ensure your Game.py contains the PacMan class and get_a_star_action
from Models.Game import PacMan, get_a_star_action 
from map import LAYOUT

# Constants
TILE_SIZE = 30
FPS = 5 # Slightly slower for better human control

def main():
    pygame.init()
    env = PacMan(LAYOUT)
    screen = pygame.display.set_mode((20 * TILE_SIZE, 20 * TILE_SIZE))
    pygame.display.set_caption("Manual Pacman vs A* Ghost")
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

    frame_counter = 0
    running = True
    current_p_action = 3 # Default start moving Right

    while running:
        frame_counter += 1
        
        # --- 2. Keyboard Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:    current_p_action = 0
                elif event.key == pygame.K_DOWN:  current_p_action = 1
                elif event.key == pygame.K_LEFT:  current_p_action = 2
                elif event.key == pygame.K_RIGHT: current_p_action = 3

        # --- 3. Decision Making ---
        # Ghost moves every 2nd frame to be "fair" to the human player
        if frame_counter % 2 == 0:
            g_action = get_a_star_action(env, is_ghost=True)
        else:
            g_action = -1 # Special signal to stay still

        # --- 4. Step Environment ---
        _, p_rew, g_rew, done = env.step(current_p_action, g_action)
        
        if done:
            print(f"Final Score -> Pacman: {p_rew:.1f} | Ghost: {g_rew:.1f}")
            pygame.time.delay(1000) 
            env.reset()
            frame_counter = 0

        # --- 5. Rendering ---
        screen.fill((0, 0, 0))
        
        # Draw Map
        for r in range(20):
            for c in range(20):
                rect = (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if env.board[r, c] == 2.0:
                    pygame.draw.rect(screen, (33, 33, 255), rect, 2) # Blue walls
                elif env.items[r, c] == 1.0: # Pellets
                    pygame.draw.circle(screen, (255, 184, 151), (c*TILE_SIZE+15, r*TILE_SIZE+15), 2)
                elif env.items[r, c] == -1.0: # Stunner
                    pygame.draw.circle(screen, (0, 255, 0), (c*TILE_SIZE+15, r*TILE_SIZE+15), 6)
                elif env.items[r, c] == -2.0: # Bomb
                    pygame.draw.circle(screen, (255, 0, 0), (c*TILE_SIZE+15, r*TILE_SIZE+15), 6)

        # Draw Agents
        p_pos = (env.pacman_pos[1]*TILE_SIZE+2, env.pacman_pos[0]*TILE_SIZE+2)
        g_pos = (env.ghost_pos[1]*TILE_SIZE+2, env.ghost_pos[0]*TILE_SIZE+2)

        if using_custom_imgs:
            angles = [90, 270, 0, 180] 
            rot_p = pygame.transform.rotate(p_img_orig, angles[current_p_action])
            screen.blit(rot_p, p_pos)
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