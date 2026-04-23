import argparse
import random
from pathlib import Path

import pygame
import torch

from map import LAYOUT
from Models.Brain import Brain
from Models.Game import PacMan


TILE_SIZE = 30


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained Pacman agent.")
    parser.add_argument("--model-path", type=str, default="p_brain.pth")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=0, help="0 means run until the window is closed.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate for evaluation runs.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def choose_action(model, env, state, device, epsilon):
    legal_actions = env.get_valid_actions(env.pacman_pos)

    if random.random() < epsilon:
        return random.choice(legal_actions)

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)[0].cpu()

    best_action = max(legal_actions, key=lambda action: q_values[action].item())
    return int(best_action)


def load_images():
    pacman_img = pygame.image.load("pacman.png").convert_alpha()
    ghost_img = pygame.image.load("ghost.png").convert_alpha()

    pacman_img = pygame.transform.scale(pacman_img, (TILE_SIZE - 6, TILE_SIZE - 6))
    ghost_img = pygame.transform.scale(ghost_img, (TILE_SIZE - 6, TILE_SIZE - 6))

    return pacman_img, ghost_img


def draw_env(screen, env, pacman_img, ghost_img):
    screen.fill((0, 0, 0))

    for row in range(env.rows):
        for col in range(env.cols):
            rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)

            # Walls
            if env.board[row, col] == 1:
                pygame.draw.rect(screen, (33, 33, 255), rect, 2)

            # Pellets
            elif env.items[row, col] == 1.0:
                pygame.draw.circle(screen, (255, 214, 160), rect.center, 3)

    # Pacman position
    pac_x = env.pacman_pos[1] * TILE_SIZE + 3
    pac_y = env.pacman_pos[0] * TILE_SIZE + 3

    # Ghost position
    ghost_x = env.ghost_pos[1] * TILE_SIZE + 3
    ghost_y = env.ghost_pos[0] * TILE_SIZE + 3

    screen.blit(pacman_img, (pac_x, pac_y))
    screen.blit(ghost_img, (ghost_x, ghost_y))


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    env = PacMan(LAYOUT)
    model = Brain(input_channels=5, num_actions=env.num_actions).to(device)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path.resolve()}. Train first with `python train.py`."
        )

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as exc:
        raise RuntimeError(
            "The saved model does not match the new paper-style DQN architecture. "
            "Run `python train.py` to generate a fresh checkpoint."
        ) from exc

    model.eval()

    pygame.init()
    screen = pygame.display.set_mode((env.cols * TILE_SIZE, env.rows * TILE_SIZE))
    pygame.display.set_caption("Pacman DQN vs Rule-Based Ghost")
    clock = pygame.time.Clock()

    # Load custom images
    pacman_img, ghost_img = load_images()

    episode = 0
    state = env.reset()
    running = True
    episode_reward = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = choose_action(model, env, state, device, args.epsilon)
        state, reward, done, info = env.step(action)
        episode_reward += reward

        draw_env(screen, env, pacman_img, ghost_img)
        pygame.display.flip()
        clock.tick(args.fps)

        if done:
            episode += 1
            print(
                f"Episode {episode} finished | reward {episode_reward:7.2f} | "
                f"pellets left {info['pellets_remaining']} | outcome {info['outcome']}"
            )

            if args.episodes and episode >= args.episodes:
                break

            pygame.time.delay(750)
            state = env.reset()
            episode_reward = 0.0

    pygame.quit()


if __name__ == "__main__":
    main()