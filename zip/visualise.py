"""
visualize.py  –  Pac-Man DQN viewer
Movement philosophy: each sprite travels from tile A to tile B over a fixed
duration using a proper easing function sampled over a normalised phase t∈[0,1].
  • Pac-Man  → cubic ease-in-out  (fast middle, soft ends — purposeful)
  • Ghost    → smootherstep (degree-5, C² continuous — slightly heavier feel)
No exponential-decay lerp anywhere.  Sprites always land exactly on the target.
"""

import argparse
import math
import random
from pathlib import Path

import pygame
import torch

from map import LAYOUT
from Models.Brain import Brain
from Models.Game import PacMan


# ── Constants ─────────────────────────────────────────────────────────────────
TILE_SIZE  = 32
HUD_HEIGHT = 48
MOVE_MS    = 120          # ms to travel one tile  (tune to taste)
RENDER_FPS = 60           # render loop; game steps at --fps

# ── Palette ───────────────────────────────────────────────────────────────────
COLOR_BG          = (10,  10,  20)
COLOR_WALL_INNER  = (20,  30,  90)
COLOR_WALL_EDGE   = (60, 100, 230)
COLOR_WALL_GLOW   = (40,  70, 180)
COLOR_PELLET      = (255, 220, 140)
COLOR_PELLET_GLOW = (255, 180,  60)
COLOR_HUD_BG      = (12,  12,  25)
COLOR_HUD_TEXT    = (220, 220, 255)
COLOR_HUD_ACC     = (100, 180, 255)
GO_BG             = (8,    4,  16)
GO_TEXT           = (255,  60,  80)
GO_SUBTEXT        = (200, 160, 255)
GO_OUTLINE        = (255,  20,  60)
WIN_TEXT          = ( 80, 255, 160)
WIN_SUBTEXT       = (180, 255, 220)
WIN_OUTLINE       = ( 40, 220, 120)


# ── Easing functions (t in [0,1] -> [0,1]) ───────────────────────────────────

def ease_cubic_inout(t: float) -> float:
    """Cubic ease-in-out.  Slow start, fast middle, slow end.  C1 at endpoints."""
    if t < 0.5:
        return 4.0 * t * t * t
    p = 2.0 * t - 2.0
    return 0.5 * p * p * p + 1.0


def smootherstep(t: float) -> float:
    """Perlin smootherstep (degree-5 polynomial).  C2 continuous."""
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# ── Sprite transition tracker ─────────────────────────────────────────────────

class SpriteMotion:
    """
    Tracks a single sprite moving between grid tiles.
    Call .push(new_tile, now_ms) when the game logic moves the sprite.
    Call .pixel_pos(now_ms)      each frame to get interpolated screen coords.
    """

    def __init__(self, tile, ease_fn):
        r, c          = tile
        self.src_px   = float(c * TILE_SIZE)
        self.src_py   = float(r * TILE_SIZE)
        self.dst_px   = self.src_px
        self.dst_py   = self.src_py
        self.start_ms = 0
        self.ease     = ease_fn

    def _phase(self, now_ms: int) -> float:
        elapsed = now_ms - self.start_ms
        return min(1.0, elapsed / MOVE_MS) if MOVE_MS > 0 else 1.0

    def push(self, new_tile, now_ms: int):
        """Begin a new move.  Source = current visual position (mid-transit ok)."""
        e             = self.ease(self._phase(now_ms))
        self.src_px   = self.src_px + (self.dst_px - self.src_px) * e
        self.src_py   = self.src_py + (self.dst_py - self.src_py) * e
        r, c          = new_tile
        self.dst_px   = float(c * TILE_SIZE)
        self.dst_py   = float(r * TILE_SIZE)
        self.start_ms = now_ms

    def snap(self, tile):
        """Hard-snap to tile, no animation (used on episode reset)."""
        r, c          = tile
        self.src_px = self.dst_px = float(c * TILE_SIZE)
        self.src_py = self.dst_py = float(r * TILE_SIZE)
        self.start_ms = pygame.time.get_ticks()

    def pixel_pos(self, now_ms: int):
        """Return top-left (px, py) in screen space (HUD offset applied)."""
        e  = self.ease(self._phase(now_ms))
        px = self.src_px + (self.dst_px - self.src_px) * e
        py = self.src_py + (self.dst_py - self.src_py) * e
        return px + 2.0, py + 2.0 + HUD_HEIGHT


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained Pacman agent.")
    parser.add_argument("--model-path", type=str,   default="p_brain.pth")
    parser.add_argument("--fps",        type=int,   default=8,
                        help="Game logic steps per second.")
    parser.add_argument("--episodes",   type=int,   default=0,
                        help="0 = run forever.")
    parser.add_argument("--epsilon",    type=float, default=0.1)
    parser.add_argument("--cpu",        action="store_true")
    return parser.parse_args()


# ── Agent ─────────────────────────────────────────────────────────────────────

def choose_action(model, env, state, device, epsilon):
    legal = env.get_valid_actions(env.pacman_pos)
    if random.random() < epsilon:
        return random.choice(legal)
    t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q = model(t)[0].cpu()
    return int(max(legal, key=lambda a: q[a].item()))


# ── Assets ────────────────────────────────────────────────────────────────────

def load_images():
    p  = pygame.image.load("pacman.png").convert_alpha()
    g  = pygame.image.load("ghost.png").convert_alpha()
    sz = TILE_SIZE - 4
    return (pygame.transform.scale(p, (sz, sz)),
            pygame.transform.scale(g, (sz, sz)))


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _glow(surface, color, center, radius, layers=4):
    for i in range(layers, 0, -1):
        alpha = int(55 * i / layers)
        r     = int(radius * (1.0 + 0.55 * i / layers))
        s     = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (r, r), r)
        surface.blit(s, (center[0] - r, center[1] - r),
                     special_flags=pygame.BLEND_RGBA_ADD)


def draw_wall(surface, rect):
    pygame.draw.rect(surface, COLOR_WALL_INNER, rect)
    pygame.draw.rect(surface, COLOR_WALL_EDGE,  rect, 2)
    for cx, cy in [(rect.left+3, rect.top+3),   (rect.right-4, rect.top+3),
                   (rect.left+3, rect.bottom-4), (rect.right-4, rect.bottom-4)]:
        pygame.draw.circle(surface, COLOR_WALL_GLOW, (cx, cy), 1)


def draw_pellet(surface, center, t):
    pulse  = 0.75 + 0.25 * math.sin(t * 3.5 + center[0] * 0.25)
    radius = max(1, int(3 * pulse))
    _glow(surface, COLOR_PELLET_GLOW, center, radius + 3, layers=3)
    pygame.draw.circle(surface, COLOR_PELLET, center, radius)


def draw_hud(surface, font_big, font_small, episode, reward, pellets, w, t):
    pygame.draw.rect(surface, COLOR_HUD_BG, (0, 0, w, HUD_HEIGHT))
    pygame.draw.line(surface, COLOR_WALL_EDGE,
                     (0, HUD_HEIGHT-1), (w, HUD_HEIGHT-1), 1)
    surface.blit(font_big.render(f"EP {episode}", True, COLOR_HUD_ACC), (12, 10))
    sc = font_big.render(f"SCORE  {reward:+.0f}", True, COLOR_HUD_TEXT)
    surface.blit(sc, (w // 2 - sc.get_width() // 2, 10))
    pl = font_small.render(f"Pellets: {pellets}", True, COLOR_HUD_TEXT)
    surface.blit(pl, (w - pl.get_width() - 12, 15))


def draw_board(surface, env, pac_motion, ghost_motion,
               pac_img, ghost_img, now_ms, t):
    surface.fill(COLOR_BG, pygame.Rect(0, HUD_HEIGHT,
                                       env.cols * TILE_SIZE,
                                       env.rows * TILE_SIZE))
    for row in range(env.rows):
        for col in range(env.cols):
            rect = pygame.Rect(col * TILE_SIZE,
                               row * TILE_SIZE + HUD_HEIGHT,
                               TILE_SIZE, TILE_SIZE)
            if   env.board[row, col] == 1:
                draw_wall(surface, rect)
            elif env.items[row, col] == 1.0:
                draw_pellet(surface, rect.center, t)

    sz = TILE_SIZE - 4

    # Ghost
    gx, gy = ghost_motion.pixel_pos(now_ms)
    _glow(surface, (220, 60, 60),
          (int(gx) + sz // 2, int(gy) + sz // 2), sz // 2 + 4, layers=4)
    surface.blit(ghost_img, (int(gx), int(gy)))

    # Pac-Man
    px, py = pac_motion.pixel_pos(now_ms)
    _glow(surface, (255, 220, 0),
          (int(px) + sz // 2, int(py) + sz // 2), sz // 2 + 4, layers=4)
    surface.blit(pac_img, (int(px), int(py)))


# ── End-of-episode overlay ────────────────────────────────────────────────────

def _end_overlay(surface, font_big, font_med, font_small,
                 outcome, reward, episode, w, h, frac):
    won     = (outcome == "win")
    txt_col = WIN_TEXT    if won else GO_TEXT
    sub_col = WIN_SUBTEXT if won else GO_SUBTEXT
    out_col = WIN_OUTLINE if won else GO_OUTLINE

    ov = pygame.Surface((w, h), pygame.SRCALPHA)
    ov.fill((*GO_BG, int(210 * frac)))
    surface.blit(ov, (0, 0))

    cx, cy = w // 2, h // 2

    # Deterministic scan-lines spaced by golden ratio — no random flicker
    if frac > 0.3:
        n = int(8 * frac)
        for i in range(n):
            ly  = int(cy - 80 + 160 * (i / max(n - 1, 1)))

            # FIX: ensure width is always valid (>0)
            lw  = max(1, int(w * (0.4 + 0.4 * math.sin(i * 1.6180339887))))

            lx  = (w - lw) // 2
            ls  = pygame.Surface((lw, 1), pygame.SRCALPHA)
            ls.fill((*out_col, int(30 * frac)))
            surface.blit(ls, (lx, ly))

    # Headline: cubic-ease drop-in from above, size grows with frac
    headline  = "YOU WIN!" if won else "GAME OVER"
    font_sz   = max(8, int(80 * frac))
    try:
        dyn = pygame.font.Font(None, font_sz)
    except Exception:
        dyn = font_big

    drop = ease_cubic_inout(frac)
    head_y = int(cy - 60 - 40 * (1.0 - drop))

    for dx, dy in [(-3, 3), (3, 3), (-3, -3), (3, -3)]:
        sh = dyn.render(headline, True, out_col)
        surface.blit(sh, (cx - sh.get_width() // 2 + dx, head_y + dy))

    head_surf = dyn.render(headline, True, txt_col)
    ha = pygame.Surface(head_surf.get_size(), pygame.SRCALPHA)
    ha.blit(head_surf, (0, 0))
    ha.set_alpha(int(255 * frac))
    surface.blit(ha, (cx - ha.get_width() // 2, head_y))

    # Sub-text: smootherstep fade-in + rise from below, starts at frac=0.55
    if frac > 0.55:
        sf        = smootherstep((frac - 0.55) / 0.45)
        sub_alpha = int(255 * sf)
        sub_y     = int(cy + 10 + 20 * (1.0 - sf))

        sub1 = font_med.render(
            f"Episode {episode}   ·   Score {reward:+.0f}", True, sub_col)
        s1 = pygame.Surface(sub1.get_size(), pygame.SRCALPHA)
        s1.blit(sub1, (0, 0))
        s1.set_alpha(sub_alpha)
        surface.blit(s1, (cx - s1.get_width() // 2, sub_y))

        sub2 = font_small.render("Continuing…", True, (160, 160, 200))
        s2 = pygame.Surface(sub2.get_size(), pygame.SRCALPHA)
        s2.blit(sub2, (0, 0))
        s2.set_alpha(sub_alpha)
        surface.blit(s2, (cx - s2.get_width() // 2, sub_y + 36))
def run_end_animation(screen, font_big, font_med, font_small,
                      outcome, reward, episode, w, h, clock,
                      env, pac_motion, ghost_motion, pac_img, ghost_img,
                      fade_ms=520, hold_ms=980):
    total = fade_ms + hold_ms
    start = pygame.time.get_ticks()
    t     = 0.0

    while True:
        dt     = clock.tick(RENDER_FPS) / 1000.0
        t     += dt
        now_ms = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit

        elapsed = pygame.time.get_ticks() - start
        frac    = smootherstep(min(1.0, elapsed / fade_ms))

        screen.fill(COLOR_BG)
        draw_board(screen, env, pac_motion, ghost_motion,
                   pac_img, ghost_img, now_ms, t)
        _end_overlay(screen, font_big, font_med, font_small,
                     outcome, reward, episode, w, h, frac)
        pygame.display.flip()

        if elapsed >= total:
            break


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    env   = PacMan(LAYOUT)
    model = Brain(input_channels=5, num_actions=env.num_actions).to(device)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path.resolve()}. "
            "Train first with `python train.py`.")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as exc:
        raise RuntimeError(
            "Saved model does not match architecture. "
            "Run `python train.py` to regenerate.") from exc

    model.eval()

    pygame.init()
    screen_w = env.cols * TILE_SIZE
    screen_h = env.rows * TILE_SIZE + HUD_HEIGHT
    screen   = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Pacman DQN  ·  Refined")
    clock    = pygame.time.Clock()

    font_big   = pygame.font.Font(None, 28)
    font_med   = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 20)

    pac_img, ghost_img = load_images()

    pac_motion   = SpriteMotion(env.pacman_pos, ease_cubic_inout)
    ghost_motion = SpriteMotion(env.ghost_pos,  smootherstep)

    # Game-step timing decoupled from render FPS
    step_interval = 1000 // max(1, args.fps)
    last_step_ms  = pygame.time.get_ticks()

    episode        = 0
    state          = env.reset()
    running        = True
    episode_reward = 0.0
    t              = 0.0
    done           = False
    info           = {"pellets_remaining": "?", "outcome": ""}

    while running:
        dt     = clock.tick(RENDER_FPS) / 1000.0
        t     += dt
        now_ms = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # ── Game logic step ───────────────────────────────────────────────────
        if not done and (now_ms - last_step_ms) >= step_interval:
            last_step_ms = now_ms
            action = choose_action(model, env, state, device, args.epsilon)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            pac_motion.push(env.pacman_pos, now_ms)
            ghost_motion.push(env.ghost_pos,  now_ms)

        # ── Render ────────────────────────────────────────────────────────────
        screen.fill(COLOR_BG)
        draw_board(screen, env, pac_motion, ghost_motion,
                   pac_img, ghost_img, now_ms, t)
        draw_hud(screen, font_big, font_small,
                 episode + 1, episode_reward,
                 info.get("pellets_remaining", "?"), screen_w, t)
        pygame.display.flip()

        # ── Episode end ───────────────────────────────────────────────────────
        if done:
            episode += 1
            outcome = info.get("outcome", "unknown")
            print(f"Episode {episode:4d} | reward {episode_reward:7.2f} | "
                  f"pellets left {info.get('pellets_remaining','?')} | "
                  f"outcome {outcome}")

            run_end_animation(screen, font_big, font_med, font_small,
                              outcome, episode_reward, episode,
                              screen_w, screen_h, clock,
                              env, pac_motion, ghost_motion, pac_img, ghost_img,
                              fade_ms=520, hold_ms=980)

            if args.episodes and episode >= args.episodes:
                break

            state = env.reset()
            episode_reward = 0.0
            done  = False
            pac_motion.snap(env.pacman_pos)
            ghost_motion.snap(env.ghost_pos)
            last_step_ms = pygame.time.get_ticks()

    pygame.quit()


if __name__ == "__main__":
    main()