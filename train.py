import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from map import LAYOUT
from Models.Brain import Brain, ReplayMemory, huber_loss
from Models.Game import PacMan


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a Pacman DQN with a rule-based ghost.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay-steps", type=int, default=12_000)
    parser.add_argument("--ghost-interval-start", type=int, default=5)
    parser.add_argument("--ghost-interval-end", type=int, default=2)
    parser.add_argument("--ghost-delay-start", type=int, default=40)
    parser.add_argument("--ghost-delay-end", type=int, default=0)
    parser.add_argument("--curriculum-episodes", type=int, default=0, help="0 scales automatically with total episodes.")
    parser.add_argument("--reward-clip", type=float, default=1.0, help="Clip rewards to +/- this value for DQN updates.")
    parser.add_argument("--random-start-distance", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="p_brain.pth")
    parser.add_argument("--cpu", action="store_true")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step, start, end, decay_steps):
    progress = min(step / max(decay_steps, 1), 1.0)
    return start + progress * (end - start)


def interpolate_schedule(episode_idx, total_episodes, start, end):
    progress = min(max(episode_idx, 0) / max(total_episodes, 1), 1.0)
    return start + (end - start) * progress


def select_action(model, state, legal_actions, epsilon, device):
    if random.random() < epsilon:
        return random.choice(legal_actions)

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)[0].cpu().numpy()

    masked_q = np.full_like(q_values, -np.inf)
    for action in legal_actions:
        masked_q[action] = q_values[action]
    return int(masked_q.argmax())


def optimize_model(model, target_model, optimizer, memory, batch_size, gamma, device):
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    states, actions, rewards, next_states, dones, next_masks = zip(*transitions)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    next_masks = torch.tensor(np.array(next_masks), dtype=torch.float32, device=device)

    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        online_next_q = model(next_states).masked_fill(next_masks <= 0, float("-inf"))
        best_next_actions = online_next_q.argmax(dim=1, keepdim=True)
        target_next_q = target_model(next_states).gather(1, best_next_actions).squeeze(1)
        best_next_q = torch.where(
            torch.isfinite(target_next_q),
            target_next_q,
            torch.zeros_like(target_next_q),
        )
        target_q = rewards + gamma * best_next_q * (1.0 - dones)

    loss = huber_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def train(args):
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    env = PacMan(
        LAYOUT,
        ghost_move_interval=args.ghost_interval_start,
        ghost_start_delay=args.ghost_delay_start,
        max_steps=args.max_steps,
    )

    policy_net = Brain(input_channels=5, num_actions=env.num_actions).to(device)
    target_net = Brain(input_channels=5, num_actions=env.num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    memory = ReplayMemory(capacity=args.replay_size)

    global_step = 0
    curriculum_episodes = args.curriculum_episodes or max(int(args.episodes * 0.8), 1)
    best_score = (-1, -1, float("-inf"))
    total_wins = 0
    model_path = Path(args.model_path)

    for episode in range(1, args.episodes + 1):
        ghost_interval = round(
            interpolate_schedule(
                episode - 1,
                curriculum_episodes,
                args.ghost_interval_start,
                args.ghost_interval_end,
            )
        )
        ghost_delay = round(
            interpolate_schedule(
                episode - 1,
                curriculum_episodes,
                args.ghost_delay_start,
                args.ghost_delay_end,
            )
        )
        env.configure_difficulty(
            ghost_move_interval=ghost_interval,
            ghost_start_delay=ghost_delay,
        )
        state = env.reset(randomize_positions=True, min_spawn_distance=args.random_start_distance)
        episode_reward = 0.0
        losses = []

        for _ in range(args.max_steps):
            epsilon = epsilon_by_step(
                global_step,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay_steps,
            )
            legal_actions = env.get_valid_actions(env.pacman_pos)
            action = select_action(policy_net, state, legal_actions, epsilon, device)

            next_state, reward, done, info = env.step(action)
            next_mask = env.legal_action_mask()
            clipped_reward = float(np.clip(reward, -args.reward_clip, args.reward_clip))
            memory.push(state, action, clipped_reward, next_state, done, next_mask)

            state = next_state
            episode_reward += reward
            global_step += 1

            if global_step >= args.warmup_steps and global_step % args.train_every == 0:
                loss = optimize_model(
                    policy_net,
                    target_net,
                    optimizer,
                    memory,
                    args.batch_size,
                    args.gamma,
                    device,
                )
                if loss is not None:
                    losses.append(loss)

            if global_step % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        avg_loss = float(np.mean(losses)) if losses else 0.0
        pellets_left = info["pellets_remaining"]
        pellets_eaten = env.total_pellets - pellets_left
        win = int(info["outcome"] == "cleared")
        total_wins += win
        print(
            f"Ep {episode:04d} | reward {episode_reward:8.2f} | "
            f"loss {avg_loss:7.4f} | epsilon {epsilon:5.3f} | "
            f"eaten {pellets_eaten:3d} | left {pellets_left:3d} | "
            f"ghost {ghost_interval}t/{ghost_delay}d | wins {total_wins:3d} | outcome {info['outcome']}"
        )

        episode_score = (win, pellets_eaten, episode_reward)
        if episode_score > best_score:
            best_score = episode_score
            torch.save(policy_net.state_dict(), model_path)

    target_net.load_state_dict(policy_net.state_dict())
    torch.save(target_net.state_dict(), model_path)
    print(f"Saved trained Pacman model to {model_path.resolve()}")


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
