# PacManDRL Project Report

## 1. Executive Summary

PacManDRL is a custom reinforcement learning project in which a Pacman agent learns to collect pellets while avoiding a ghost in a 20x20 maze. The project was inspired by the Stanford CS229 Pacman reinforcement learning paper, but it was adapted into a focused single-agent learning setup. In this version, Pacman is the only learning agent, while the ghost follows a deterministic rule-based shortest-path policy and moves at half of Pacman's speed.

The goal of the project is to train an agent that balances two competing objectives: maximize pellet collection and survive long enough to clear the board. To make this possible, the project combines a custom game environment, a convolutional neural network, replay-based DQN training, curriculum learning, and practical stability improvements such as target networks, reward clipping, randomized starts, and checkpoint-based resumption.

This report explains how the project works, what architecture it uses, how training is organized, what design decisions were made, and why the system is relevant to real-world sequential decision-making problems.

## 2. Problem Definition

The project models Pacman as a sequential decision-making problem in a grid world:

- The environment is a maze with walls, pellets, a Pacman agent, and one ghost.
- Pacman must learn a policy that collects as many pellets as possible while avoiding collision with the ghost.
- The ghost does not learn. Instead, it pursues Pacman using shortest-path movement and only moves every other Pacman step in the final target configuration.

This creates a constrained planning problem under threat. Pacman must learn:

- short-term movement decisions
- long-term path planning
- local risk avoidance
- trade-offs between reward collection and survival

The learning objective is not simply to maximize immediate pellet rewards, but to discover a policy that clears the maze while remaining safe.

## 3. Relation to the CS229 Paper

The project was inspired by the Stanford CS229 paper "Reinforcement Learning in Pacman." That paper explored multiple approaches, including tabular Q-learning, approximate Q-learning, and deep Q-learning.

This implementation does not reproduce the paper line by line. Instead, it adapts the paper's core reinforcement learning ideas into a cleaner engineering-focused system:

- one learning agent instead of two competing learners
- a rule-based ghost instead of a learned adversary
- an image-like grid state for deep learning
- curriculum learning and replay-based training
- practical reward shaping for survival and pellet completion

The resulting system is closer to a robust applied RL prototype than a pure reproduction of the research benchmark.

## 4. Environment Design

### 4.1 Maze Layout

The environment is defined on a 20x20 grid. Each cell is either:

- a wall
- an empty traversable space
- a pellet location

Pacman and the ghost occupy valid open cells. At reset, pellets are placed in all traversable cells except the current spawn positions.

### 4.2 Episode Start

The environment supports two reset modes:

- deterministic start positions for evaluation and visualization
- randomized start positions for training

Randomized starts were added to prevent route memorization. Pacman and the ghost are also forced to begin a minimum maze distance apart during training so the agent has time to explore meaningfully.

### 4.3 Action Space

Pacman has four discrete actions:

- up
- down
- left
- right

Illegal actions that would move into walls are masked during action selection.

### 4.4 Ghost Behavior

The ghost is rule-based rather than learned. It computes a shortest path toward Pacman and takes the first step on that path. This makes the ghost deterministic, interpretable, and consistent.

The final target design is:

- Pacman moves every frame
- ghost moves every 2 frames

During training, curriculum learning starts with an easier ghost:

- longer delay before the ghost starts moving
- slower movement intervals early in training
- gradual transition back to the final half-speed ghost

## 5. State Representation

The environment exposes a 5-channel tensor of shape 5 x 20 x 20. This acts like a small image fed to the neural network.

The channels are:

- Channel 0: walls
- Channel 1: pellets
- Channel 2: Pacman position
- Channel 3: ghost position
- Channel 4: ghost timing cue

The timing cue indicates whether the ghost is about to move, allowing the network to reason not only about where danger is, but also when danger is likely to change.

This representation gives the agent explicit awareness of:

- geometry
- food locations
- its own position
- the ghost's position
- movement rhythm of the ghost

## 6. Reward System

The reward function was tuned to reflect the desired gameplay behavior while remaining stable enough for DQN training.

The final main rewards are:

- Empty cell: -0.1
- Pellet collected: +10
- Win by clearing all pellets: +500
- Lose by getting caught: -450

In addition to these core rewards, the environment includes small shaping terms:

- a small positive bonus when Pacman increases its distance from the ghost
- no direct penalty for moving toward the ghost
- tiny anti-oscillation penalties for standing still, immediate backtracking, and revisiting recent cells
- a small bonus for moving closer to the nearest pellet

The purpose of these shaping terms is not to replace the main objective, but to help the network discover policies that are both productive and stable.

## 7. Neural Network Architecture

### 7.1 Model Type

The agent uses a convolutional Deep Q-Network implemented in PyTorch.

### 7.2 Feature Extractor

The state tensor is passed through stacked convolutional layers:

- Conv2d(5 -> 8)
- Conv2d(8 -> 16)
- Conv2d(16 -> 32)

Each convolution is followed by a ReLU activation. These layers allow the model to detect local spatial patterns such as:

- corridors
- dead ends
- pellet clusters
- relative ghost danger
- safe escape routes

### 7.3 Dueling Architecture

The current implementation uses a dueling DQN structure. After convolutional features are flattened, the network splits into:

- a value stream V(s)
- an advantage stream A(s, a)

The final Q-value is computed as:

Q(s, a) = V(s) + A(s, a) - mean(A(s, a))

This helps the model learn which states are generally good or bad independently of the immediate action ranking, which can improve stability in tasks with many similar actions.

### 7.4 Action Selection

Action selection uses epsilon-greedy exploration:

- with probability epsilon, choose a random legal action
- otherwise, choose the legal action with highest Q-value

Legal-action masking prevents the network from choosing wall collisions as valid decisions.

## 8. Training Pipeline

### 8.1 Experience Replay

Training uses a replay buffer that stores transitions:

- state
- action
- reward
- next state
- done flag
- mask of legal next actions

Replay reduces temporal correlation between consecutive samples and improves sample efficiency.

### 8.2 Target Network

A separate target network is used for bootstrap targets. This stabilizes learning by preventing the network from chasing its own rapidly changing estimates at every step.

### 8.3 Double DQN Update

The project uses a Double DQN style target:

- the online network selects the best next action
- the target network evaluates that action

This reduces overestimation bias compared with naive max-Q targets.

### 8.4 Loss Function

Training uses Huber loss, which is less sensitive to large outliers than mean squared error and is often more stable for Q-learning.

### 8.5 Reward Clipping for Training

To improve stability, training stores clipped rewards in replay even though the environment logs the original rewards. This preserves interpretability in the logs while preventing large reward magnitudes from destabilizing value estimation.

### 8.6 Curriculum Learning

Curriculum learning was introduced after early experiments showed that Pacman was learning only short pellet-harvesting routes before dying.

The curriculum makes training easier in the beginning by:

- slowing ghost movement further than the final target speed
- delaying ghost activation
- gradually restoring final difficulty

This encourages the agent to first learn:

- safe exploration
- route building
- pellet prioritization

before it has to solve the full end-game survival problem.

### 8.7 Randomized Starts

Randomized start states were added to reduce overfitting to a single opening trajectory. This makes the policy more robust and more representative of a general decision-making system rather than a memorized route.

### 8.8 Checkpointing and Resume Support

The project supports:

- saving the main model weights to a `.pth` file
- saving a richer training checkpoint to a `.ckpt` file
- resuming training without starting from scratch

The checkpoint stores:

- model weights
- target network weights
- optimizer state
- total training step count
- win count
- best-score metadata

## 9. Evaluation and Visualization

The project includes a visualization script built with Pygame. This allows the trained agent to be observed directly in the environment.

The viewer:

- loads a trained model
- runs the policy in the maze
- renders walls, pellets, Pacman, and the ghost
- optionally injects small evaluation-time randomness

This visual feedback was important because it revealed several key learning stages:

- early random wandering
- corner oscillation
- repeated fixed pellet routes
- meaningful ghost avoidance
- longer strategic arcs through the maze

Behavioral visualization was essential for diagnosing training plateaus that were not obvious from scalar rewards alone.

## 10. Key Engineering Challenges

Several practical challenges emerged during development:

### 10.1 Local Optima

Pacman often learned to collect a moderate number of pellets and then accept death, because that policy produced higher short-term return than exploring a riskier long route.

### 10.2 Oscillation

The agent sometimes learned to move back and forth in safe corners. This was mitigated with small anti-loop penalties and improved curriculum design.

### 10.3 Deterministic Memorization

Fixed start states made it easy for Pacman to memorize narrow opening sequences. Randomized starts reduced this issue.

### 10.4 Instability in Deep Q-Learning

Large reward magnitudes and bootstrapped targets can create unstable training. Reward clipping, target networks, and Huber loss improved this significantly.

## 11. Current System Behavior

At its improved stage, the agent demonstrates:

- purposeful pellet collection
- meaningful ghost avoidance
- route-level planning over multiple corridors
- better survival than the initial versions

Although the agent may still fail to clear the full map consistently, it has clearly learned:

- threat-aware navigation
- reward-seeking under constraints
- temporal decision-making with delayed consequences

This makes the project successful as a reinforcement learning system even before perfect win rates are reached.

## 12. Real-World Applications

Even though the environment is a game, the learning problem maps well to real-world domains where an agent must balance reward seeking and risk avoidance.

### 12.1 Robotics Navigation

A robot navigating a warehouse or building may need to:

- collect or deliver items
- avoid moving obstacles
- reason over corridors and bottlenecks
- choose between safe and risky routes

This is conceptually very similar to Pacman collecting pellets while avoiding the ghost.

### 12.2 Autonomous Vehicles

Autonomous systems must continuously trade off:

- progress toward a goal
- safety around dynamic hazards
- path efficiency
- local versus long-term outcomes

The Pacman task is a simplified version of this decision structure.

### 12.3 Supply Chain and Routing

Route planning under uncertainty often involves maximizing pickups or deliveries while minimizing exposure to congestion, delays, or failures. The same structure appears in Pacman:

- collect rewards efficiently
- avoid costly interactions
- use the map intelligently

### 12.4 Cybersecurity

Automated defense systems often need to:

- gather information
- preserve system availability
- avoid high-risk actions
- react to adversarial behavior

This project reflects the same kind of strategic balance between progress and avoidance.

### 12.5 Resource-Constrained Planning

Many business and engineering problems require optimizing under threat or uncertainty rather than simply maximizing raw gain. PacManDRL serves as a compact demonstration of this broader RL principle.

## 13. Skills Demonstrated by the Project

This project demonstrates strong practical ability in:

- reinforcement learning
- environment design
- reward engineering
- PyTorch modeling
- debugging and training stabilization
- curriculum learning
- visual simulation with Pygame
- checkpoint management and reproducible experimentation

It also demonstrates the ability to translate a research idea into an applied engineering system and iterate based on observed agent behavior.

## 14. Limitations

The current system still has important limitations:

- one ghost instead of multiple ghosts
- deterministic ghost policy rather than stochastic behavior
- single fixed maze layout
- no transfer learning across maps
- DQN limitations in very sparse or long-horizon settings

These limitations do not invalidate the project, but they define its scope. This is a focused RL system, not a full clone of commercial Pac-Man.

## 15. Future Improvements

Several extensions would make the project stronger:

- periodic evaluation episodes with fixed settings and epsilon = 0
- checkpointing based on evaluation wins rather than training reward
- multiple maze layouts for better generalization
- prioritized replay
- n-step returns
- explicit danger maps or learned distance features
- multi-ghost environments
- transfer learning from smaller maps to the full map

## 16. Conclusion

PacManDRL is a strong end-to-end reinforcement learning project that combines environment engineering, neural network design, curriculum learning, and practical debugging into a cohesive system. It goes beyond a toy example because the agent learns non-trivial spatial and temporal behavior: it collects pellets, avoids a pursuing ghost, adapts to maze structure, and shows evidence of meaningful policy learning.

The final system is especially valuable because it demonstrates the full workflow of applied reinforcement learning:

- define the environment
- design the state space
- select and implement the model
- create a reward function
- stabilize training
- evaluate behavior visually
- iterate based on failures

That workflow is directly relevant to many real-world AI problems where intelligent action under risk is more important than static prediction.
