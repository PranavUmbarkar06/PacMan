# PacManDRL

Paper-inspired Pacman DQN:

- Pacman is the only learning agent.
- The ghost is rule-based and moves at half of Pacman's speed.
- Rewards emphasize eating pellets, clearing the map, and staying away from the ghost.

Run training:

```bash
python train.py --episodes 400
```

Watch the trained model:

```bash
python main.py
```
