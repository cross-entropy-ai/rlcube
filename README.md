---
title: "Rubik's Cube Solver"
emoji: 🧊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

## 🧊 Rubik's Cube Solver

Solve the Rubik's Cube using Reinforcement Learning! 🚀

https://github.com/user-attachments/assets/0f6cdf0c-37f5-4f7c-a367-c696e21ad4dc

## How it works

We train a value network with Deep Approximate Value Iteration (DAVI), following [Solving the Rubik's Cube with Deep Reinforcement Learning and Search](https://arxiv.org/abs/1805.07470) (DeepCubeA). The network learns the *cost-to-go* of a state, i.e. how many moves it is from solved, purely by bootstrapping: each training state's target is `1 + min` over its neighbours' cost-to-go as estimated by a periodically-updated target network, with the solved state anchored at 0. No ground-truth distances are used.

To keep the state space small, the solver only turns the Front, Right and Up faces, so the back-left-down corner never moves and whole-cube orientations disappear from the state space. Solutions are then found with **batch weighted A\*** search, using the network as the heuristic.

## 🏋️‍♂️ Train the Model

1. Install dependencies:
   ```
   uv sync
   ```
2. Start training (about an hour on an Apple Silicon Mac):
   ```
   uv run python -m rlcube.train.train
   ```
   The script prints the loss and, every 2,000 iterations, the A\* solve rate on 100 random cubes.

Checkpoints are written to `checkpoints/`. Copy `checkpoint_best.pth` to `checkpoint_final.pth` so the API uses it (or point `RLCUBE_CHECKPOINT` at any checkpoint). 🎯
