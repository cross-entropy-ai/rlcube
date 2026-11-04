"""Deep Approximate Value Iteration (DeepCubeA-style) for the 2x2 cube.

Each iteration:
  1. Generate a fresh batch of states scrambled 1..max_depth moves from solved.
  2. Bootstrap targets with a frozen target network:
       J(s) = 0                               if s is solved
       J(s) = min_a [1 + J_target(step(s, a))] otherwise
  3. Regress the online network onto those targets (MSE).
The target network is refreshed whenever the online network has fit the
current targets (loss < threshold), or after `max_wait` iterations.

No ground-truth distances are used anywhere: progress is measured only by
how well A* with the current network solves a fixed set of scrambled cubes.
"""

import argparse
import copy
import os
import time

import numpy as np
import torch

from rlcube.envs.cube2 import (
    REDUCED_PERMS,
    SOLVED_STATE,
    random_scrambles,
)
from rlcube.models.models import ValueNet
from rlcube.models.search import astar_solve

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

ONE_HOT = torch.eye(6, device=device)


def encode(states: torch.Tensor) -> torch.Tensor:
    """(B, 24) int64 -> (B, 24, 6) float one-hot, same as batch_one_hot."""
    return ONE_HOT[states]


@torch.no_grad()
def bootstrap_targets(target_net, states: torch.Tensor, perms, solved_state):
    """One-step Bellman targets J(s) = min_a 1 + J_target(s_a), J(solved) = 0."""
    children = states[:, perms].reshape(-1, 24)  # (B*6, 24)
    child_cost = target_net(encode(children)).clamp(min=0)
    child_cost[(children == solved_state).all(dim=1)] = 0
    targets = 1 + child_cost.view(len(states), -1).min(dim=1).values
    targets[(states == solved_state).all(dim=1)] = 0
    return targets


def evaluate(net: ValueNet, eval_states: np.ndarray, max_nodes: int):
    """Self-assessment: solve rate / avg solution length / avg nodes with A*."""
    cpu_net = copy.deepcopy(net).to("cpu").eval()
    solved, lengths, nodes = 0, [], []
    for s in eval_states:
        path, n = astar_solve(s, net=cpu_net, max_nodes=max_nodes, time_limit=30)
        nodes.append(n)
        if path is not None:
            solved += 1
            lengths.append(len(path))
    return (
        solved / len(eval_states),
        float(np.mean(lengths)) if lengths else float("nan"),
        float(np.mean(nodes)),
    )


def train(
    iterations: int = 30_000,
    batch_size: int = 10_000,
    max_depth: int = 20,
    lr: float = 1e-3,
    loss_threshold: float = 0.05,
    check_every: int = 100,
    max_wait: int = 500,
    eval_every: int = 2_000,
    eval_size: int = 100,
    eval_max_nodes: int = 50_000,
    out_dir: str = "checkpoints",
    seed: int = 0,
):
    print(f"Using device: {device}")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    perms = torch.as_tensor(REDUCED_PERMS, device=device)
    solved_state = torch.as_tensor(SOLVED_STATE, device=device, dtype=torch.int64)
    # Fixed evaluation set: long random walks, i.e. effectively random states.
    eval_states, _ = random_scrambles(eval_size, 100, np.random.default_rng(1234))

    net = ValueNet().to(device)
    target_net = copy.deepcopy(net).eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, iterations, eta_min=lr * 0.05
    )

    os.makedirs(out_dir, exist_ok=True)
    best = (-1.0, float("inf"))  # (solve rate, -nodes) ordering
    window_loss, window_n, last_update, num_updates = 0.0, 0, 0, 0
    start_time = time.time()

    for it in range(1, iterations + 1):
        states_np, _ = random_scrambles(batch_size, max_depth, rng)
        states = torch.as_tensor(states_np, device=device, dtype=torch.int64)
        targets = bootstrap_targets(target_net, states, perms, solved_state)

        loss = torch.nn.functional.mse_loss(net(encode(states)), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        window_loss += loss.item()
        window_n += 1

        if it % check_every == 0:
            avg = window_loss / window_n
            window_loss, window_n = 0.0, 0
            updated = avg < loss_threshold or it - last_update >= max_wait
            if updated:
                target_net.load_state_dict(net.state_dict())
                last_update, num_updates = it, num_updates + 1
            print(
                f"it {it:6d} | loss {avg:.4f} | target updates {num_updates:3d}"
                f"{' (updated)' if updated else ''} | "
                f"mean target {targets.mean().item():.2f} | "
                f"{time.time() - start_time:.0f}s",
                flush=True,
            )

        if it % eval_every == 0 or it == iterations:
            rate, length, nodes = evaluate(net, eval_states, eval_max_nodes)
            print(
                f"[eval] it {it} | solved {rate:.0%} | avg length {length:.2f} | "
                f"avg nodes {nodes:.0f}",
                flush=True,
            )
            net.save(os.path.join(out_dir, "checkpoint_last.pth"))
            if (rate, -nodes) > (best[0], -best[1]):
                best = (rate, nodes)
                net.save(os.path.join(out_dir, "checkpoint_best.pth"))
                print("[eval] new best, saved checkpoint_best.pth", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=2_000)
    args = parser.parse_args()
    train(
        iterations=args.iterations,
        batch_size=args.batch_size,
        max_depth=args.max_depth,
        lr=args.lr,
        eval_every=args.eval_every,
    )
