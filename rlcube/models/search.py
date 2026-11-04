import heapq
import itertools
import os
import time
from functools import lru_cache

import numpy as np
import torch

from rlcube.envs.cube2 import (
    REDUCED_ACTIONS,
    REDUCED_PERMS,
    SOLVED_STATE,
    batch_one_hot,
    canonicalize,
)
from rlcube.models.models import ValueNet

NUM_ACTIONS = len(REDUCED_ACTIONS)
CHECKPOINT = os.environ.get("RLCUBE_CHECKPOINT", "checkpoints/checkpoint_final.pth")


@lru_cache(maxsize=1)
def get_net() -> ValueNet:
    # Loaded lazily so importing is cheap; override the path with RLCUBE_CHECKPOINT.
    net = ValueNet()
    net.load(CHECKPOINT)
    net.eval()
    return net


def to_state(obs) -> np.ndarray:
    """(24, 6) one-hot obs or (6, 4) / (24,) color array -> flat (24,) int8 state."""
    obs = np.asarray(obs)
    if obs.shape == (24, 6):
        return obs.argmax(axis=1).astype(np.int8)
    return obs.reshape(24).astype(np.int8)


@torch.no_grad()
def cost_to_go(net: ValueNet, states: np.ndarray) -> np.ndarray:
    """Heuristic h(s) for canonical states (B, 24) -> (B,), with h(solved) = 0."""
    h = net(torch.from_numpy(batch_one_hot(states))).clamp_(min=0).numpy()
    h[(states == SOLVED_STATE).all(axis=1)] = 0
    return h


def astar_solve(
    obs,
    net: ValueNet | None = None,
    weight=1.0,
    batch_size=16,
    max_nodes=200_000,
    time_limit=10.0,
):
    """Batch weighted A*: expand the `batch_size` nodes with lowest
    f(s) = weight * g(s) + h(s) per iteration, so the network sees full batches.

    The search runs on the canonical state with the 6 reduced actions; the
    returned actions are env action ids (0-11) and work on the original cube.
    Returns (actions, generated nodes); actions is None if no solution was found.
    """
    net = net or get_net()
    start = canonicalize(to_state(obs))[0]
    if (start == SOLVED_STATE).all():
        return [], 1

    counter = itertools.count()  # tie-breaker so heapq never compares arrays
    g_best = {start.tobytes(): 0}
    parent = {start.tobytes(): (None, None)}
    open_heap = [(0.0, next(counter), start)]
    deadline = time.monotonic() + time_limit

    def path_to(key, last_action):
        actions = [last_action]
        while parent[key][0] is not None:
            key, a = parent[key]
            actions.append(a)
        return [int(REDUCED_ACTIONS[a]) for a in reversed(actions)]

    while open_heap and len(g_best) < max_nodes and time.monotonic() < deadline:
        n = min(batch_size, len(open_heap))
        states = np.stack([heapq.heappop(open_heap)[2] for _ in range(n)])
        children = states[:, REDUCED_PERMS].reshape(-1, 24)  # (n*6, 24)
        solved = (children == SOLVED_STATE).all(axis=1)
        if solved.any():
            i = int(np.argmax(solved))
            return path_to(states[i // NUM_ACTIONS].tobytes(), i % NUM_ACTIONS), len(
                g_best
            )

        new_children, new_g = [], []
        for i, child in enumerate(children):
            p_key = states[i // NUM_ACTIONS].tobytes()
            g = g_best[p_key] + 1
            key = child.tobytes()
            if g < g_best.get(key, np.inf):
                g_best[key] = g
                parent[key] = (p_key, i % NUM_ACTIONS)
                new_children.append(child)
                new_g.append(g)
        if not new_children:
            continue
        h = cost_to_go(net, np.stack(new_children))
        for child, g, hv in zip(new_children, new_g, h):
            heapq.heappush(open_heap, (weight * g + float(hv), next(counter), child))
    return None, len(g_best)


if __name__ == "__main__":
    from rlcube.envs.cube2 import Cube2Env

    env = Cube2Env()
    for _ in range(30):
        env.step(env.action_space.sample())
    print("A*:", astar_solve(env.obs()))
