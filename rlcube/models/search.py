import heapq
import itertools
import os
import time
from functools import lru_cache

import numpy as np
import torch

from rlcube.envs.cube2 import PERMS, batch_is_solved, batch_one_hot
from rlcube.models.models import DNN

ACTIONS = 12
C_PUCT = 1.414
CHECKPOINT = os.environ.get("RLCUBE_CHECKPOINT", "checkpoints/checkpoint_final.pth")


@lru_cache(maxsize=1)
def get_net() -> DNN:
    # Loaded lazily (not at import time) so importing this module is cheap and
    # the checkpoint path can be overridden with RLCUBE_CHECKPOINT.
    net = DNN()
    net.load(CHECKPOINT)
    net.eval()
    return net


def to_states(obs) -> np.ndarray:
    """(24, 6) one-hot obs or (6, 4) / (24,) color array -> flat (24,) int8 state."""
    obs = np.asarray(obs)
    if obs.shape == (24, 6):
        return obs.argmax(axis=1).astype(np.int8)
    return obs.reshape(24).astype(np.int8)


@torch.no_grad()
def evaluate(states: np.ndarray):
    """Batched network call. states: (B, 24) -> values (B,), policy probs (B, 12)."""
    x = torch.from_numpy(batch_one_hot(states))
    values, logits = get_net()(x)
    return values.view(-1).numpy(), torch.softmax(logits, dim=1).numpy()


def expand(state: np.ndarray) -> np.ndarray:
    """(24,) -> (12, 24): the state after each of the 12 actions."""
    return state[PERMS]


# ---------------------------------------------------------------------------
# MCTS (fixed version of the original implementation)
# ---------------------------------------------------------------------------


class Node:
    def __init__(self, state, value, policy, parent=None):
        self.state = state
        self.parent = parent
        self.value = value
        self.policy = policy
        self.children = {}
        self.N = np.zeros(ACTIONS, dtype=np.int32)
        # FIX 1: Q per action starts at -inf and is filled with the child's own
        # network value on expansion. The old code started W at 0 and backed up
        # with max(W, leaf_value); since values far from the goal are negative,
        # W stayed 0 forever and the value head was ignored completely.
        self.W = np.full(ACTIONS, -np.inf, dtype=np.float32)

    def is_leaf(self):
        return len(self.children) == 0

    def select_action(self):
        n_sum = np.sum(self.N) + 1
        scores = self.policy * C_PUCT * np.sqrt(n_sum) / (self.N + 1) + self.W
        return int(np.argmax(scores))


class MonteCarloTree:
    def __init__(self, obs, max_simulations=300):
        self.max_simulations = max_simulations
        state = to_states(obs)
        value, policy = evaluate(state[None])
        self.root = Node(state, float(value[0]), policy[0])
        self.is_solved = bool(batch_is_solved(state[None])[0])
        self.solved_path = []
        # FIX 2: transposition table. ~40% of the old tree were duplicate states
        # (e.g. F then F' returns to the parent).
        self.seen = {state.tobytes()}
        self.num_nodes = 1
        if not self.is_solved:
            self._build()

    def _build(self):
        for _ in range(self.max_simulations):
            node = self.root
            path = []

            # Selection
            while not node.is_leaf():
                a = node.select_action()
                if node.W[a] == -np.inf:  # every option here is a dead end
                    break
                path.append((node, a))
                node = node.children[a]
            if not node.is_leaf():
                if not path:
                    return  # whole tree exhausted
                parent, a = path[-1]
                parent.W[a] = -np.inf
                continue

            # Expansion: evaluate all 12 children in ONE batched network call
            # (FIX 3: the old code made 12 separate calls + 12 env copies).
            child_states = expand(node.state)
            solved = batch_is_solved(child_states)
            if solved.any():
                a = int(np.argmax(solved))
                self.is_solved = True
                self.solved_path = [a for _, a in path] + [a]
                return
            values, policies = evaluate(child_states)
            for a in range(ACTIONS):
                key = child_states[a].tobytes()
                if key in self.seen:
                    continue  # duplicate: leave W[a] = -inf so it is never chosen
                self.seen.add(key)
                node.children[a] = Node(
                    child_states[a], float(values[a]), policies[a], node
                )
                node.W[a] = values[a]
                self.num_nodes += 1

            # Backup
            # FIX 4: back up the best value among the freshly evaluated children
            # (one-step lookahead), not the value of the node itself. The old
            # root-only special case is no longer needed.
            # FIX 5: W must be allowed to go DOWN. A branch that looked good but
            # turned out worse after expansion should lose priority, so W[a] is
            # set to the best value of the subtree below it rather than max'ed
            # with its stale previous value.
            for parent, a in reversed(path):
                parent.N[a] += 1
                parent.W[a] = np.max(parent.children[a].W)


# ---------------------------------------------------------------------------
# Batch Weighted A* (the search DeepCubeA actually uses)
# ---------------------------------------------------------------------------


def astar_solve(obs, weight=0.6, batch_size=64, max_nodes=200_000, time_limit=10.0):
    """Batch weighted A*: f(s) = weight * g(s) + h(s), with h = -value(s).

    Pops `batch_size` nodes per iteration so the network runs on full batches.
    Unlike MCTS this is complete: given enough nodes it always finds a solution.
    Returns (list of actions, number of generated nodes) or (None, nodes).
    """
    start = to_states(obs)
    if batch_is_solved(start[None])[0]:
        return [], 1

    counter = itertools.count()  # tie-breaker so heapq never compares arrays
    g_best = {start.tobytes(): 0}
    parent = {start.tobytes(): (None, None)}
    value, _ = evaluate(start[None])
    open_heap = [(-float(value[0]), next(counter), start)]
    deadline = time.monotonic() + time_limit

    def path_to(key, last_action):
        actions = [last_action]
        while parent[key][0] is not None:
            key, a = parent[key]
            actions.append(a)
        return actions[::-1]

    while open_heap and len(g_best) < max_nodes and time.monotonic() < deadline:
        batch = [
            heapq.heappop(open_heap) for _ in range(min(batch_size, len(open_heap)))
        ]
        states = np.stack([s for _, _, s in batch])
        children = states[:, PERMS].reshape(-1, 24)  # (B*12, 24)
        solved = batch_is_solved(children)
        if solved.any():
            i = int(np.argmax(solved))
            return path_to(states[i // ACTIONS].tobytes(), i % ACTIONS), len(g_best)

        new_children, new_g = [], []
        for i, child in enumerate(children):
            p_key = states[i // ACTIONS].tobytes()
            g = g_best[p_key] + 1
            key = child.tobytes()
            if g < g_best.get(key, np.inf):
                g_best[key] = g
                parent[key] = (p_key, i % ACTIONS)
                new_children.append(child)
                new_g.append(g)
        if not new_children:
            continue
        values, _ = evaluate(np.stack(new_children))
        for child, g, v in zip(new_children, new_g, values):
            heapq.heappush(open_heap, (weight * g - float(v), next(counter), child))
    return None, len(g_best)


if __name__ == "__main__":
    from rlcube.envs.cube2 import Cube2Env

    env = Cube2Env()
    for _ in range(14):
        env.step(env.action_space.sample())
    tree = MonteCarloTree(env.obs())
    print("MCTS:", tree.is_solved, tree.solved_path)
    print("A*:", astar_solve(env.obs()))
