import gymnasium as gym
import numpy as np

F = 0
B = 1
R = 2
L = 3
U = 4
D = 5


class Cube2Env(gym.Env):
    def __init__(self):
        super(Cube2Env, self).__init__()
        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(24, 6), dtype=np.int8
        )
        self.state = np.zeros((6, 4), dtype=np.int8)
        self.reset()

    def from_obs(obs):
        state = np.zeros((6, 4), dtype=np.int8)
        obs = np.asarray(obs)
        assert obs.shape == (24, 6)
        for i in range(6):
            for j in range(4):
                idx = i * 4 + j
                state[i, j] = np.argmax(obs[idx])
        env = Cube2Env()
        env.reset(state=state)
        return env

    def reset(self, seed=None, options=None, state: np.ndarray = None):
        super().reset(seed=seed, options=options)
        if state is None:
            self.state = np.zeros((6, 4), dtype=np.int8)
            self.state[0] = np.ones(4, dtype=np.int8) * F
            self.state[1] = np.ones(4, dtype=np.int8) * B
            self.state[2] = np.ones(4, dtype=np.int8) * R
            self.state[3] = np.ones(4, dtype=np.int8) * L
            self.state[4] = np.ones(4, dtype=np.int8) * U
            self.state[5] = np.ones(4, dtype=np.int8) * D
        else:
            assert state.shape == (6, 4) and state.dtype == np.int8
            self.state = state
        self.step_count = 0
        return self.obs(), {}

    def step(self, action):
        self.step_count += 1
        new_state = self.state.copy()

        # Front Clockwise
        if action == 0:
            new_state[F, 0] = self.state[F, 2]
            new_state[F, 1] = self.state[F, 0]
            new_state[F, 2] = self.state[F, 3]
            new_state[F, 3] = self.state[F, 1]
            new_state[R, 1] = self.state[U, 3]
            new_state[R, 3] = self.state[U, 1]
            new_state[L, 1] = self.state[D, 3]
            new_state[L, 3] = self.state[D, 1]
            new_state[U, 1] = self.state[L, 1]
            new_state[U, 3] = self.state[L, 3]
            new_state[D, 1] = self.state[R, 1]
            new_state[D, 3] = self.state[R, 3]
        # Front Counter-Clockwise
        elif action == 1:
            new_state[F, 0] = self.state[F, 1]
            new_state[F, 1] = self.state[F, 3]
            new_state[F, 2] = self.state[F, 0]
            new_state[F, 3] = self.state[F, 2]
            new_state[R, 1] = self.state[D, 1]
            new_state[R, 3] = self.state[D, 3]
            new_state[L, 1] = self.state[U, 1]
            new_state[L, 3] = self.state[U, 3]
            new_state[U, 1] = self.state[R, 3]
            new_state[U, 3] = self.state[R, 1]
            new_state[D, 1] = self.state[L, 3]
            new_state[D, 3] = self.state[L, 1]
        # Back Clockwise
        elif action == 2:
            new_state[B, 0] = self.state[B, 1]
            new_state[B, 1] = self.state[B, 3]
            new_state[B, 2] = self.state[B, 0]
            new_state[B, 3] = self.state[B, 2]
            new_state[R, 0] = self.state[D, 0]
            new_state[R, 2] = self.state[D, 2]
            new_state[L, 0] = self.state[U, 0]
            new_state[L, 2] = self.state[U, 2]
            new_state[U, 0] = self.state[R, 2]
            new_state[U, 2] = self.state[R, 0]
            new_state[D, 0] = self.state[L, 2]
            new_state[D, 2] = self.state[L, 0]
        # Back Counter-Clockwise
        elif action == 3:
            new_state[B, 0] = self.state[B, 2]
            new_state[B, 1] = self.state[B, 0]
            new_state[B, 2] = self.state[B, 3]
            new_state[B, 3] = self.state[B, 1]
            new_state[R, 0] = self.state[U, 2]
            new_state[R, 2] = self.state[U, 0]
            new_state[L, 0] = self.state[D, 2]
            new_state[L, 2] = self.state[D, 0]
            new_state[U, 0] = self.state[L, 0]
            new_state[U, 2] = self.state[L, 2]
            new_state[D, 0] = self.state[R, 0]
            new_state[D, 2] = self.state[R, 2]
        # Right Clockwise
        elif action == 4:
            new_state[F, 2] = self.state[D, 2]
            new_state[F, 3] = self.state[D, 3]
            new_state[B, 2] = self.state[U, 2]
            new_state[B, 3] = self.state[U, 3]
            new_state[R, 0] = self.state[R, 2]
            new_state[R, 1] = self.state[R, 0]
            new_state[R, 2] = self.state[R, 3]
            new_state[R, 3] = self.state[R, 1]
            new_state[U, 2] = self.state[F, 3]
            new_state[U, 3] = self.state[F, 2]
            new_state[D, 2] = self.state[B, 3]
            new_state[D, 3] = self.state[B, 2]
        # Right Counter-Clockwise
        elif action == 5:
            new_state[F, 2] = self.state[U, 3]
            new_state[F, 3] = self.state[U, 2]
            new_state[B, 2] = self.state[D, 3]
            new_state[B, 3] = self.state[D, 2]
            new_state[R, 0] = self.state[R, 1]
            new_state[R, 1] = self.state[R, 3]
            new_state[R, 2] = self.state[R, 0]
            new_state[R, 3] = self.state[R, 2]
            new_state[U, 2] = self.state[B, 2]
            new_state[U, 3] = self.state[B, 3]
            new_state[D, 2] = self.state[F, 2]
            new_state[D, 3] = self.state[F, 3]
        # Left Clockwise
        elif action == 6:
            new_state[F, 0] = self.state[U, 1]
            new_state[F, 1] = self.state[U, 0]
            new_state[B, 0] = self.state[D, 1]
            new_state[B, 1] = self.state[D, 0]
            new_state[L, 0] = self.state[L, 1]
            new_state[L, 1] = self.state[L, 3]
            new_state[L, 2] = self.state[L, 0]
            new_state[L, 3] = self.state[L, 2]
            new_state[U, 0] = self.state[B, 0]
            new_state[U, 1] = self.state[B, 1]
            new_state[D, 0] = self.state[F, 0]
            new_state[D, 1] = self.state[F, 1]
        # Left Counter-Clockwise
        elif action == 7:
            new_state[F, 0] = self.state[D, 0]
            new_state[F, 1] = self.state[D, 1]
            new_state[B, 0] = self.state[U, 0]
            new_state[B, 1] = self.state[U, 1]
            new_state[L, 0] = self.state[L, 2]
            new_state[L, 1] = self.state[L, 0]
            new_state[L, 2] = self.state[L, 3]
            new_state[L, 3] = self.state[L, 1]
            new_state[U, 0] = self.state[F, 1]
            new_state[U, 1] = self.state[F, 0]
            new_state[D, 0] = self.state[B, 1]
            new_state[D, 1] = self.state[B, 0]
        # Up Clockwise
        elif action == 8:
            new_state[F, 1] = self.state[R, 3]
            new_state[F, 3] = self.state[R, 2]
            new_state[B, 1] = self.state[L, 3]
            new_state[B, 3] = self.state[L, 2]
            new_state[R, 2] = self.state[B, 1]
            new_state[R, 3] = self.state[B, 3]
            new_state[L, 2] = self.state[F, 1]
            new_state[L, 3] = self.state[F, 3]
            new_state[U, 0] = self.state[U, 1]
            new_state[U, 1] = self.state[U, 3]
            new_state[U, 2] = self.state[U, 0]
            new_state[U, 3] = self.state[U, 2]
        # Up Counter-Clockwise
        elif action == 9:
            new_state[F, 1] = self.state[L, 2]
            new_state[F, 3] = self.state[L, 3]
            new_state[B, 1] = self.state[R, 2]
            new_state[B, 3] = self.state[R, 3]
            new_state[R, 2] = self.state[F, 3]
            new_state[R, 3] = self.state[F, 1]
            new_state[L, 2] = self.state[B, 3]
            new_state[L, 3] = self.state[B, 1]
            new_state[U, 0] = self.state[U, 2]
            new_state[U, 1] = self.state[U, 0]
            new_state[U, 2] = self.state[U, 3]
            new_state[U, 3] = self.state[U, 1]
        # Bottom Clockwise
        elif action == 10:
            new_state[F, 0] = self.state[L, 0]
            new_state[F, 2] = self.state[L, 1]
            new_state[B, 0] = self.state[R, 0]
            new_state[B, 2] = self.state[R, 1]
            new_state[R, 0] = self.state[F, 2]
            new_state[R, 1] = self.state[F, 0]
            new_state[L, 0] = self.state[B, 2]
            new_state[L, 1] = self.state[B, 0]
            new_state[D, 0] = self.state[D, 2]
            new_state[D, 1] = self.state[D, 0]
            new_state[D, 2] = self.state[D, 3]
            new_state[D, 3] = self.state[D, 1]
        # Bottom Counter-Clockwise
        elif action == 11:
            new_state[F, 0] = self.state[R, 1]
            new_state[F, 2] = self.state[R, 0]
            new_state[B, 0] = self.state[L, 1]
            new_state[B, 2] = self.state[L, 0]
            new_state[R, 0] = self.state[B, 0]
            new_state[R, 1] = self.state[B, 2]
            new_state[L, 0] = self.state[F, 0]
            new_state[L, 1] = self.state[F, 2]
            new_state[D, 0] = self.state[D, 1]
            new_state[D, 1] = self.state[D, 3]
            new_state[D, 2] = self.state[D, 0]
            new_state[D, 3] = self.state[D, 2]
        self.state = new_state
        return (
            self.obs(),
            1 if self.is_solved() else -1,
            self.is_solved(),
            self.step_count >= 100,
            {},
        )

    def adjacent_obs(self):
        neighbors = []
        for i in range(12):
            env = Cube2Env()
            env.reset(state=self.state)
            obs, _, _, _, _ = env.step(i)
            neighbors.append(obs)
        return np.array(neighbors)

    def obs(self):
        one_hots = []
        for i in range(6):
            for j in range(4):
                label = int(self.state[i, j])
                zeros = np.zeros(6, dtype=np.int8)
                zeros[label] = 1
                one_hots.append(zeros)
        return np.array(one_hots)

    def is_solved(self):
        for i in range(6):
            if np.mean(self.state[i]) != self.state[i][0]:
                return False
        return True

    def print_js_code(self):
        print(f"rotationController.setState({self.state.tolist()});")


def _build_permutations() -> np.ndarray:
    """PERMS[a][i] = index of the sticker that moves to position i under action a.

    Derived from Cube2Env.step so the two can never drift apart. With flattened
    (24,) states this gives `new_state = state[PERMS[a]]`, which also works on
    batches: `states[:, PERMS[a]]`.
    """
    env = Cube2Env()
    perms = []
    for a in range(12):
        env.state = np.arange(24, dtype=np.int8).reshape(6, 4)
        env.obs = lambda: None  # labels 0..23 are not valid colors; skip encoding
        env.is_solved = lambda: False
        Cube2Env.step(env, a)
        perms.append(env.state.reshape(-1).astype(np.int64))
    return np.stack(perms)


PERMS = _build_permutations()
INVERSE_ACTION = np.array([a ^ 1 for a in range(12)])


def batch_is_solved(states: np.ndarray) -> np.ndarray:
    """states: (B, 24) color indices -> (B,) bool."""
    faces = states.reshape(-1, 6, 4)
    return (faces == faces[:, :, :1]).all(axis=(1, 2))


def batch_one_hot(states: np.ndarray) -> np.ndarray:
    """states: (B, 24) color indices -> (B, 24, 6) float32, same encoding as Cube2Env.obs."""
    return np.eye(6, dtype=np.float32)[states]
