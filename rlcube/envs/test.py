from rlcube.envs.cube2 import Cube2Env

if __name__ == "__main__":
    print("Testing Cube2Env.from_obs")
    env = Cube2Env()
    env2 = Cube2Env.from_obs(env.obs())
    print(env2.state, env2.is_solved())
    env2.print_js_code()
    print()

    print("Testing Cube2Env.adjacent_obs")
    env = Cube2Env()
    adjacent_obs = env.adjacent_obs()
    for i in range(12):
        env = Cube2Env.from_obs(adjacent_obs[i])
        env.print_js_code()
