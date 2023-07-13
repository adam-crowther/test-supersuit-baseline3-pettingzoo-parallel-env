import supersuit as ss
from pettingzoo.test import parallel_api_test
from stable_baselines3 import PPO

from dummy_env import dummy

if __name__ == '__main__':
    env_parallel = dummy.DummyParallelEnv()
    parallel_api_test(env_parallel)

    # env_parallel = ss.flatten_v0(env_parallel)
    env_parallel = ss.pettingzoo_env_to_vec_env_v1(env_parallel)
    env_parallel = ss.concat_vec_envs_v1(env_parallel, 1, base_class="stable_baselines3")

    model = PPO("MlpPolicy", env_parallel, verbose=1)
    
    model.learn(total_timesteps=10_000)
