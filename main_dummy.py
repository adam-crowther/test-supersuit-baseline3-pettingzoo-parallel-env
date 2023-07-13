import supersuit as ss
from pettingzoo.test import parallel_api_test
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from dummy_env import dummy
from dummy_env.sb3_shim_wrapper import Sb3ShimWrapper

if __name__ == '__main__':
    env_parallel = dummy.DummyParallelEnv()
    parallel_api_test(env_parallel)

    # env_parallel = ss.flatten_v0(env_parallel)
    env_parallel = ss.pettingzoo_env_to_vec_env_v1(env_parallel)
    env_parallel = ss.concat_vec_envs_v1(env_parallel, 1, base_class="stable_baselines3")

    # This shim is required to work around Issue #222
    # https://github.com/Farama-Foundation/SuperSuit/issues/222
    env_parallel = Sb3ShimWrapper(env_parallel)

    model = PPO("MlpPolicy", env_parallel, verbose=1)

    # This EvalCallback is included to demonstrate Issue #223
    # https://github.com/Farama-Foundation/SuperSuit/issues/223
    eval_callback = EvalCallback(env_parallel,
                                 best_model_save_path="./logs/",
                                 log_path="./logs/",
                                 eval_freq=50,
                                 deterministic=True,
                                 render=True,
                                 verbose=1)
    model.learn(total_timesteps=10_000, callback=eval_callback)
