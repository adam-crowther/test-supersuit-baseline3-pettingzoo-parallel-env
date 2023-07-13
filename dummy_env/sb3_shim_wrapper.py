from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvWrapper


class Sb3ShimWrapper(VecEnvWrapper):
    metadata = {'render_modes': ['human', 'files', 'none'], "name": "Sb3ShimWrapper-v0"}

    def __init__(self, venv):
        super().__init__(venv)

    def reset(self, seed=None, options=None):
        return self.venv.reset()[0]

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()
