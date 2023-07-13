import random
from typing import Dict

import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ObsDict, ActionDict


class DummyParallelEnv(ParallelEnv, EzPickle):
    metadata = {'render_modes': ['ansi'], "name": "TestParallelEnv-v0"}

    def __init__(self, n_agents: int = 20, new_step_api: bool = True) -> None:
        EzPickle.__init__(
            self,
            n_agents,
            new_step_api
        )

        self._terminated = False
        self.current_step = 0

        self.n_agents = n_agents
        self.possible_agents = [f"player_{idx}" for idx in range(n_agents)]
        self.agents = self.possible_agents[:]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.observation_spaces = {
            agent: spaces.Box(shape=(len(self.agents),), dtype=np.float64, low=0.0, high=1.0)
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(4) for agent in self.possible_agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, actions: ActionDict) \
            -> tuple[ObsDict, dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        self.current_step += 1
        self._terminated = self.current_step >= 100

        observations = self.__calculate_observations()
        rewards = {
            self.agents[agent]: random.randint(0, 100) for agent in range(len(self.agents))
        }
        terminated = {agent: self._terminated for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if self._terminated:
            self.agents = []

        return observations, rewards, terminated, truncated, infos

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[ObsDict, dict[str, dict]]:
        self.agents = self.possible_agents[:]
        self._terminated = False
        self.current_step = 0
        observations = self.__calculate_observations()
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def __calculate_observations(self) -> Dict[str, np.ndarray]:
        return {
            agent: self.observation_space(agent).sample() for agent in self.agents
        }
