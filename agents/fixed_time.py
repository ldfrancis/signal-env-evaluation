import numpy as np


class FixedTime:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._fixed_time = cfg["fixed_time"]
        self._observation_space = cfg["observation_space"]
        self._action_space = cfg["action_space"]
    
    def reset(self):
        self._cur_time = 0
        self.stats = {

        }

    def train(self): return
    def eval(self): return
    def observe(self, *args): return
    def update(self, *args): return
    def save_model(self, *args): return
    def load_model(self, *args): return

    def act(self, obs):
        if isinstance(obs, dict):
            phase = np.argmax(obs["phase"])
        else:
            phase = np.argmax(obs[-self._action_space.n:])

        
        if self._cur_time % self._fixed_time == 0:
            phase = (phase + 1) % self._action_space.n

        self._cur_time += self._cfg["delta_time"]
        return phase
        

    