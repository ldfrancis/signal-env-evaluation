import torch

from rollouts import Rollout

class Evaluator:
    def __init__(self, agents, env, metrics=[]):
        self._env = env
        self._agents = agents
        self._rollout = Rollout(agents, env, metrics, False)
        

    def __call__(self):
        for agent in self._agents.values():
            agent.eval()

        return self._rollout()

    def load_agents(self, load_dir):
        for k, v in self._agents.items():
            v.load_model(load_dir+f"/{k}.pt")

    

