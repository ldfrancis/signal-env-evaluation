

from rollouts import Rollout
import pprint
import pandas as pd
import torch
from evaluator import Evaluator


class Trainer:
    def __init__(self, agents, env, metrics, log_file=""):
        self._agents = agents
        self._env = env
        self._metrics = metrics
        self._rollout = Rollout(agents, env, metrics)
        self.results = {}
        self._log_file = log_file
        self._evaluator = Evaluator(agents, env, metrics)

    def __call__(self, episodes):
        steps = 0
        for episode in range(1, episodes+1):
            for agent in self._agents.values():
                agent.train()

            stats = self._rollout(steps)
            stats.update(
                {
                    "episode": episode
                }
            )
            steps = stats["steps"]

            # evaluation
            # eval_stats = self._evaluator()
            # eval_stats.update({
            #     "episode": episode
            # })

            for k,v in stats.items():
                self.results[k] = self.results.get(k,[]) + [v]

            # stats["evaluation"] = eval_stats

            pprint.pprint(stats)
        
        if self._log_file:
            df = pd.DataFrame(self.results)
            df.to_csv(self._log_file, index=False)
        else:
            return pd.DataFrame(self.results)

    def save_agents(self, save_dir):
        for k,v in self._agents.items():
            v.save_model(save_dir+f"/{k}.pt")
            

            