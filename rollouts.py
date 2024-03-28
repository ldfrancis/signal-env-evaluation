import numpy as np


class Rollout:
    def __init__(self, agents, env, metrics=[], train=True) -> None:
        self._agents = agents
        self._env = env
        self._metrics = metrics
        self._train = train

    def __call__(self, steps=0):
        obs = self._env.reset()
        done = False
        returns = 0
        while not done:
            actions = {
                name: agent.act(obs[name]) for name,agent in self._agents.items()
            }
            obs, rews, dones, _ = self._env.step(actions)
            done = dones["__all__"]
            steps += 1
            if self._train:
                for name, agent in self._agents.items():
                    agent.observe(obs[name], rews[name], done)
                    agent.update(steps)
                    returns += rews[name]
        stats = {}
        for metric in self._metrics:
            if metric == "throughput":
                stats["throughput"] = self._env.obtain_metric(metric)
            else:
                agg, name = metric.split("_",1)
                if agg == "total": agg = "sum"
                stats[metric] = self._env.obtain_metric(name, agg)

        stats.update({
            "returns":returns
        })

        if self._train:
            a_stats = {}
            for agent in self._agents.values():
                for s,v in agent.stats.items():
                    a_stats[s] = a_stats.get(s, []) + [v]
            for k,v in a_stats.items():
                a_stats[k] = np.mean(v)

            stats.update({
                **a_stats,
                "steps":steps,
                })

        return stats


