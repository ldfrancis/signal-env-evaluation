import hydra
from omegaconf import DictConfig, OmegaConf
from environment import SignalEnvironment
from evaluator import Evaluator
from observations import get_observation_class
import reward_functions
from agents import get_agent_class
from trainer import Trainer
import numpy as np
import json

import os
from plot_demand import demand

hydra_run = 0

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg : DictConfig) -> None:
    global hydra_run

    print(OmegaConf.to_yaml(cfg))
    if not cfg.experiment.name:
        print("Error! Please specify an experiment name. Set exp.name")
        return
    
    try:
        cfg.experiment.net_demand
        demand(cfg.road_network.name, cfg.road_network.route_file, cfg.road_network.begin_time, cfg.road_network.num_seconds)
        return
    except:
        pass

    # if cfg.experiment.hpo:
        

    
    # environment
    env = SignalEnvironment(
        net_file=cfg.road_network.net_file,
        route_file=cfg.road_network.route_file,
        use_gui=cfg.environment.use_gui,
        begin_time=cfg.road_network.begin_time,
        num_seconds=cfg.road_network.num_seconds,
        delta_time=cfg.environment.delta_time,
        yellow_time=cfg.environment.yellow_time,
        min_green=cfg.environment.min_green,
        sumo_warnings=False,
        sumo_seed=cfg.environment.sumo_seed,
        observation_class = get_observation_class(cfg.environment.observation),
        reward_fn=getattr(reward_functions, f"{cfg.environment.reward_fn}_reward")
    )

    # agents
    agents = {}
    for ts_id in env.ts_ids:
        Agent_cls = get_agent_class(cfg.agent.name)
        agent_cfg = {}
        agent_cfg.update(
            {
                **cfg.agent,
                **cfg.trainer,
                "delta_time": cfg.environment.delta_time,
                "observation_space": env.observation_spaces(ts_id),
                "action_space": env.action_spaces(ts_id),
                "steps_per_episode": cfg.road_network.num_seconds
            }
        )
        agents[ts_id] = Agent_cls(agent_cfg)

    # hyperparameter search
    if cfg.experiment.get("hypsrch"):
        learning_rates = [0.1, 0.01, 0.001]
        gammas = [0.9, 0.95, 0.99]
        target_updates = [5,10,15]
        search_result = {}
        for lr in learning_rates:
            for gamma in gammas:
                for tu in target_updates:
                    agents = {}
                    for ts_id in env.ts_ids:
                        Agent_cls = get_agent_class(cfg.agent.name)
                        agent_cfg = {}
                        agent_cfg.update(
                            {
                                **cfg.agent,
                                **cfg.trainer,
                                "delta_time": cfg.environment.delta_time,
                                "observation_space": env.observation_spaces(ts_id),
                                "action_space": env.action_spaces(ts_id),
                                "steps_per_episode": cfg.road_network.num_seconds
                            }
                        )
                        agent_cfg.update(
                            {
                                "learning_rate":lr,
                                "gamma": gamma,
                                "target_update_frequency":tu
                            }
                        )
                        agents[ts_id] = Agent_cls(agent_cfg)
                        metrics = ["average_waiting_time", "average_travel_time", "average_delay", "average_queue_length", "throughput"]
                        trainer = Trainer(agents, env, metrics)
                        result = trainer(200)
                        result = np.min(np.sum(result[metrics].values, axis=1))
                        search_result[f"{lr}, {gamma}, {tu}"] = result

        log_dir = f"logs/{cfg.experiment.name}/{cfg.road_network.name}-{cfg.environment.observation}-{cfg.environment.reward_fn}-{cfg.agent.name}"
        os.makedirs(log_dir, exist_ok=True)

        best_score = np.inf
        best_result = None
        for k, v in search_result.items():
            if v < best_score:
                best_result = (k, v)
                best_score = v

        with open(log_dir+"/hypsrch.yaml", "w") as f:
            k, v = best_result
            f.write(OmegaConf.to_yaml({k:float(v)}))

        return  

    # trainer
    if cfg.experiment.train:
        for run in range(cfg.experiment.runs):
            log_file = ""
            log_dir = f"logs/{cfg.experiment.name}/{cfg.road_network.name}-{cfg.environment.observation}-{cfg.environment.reward_fn}-{cfg.agent.name}"
            os.makedirs(log_dir, exist_ok=True)
            if cfg.experiment.should_log:
                with open(log_dir+"/config.yaml", "w+") as f:
                    f.write(str(OmegaConf.to_yaml(cfg)))
                log_file = log_dir+f"/{run}.csv"
            for agent in agents.values():
                agent.reset()
            env.reset()
            trainer = Trainer(agents, env, ["average_waiting_time", "average_travel_time", "average_delay", "average_queue_length", "throughput"], log_file=log_file)
            trainer(cfg.trainer.episodes)
            
            os.makedirs(log_dir+"/models", exist_ok=True)
            trainer.save_agents(log_dir+"/models")
            env.close()
    else:
        load_dir = cfg.experiment.load_dir
        evaluator = Evaluator(agents, env,["average_waiting_time", "average_travel_time", "average_delay", "average_queue_length", "throughput"])
        evaluator.load_agents(load_dir)
        print(evaluator())

    hydra_run += 1

    

    
    

if __name__=="__main__":
    main()