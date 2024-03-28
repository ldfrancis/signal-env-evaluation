import hydra
from omegaconf import DictConfig, OmegaConf
from rlsignal.agents.dqn.dqn_trainer import DQNTrainer
from run import Runner
from common.registry import Registry
from argparse import Namespace
import wandb
import os
from _helpers import register_reward_function


# parser = argparse.ArgumentParser(description='Run Experiment')
# parser.add_argument('--thread_num', type=int, default=4, help='number of threads')  # used in cityflow
# parser.add_argument('--ngpu', type=str, default="0", help='gpu to be used')  # choose gpu card
# parser.add_argument('--prefix', type=str, default='test', help="the number of prefix in this running process")
# parser.add_argument('--seed', type=int, default=None, help="seed for pytorch backend")
# parser.add_argument('--debug', type=bool, default=True)
# parser.add_argument('--interface', type=str, default="libsumo", choices=['libsumo','traci'], help="interface type") # libsumo(fast) or traci(slow)
# parser.add_argument('--delay_type', type=str, default="apx", choices=['apx','real'], help="method of calculating delay") # apx(approximate) or real

# parser.add_argument('-t', '--task', type=str, default="tsc", help="task type to run")
# parser.add_argument('-a', '--agent', type=str, default="dqn", help="agent type of agents in RL environment")
# parser.add_argument('-w', '--world', type=str, default="cityflow", choices=['cityflow','sumo'], help="simulator type")
# parser.add_argument('-n', '--network', type=str, default="cityflow1x1", help="network name")
# parser.add_argument('-d', '--dataset', type=str, default='onfly', help='type of dataset in training process')

# self.episodes = trainer_config["episodes"]
#         self.action_interval = trainer_config["action_interval"]
#         self.steps =trainer_config['steps']
#         self.eval_steps = trainer_config['eval_steps']
#         self.buffer_size = trainer_config['buffer_size']
#         self.target_update = trainer_config["target_update"]
#         self.batch_size = trainer_config["batch_size"]
#         self.epsilon = trainer_config["max_epsilon"]
#         self.epsilon_decay = trainer_config["epsilon_decay"]
#         self.min_epsilon = trainer_config["min_epsilon"]
#         self.gamma = trainer_config["gamma"]
#         self.exp_name = trainer_config["exp_name"]

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    if not cfg.exp.name:
        print("Error! Please specify an experiment name. Set exp.name")
        return

    if cfg.exp.use_wandb:
        wandb.login(key=os.environ["WANDBKEY"])
        wandb.init(name=f"{cfg.wandb.name}-{cfg.agent.name}-{cfg.exp.name}-{cfg.env.network}", reinit = True, project = cfg.wandb.project, entity = cfg.wandb.entity, config = dict(cfg))

    if False:#cfg.agent.name == "dqn":
        env_config = OmegaConf.to_container(cfg.env)
        env_config["sumo_cfg_path"] = f"LibSignal/configs/sim/{env_config['network']}.cfg"
        if env_config["reward_function"] == "queue_length":
            env_config["reward_function"] = "lane_waiting_count" 
        trainer_config = {
            "env_config":env_config,
            "dqn_config":cfg.agent,
            "exp_config":cfg.exp
        }
        dqn_trainer = DQNTrainer(trainer_config)
        dqn_trainer()
        breakpoint()
    
    # register config
    Registry.mapping["config"] = cfg
    register_reward_function(cfg.env.reward_function)
    
    runner = Runner(Namespace(
        **{
            "thread_num": 4,
            "ngpu": 0,
            "prefix": cfg.exp.name,
            "seed": cfg.exp.seed,
            "debug": True,
            "interface": cfg.exp.interface,
            "delay_type": "apx",
            "task":"tsc",
            "agent":cfg.agent.name,
            "world":"sumo",
            "network":cfg.env.network,
            "dataset": "onfly"
        }
    ))

    Registry.mapping['model_mapping']['setting'].param["load_model"] = cfg.exp.load_model
    Registry.mapping['model_mapping']['setting'].param["train_model"] = cfg.exp.train
    Registry.mapping['model_mapping']['setting'].param["test_model"] = cfg.exp.test
    Registry.mapping["world_mapping"]["setting"].param["gui"] = cfg.exp.use_gui

    runner.run()
    
    


if __name__ == "__main__":
    main()