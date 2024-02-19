import hydra
from omegaconf import DictConfig, OmegaConf
from run import Runner
from common.registry import Registry
from argparse import Namespace
import wandb
import os


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


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    if not cfg.exp.name:
        print("Error! Please specify an experiment name. Set exp.name")
        return
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
    
    if cfg.exp.use_wandb:
        wandb.login(key=os.environ["WANDBKEY"])
        wandb.init(name=f"{cfg.wandb.name}-{cfg.agent.name}-{cfg.exp.name}", reinit = True, project = cfg.wandb.project, entity = cfg.wandb.entity, config = dict(cfg))

    runner.run()


if __name__ == "__main__":
    main()