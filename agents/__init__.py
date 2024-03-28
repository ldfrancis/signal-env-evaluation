from .dqn import DQN
from .ppo import PPO
from .fixed_time import FixedTime


def get_agent_class(name):
    if name == "dqn":
        return DQN
    elif name == "ppo":
        return PPO
    elif name == "fixed_time":
        return FixedTime
    