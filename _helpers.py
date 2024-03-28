
from common.registry import Registry


def register_reward_function(rew_func):
    if rew_func == "queue_length": rew_func = "lane_waiting_count"
    Registry.mapping["config"].env.reward_function = rew_func

