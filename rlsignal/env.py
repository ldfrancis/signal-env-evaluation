from LibSignal.generator.intersection_phase import IntersectionPhaseGenerator
from LibSignal.generator.lane_vehicle import LaneVehicleGenerator
from LibSignal.world.world_sumo import World
from rlsignal.utils.metrics import Metrics
import numpy as np


class Environment:
    def __init__(self, env_config) -> None:
        interface = "traci" if env_config["use_gui"] else "libsumo"
        self.world = World(
            env_config["sumo_cfg_path"], 
            interface=interface
        )
        self.env_config = env_config
        self.init()
        
    def init(self):
        self.n_intersections = len(self.world.intersection_ids)

        self.queue_fns, self.delay_fns = [], []
        self.ob_gens, self.phase_gens, self.rew_gens = [],[],[]
        self.intersections = []
        self.n_actions = []
        for i in range(self.n_intersections):
            intersection_id = self.world.intersection_ids[i]
            intersection = self.world.id2intersection[intersection_id]

            ob_gen = LaneVehicleGenerator(
                self.world, 
                intersection,
                [self.env_config["obs_repr"]], 
                in_only=True, 
                average=None
            )
            phase_gen = IntersectionPhaseGenerator(
                self.world, 
                intersection,
                ['phase'], 
                targets=['cur_phase'], 
                negative=False
            )
            rew_gen = LaneVehicleGenerator(
                self.world, 
                intersection,
                [self.env_config["reward_function"]], 
                in_only=True, 
                average="all",
                negative=True
            )
            queue = LaneVehicleGenerator(
                self.world, 
                intersection,
                ["lane_waiting_count"], 
                in_only=True,
                negative=False
            )
            delay = LaneVehicleGenerator(
                self.world, 
                intersection,
                ["lane_delay"], 
                in_only=True, 
                average="all",
                negative=False
            )

            self.intersections += [intersection]
            self.ob_gens += [ob_gen]
            self.rew_gens += [rew_gen]
            self.phase_gens += [phase_gen]
            self.queue_fns += [queue]
            self.delay_fns += [delay]
            self.n_actions += [len(intersection.phases)]

        lane_metrics = ['rewards', 'queue', 'delay']
        world_metrics = ['real avg travel time', 'throughput']
        self.metric = Metrics(lane_metrics, world_metrics, self)
        self.action_interval = self.env_config["action_interval"]
        self.steps = 0

    def reset(self):
        self.world.reset()
        self.init()
        return [ob_gen.generate() for ob_gen in self.ob_gens]

    def step(self, actions):
        rews = []
        for _ in range(self.action_interval):
            self.world.step(actions)
            rews += [[rew_gen.generate().item() for rew_gen in self.rew_gens]]
            self.steps +=1

        obs = [ob_gen.generate() for ob_gen in self.ob_gens]
        rews = np.mean(rews, axis=0)
        dones = [False] * self.n_intersections
        info = {}
        return obs, rews, dones, info

    def get_queues(self):
        return [np.sum(np.array(queue_gen.generate())) for queue_gen in self.queue_fns]

    def get_delays(self):
        return [np.sum(np.array(delay_gen.generate())) for delay_gen in self.delay_fns]
    
    def get_phases(self):
        return [phase_gen.generate()[0] for phase_gen in self.phase_gens]

    def get_rewards(self):
        return [rew_gen.generate().item() for rew_gen in self.rew_gens]

    