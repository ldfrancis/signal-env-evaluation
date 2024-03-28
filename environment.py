from typing import Callable, Tuple
from observations import ImageLikeLaneVehicleInfo, LaneVehicleCountObservation
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment.observations import DefaultObservationFunction, ObservationFunction
import numpy as np
import time


class SignalEnvironment(SumoEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # containers for vehicle info and lane info
        self.init()

    def init(self):
        self._vehicles = {}
        self._lanes = {l:{} for ts in self.traffic_signals.values() for l in ts.lanes}
        self._is_done = False

    def _sumo_step(self):
        return super()._sumo_step()

    def reset(self, seed: int | None = None, **kwargs):
        self.init()
        return super().reset(seed, **kwargs)

    def step(self, action):
        observations, rewards, dones, info = super().step(action)
        
        # metrics
        # travel_time, delay, throughput, waiting_time, queue
        
        self._is_done = dones["__all__"]
        self.update_lanes_and_vehicles()
        
        return observations, rewards, dones, info
    
    def update_lanes_and_vehicles(self):
        vehicles = self.sumo.vehicle.getIDList()
        cur_simulation_time = self.sumo.simulation.getTime()

        # obtain waiting time, and speed info for each vehicle
        # also update the queue length for lanes
        for v in vehicles:
            # waiting time, departure time, travel time, speed, and last time in simulation
            v_info = self._vehicles.get(v, {})

            # waiting time
            _waiting_time = self.sumo.vehicle.getWaitingTime(v)
            cur_waiting_time = v_info.get("cur_waiting_time", 0)
            waiting_time = v_info.get("waiting_time",0)
            if _waiting_time == 0:
                # waiting time not accounted for due to the step length of the environment
                if cur_waiting_time != 0:
                    delta = self.delta_time 
                else:
                    delta = 0
                waiting_time += (cur_waiting_time+delta)
            v_info["waiting_time"] = waiting_time
            v_info["cur_waiting_time"] = _waiting_time
            
            # travel time
            if not v in self._vehicles:
                v_info["departure_time"] = self.sumo.vehicle.getDeparture(v)
            v_info["last_time"] = cur_simulation_time 
            v_info["travel_time"] = cur_simulation_time-v_info["departure_time"]

            # trajectory info
            cur_lane = self.sumo.vehicle.getLaneID(v)
            cur_lane_pos = self.sumo.vehicle.getLanePosition(v)
            if cur_lane != v_info.get("cur_lane", ""): # entered a new lane
                v_info["trajectory"] = v_info.get("trajectory",[]) + [[cur_lane, self.sumo.vehicle.getAllowedSpeed(v), cur_lane_pos, cur_lane_pos]]
            else:
                v_info["trajectory"][-1][-1] = cur_lane_pos
            v_info["cur_lane"] = cur_lane

            # speed
            speed = self.sumo.vehicle.getSpeed(v)
            v_info["cur_speed"] = speed

            self._vehicles[v] = v_info

            # lane queue lengths
            # lane = self.sumo.vehicle.getLaneID(v)
            # queued_vehicles = self._lanes[lane].get("queued_vehicles", set())
            # if not v in queued_vehicles and speed < 0.1:
            #     queued_vehicles.add(v)
            #     self._lanes[lane]["queued_vehicles"] = queued_vehicles
            #     self._lanes[lane]["total_queue_length"] = len(queued_vehicles)

        for lane in self._lanes:
            l_vehicles = self.sumo.lane.getLastStepVehicleIDs(lane)
            stopped_vehicles = [v for v in l_vehicles if self.sumo.vehicle.getSpeed(v) < 0.1]
            n_stopped_vehilces = len(stopped_vehicles)
            queue_length = max(n_stopped_vehilces, self._lanes[lane].get("queue_length",0))
            self._lanes[lane]["queue_length"] = queue_length
            

    def obtain_metric(self, name, aggregation="sum")->float:
        """Obtain a particular metric from the environment after simulation is done

        Args:
            name: the name of the metric to obtain. possible metrics include
                - travel time
                - delay
                - throughput
                - waiting time
                - queue length
            aggregation: The type of aggregation to apply for the chosen metrics. Most of
                this metrics are calculated either per vehicle or per lane, and an aggregation
                returns a single number for all values obtained per vehicle or lane. Possible
                aggregations are:
                - sum
                - average
        Returns:
            metric (float): the aggregated metric for the simulation
        """
        assert self._is_done, "Simulation must be done before metrics are obtained"

        if name == "travel_time":
            results = [v["travel_time"] for v in self._vehicles.values()]
            
        elif name == "throughput":
            cur_time = self.sumo.simulation.getTime()
            v_completed_trips = [v for v in self._vehicles.values() if v["last_time"] < cur_time]
            # There is no aggregation for throughput
            num_completed_trips = len(v_completed_trips)
            return num_completed_trips
        
        elif name == "waiting_time":
            results = [v["waiting_time"] for v in self._vehicles.values()]

        elif name == "queue_length":
            results = [l["queue_length"] for l in self._lanes.values()]

        elif name == "delay":
            # ideal_travel_time
            results = []
            for v in self._vehicles.values():
                trajectory = v["trajectory"]
                dist_maxspeed = [(abs(d1-d2), s) for _, s, d1, d2 in trajectory]
                ideal_travel_times_on_lanes = [d/s for d,s in dist_maxspeed]
                results += [v["travel_time"]-sum(ideal_travel_times_on_lanes)]
        else:
            raise Exception(f"Invalid metric {name}. Possible metrics are ['delay','queue_length','waiting_time','throughput',and 'travel_time']")
            
        # Perform aggregation
        if aggregation == "sum":
            return sum(results)
        elif aggregation == "average":
            return np.mean(results)
        else:
            raise Exception(f"Invalid aggregation type {aggregation}. Possible aggregations are ['sum', 'average']")




if __name__=="__main__":

    env = SignalEnvironment(
        net_file="sumorl/nets/RESCO/grid4x4/grid4x4.net.xml",
        route_file='sumorl/nets/RESCO/grid4x4/grid4x4_1.rou.xml',
        use_gui=True,
        num_seconds=3600,
        observation_class = ImageLikeLaneVehicleInfo
    )

    obs = env.reset()
    breakpoint()


    