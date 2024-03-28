from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
import numpy as np
from gymnasium import spaces


class LaneVehicleCountObservation(ObservationFunction):
    
    def __init__(self, ts: TrafficSignal, distance=200):
        super().__init__(ts)
        self._distance = distance

    def __call__(self):
        lanes = self.ts.lanes
        n_lane_vehicles = []
        for lane in lanes:
            all_vehicles = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            detected_vehicles = []
            for v in all_vehicles:
                tls = self.ts.sumo.vehicle.getNextTLS(v)
                if len(tls) == 0:
                    continue
                closest_tls = tls[0]
                _, _, distance, _ = closest_tls
                if distance <= self._distance:
                    detected_vehicles += [v]
            n_lane_vehicles += [len(detected_vehicles)]
        norm_n_lane_vehicles = [n/(self._distance/3) for n in n_lane_vehicles]

        # phase id - one hot
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        return np.array(norm_n_lane_vehicles+phase_id)
        
    def observation_space(self):
        return spaces.Box(
            low=np.zeros(len(self.ts.lanes)+self.ts.num_green_phases, dtype=np.float32),
            high=np.ones(len(self.ts.lanes)+self.ts.num_green_phases, dtype=np.float32)*10,
        )


class LaneVehicleWaitingTimesObservation(ObservationFunction):
    def __init__(self, ts: TrafficSignal, distance=200):
        super().__init__(ts)
        self._distance = distance

    def __call__(self):
        lanes = self.ts.lanes
        norm_lane_waiting_times = []
        for lane in lanes:
            all_vehicles = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            waiting_times = []
            n_waiting_vehicles = 0
            for v in all_vehicles:
                tls = self.ts.sumo.vehicle.getNextTLS(v)
                if len(tls) == 0:
                    continue
                closest_tls = tls[0]
                _, _, distance, _ = closest_tls
                if distance <= self._distance:
                    waiting_time = self.ts.sumo.vehicle.getWaitingTime(v)
                    if waiting_time > 0: 
                        waiting_times += [waiting_time]
                        n_waiting_vehicles += 1
            lane_waiting_time = np.sum(waiting_times)
            norm_lane_waiting_times += [lane_waiting_time/(100*max(1,n_waiting_vehicles))]

        # phase id - one hot
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        result = np.minimum(np.array(norm_lane_waiting_times+phase_id), 1)
        return result
    
    def observation_space(self):
        return spaces.Box(
            low=np.zeros(len(self.ts.lanes)+self.ts.num_green_phases, dtype=np.float32),
            high=np.ones(len(self.ts.lanes)+self.ts.num_green_phases, dtype=np.float32)*10,
        ) 
    

class ImageLikeLaneVehicleInfo(ObservationFunction):
    def __init__(self, ts: TrafficSignal, distance=200):
        super().__init__(ts)
        self._distance = distance

    def __call__(self):
        lanes = self.ts.lanes
        observation = np.zeros((self._distance, len(lanes), 3))
        for i,lane in enumerate(lanes):
            all_vehicles = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            for v in all_vehicles:
                tls = self.ts.sumo.vehicle.getNextTLS(v)
                if len(tls) == 0:
                    continue
                closest_tls = tls[0]
                _, _, distance, _ = closest_tls
                if distance <= self._distance: # detected vehicles
                    speed = self.ts.sumo.vehicle.getSpeed(v)
                    w_time = self.ts.sumo.vehicle.getWaitingTime(v)
                    observation[int(distance), i, 0] = 1
                    observation[int(distance), i, 1] = speed/max(1,self.ts.sumo.lane.getMaxSpeed(lane))
                    observation[int(distance), i, 2] = w_time/(100)
        return {
            "obs": observation,
            "phase": np.array(
                [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
            )
        }
        
    
    def observation_space(self):
        dict = {
            "obs":spaces.Box(
                low=np.zeros((self._distance, len(self.ts.lanes), 2), dtype=np.float32),
                high=np.ones((self._distance, len(self.ts.lanes), 2), dtype=np.float32)*10
            ),
            "phase":spaces.Box(
                low=np.zeros(self.ts.num_green_phases, dtype=np.float32),
                high = np.zeros(self.ts.num_green_phases, dtype=np.float32)
            )
        }
        return dict
        

def get_observation_class(name):
    if name == "lane_vehicle_count":
        return LaneVehicleCountObservation
    elif name == "lane_waiting_time":
        return LaneVehicleWaitingTimesObservation
    elif name == "image_like_vehicle_info":
        return ImageLikeLaneVehicleInfo
    else:
        raise Exception(f"Invalid observation type {name}. Possible choices are ['lane_vehicle_count','lane_waiting_time', and 'image_like_vehicle_info']")