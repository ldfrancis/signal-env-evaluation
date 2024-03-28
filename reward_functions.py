from sumo_rl.environment.traffic_signal import TrafficSignal


def pressure_reward(traffic_signal:TrafficSignal):
    in_lanes = traffic_signal.lanes
    out_lanes = traffic_signal.out_lanes
    n_incoming_vehicles = sum(traffic_signal.sumo.lane.getLastStepVehicleNumber(l) for l in in_lanes)
    n_outgoing_vehicles = sum(traffic_signal.sumo.lane.getLastStepVehicleNumber(l) for l in out_lanes)
    return -(n_incoming_vehicles-n_outgoing_vehicles)


def queue_length_reward(traffic_signal:TrafficSignal):
    in_lanes = traffic_signal.lanes
    queue_length = sum(traffic_signal.sumo.lane.getLastStepHaltingNumber(l) for l in in_lanes)
    return -queue_length


def diff_queue_length_reward(traffic_signal:TrafficSignal):
    if not hasattr(traffic_signal,"prev_queue_length"):
        traffic_signal.prev_queue_length = 0
    queue_length = sum(traffic_signal.sumo.lane.getLastStepHaltingNumber(l) for l in traffic_signal.lanes)
    result = traffic_signal.prev_queue_length - queue_length
    traffic_signal.prev_queue_length = queue_length
    return result
    