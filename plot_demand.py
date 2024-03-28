import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


def demand(name, rou_file, begin_time, num_seconds):
        tree = ET.parse(rou_file)
        result = {}
        number_of_vehicles = 0
        for t in range(1, num_seconds + 1):
            for elem in tree.getroot():
                if elem.tag in ["trip","vehicle"]:
                    depart_time = float(elem.attrib["depart"])
                    arrival_time = depart_time + 50#float(elem.attrib["arrival"])
                    _time = int(depart_time - begin_time)
                    arrival_time = int(arrival_time - begin_time)
                    if _time == t:
                        number_of_vehicles += 1
                    if arrival_time == t:
                        number_of_vehicles -= 1

            result[t] = number_of_vehicles
        results =  [None]*len(result)
        for k,v in result.items():
            results[k-1] = v

        plt.figure(figsize=)
        plt.plot(range(1, num_seconds+1), results)
        plt.ylabel("Number of vehicles")
        plt.xlabel("Time (seconds)")
        plt.title(f"Vehicle demand for the {name} road network")
        plt.show()

        