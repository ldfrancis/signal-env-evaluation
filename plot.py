
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


arg = argparse.ArgumentParser()
arg.add_argument("--dir", type=str)
arg = arg.parse_args()

if arg.dir and os.path.exists(arg.dir):
    road_net, observation, reward_fn, alagent = arg.dir.rsplit("/")[-1].split("-")
    files = os.listdir(arg.dir)
    dfs = [pd.read_csv(arg.dir+f"/{f}") for f in files if f[-4:]==".csv"]
    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    for metric,ax in zip(["average_waiting_time", "average_travel_time", "average_delay", "average_queue_length"],axes.ravel()):
        values = np.stack([df[metric].values for df in dfs], axis=1)
        
        if alagent == "fixed_time":
            avg = np.mean(values)
            ax.axhline(y=avg, color='r', linestyle='--', linewidth=2)
        else:
            mean = np.mean(values, axis=1)
            std = np.std(values, axis=1)
            episodes = dfs[0]["episode"]
            ax.plot(episodes, mean)
            ax.fill_between(episodes, mean+std, mean-std, facecolor="C0", alpha=0.4)
            avg = np.min(mean)

        ax.text(0.3,0.9, f"{metric}={avg:.2f}", ha='left', va='top', transform=ax.transAxes)
        ax.set_ylabel(metric)
        ax.set_xlabel("episodes")
    fig.suptitle(f'Results for {alagent.upper()} on {road_net} road network,\n using {observation} as observation and {reward_fn} as reward function', fontsize=10)
    fig.savefig(arg.dir+"/fig.png")
    plt.show()