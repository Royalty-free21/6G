"""
run_experiments.py
Convenience script to run a set of experiments and produce CSV + plots.

Usage:
    python run_experiments.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from simulator import PROFILE_5G, PROFILE_6G, run_sweep

def main():
    os.makedirs("results", exist_ok=True)
    loads = [1, 2, 5, 10, 20, 50, 100]
    n_users = 50
    sim_time_s = 10.0
    pkt_size_bytes = 1200
    distances = [10, 50, 200]

    all_results = []
    for distance in distances:
        for profile in [PROFILE_5G, PROFILE_6G]:
            df = run_sweep(profile, loads_pps=loads, n_users=n_users, sim_time_s=sim_time_s,
                           pkt_size_bytes=pkt_size_bytes, distance_m=distance)
            df['distance_m'] = distance
            all_results.append(df)
            # save individual
            df.to_csv(f"results/{profile.name}_dist{distance}m.csv", index=False)

    big = pd.concat(all_results, ignore_index=True)
    big.to_csv("results/all_experiments.csv", index=False)
    print("Saved results to results/all_experiments.csv")

    # plotting example: throughput vs load for each tech at each distance
    for distance in distances:
        sub = big[big['distance_m'] == distance]
        fig, ax = plt.subplots(1,3, figsize=(15,4))
        for tech, group in sub.groupby('tech'):
            ax[0].plot(group['input_load_pps'], group['throughput_mbps'], marker='o', label=tech)
            ax[1].plot(group['input_load_pps'], group['avg_latency_ms'], marker='o', label=tech)
            ax[2].plot(group['input_load_pps'], group['packet_loss_pct'], marker='o', label=tech)
        ax[0].set_title(f"Throughput @ {distance} m")
        ax[1].set_title(f"Avg Latency @ {distance} m")
        ax[2].set_title(f"Packet Loss @ {distance} m")
        ax[0].set_xlabel("packets/sec per user")
        ax[1].set_xlabel("packets/sec per user")
        ax[2].set_xlabel("packets/sec per user")
        ax[0].legend()
        plt.tight_layout()
        plt.savefig(f"results/summary_{distance}m.png")
        plt.close(fig)

    print("Plots saved in results/")

if __name__ == "__main__":
    main()
