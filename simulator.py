"""
simulator.py
Simplified 5G vs 6G performance simulator.

Usage:
    python simulator.py        # runs a quick demo scenario

Functions:
- TechProfile: defines the technology parameters (5G, 6G)
- simulate_one: runs a simulation for given parameters and returns metrics
- run_demo: run a small demonstration and plot results

This file is intentionally single-file for ease of use and extension.
"""

import math
import random
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from collections import deque

# ---------- Technology profiles ----------
class TechProfile:
    def __init__(self, name, freq_ghz, bandwidth_mhz, spectral_efficiency_bpshz,
                 base_latency_ms, noise_figure_db):
        """
        freq_ghz: center frequency in GHz
        bandwidth_mhz: channel bandwidth in MHz
        spectral_efficiency_bpshz: baseline bits/s/Hz under good conditions
        base_latency_ms: baseline processing + air latency
        noise_figure_db: receiver NF
        """
        self.name = name
        self.freq_ghz = freq_ghz
        self.bandwidth_mhz = bandwidth_mhz
        self.spectral_efficiency_bpshz = spectral_efficiency_bpshz
        self.base_latency_ms = base_latency_ms
        self.noise_figure_db = noise_figure_db

    @property
    def bandwidth_hz(self):
        return self.bandwidth_mhz * 1e6

    @property
    def peak_capacity_bps(self):
        """Theoretical peak capacity in bps under ideal conditions"""
        return self.bandwidth_hz * self.spectral_efficiency_bpshz

# Predefined profiles (tunable)
PROFILE_5G = TechProfile(
    name="5G",
    freq_ghz=3.5,               # mid-band 5G example
    bandwidth_mhz=100,         # 100 MHz channel
    spectral_efficiency_bpshz=6.0,  # bits/s/Hz (approx)
    base_latency_ms=5.0,
    noise_figure_db=5.0
)

PROFILE_6G = TechProfile(
    name="6G",
    freq_ghz=140.0,            # example sub-THz/THz band (simplified)
    bandwidth_mhz=2000,       # 2 GHz channel (massive)
    spectral_efficiency_bpshz=12.0, # higher efficiency via advanced PHY
    base_latency_ms=1.0,
    noise_figure_db=7.0
)

# ---------- Channel & pathloss model ----------
def path_loss_db(distance_m, freq_ghz, path_loss_exponent=2.7, reference_loss_db=30.0):
    """
    Simple log-distance path loss model with frequency-dependent extra loss.
    reference_loss_db is PL at 1 m for a reference frequency (we add frequency term).
    """
    if distance_m < 1.0:
        distance_m = 1.0
    # free-space-ish with exponent
    fs_component = reference_loss_db + 10.0 * path_loss_exponent * math.log10(distance_m)
    # extra frequency attenuation (higher freq -> more attenuation)
    freq_factor_db = 20.0 * math.log10(freq_ghz / 1.0)  # relative to 1 GHz
    return fs_component + freq_factor_db

def snr_db(transmit_power_dbm, path_loss_db, noise_figure_db, bandwidth_hz):
    """
    Very simplified SNR calculation.
    - transmit_power_dbm: e.g., 30 dBm
    - path_loss_db: from path_loss_db()
    - noise_figure_db: receiver NF
    - bandwidth_hz: channel bandwidth
    Returns SNR in dB.
    """
    # Thermal noise (dBm) = -174 dBm/Hz + 10*log10(BW)
    thermal_noise_dbm = -174.0 + 10.0 * math.log10(bandwidth_hz)
    rx_power_dbm = transmit_power_dbm - path_loss_db
    noise_total_dbm = thermal_noise_dbm + noise_figure_db
    return rx_power_dbm - noise_total_dbm

def spectral_efficiency_from_snr(snr_db, max_se):
    """
    Simple mapping from SNR to spectral efficiency (bps/Hz).
    Use Shannon-like saturating curve but capped by max_se.
    """
    # Shannon capacity per Hz (bits/s/Hz) approximation with some practical gap
    gap_db = 3.0  # practical gap from Shannon
    snr_linear = 10**((snr_db - gap_db) / 10.0)
    se = math.log2(1 + snr_linear)
    # saturate to max_se (technology limit)
    return max(0.1, min(se, max_se))

# ---------- Queueing & simulation ----------
class Packet:
    def __init__(self, size_bytes, arrival_time_s, id=None):
        self.size_bytes = size_bytes
        self.arrival_time_s = arrival_time_s
        self.start_tx_time_s = None
        self.end_tx_time_s = None
        self.id = id

def simulate_one(profile, n_users=50, sim_time_s=10.0, pkt_size_bytes=1500,
                 mean_arrival_rate_per_user=50.0,  # packets per second per user - heavy load
                 tx_power_dbm=30.0, distance_m=50.0,
                 queue_limit_packets=1000, seed=None):
    """
    Simulate simple shared channel for 'sim_time_s' seconds.
    - mean_arrival_rate_per_user: Poisson packet arrival rate per user
    Returns: dict with metrics (avg throughput Mbps, avg latency ms, packet loss %)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Calculate path & spectral efficiency
    pl_db = path_loss_db(distance_m, profile.freq_ghz)
    snr = snr_db(tx_power_dbm, pl_db, profile.noise_figure_db, profile.bandwidth_hz)
    se = spectral_efficiency_from_snr(snr, profile.spectral_efficiency_bpshz)
    # Effective capacity in bps under these channel conditions
    capacity_bps = profile.bandwidth_hz * se

    # Simulation parameters
    t = 0.0
    time_step = 0.001  # 1 ms granularity for scheduling (improves accuracy)
    # Arrival processes: for each user generate Poisson arrivals (approx with exponential inter-arrival)
    user_next_arrival = [np.random.exponential(1.0/mean_arrival_rate_per_user) for _ in range(n_users)]
    queue = deque()
    next_packet_id = 0

    total_bits_transmitted = 0
    total_packets = 0
    packets_dropped = 0
    latencies = []

    # Manager: each timestep, generate arrivals, and serve channel capacity*time_step bits
    while t < sim_time_s:
        # arrivals
        for u in range(n_users):
            if user_next_arrival[u] <= 0.0:
                # generate packet
                pkt = Packet(size_bytes=pkt_size_bytes, arrival_time_s=t, id=next_packet_id)
                next_packet_id += 1
                if len(queue) < queue_limit_packets:
                    queue.append(pkt)
                else:
                    packets_dropped += 1
                # schedule next arrival
                user_next_arrival[u] = np.random.exponential(1.0/mean_arrival_rate_per_user)
                total_packets += 1
            else:
                user_next_arrival[u] -= time_step

        # serve channel for this timeslot
        bits_available = capacity_bps * time_step
        while bits_available > 0 and queue:
            cur = queue[0]
            pkt_bits = cur.size_bytes * 8
            if cur.start_tx_time_s is None:
                cur.start_tx_time_s = t
            if pkt_bits <= bits_available:
                # transmit whole packet
                bits_available -= pkt_bits
                total_bits_transmitted += pkt_bits
                cur.end_tx_time_s = t + ( (pkt_bits) / capacity_bps )
                # compute latency
                latency_s = (cur.end_tx_time_s - cur.arrival_time_s)
                latencies.append(latency_s * 1000.0)  # ms
                queue.popleft()
            else:
                # partial transmission: reduce remaining size (simulate fragmentation)
                # For simplicity, we consider packet will be completed in later slots:
                # subtract the bits and reduce packet size for future
                remaining_bits = pkt_bits - bits_available
                transmitted_bits = bits_available
                total_bits_transmitted += transmitted_bits
                # update packet size for future
                cur.size_bytes = math.ceil(remaining_bits / 8.0)
                bits_available = 0

        # simple queue delay drop: if any packet waited too long, drop it (maxDelay)
        max_delay_s = 0.5  # 500 ms threshold for drop (tunable)
        while queue and (t - queue[0].arrival_time_s) > max_delay_s:
            _ = queue.popleft()
            packets_dropped += 1

        t += time_step

    # metrics
    sim_duration_s = sim_time_s
    throughput_mbps = (total_bits_transmitted / sim_duration_s) / 1e6
    avg_latency_ms = np.mean(latencies) if len(latencies) > 0 else float('nan')
    packet_loss_pct = (packets_dropped / max(1, total_packets)) * 100.0 if total_packets > 0 else 0.0
    # Also compute utilization (approx)
    utilization = min(1.0, ( (mean_arrival_rate_per_user * n_users * pkt_size_bytes * 8.0) / capacity_bps))

    return {
        "tech": profile.name,
        "freq_ghz": profile.freq_ghz,
        "bandwidth_mhz": profile.bandwidth_mhz,
        "spectral_efficiency_bpshz": se,
        "capacity_mbps": capacity_bps/1e6,
        "throughput_mbps": throughput_mbps,
        "avg_latency_ms": avg_latency_ms,
        "packet_loss_pct": packet_loss_pct,
        "utilization": utilization,
        "n_users": n_users,
        "mean_arrival_rate_per_user": mean_arrival_rate_per_user,
        "distance_m": distance_m,
        "snr_db": snr
    }


def run_sweep(profile, loads_pps, n_users=50, sim_time_s=10.0, pkt_size_bytes=1500, distance_m=50.0):
    """Run multiple simulations over loads (packets per second per user)"""
    results = []
    for lam in loads_pps:
        res = simulate_one(profile, n_users=n_users, sim_time_s=sim_time_s, pkt_size_bytes=pkt_size_bytes,
                           mean_arrival_rate_per_user=lam, distance_m=distance_m, seed=42)
        res['input_load_pps'] = lam
        results.append(res)
    return pd.DataFrame(results)

# ---------- Demo entry point ----------
def run_demo():
    import os
    os.makedirs("results", exist_ok=True)

    # parameters
    loads = [1, 5, 10, 20, 50, 100]  # packets per second per user
    n_users = 40
    sim_time_s = 8.0
    pkt_size_bytes = 1200
    distances = [10, 50, 200]  # meters

    all_rows = []
    for d in distances:
        df5 = run_sweep(PROFILE_5G, loads, n_users=n_users, sim_time_s=sim_time_s, pkt_size_bytes=pkt_size_bytes, distance_m=d)
        df6 = run_sweep(PROFILE_6G, loads, n_users=n_users, sim_time_s=sim_time_s, pkt_size_bytes=pkt_size_bytes, distance_m=d)
        df = pd.concat([df5, df6], ignore_index=True)
        df['distance_m'] = d
        all_rows.append(df)
        # save CSV
        df.to_csv(f"results/perf_distance_{d}m.csv", index=False)

    big = pd.concat(all_rows, ignore_index=True)
    big.to_csv("results/perf_all.csv", index=False)
    print("Saved results to results/*.csv")

    # quick plots: throughput vs load for distance 50 m
    plot_df = big[big['distance_m'] == 50]
    fig, ax = plt.subplots(1,3, figsize=(15,4))
    for tech, g in plot_df.groupby('tech'):
        ax[0].plot(g['input_load_pps'], g['throughput_mbps'], marker='o', label=tech)
        ax[1].plot(g['input_load_pps'], g['avg_latency_ms'], marker='o', label=tech)
        ax[2].plot(g['input_load_pps'], g['packet_loss_pct'], marker='o', label=tech)
    ax[0].set_title("Throughput (Mbps) vs load (pps/user)")
    ax[0].set_xlabel("packets/sec per user")
    ax[0].set_ylabel("Throughput (Mbps)")
    ax[1].set_title("Avg latency (ms)")
    ax[1].set_xlabel("packets/sec per user")
    ax[2].set_title("Packet loss (%)")
    ax[2].set_xlabel("packets/sec per user")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig("results/summary_plots_50m.png")
    plt.show()

if __name__ == "__main__":
    run_demo()
