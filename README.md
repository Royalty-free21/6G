# 6G
It uses a simplified channel model (frequency-dependent path loss and noise), queueing at the base station, and Monte-Carlo runs to produce metrics and plots.
6G_vs_5G_Simulator
==================

Requirements
------------
- Python 3.9+
- Install dependencies:
    pip install -r requirements.txt

Files
-----
- simulator.py            : main simulator and demo runner
- run_experiments.py      : runs a set of experiments and produces CSV + plots
- requirements.txt

Run quick demo
--------------
python simulator.py

This will:
- run a few scenarios for 5G and 6G at distances 10m, 50m, 200m
- save CSV results under ./results/
- generate a summary PNG (results/summary_plots_50m.png) and show plots

Run full experiments
--------------------
python run_experiments.py

Outputs:
- results/all_experiments.csv
- results/{TECH}_dist{D}m.csv
- results/summary_{D}m.png for D in distances

Interpretation tips
-------------------
- throughput_mbps : measured bits delivered per second (effective)
- avg_latency_ms  : average packet end-to-end latency observed for transmitted packets
- packet_loss_pct : percent of packets dropped due to queue overflow or delay timeout
- Compare how 6G vs 5G behave under higher load and at larger distances.

Extension ideas
---------------
- Replace simple channel model with per-packet SNR variation (fading).
- Implement per-user mobility and variable distances.
- Add scheduler policies (round-robin, proportional fairness).
- Integrate more realistic PHY/MAC from ns-3 or MATLAB.<img width="850" height="478" alt="image" src="https://github.com/user-attachments/assets/8de893fa-a92b-4668-a5ab-f24747ff343e" />

