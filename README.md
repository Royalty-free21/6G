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



<img width="850" height="385" alt="image" src="https://github.com/user-attachments/assets/2091faaa-dcf5-4203-9042-1c40315ff913" />

<img width="2744" height="1886" alt="image" src="https://github.com/user-attachments/assets/0c2a370d-ffa8-475b-95c0-2c6243c644ac" />


<img width="3692" height="2922" alt="image" src="https://github.com/user-attachments/assets/7e7ef057-9322-41ba-9352-7190b59ed49b" />
<img width="850" height="429" alt="image" src="https://github.com/user-attachments/assets/ddb6e911-a29e-4a9f-b147-41394b87512a" />
<img width="1200" height="675" alt="image" src="https://github.com/user-attachments/assets/934d72cf-b8b8-45b3-9f4b-31c17f6cbb0b" /><img width="1195" height="399" alt="image" src="https://github.com/user-attachments/assets/e6315fd0-c7c7-48f6-8179-47833776ea6a" />

<img width="2048" height="1152" alt="image" src="https://github.com/user-attachments/assets/79f4c056-09a3-4576-91b8-0557aab228a9" />












