import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# 1. Kirchhoff Check (KCL)
# -------------------------------------------------------------
cases = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
dfs = {}

print("=" * 60)
print("DELIVERABLE 3: KIRCHHOFF CURRENT LAW (KCL) CHECK")
print("=" * 60)

for case in cases:
  csv_file = f"{case}.csv"
  if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found!")
    continue

  df = pd.read_csv(csv_file)
  dfs[case] = df

  # Total sum of terminal currents: Id + Ig + Is + Ib
  kcl_sum = df["id"] + df["ig"] + df["is"] + df["ib"]
  max_violation = np.abs(kcl_sum).max()
  p99_violation = np.percentile(np.abs(kcl_sum), 99)

  print(f"[{case:8s}] Max |Sum(I)| = {max_violation:.3e} A | 99th%: {p99_violation:.3e} A")

# -------------------------------------------------------------
# 2. Generate Required Trend Plots
# -------------------------------------------------------------
os.makedirs("plots", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Plot 1: OFF Current vs Temperature (NMOS_OFF and PMOS_OFF)
fig, ax = plt.subplots(figsize=(7, 5))
if "NMOS_OFF" in dfs:
  df_n_off = dfs["NMOS_OFF"]
  ax.scatter(df_n_off["temp"], np.abs(df_n_off["id"]), alpha=0.25, s=8, label="NMOS OFF |Id|", color="crimson")
if "PMOS_OFF" in dfs:
  df_p_off = dfs["PMOS_OFF"]
  ax.scatter(df_p_off["temp"], np.abs(df_p_off["id"]), alpha=0.25, s=8, label="PMOS OFF |Id|", color="navy")

ax.set_yscale("log")
ax.set_title("OFF Current vs Temperature (-40°C to 125°C)", fontsize=13, fontweight="bold")
ax.set_xlabel("Temperature (°C)", fontsize=11)
ax.set_ylabel("Drain Current |Id| (A) [Log Scale]", fontsize=11)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig("plots/trend_off_current_vs_temp.png", dpi=300)
plt.close()
print("Saved: plots/trend_off_current_vs_temp.png")

# Plot 2: OFF Current vs vtho (NMOS_OFF and PMOS_OFF)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
if "NMOS_OFF" in dfs:
  df_n_off = dfs["NMOS_OFF"]
  ax1.scatter(df_n_off["vtho"], np.abs(df_n_off["id"]), alpha=0.3, s=8, color="crimson")
  ax1.set_yscale("log")
  ax1.set_title("NMOS OFF: |Id| vs vth0", fontsize=12, fontweight="bold")
  ax1.set_xlabel("vth0 (V)", fontsize=11)
  ax1.set_ylabel("|Id| (A) [Log Scale]", fontsize=11)

if "PMOS_OFF" in dfs:
  df_p_off = dfs["PMOS_OFF"]
  ax2.scatter(df_p_off["vtho"], np.abs(df_p_off["id"]), alpha=0.3, s=8, color="navy")
  ax2.set_yscale("log")
  ax2.set_title("PMOS OFF: |Id| vs vth0", fontsize=12, fontweight="bold")
  ax2.set_xlabel("vth0 (V)", fontsize=11)
  ax2.set_ylabel("|Id| (A) [Log Scale]", fontsize=11)

plt.suptitle("Subthreshold Leakage vs Threshold Voltage", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/trend_off_current_vs_vtho.png", dpi=300)
plt.close()
print("Saved: plots/trend_off_current_vs_vtho.png")

# Plot 3: Gate Current vs toxe (NMOS_ON and PMOS_ON)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
if "NMOS_ON" in dfs:
  df_n_on = dfs["NMOS_ON"]
  ax1.scatter(df_n_on["toxe"] * 1e9, np.abs(df_n_on["ig"]), alpha=0.3, s=8, color="darkgreen")
  ax1.set_yscale("log")
  ax1.set_title("NMOS ON: |Ig| vs toxe", fontsize=12, fontweight="bold")
  ax1.set_xlabel("toxe (nm)", fontsize=11)
  ax1.set_ylabel("|Ig| (A) [Log Scale]", fontsize=11)

if "PMOS_ON" in dfs:
  df_p_on = dfs["PMOS_ON"]
  ax2.scatter(df_p_on["toxe"] * 1e9, np.abs(df_p_on["ig"]), alpha=0.3, s=8, color="purple")
  ax2.set_yscale("log")
  ax2.set_title("PMOS ON: |Ig| vs toxe", fontsize=12, fontweight="bold")
  ax2.set_xlabel("toxe (nm)", fontsize=11)
  ax2.set_ylabel("|Ig| (A) [Log Scale]", fontsize=11)

plt.suptitle("Gate Tunneling Leakage vs Oxide Thickness", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/trend_gate_current_vs_toxe.png", dpi=300)
plt.close()
print("Saved: plots/trend_gate_current_vs_toxe.png")

# -------------------------------------------------------------
# 3. Deliverable 2: Example Transistor Netlists
# -------------------------------------------------------------
os.makedirs("netlists/transistor", exist_ok=True)

configs = {
    "nmos_on":  {"type": "nmos", "vd": 0.9, "vg": 0.9, "vs": 0.0, "vb": 0.0},
    "nmos_off": {"type": "nmos", "vd": 0.9, "vg": 0.0, "vs": 0.0, "vb": 0.0},
    "pmos_on":  {"type": "pmos", "vd": 0.0, "vg": 0.0, "vs": 0.9, "vb": 0.9},
    "pmos_off": {"type": "pmos", "vd": 0.0, "vg": 0.9, "vs": 0.9, "vb": 0.9},
}

for name, cfg in configs.items():
  dev = "n" if cfg["type"] == "nmos" else "p"
  netlist_text = f"""* Example Single-Transistor Netlist: {name.upper()}
* Tech node: 16nm_HP | Nominal VDD = 0.9V

.INCLUDE ../../16nm_HP.pm
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9
.TEMP 27.0

M{dev} drain gate source body {cfg['type']} W=32n L=16n

Vd drain  0 {cfg['vd']}
Vg gate   0 {cfg['vg']}
Vs source 0 {cfg['vs']}
Vb body   0 {cfg['vb']}

.control
op
print i(Vd) i(Vg) i(Vs) i(Vb)
.endc
.END
"""
  filepath = f"netlists/transistor/{name}.sp"
  with open(filepath, "w") as f:
    f.write(netlist_text)
  print(f"Generated: {filepath}")

print("=" * 60)
print("Data check and initial deliverables completed successfully.")
print("=" * 60)