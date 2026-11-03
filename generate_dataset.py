import concurrent.futures
import os
import re
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------
# Configuration & Specifications
# ---------------------------------------------------------
NUM_SAMPLES = 10000
RANDOM_SEED = 42
LMIN = 16e-9  # 16 nm
VDD_NOM = 0.9  # Assigned nominal VDD = 0.9 V

# Locate ngspice_con executable
NGSPICE_EXE = shutil.which("ngspice_con") or shutil.which("ngspice")
if not NGSPICE_EXE:
  fallback = r"C:\Spice64\bin\ngspice_con.exe"
  if os.path.exists(fallback):
    NGSPICE_EXE = fallback
  else:
    raise FileNotFoundError(
        "Could not find ngspice_con.exe. Ensure C:\\Spice64\\bin is in your"
        " PATH."
    )

# Baseline parameters extracted from 16nm_HP.pm
NOMINALS = {
    "nmos": {
        "vtho": 0.47965,
        "toxe": 9.5e-10,
        "ndep": 7.0e18,
        "u0": 0.03,
        "xj": 5.0e-9,
    },
    "pmos": {
        "vtho": -0.43121,
        "toxe": 1.0e-9,
        "ndep": 5.5e18,
        "u0": 0.006,
        "xj": 5.0e-9,
    },
}

if not os.path.exists("16nm_HP.pm"):
  raise FileNotFoundError(
      "16nm_HP.pm not found in the current working directory."
  )

with open("16nm_HP.pm", "r") as f:
  RAW_MODEL_TEXT = f.read()

# Split 16nm_HP.pm into NMOS and PMOS sections so substitutions don't cross-contaminate
parts = re.split(r"(?i)(?=\.model\s+pmos)", RAW_MODEL_TEXT)
NMOS_MODEL_TEMPLATE = parts[0]
PMOS_MODEL_TEMPLATE = parts[1] if len(parts) > 1 else RAW_MODEL_TEXT


# ---------------------------------------------------------
# Netlist Generation & SPICE Execution
# ---------------------------------------------------------
def build_netlist(row, mos_type):
  """Builds a standalone SPICE netlist with updated model parameters."""
  model_str = (
      NMOS_MODEL_TEMPLATE if mos_type == "nmos" else PMOS_MODEL_TEMPLATE
  )

  # Map DataFrame column names to exact SPICE parameter keywords
  spice_params = {
      "vth0": row["vtho"],  # Map 'vtho' (column) to 'vth0' (SPICE)
      "toxe": row["toxe"],
      "ndep": row["ndep"],
      "u0": row["u0"],
      "xj": row["xj"],
  }

  for param, val in spice_params.items():
    model_str = re.sub(
        rf"(\b{param}\s*=\s*)([^\s]+)", rf"\g<1>{val:.6e}", model_str
    )

  dev_letter = "n" if mos_type == "nmos" else "p"

  netlist = f"""* Single Transistor PVT Leakage Simulation
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9
.TEMP {row['temp']:.2f}

{model_str}

M{dev_letter} drain gate source body {mos_type} W={row['w']:.6e} L={LMIN:.6e}
Vd drain 0 {row['vd']:.6f}
Vg gate  0 {row['vg']:.6f}
Vs source 0 {row['vs']:.6f}
Vb body  0 {row['vb']:.6f}

.control
op
print i(Vd)
print i(Vg)
print i(Vs)
print i(Vb)
.endc
.END
"""
  return netlist


def simulate_single(sample_dict, mos_type):
  """Writes temporary netlist, runs ngspice_con, and extracts terminal currents."""
  netlist = build_netlist(sample_dict, mos_type)

  # Use a self-deleting temp file for reliable Windows execution
  with tempfile.NamedTemporaryFile(
      mode="w", suffix=".cir", delete=False
  ) as tmp_file:
    tmp_file.write(netlist)
    tmp_path = tmp_file.name

  try:
    proc = subprocess.run(
        [NGSPICE_EXE, "-b", tmp_path], capture_output=True, text=True
    )
    stdout = proc.stdout
  finally:
    if os.path.exists(tmp_path):
      try:
        os.remove(tmp_path)
      except OSError:
        pass

  # Match both i(vx) and vx#branch syntax
  match_d = re.search(
      r"(?:i\(vd\)|vd#branch)\s*=\s*([+-]?[\d\.]+(?:[eE][+-]?\d+)?)",
      stdout,
      re.IGNORECASE,
  )
  match_g = re.search(
      r"(?:i\(vg\)|vg#branch)\s*=\s*([+-]?[\d\.]+(?:[eE][+-]?\d+)?)",
      stdout,
      re.IGNORECASE,
  )
  match_s = re.search(
      r"(?:i\(vs\)|vs#branch)\s*=\s*([+-]?[\d\.]+(?:[eE][+-]?\d+)?)",
      stdout,
      re.IGNORECASE,
  )
  match_b = re.search(
      r"(?:i\(vb\)|vb#branch)\s*=\s*([+-]?[\d\.]+(?:[eE][+-]?\d+)?)",
      stdout,
      re.IGNORECASE,
  )

  if match_d and match_g and match_s and match_b:
    # Terminal current entering device = -I(V_source)
    id_val = -float(match_d.group(1))
    ig_val = -float(match_g.group(1))
    is_val = -float(match_s.group(1))
    ib_val = -float(match_b.group(1))
  else:
    id_val, ig_val, is_val, ib_val = np.nan, np.nan, np.nan, np.nan

  result = sample_dict.copy()
  result["id"] = id_val
  result["ig"] = ig_val
  result["is"] = is_val
  result["ib"] = ib_val
  return result


# ---------------------------------------------------------
# Sample Generation
# ---------------------------------------------------------
def generate_samples(case_name, n_samples, seed_offset=0):
  np.random.seed(RANDOM_SEED + seed_offset)

  # 1. VDD: Uniform, assigned VDD +- 10%
  vdd = np.random.uniform(0.9 * VDD_NOM, 1.1 * VDD_NOM, n_samples)

  # 2. Vd and Vs: Uniform, 0 to VDD
  vd = np.random.uniform(0.0, vdd)
  vs = np.random.uniform(0.0, vdd)

  # 3. Width: Uniform, 1 * Lmin to 8 * Lmin
  w = np.random.uniform(1.0 * LMIN, 8.0 * LMIN, n_samples)

  # 4. Temperature: Uniform, -40 to 125 C
  temp = np.random.uniform(-40.0, 125.0, n_samples)

  mos_type = "nmos" if "NMOS" in case_name else "pmos"
  nom = NOMINALS[mos_type]

  def sample_param(nom_val):
    low = min(nom_val * 0.9, nom_val * 1.1)
    high = max(nom_val * 0.9, nom_val * 1.1)
    return np.random.uniform(low, high, n_samples)

  vtho = sample_param(nom["vtho"])
  toxe = sample_param(nom["toxe"])
  ndep = sample_param(nom["ndep"])
  u0 = sample_param(nom["u0"])
  xj = sample_param(nom["xj"])

  # 5. Gate and Body voltages fixed by case
  if case_name == "NMOS_ON":
    vg = vdd.copy()
    vb = np.zeros(n_samples)
  elif case_name == "NMOS_OFF":
    vg = np.zeros(n_samples)
    vb = np.zeros(n_samples)
  elif case_name == "PMOS_ON":
    vg = np.zeros(n_samples)
    vb = vdd.copy()
  elif case_name == "PMOS_OFF":
    vg = vdd.copy()
    vb = vdd.copy()

  df = pd.DataFrame({
      "vdd": vdd,
      "temp": temp,
      "w": w,
      "vd": vd,
      "vs": vs,
      "vg": vg,
      "vb": vb,
      "vtho": vtho,
      "toxe": toxe,
      "ndep": ndep,
      "u0": u0,
      "xj": xj,
  })
  return df, mos_type


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
  cases = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
  workers = max(1, (os.cpu_count() or 4) - 1)
  print(f"Using {workers} parallel processes with {NGSPICE_EXE}")

  # Pre-flight check: Run 1 test sample to ensure SPICE output parses cleanly
  test_df, test_type = generate_samples("NMOS_ON", 1)
  test_res = simulate_single(test_df.to_dict(orient="records")[0], test_type)
  if np.isnan(test_res["id"]):
    raise RuntimeError(
        "Pre-flight simulation test failed to extract currents. Check your"
        " Ngspice installation."
    )
  print("Pre-flight SPICE test passed successfully.")

  for idx, case in enumerate(cases):
    print(f"\n--- Generating {NUM_SAMPLES} samples for {case} ---")
    df_samples, mos_type = generate_samples(case, NUM_SAMPLES, seed_offset=idx)
    sample_dicts = df_samples.to_dict(orient="records")

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
      futures = [
          executor.submit(simulate_single, s, mos_type) for s in sample_dicts
      ]
      for f in tqdm(
          concurrent.futures.as_completed(futures),
          total=len(futures),
          desc=case,
      ):
        results.append(f.result())

    df_out = pd.DataFrame(results)

    # Deliverable 1 exact column ordering
    cols = [
        "vdd",
        "temp",
        "w",
        "vd",
        "vs",
        "vg",
        "vb",
        "vtho",
        "toxe",
        "ndep",
        "u0",
        "xj",
        "id",
        "ig",
        "is",
        "ib",
    ]
    df_out = df_out[cols]

    # Verify KCL: |Id + Ig + Is + Ib| ~ 0
    kcl_err = np.abs(
        df_out["id"] + df_out["ig"] + df_out["is"] + df_out["ib"]
    ).max()
    print(f"Max Kirchhoff current sum violation: {kcl_err:.3e} A")

    filename = f"{case}.csv"
    df_out.to_csv(filename, index=False)
    print(f"Saved: {filename} ({len(df_out)} rows)")


if __name__ == "__main__":
  main()