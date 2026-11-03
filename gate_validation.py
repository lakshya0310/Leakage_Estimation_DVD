import concurrent.futures
import os
import re
import shutil
import subprocess
import tempfile
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from train_models import CanonicalMOSModel

# ---------------------------------------------------------------------------
# Configuration & Specifications
# ---------------------------------------------------------------------------
NUM_CORNERS = 200
RANDOM_SEED = 42
LMIN = 16e-9  # 16 nm
VDD_NOM = 0.9  # Nominal VDD = 0.9 V

os.makedirs("netlists/gates", exist_ok=True)
os.makedirs("plots/gates", exist_ok=True)

# Locate ngspice executable
NGSPICE_EXE = shutil.which("ngspice_con") or shutil.which("ngspice")
if not NGSPICE_EXE:
  fallback = r"C:\Spice64\bin\ngspice_con.exe"
  if os.path.exists(fallback):
    NGSPICE_EXE = fallback
  else:
    NGSPICE_EXE = "ngspice_con"

if not os.path.exists("16nm_HP.pm"):
  raise FileNotFoundError("16nm_HP.pm not found.")
with open("16nm_HP.pm", "r") as f:
  RAW_MODEL_TEXT = f.read()

parts = re.split(r"(?i)(?=\.model\s+pmos)", RAW_MODEL_TEXT)
NMOS_MODEL_TEMPLATE = parts[0]
PMOS_MODEL_TEMPLATE = parts[1] if len(parts) > 1 else RAW_MODEL_TEXT

print("Loading trained models from models/...")
models = {
    "NMOS_ON": joblib.load("models/NMOS_ON_model.joblib"),
    "NMOS_OFF": joblib.load("models/NMOS_OFF_model.joblib"),
    "PMOS_ON": joblib.load("models/PMOS_ON_model.joblib"),
    "PMOS_OFF": joblib.load("models/PMOS_OFF_model.joblib"),
}
print("Models loaded successfully!")

# ---------------------------------------------------------------------------
# Deliverable 2: Generate Example SPICE Netlists for Each Gate
# ---------------------------------------------------------------------------
def generate_sample_gate_netlists():
  """Generates standalone example SPICE netlists for INV, NAND2, NOR2, AND2, and OR2."""
  os.makedirs("netlists/gates", exist_ok=True)
  
  sample_gates = {
      "inv": f"""* Example SPICE Netlist: INV
* Tech node: 16nm_HP | Nominal VDD = {VDD_NOM}V

.INCLUDE ../../16nm_HP.pm
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9
.TEMP 27.0

Vdd supply 0 {VDD_NOM}
Va a 0 0.0

Mp out a supply supply pmos W=64n L={LMIN}
Mn out a 0 0 nmos W=32n L={LMIN}

.control
op
print i(Vdd) v(out)
.endc
.END
""",
      "nand2": f"""* Example SPICE Netlist: NAND2
* Tech node: 16nm_HP | Nominal VDD = {VDD_NOM}V

.INCLUDE ../../16nm_HP.pm
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9
.TEMP 27.0

Vdd supply 0 {VDD_NOM}
Va a 0 0.0
Vb b 0 0.0

Mp1 out a supply supply pmos W=64n L={LMIN}
Mp2 out b supply supply pmos W=64n L={LMIN}
Mn1 out a x 0 nmos W=64n L={LMIN}
Mn2 x b 0 0 nmos W=64n L={LMIN}

.control
op
print i(Vdd) v(out) v(x)
.endc
.END
""",
      "nor2": f"""* Example SPICE Netlist: NOR2
* Tech node: 16nm_HP | Nominal VDD = {VDD_NOM}V

.INCLUDE ../../16nm_HP.pm
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9
.TEMP 27.0

Vdd supply 0 {VDD_NOM}
Va a 0 0.0
Vb b 0 0.0

Mp1 y a supply supply pmos W=128n L={LMIN}
Mp2 out b y supply pmos W=128n L={LMIN}
Mn1 out a 0 0 nmos W=32n L={LMIN}
Mn2 out b 0 0 nmos W=32n L={LMIN}

.control
op
print i(Vdd) v(out) v(y)
.endc
.END
""",
      "and2": f"""* Example SPICE Netlist: AND2
* Tech node: 16nm_HP | Nominal VDD = {VDD_NOM}V

.INCLUDE ../../16nm_HP.pm
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9
.TEMP 27.0

Vdd supply 0 {VDD_NOM}
Va a 0 0.0
Vb b 0 0.0

Mp1 nand_out a supply supply pmos W=64n L={LMIN}
Mp2 nand_out b supply supply pmos W=64n L={LMIN}
Mn1 nand_out a x 0 nmos W=64n L={LMIN}
Mn2 x b 0 0 nmos W=64n L={LMIN}
Mp3 out nand_out supply supply pmos W=64n L={LMIN}
Mn3 out nand_out 0 0 nmos W=32n L={LMIN}

.control
op
print i(Vdd) v(out) v(nand_out) v(x)
.endc
.END
""",
      "or2": f"""* Example SPICE Netlist: OR2
* Tech node: 16nm_HP | Nominal VDD = {VDD_NOM}V

.INCLUDE ../../16nm_HP.pm
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9
.TEMP 27.0

Vdd supply 0 {VDD_NOM}
Va a 0 0.0
Vb b 0 0.0

Mp1 y a supply supply pmos W=128n L={LMIN}
Mp2 nor_out b y supply pmos W=128n L={LMIN}
Mn1 nor_out a 0 0 nmos W=32n L={LMIN}
Mn2 nor_out b 0 0 nmos W=32n L={LMIN}
Mp3 out nor_out supply supply pmos W=64n L={LMIN}
Mn3 out nor_out 0 0 nmos W=32n L={LMIN}

.control
op
print i(Vdd) v(out) v(nor_out) v(y)
.endc
.END
""",
  }

  for name, text in sample_gates.items():
    path = f"netlists/gates/{name}.sp"
    with open(path, "w") as f:
      f.write(text)
    print(f"Generated sample netlist: {path}")

# ---------------------------------------------------------------------------
# Corner Generation across 200 Process Corners
# ---------------------------------------------------------------------------
def generate_corners(n_corners=200):
  np.random.seed(RANDOM_SEED)
  vdd = np.random.uniform(0.9 * VDD_NOM, 1.1 * VDD_NOM, n_corners)
  temp = np.random.uniform(-40.0, 125.0, n_corners)

  n_vtho = np.random.uniform(0.431685, 0.527615, n_corners)
  n_toxe = np.random.uniform(8.55e-10, 1.045e-9, n_corners)
  n_ndep = np.random.uniform(6.3e18, 7.7e18, n_corners)
  n_u0 = np.random.uniform(0.027, 0.033, n_corners)
  n_xj = np.random.uniform(4.5e-9, 5.5e-9, n_corners)

  p_vtho = np.random.uniform(-0.474331, -0.388089, n_corners)
  p_toxe = np.random.uniform(9.0e-10, 1.1e-9, n_corners)
  p_ndep = np.random.uniform(4.95e18, 6.05e18, n_corners)
  p_u0 = np.random.uniform(0.0054, 0.0066, n_corners)
  p_xj = np.random.uniform(4.5e-9, 5.5e-9, n_corners)

  corners = []
  for i in range(n_corners):
    corners.append({
        "vdd": vdd[i], "temp": temp[i],
        "n_vtho": n_vtho[i], "n_toxe": n_toxe[i], "n_ndep": n_ndep[i], "n_u0": n_u0[i], "n_xj": n_xj[i],
        "p_vtho": p_vtho[i], "p_toxe": p_toxe[i], "p_ndep": p_ndep[i], "p_u0": p_u0[i], "p_xj": p_xj[i],
    })
  return corners

GATE_CASES = [
    ("INV", "0", {"A": 0.0}),
    ("INV", "1", {"A": 1.0}),
    ("NAND2", "00", {"A": 0.0, "B": 0.0}),
    ("NAND2", "01", {"A": 0.0, "B": 1.0}),
    ("NAND2", "10", {"A": 1.0, "B": 0.0}),
    ("NAND2", "11", {"A": 1.0, "B": 1.0}),
    ("NOR2", "00", {"A": 0.0, "B": 0.0}),
    ("NOR2", "01", {"A": 0.0, "B": 1.0}),
    ("NOR2", "10", {"A": 1.0, "B": 0.0}),
    ("NOR2", "11", {"A": 1.0, "B": 1.0}),
    ("AND2", "00", {"A": 0.0, "B": 0.0}),
    ("AND2", "01", {"A": 0.0, "B": 1.0}),
    ("AND2", "10", {"A": 1.0, "B": 0.0}),
    ("AND2", "11", {"A": 1.0, "B": 1.0}),
    ("OR2", "00", {"A": 0.0, "B": 0.0}),
    ("OR2", "01", {"A": 0.0, "B": 1.0}),
    ("OR2", "10", {"A": 1.0, "B": 0.0}),
    ("OR2", "11", {"A": 1.0, "B": 1.0}),
]

def build_gate_netlist(gate_name, inputs, corner):
  vdd = corner["vdd"]
  va = inputs.get("A", 0.0) * vdd
  vb = inputs.get("B", 0.0) * vdd

  n_model = NMOS_MODEL_TEMPLATE
  for p, val in [("vth0", corner["n_vtho"]), ("toxe", corner["n_toxe"]), ("ndep", corner["n_ndep"]), ("u0", corner["n_u0"]), ("xj", corner["n_xj"])]:
    n_model = re.sub(rf"(\b{p}\s*=\s*)([^\s]+)", rf"\g<1>{val:.6e}", n_model)

  p_model = PMOS_MODEL_TEMPLATE
  for p, val in [("vth0", corner["p_vtho"]), ("toxe", corner["p_toxe"]), ("ndep", corner["p_ndep"]), ("u0", corner["p_u0"]), ("xj", corner["p_xj"])]:
    p_model = re.sub(rf"(\b{p}\s*=\s*)([^\s]+)", rf"\g<1>{val:.6e}", p_model)

  if gate_name == "INV":
    circuit = f"""
Mp out a supply supply pmos W=64n L={LMIN}
Mn out a 0 0 nmos W=32n L={LMIN}
Va a 0 {va}
"""
    print_nodes = "print i(Vdd) v(out)"
  elif gate_name == "NAND2":
    circuit = f"""
Mp1 out a supply supply pmos W=64n L={LMIN}
Mp2 out b supply supply pmos W=64n L={LMIN}
Mn1 out a x 0 nmos W=64n L={LMIN}
Mn2 x b 0 0 nmos W=64n L={LMIN}
Va a 0 {va}
Vb b 0 {vb}
"""
    print_nodes = "print i(Vdd) v(out) v(x)"
  elif gate_name == "NOR2":
    circuit = f"""
Mp1 y a supply supply pmos W=128n L={LMIN}
Mp2 out b y supply pmos W=128n L={LMIN}
Mn1 out a 0 0 nmos W=32n L={LMIN}
Mn2 out b 0 0 nmos W=32n L={LMIN}
Va a 0 {va}
Vb b 0 {vb}
"""
    print_nodes = "print i(Vdd) v(out) v(y)"
  elif gate_name == "AND2":
    circuit = f"""
Mp1 nand_out a supply supply pmos W=64n L={LMIN}
Mp2 nand_out b supply supply pmos W=64n L={LMIN}
Mn1 nand_out a x 0 nmos W=64n L={LMIN}
Mn2 x b 0 0 nmos W=64n L={LMIN}
Mp3 out nand_out supply supply pmos W=64n L={LMIN}
Mn3 out nand_out 0 0 nmos W=32n L={LMIN}
Va a 0 {va}
Vb b 0 {vb}
"""
    print_nodes = "print i(Vdd) v(out) v(nand_out) v(x)"
  elif gate_name == "OR2":
    circuit = f"""
Mp1 y a supply supply pmos W=128n L={LMIN}
Mp2 nor_out b y supply pmos W=128n L={LMIN}
Mn1 nor_out a 0 0 nmos W=32n L={LMIN}
Mn2 nor_out b 0 0 nmos W=32n L={LMIN}
Mp3 out nor_out supply supply pmos W=64n L={LMIN}
Mn3 out nor_out 0 0 nmos W=32n L={LMIN}
Va a 0 {va}
Vb b 0 {vb}
"""
    print_nodes = "print i(Vdd) v(out) v(nor_out) v(y)"
  else:
    raise ValueError(f"Unknown gate {gate_name}")

  netlist = f"""* Gate SPICE Simulation: {gate_name} Inputs {inputs}
.OPTIONS GMIN=1e-18 ABSTOL=1e-18 RELTOL=1e-6 VNTOL=1e-9
.TEMP {corner['temp']:.2f}

{n_model}
{p_model}

Vdd supply 0 {vdd}
{circuit}

.control
op
{print_nodes}
.endc
.END
"""
  return netlist

def simulate_corner_spice(args):
  """Worker function to simulate a single corner in ngspice and extract results."""
  gate_name, inputs, corner = args
  netlist = build_gate_netlist(gate_name, inputs, corner)
  with tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False) as tf:
    tf.write(netlist)
    tpath = tf.name

  try:
    proc = subprocess.run([NGSPICE_EXE, "-b", tpath], capture_output=True, text=True)
    stdout = proc.stdout
  finally:
    if os.path.exists(tpath):
      try:
        os.remove(tpath)
      except OSError:
        pass

  m_i = re.search(r"i\(vdd\)\s*=\s*([+-]?[\d\.]+(?:[eE][+-]?\d+)?)", stdout, re.IGNORECASE)
  spice_i = abs(float(m_i.group(1))) if m_i else 1e-15

  nd = {}
  for m in re.finditer(r"(v\(\w+\))\s*=\s*([+-]?[\d\.]+(?:[eE][+-]?\d+)?)", stdout, re.IGNORECASE):
    nd[m.group(1).lower()] = float(m.group(2))

  return max(spice_i, 1e-15), nd

# ---------------------------------------------------------------------------
# Batched Hierarchical ML Prediction using Trained Transistor Models
# ---------------------------------------------------------------------------
def predict_fets_batch(mos_type, is_on, w, vds, vgs, vss, vbs, corners_list):
  """Batched inference through trained CanonicalMOSModel for a group of FETs."""
  key = f"{mos_type.upper()}_{'ON' if is_on else 'OFF'}"
  p = 'n_' if mos_type == 'nmos' else 'p_'
  n = len(corners_list)
  df = pd.DataFrame({
      'vdd': [c['vdd'] for c in corners_list],
      'temp': [c['temp'] for c in corners_list],
      'w': np.full(n, w),
      'vd': vds,
      'vs': vss,
      'vg': vgs,
      'vb': vbs,
      'vtho': [c[p + 'vtho'] for c in corners_list],
      'toxe': [c[p + 'toxe'] for c in corners_list],
      'ndep': [c[p + 'ndep'] for c in corners_list],
      'u0': [c[p + 'u0'] for c in corners_list],
      'xj': [c[p + 'xj'] for c in corners_list],
  })
  return models[key].predict(df)

def evaluate_gate_ml(gate_name, inp_str, inputs, corners, node_records):
  """
  Calculates logic gate leakage using trained transistor models and
  pre-recorded internal node and output voltages as specified in Part B Step 4.
  """
  n_pts = len(corners)
  vdd_arr = np.array([c["vdd"] for c in corners])
  va_arr = inputs.get("A", 0.0) * vdd_arr
  vb_arr = inputs.get("B", 0.0) * vdd_arr
  vout_arr = np.array([nd.get("v(out)", 0.0) for nd in node_records])

  if gate_name == "INV":
    if inp_str == "0":
      preds = predict_fets_batch('nmos', False, 32e-9, vds=vout_arr, vgs=va_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      return np.abs(preds["id"].values)
    else:
      preds = predict_fets_batch('pmos', False, 64e-9, vds=vout_arr, vgs=va_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      return np.abs(preds["is"].values)

  elif gate_name == "NAND2":
    vx_arr = np.array([nd.get("v(x)", 0.0) for nd in node_records])
    if inp_str == "00":
      p1 = predict_fets_batch('nmos', False, 64e-9, vds=vout_arr, vgs=va_arr, vss=vx_arr, vbs=np.zeros(n_pts), corners_list=corners)
      p2 = predict_fets_batch('nmos', False, 64e-9, vds=vx_arr, vgs=vb_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      return (np.abs(p1["id"].values) + np.abs(p2["id"].values)) / 2.0
    elif inp_str == "01":
      p1 = predict_fets_batch('nmos', False, 64e-9, vds=vout_arr, vgs=va_arr, vss=vx_arr, vbs=np.zeros(n_pts), corners_list=corners)
      return np.abs(p1["id"].values)
    elif inp_str == "10":
      p2 = predict_fets_batch('nmos', False, 64e-9, vds=vx_arr, vgs=vb_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      return np.abs(p2["id"].values)
    else:  # "11"
      p1 = predict_fets_batch('pmos', False, 64e-9, vds=vout_arr, vgs=va_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      p2 = predict_fets_batch('pmos', False, 64e-9, vds=vout_arr, vgs=vb_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      return np.abs(p1["is"].values) + np.abs(p2["is"].values)

  elif gate_name == "NOR2":
    vy_arr = np.array([nd.get("v(y)", 0.0) for nd in node_records])
    if inp_str == "00":
      p1 = predict_fets_batch('nmos', False, 32e-9, vds=vout_arr, vgs=va_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      p2 = predict_fets_batch('nmos', False, 32e-9, vds=vout_arr, vgs=vb_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      return np.abs(p1["id"].values) + np.abs(p2["id"].values)
    elif inp_str == "01":
      p2 = predict_fets_batch('pmos', False, 128e-9, vds=vout_arr, vgs=vb_arr, vss=vy_arr, vbs=vdd_arr, corners_list=corners)
      return np.abs(p2["is"].values)
    elif inp_str == "10":
      p1 = predict_fets_batch('pmos', False, 128e-9, vds=vy_arr, vgs=va_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      return np.abs(p1["is"].values)
    else:  # "11"
      p1 = predict_fets_batch('pmos', False, 128e-9, vds=vy_arr, vgs=va_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      p2 = predict_fets_batch('pmos', False, 128e-9, vds=vout_arr, vgs=vb_arr, vss=vy_arr, vbs=vdd_arr, corners_list=corners)
      return (np.abs(p1["is"].values) + np.abs(p2["is"].values)) / 2.0

  elif gate_name == "AND2":
    vnand_arr = np.array([nd.get("v(nand_out)", 0.0) for nd in node_records])
    vx_arr = np.array([nd.get("v(x)", 0.0) for nd in node_records])
    if inp_str == "00":
      pn1 = predict_fets_batch('nmos', False, 64e-9, vds=vnand_arr, vgs=va_arr, vss=vx_arr, vbs=np.zeros(n_pts), corners_list=corners)
      pn2 = predict_fets_batch('nmos', False, 64e-9, vds=vx_arr, vgs=vb_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      ml_nand = (np.abs(pn1["id"].values) + np.abs(pn2["id"].values)) / 2.0
      pi = predict_fets_batch('pmos', False, 64e-9, vds=vout_arr, vgs=vnand_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      ml_inv = np.abs(pi["is"].values)
    elif inp_str == "01":
      pn1 = predict_fets_batch('nmos', False, 64e-9, vds=vnand_arr, vgs=va_arr, vss=vx_arr, vbs=np.zeros(n_pts), corners_list=corners)
      ml_nand = np.abs(pn1["id"].values)
      pi = predict_fets_batch('pmos', False, 64e-9, vds=vout_arr, vgs=vnand_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      ml_inv = np.abs(pi["is"].values)
    elif inp_str == "10":
      pn2 = predict_fets_batch('nmos', False, 64e-9, vds=vx_arr, vgs=vb_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      ml_nand = np.abs(pn2["id"].values)
      pi = predict_fets_batch('pmos', False, 64e-9, vds=vout_arr, vgs=vnand_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      ml_inv = np.abs(pi["is"].values)
    else:  # "11"
      p1 = predict_fets_batch('pmos', False, 64e-9, vds=vnand_arr, vgs=va_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      p2 = predict_fets_batch('pmos', False, 64e-9, vds=vnand_arr, vgs=vb_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      ml_nand = np.abs(p1["is"].values) + np.abs(p2["is"].values)
      pi = predict_fets_batch('nmos', False, 32e-9, vds=vout_arr, vgs=vnand_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      ml_inv = np.abs(pi["id"].values)
    return ml_nand + ml_inv

  elif gate_name == "OR2":
    vnor_arr = np.array([nd.get("v(nor_out)", 0.0) for nd in node_records])
    vy_arr = np.array([nd.get("v(y)", 0.0) for nd in node_records])
    if inp_str == "00":
      pn1 = predict_fets_batch('nmos', False, 32e-9, vds=vnor_arr, vgs=va_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      pn2 = predict_fets_batch('nmos', False, 32e-9, vds=vnor_arr, vgs=vb_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      ml_nor = np.abs(pn1["id"].values) + np.abs(pn2["id"].values)
      pi = predict_fets_batch('pmos', False, 64e-9, vds=vout_arr, vgs=vnor_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      ml_inv = np.abs(pi["is"].values)
    elif inp_str == "01":
      p2 = predict_fets_batch('pmos', False, 128e-9, vds=vnor_arr, vgs=vb_arr, vss=vy_arr, vbs=vdd_arr, corners_list=corners)
      ml_nor = np.abs(p2["is"].values)
      pi = predict_fets_batch('nmos', False, 32e-9, vds=vout_arr, vgs=vnor_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      ml_inv = np.abs(pi["id"].values)
    elif inp_str == "10":
      p1 = predict_fets_batch('pmos', False, 128e-9, vds=vy_arr, vgs=va_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      ml_nor = np.abs(p1["is"].values)
      pi = predict_fets_batch('nmos', False, 32e-9, vds=vout_arr, vgs=vnor_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      ml_inv = np.abs(pi["id"].values)
    else:  # "11"
      p1 = predict_fets_batch('pmos', False, 128e-9, vds=vy_arr, vgs=va_arr, vss=vdd_arr, vbs=vdd_arr, corners_list=corners)
      p2 = predict_fets_batch('pmos', False, 128e-9, vds=vnor_arr, vgs=vb_arr, vss=vy_arr, vbs=vdd_arr, corners_list=corners)
      ml_nor = (np.abs(p1["is"].values) + np.abs(p2["is"].values)) / 2.0
      pi = predict_fets_batch('nmos', False, 32e-9, vds=vout_arr, vgs=vnor_arr, vss=np.zeros(n_pts), vbs=np.zeros(n_pts), corners_list=corners)
      ml_inv = np.abs(pi["id"].values)
    return ml_nor + ml_inv

  else:
    raise ValueError(f"Unknown gate: {gate_name}")

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
  print("=" * 85)
  print("PART B: HIERARCHICAL GATE-LEVEL VALIDATION ACROSS 200 CORNERS")
  print("=" * 85)

  # Generate Deliverable 2 sample netlists for each gate
  print("Generating sample SPICE netlists for all gates (Deliverable 2)...")
  generate_sample_gate_netlists()
  print("-" * 85)

  corners = generate_corners(NUM_CORNERS)
  results_records = []
  all_spice_vals = []
  all_ml_vals = []

  workers = max(1, min(os.cpu_count() or 4, 8))
  print(f"Executing parallel SPICE evaluations using {workers} threads...")

  for gate_name, input_str, inputs in GATE_CASES:
    print(f">> Simulating Gate: {gate_name} with Inputs [{input_str}] across {NUM_CORNERS} corners...")

    work_items = [(gate_name, inputs, c) for c in corners]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
      sim_results = list(executor.map(simulate_corner_spice, work_items))

    spice_arr = np.array([res[0] for res in sim_results])
    node_records = [res[1] for res in sim_results]

    ml_arr = evaluate_gate_ml(gate_name, input_str, inputs, corners, node_records)

    all_spice_vals.extend(spice_arr)
    all_ml_vals.extend(ml_arr)

    rel_err = np.abs(ml_arr - spice_arr) / spice_arr * 100.0
    med_spice = float(np.median(spice_arr))
    med_ml = float(np.median(ml_arr))
    bias = float(np.mean(ml_arr - spice_arr))
    med_err = float(np.median(rel_err))
    p95_err = float(np.percentile(rel_err, 95))

    results_records.append({
        "Gate": gate_name,
        "Inputs": input_str,
        "SPICE Median (A)": med_spice,
        "ML Median (A)": med_ml,
        "Bias (A)": bias,
        "Median Error (%)": med_err,
        "95th% Error (%)": p95_err
    })

    plt.figure(figsize=(6, 6))
    plt.scatter(spice_arr, ml_arr, alpha=0.5, s=18, color="darkorange", edgecolors="none")
    lims = [min(spice_arr.min(), ml_arr.min()), max(spice_arr.max(), ml_arr.max())]
    plt.plot(lims, lims, "r--", lw=1.5, label="Ideal")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"{gate_name} Inputs [{input_str}] Parity", fontsize=12, fontweight="bold")
    plt.xlabel("SPICE Leakage (A)", fontsize=11)
    plt.ylabel("ML Predicted Leakage (A)", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/gates/parity_{gate_name}_{input_str}.png", dpi=300)
    plt.close()

  # Combined Gate Parity Plot (Deliverable 7)
  all_spice = np.array(all_spice_vals)
  all_ml = np.array(all_ml_vals)
  plt.figure(figsize=(7, 7))
  plt.scatter(all_spice, all_ml, alpha=0.3, s=14, color="dodgerblue", edgecolors="none")
  lims = [min(all_spice.min(), all_ml.min()), max(all_spice.max(), all_ml.max())]
  plt.plot(lims, lims, "r--", lw=1.8, label="Ideal (x=y)")
  plt.xscale("log")
  plt.yscale("log")
  plt.title("Deliverable 7: Gate Parity Plot (All 18 Gates across 200 Corners)", fontsize=13, fontweight="bold")
  plt.xlabel("SPICE Leakage Current (A)", fontsize=11)
  plt.ylabel("ML Predicted Leakage Current (A)", fontsize=11)
  plt.legend(loc="upper left")
  plt.grid(True, which="both", ls="--", alpha=0.5)
  plt.tight_layout()
  plt.savefig("plots/gates/gate_parity_all.png", dpi=300)
  plt.close()

  df_results = pd.DataFrame(results_records)
  df_results.to_csv("deliverable6_gate_results.csv", index=False)

  # Approach A: Save systematic gate calibration factors for hierarchical circuit composition
  gate_calibs = {}
  for _, r in df_results.iterrows():
    g = r["Gate"]
    inp = str(r["Inputs"])
    if len(inp) == 1 and g != "INV":
      inp = "0" + inp
    gate_calibs[f"{g}_{inp}"] = float(r["SPICE Median (A)"] / r["ML Median (A)"]) if r["ML Median (A)"] > 0 else 1.0

  import json
  with open("gate_calibrations.json", "w") as f:
    json.dump(gate_calibs, f, indent=2)

  print("\n" + "=" * 85)
  print("DELIVERABLE 6: GATE RESULTS TABLE")
  print("=" * 85)
  print(df_results.to_string(index=False))
  print("\nSaved gate table to: deliverable6_gate_results.csv")
  print("Saved gate calibration factors to: gate_calibrations.json")
  print("Saved gate parity plots to: plots/gates/")
  print("Saved combined gate parity plot to: plots/gates/gate_parity_all.png")
  print("=" * 85)
  print("Part B completed successfully!")

if __name__ == "__main__":
  main()
