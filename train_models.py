import os
import time
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
TARGET_COLS = ["id", "ig", "is", "ib"]
LOG_FLOOR = 1e-18  # Matches SPICE ABSTOL = 1e-18

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical Physics Feature Engineering
# ---------------------------------------------------------------------------
def build_canonical_features(df):
  """
  Exploits MOSFET Drain-Source symmetry by defining:
    vd_eff = max(vd, vs), vs_eff = min(vd, vs)
  and adding physical BSIM4 terms (thermal voltage kT/q, 1/toxe, etc.).
  """
  vd = df["vd"].values
  vs = df["vs"].values
  vg = df["vg"].values
  vb = df["vb"].values

  # Canonical (unidirectional) drain and source potentials
  vd_eff = np.maximum(vd, vs)
  vs_eff = np.minimum(vd, vs)

  vds_eff = vd_eff - vs_eff
  vgs_eff = vg - vs_eff
  vbs_eff = vb - vs_eff
  vgd_eff = vg - vd_eff
  vbd_eff = vb - vd_eff

  # Physical temperature in Kelvin & inverse thermal voltage (q / kT)
  temp_k = df["temp"].values + 273.15
  inv_vt = 11604.5 / temp_k  # q / (k * T) in V^-1

  w = df["w"].values
  vtho = df["vtho"].values
  toxe = df["toxe"].values
  ndep = df["ndep"].values
  u0 = df["u0"].values
  xj = df["xj"].values

  # Effective overdrive & subthreshold coupling terms
  vov_s = vgs_eff - np.abs(vtho)
  vov_d = vgd_eff - np.abs(vtho)

  feat_dict = {
      "vdd": df["vdd"].values,
      "temp_k": temp_k,
      "inv_vt": inv_vt,
      "log_w": np.log10(w),
      "vd_eff": vd_eff,
      "vs_eff": vs_eff,
      "vg": vg,
      "vb": vb,
      "vds_eff": vds_eff,
      "vgs_eff": vgs_eff,
      "vbs_eff": vbs_eff,
      "vgd_eff": vgd_eff,
      "vbd_eff": vbd_eff,
      "vtho": vtho,
      "inv_toxe": 1.0 / toxe,
      "log_ndep": np.log10(ndep),
      "log_u0": np.log10(u0),
      "xj": xj,
      "vov_s_vt": vov_s * inv_vt,
      "vov_d_vt": vov_d * inv_vt,
      "vds_vt": vds_eff * inv_vt,
      "vbs_vt": vbs_eff * inv_vt,
      "sqrt_vds": np.sqrt(vds_eff + 1e-9),
  }
  return pd.DataFrame(feat_dict)

def to_canonical_targets(raw_df, Y_raw_df):
  """
  Swaps Id and Is when Vd < Vs so that the canonical drain is always at
  max(Vd, Vs).
  """
  swap_mask = (raw_df["vd"].values < raw_df["vs"].values)
  id_orig = Y_raw_df["id"].values.copy()
  is_orig = Y_raw_df["is"].values.copy()

  id_can = np.where(swap_mask, is_orig, id_orig)
  is_can = np.where(swap_mask, id_orig, is_orig)
  ig_can = Y_raw_df["ig"].values.copy()
  ib_can = Y_raw_df["ib"].values.copy()

  return pd.DataFrame({
      "id": id_can,
      "ig": ig_can,
      "is": is_can,
      "ib": ib_can
  })

# ---------------------------------------------------------------------------
# Metrics Calculation
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
  """
  Computes R2 (on log10 magnitude) and Relative Error (%)
  for active leakage/conduction currents above the SPICE numerical noise floor.
  """
  mag_true = np.abs(y_true)
  mag_pred = np.abs(y_pred)

  # R2 on log10 magnitude across 10+ decades of leakage
  r2 = r2_score(np.log10(mag_true + 1e-16), np.log10(mag_pred + 1e-16))

  # Relative error (%) on non-zero currents (> 1e-14 A)
  active = mag_true > 1e-14
  if np.sum(active) > 10:
    rel_err = np.abs(mag_pred[active] - mag_true[active]) / mag_true[active] * 100.0
  else:
    rel_err = np.abs(mag_pred - mag_true) / (mag_true + 1e-14) * 100.0

  med_err = float(np.median(rel_err))
  p95_err = float(np.percentile(rel_err, 95))
  return r2, med_err, p95_err

# ---------------------------------------------------------------------------
# Canonical Multi-Terminal Regressor (Pickle-Safe)
# ---------------------------------------------------------------------------
class CanonicalMOSModel:
  """
  Predicts log10(|I|) in canonical (Vd >= Vs) coordinates and maps signs
  and Drain-Source swaps back to the original circuit coordinates.
  """
  def __init__(self, base_estimator):
    self.base_estimator = base_estimator
    self.mag_models = {}
    self.sign_models = {}
    self.constant_signs = {}

  def fit(self, raw_df, Y_raw_df):
    X_can = build_canonical_features(raw_df)
    Y_can = to_canonical_targets(raw_df, Y_raw_df)

    for col in ["id", "ig", "is", "ib"]:
      y_vals = Y_can[col].values
      y_log_mag = np.log10(np.abs(y_vals) + LOG_FLOOR)

      reg = clone(self.base_estimator)
      reg.fit(X_can, y_log_mag)
      self.mag_models[col] = reg

      # Determine sign behavior
      signs = np.where(y_vals >= 0, 1.0, -1.0)
      pos_ratio = np.mean(signs > 0)
      if pos_ratio > 0.995:
        self.constant_signs[col] = 1.0
      elif pos_ratio < 0.005:
        self.constant_signs[col] = -1.0
      else:
        # Train a sign regressor for mixed-polarity terminals
        sign_reg = clone(self.base_estimator)
        sign_reg.fit(X_can, np.sign(y_vals) * np.log10(1.0 + np.abs(y_vals) / 1e-15))
        self.sign_models[col] = sign_reg

    return self

  def predict(self, raw_df):
    X_can = build_canonical_features(raw_df)
    swap_mask = (raw_df["vd"].values < raw_df["vs"].values)

    can_preds = {}
    for col in ["id", "ig", "is", "ib"]:
      log_mag = self.mag_models[col].predict(X_can)
      mag = np.maximum(0.0, np.power(10.0, log_mag) - LOG_FLOOR)

      if col in self.constant_signs:
        sgn = self.constant_signs[col]
      else:
        sgn_pred = self.sign_models[col].predict(X_can)
        sgn = np.where(sgn_pred >= 0, 1.0, -1.0)

      can_preds[col] = sgn * mag

    # Swap Id and Is back where Vd < Vs
    id_final = np.where(swap_mask, can_preds["is"], can_preds["id"])
    is_final = np.where(swap_mask, can_preds["id"], can_preds["is"])

    return pd.DataFrame({
        "id": id_final,
        "ig": can_preds["ig"],
        "is": is_final,
        "ib": can_preds["ib"]
    })

# ---------------------------------------------------------------------------
# 4 Model Architectures
# ---------------------------------------------------------------------------
def get_base_models():
  return {
      "Random Forest": RandomForestRegressor(
          n_estimators=200, max_depth=22, min_samples_leaf=1,
          n_jobs=-1, random_state=RANDOM_SEED
      ),
      "XGBoost": XGBRegressor(
          n_estimators=600, max_depth=7, learning_rate=0.04,
          subsample=0.9, colsample_bytree=0.9,
          n_jobs=-1, random_state=RANDOM_SEED
      ),
      "LightGBM": LGBMRegressor(
          n_estimators=700, max_depth=8, num_leaves=63, learning_rate=0.04,
          subsample_for_bin=50000, min_child_samples=10,
          n_jobs=-1, random_state=RANDOM_SEED, verbose=-1
      ),
      "MLP": Pipeline([
          ("scaler", StandardScaler()),
          ("mlp", MLPRegressor(
              hidden_layer_sizes=(128, 64, 32),
              activation="tanh",
              solver="lbfgs",
              alpha=1e-4,
              max_iter=600,
              random_state=RANDOM_SEED
          ))
      ])
  }

# ---------------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------------
def load_case_dataset(case):
  path1 = os.path.join("datasets", f"{case}.csv")
  path2 = f"{case}.csv"
  filepath = path1 if os.path.exists(path1) else path2
  if not os.path.exists(filepath):
    raise FileNotFoundError(f"Could not locate {filepath}")
  return pd.read_csv(filepath)

# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
  print("=" * 85)
  print("PART A: HIGH-PRECISION ML MODEL TRAINING & COMPARISON (TARGET < 1% ERROR)")
  print("=" * 85)

  comparison_records = []
  best_models_per_case = {}

  for case in CASES:
    print(f"\n>>> Processing Case: {case} <<<")
    df = load_case_dataset(case)

    # 70% Train (7,000), 15% Val (1,500), 15% Test (1,500)
    df_train, df_temp = train_test_split(df, test_size=0.30, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=RANDOM_SEED)

    Y_train = df_train[TARGET_COLS]
    Y_val = df_val[TARGET_COLS]
    Y_test = df_test[TARGET_COLS]

    models_dict = get_base_models()
    case_results = []

    for name, base_est in models_dict.items():
      t0 = time.time()
      mos_model = CanonicalMOSModel(base_est)
      mos_model.fit(df_train, Y_train)
      elapsed = time.time() - t0

      val_preds = mos_model.predict(df_val)

      r2_list, med_list, p95_list = [], [], []
      for col in TARGET_COLS:
        r2, med, p95 = compute_metrics(Y_val[col].values, val_preds[col].values)
        r2_list.append(r2)
        med_list.append(med)
        p95_list.append(p95)

      avg_r2 = float(np.mean(r2_list))
      avg_med = float(np.mean(med_list))
      avg_p95 = float(np.mean(p95_list))

      record = {
          "Case": case,
          "Model": name,
          "Val R2": avg_r2,
          "Val Median Error (%)": avg_med,
          "Val 95th% Error (%)": avg_p95,
          "Train Time (s)": round(elapsed, 2),
          "_model_obj": mos_model,
      }
      comparison_records.append(record)
      case_results.append(record)
      print(
          f"  [{name:14s}] Val R2: {avg_r2:.5f} | Median Err: {avg_med:6.3f}% |"
          f" 95th% Err: {avg_p95:6.2f}% | Time: {elapsed:5.1f}s"
      )

    # Select best model by lowest Validation Median Error
    best_candidate = sorted(case_results, key=lambda x: x["Val Median Error (%)"])[0]
    best_name = best_candidate["Model"]
    best_model = best_candidate["_model_obj"]
    print(f"  >> Selected Best Model for {case}: {best_name} (Val Median Err: {best_candidate['Val Median Error (%)']:.3f}%)")

    # Evaluate on unseen Test Set (Deliverable 5)
    test_preds = best_model.predict(df_test)
    test_r2_list, test_med_list, test_p95_list = [], [], []
    for col in TARGET_COLS:
      r2, med, p95 = compute_metrics(Y_test[col].values, test_preds[col].values)
      test_r2_list.append(r2)
      test_med_list.append(med)
      test_p95_list.append(p95)

    best_models_per_case[case] = {
        "model_name": best_name,
        "model_obj": best_model,
        "test_r2": float(np.mean(test_r2_list)),
        "test_med": float(np.mean(test_med_list)),
        "test_p95": float(np.mean(test_p95_list)),
        "Y_test": Y_test,
        "test_preds": test_preds
    }

    model_save_path = f"models/{case}_model.joblib"
    joblib.dump(best_model, model_save_path)
    print(f"  >> Saved: {model_save_path}")

  # -------------------------------------------------------------------------
  # Deliverable 4: Model Comparison Table
  # -------------------------------------------------------------------------
  comp_df = pd.DataFrame(comparison_records).drop(columns=["_model_obj"])
  comp_df.to_csv("deliverable4_model_comparison.csv", index=False)
  print("\n" + "=" * 85)
  print("DELIVERABLE 4: MODEL COMPARISON TABLE (VALIDATION SET)")
  print("=" * 85)
  print(comp_df.to_string(index=False))
  print("\nSaved comparison table to: deliverable4_model_comparison.csv")

  # -------------------------------------------------------------------------
  # Deliverable 5: Chosen Models Test Set Performance & Parity Plots
  # -------------------------------------------------------------------------
  print("\n" + "=" * 85)
  print("DELIVERABLE 5: CHOSEN MODELS TEST SET PERFORMANCE")
  print("=" * 85)
  test_summary = []
  for case, info in best_models_per_case.items():
    print(
        f"[{case:8s}] Model: {info['model_name']:12s} | Test R2: {info['test_r2']:.5f} |"
        f" Test Median Err: {info['test_med']:5.3f}% | Test 95th% Err: {info['test_p95']:5.2f}%"
    )
    test_summary.append({
        "Case": case,
        "Selected Model": info["model_name"],
        "Test R2": info["test_r2"],
        "Test Median Error (%)": info["test_med"],
        "Test 95th% Error (%)": info["test_p95"]
    })

    # Parity Plots on Log Axes
    Y_true = info["Y_test"]
    Y_pred = info["test_preds"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, col in enumerate(TARGET_COLS):
      ax = axes[idx]
      y_t = np.abs(Y_true[col].values)
      y_p = np.abs(Y_pred[col].values)

      valid = (y_t > 1e-16) & (y_p > 1e-16)
      ax.scatter(y_t[valid], y_p[valid], alpha=0.35, s=10, color="royalblue", edgecolors="none")

      min_val = min(y_t[valid].min(), y_p[valid].min())
      max_val = max(y_t[valid].max(), y_p[valid].max())
      ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Ideal (SPICE = ML)")

      ax.set_xscale("log")
      ax.set_yscale("log")
      ax.set_title(f"Terminal Current |{col.upper()}|", fontsize=11, fontweight="bold")
      ax.set_xlabel("SPICE Reference |I| (A)", fontsize=10)
      ax.set_ylabel("ML Predicted |I| (A)", fontsize=10)
      ax.legend(loc="upper left", frameon=True)

    plt.suptitle(
        f"{case} Parity Plots (Test Set - {info['model_name']}, Median Err: {info['test_med']:.2f}%)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    parity_plot_path = f"plots/parity_{case}.png"
    plt.savefig(parity_plot_path, dpi=300)
    plt.close()

  pd.DataFrame(test_summary).to_csv("deliverable5_test_summary.csv", index=False)
  print("\nSaved test summary to: deliverable5_test_summary.csv")
  print("=" * 85)

if __name__ == "__main__":
  main()