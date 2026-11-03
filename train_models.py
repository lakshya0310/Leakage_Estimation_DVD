import os
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ---------------------------------------------------------------------------
# Setup & Paths
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
TARGET_COLS = ["id", "ig", "is", "ib"]
I_FLOOR = 1e-15  # 1 fA floor for symlog & relative error denominator

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Feature Engineering & Target Transformations
# ---------------------------------------------------------------------------
def engineer_features(df):
  """Computes physical terminal potential differences."""
  df_feat = df.copy()
  df_feat["vds"] = df_feat["vd"] - df_feat["vs"]
  df_feat["vgs"] = df_feat["vg"] - df_feat["vs"]
  df_feat["vbs"] = df_feat["vb"] - df_feat["vs"]
  df_feat["vgd"] = df_feat["vg"] - df_feat["vd"]
  feature_cols = [
      "vdd", "temp", "w", "vd", "vs", "vg", "vb",
      "vtho", "toxe", "ndep", "u0", "xj",
      "vds", "vgs", "vbs", "vgd"
  ]
  return df_feat[feature_cols]

def symlog(y, floor=I_FLOOR):
  """Symmetric log10 transform for values spanning multiple decades."""
  y = np.asarray(y)
  return np.sign(y) * np.log10(1.0 + np.abs(y) / floor)

def inv_symlog(z, floor=I_FLOOR):
  """Inverse symmetric log10 transform returning physical currents in Amperes."""
  z = np.asarray(z)
  return np.sign(z) * floor * (np.power(10.0, np.abs(z)) - 1.0)

def compute_metrics(y_true, y_pred, floor=1e-14):
  """Calculates R2, Median Relative Error (%), and 95th Percentile Relative Error (%)."""
  r2 = r2_score(y_true, y_pred)
  # Relative error per sample (bounded away from numerical zero)
  rel_error = np.abs(y_pred - y_true) / (np.abs(y_true) + floor) * 100.0
  med_err = float(np.median(rel_error))
  p95_err = float(np.percentile(rel_error, 95))
  return r2, med_err, p95_err

# ---------------------------------------------------------------------------
# Multi-Terminal Regressor Container
# ---------------------------------------------------------------------------
class MultiTerminalModel:
  """Wrapper to train and predict individual models for Id, Ig, Is, Ib."""
  def __init__(self, model_factory):
    self.model_factory = model_factory
    self.models = {}

  def fit(self, X, Y_df):
    for col in TARGET_COLS:
      y_sym = symlog(Y_df[col].values)
      reg = self.model_factory()
      reg.fit(X, y_sym)
      self.models[col] = reg
    return self

  def predict(self, X):
    preds = {}
    for col in TARGET_COLS:
      z_pred = self.models[col].predict(X)
      preds[col] = inv_symlog(z_pred)
    return pd.DataFrame(preds)

# ---------------------------------------------------------------------------
# Model Factories
# ---------------------------------------------------------------------------
def get_model_factories():
  return {
      "Random Forest": lambda: RandomForestRegressor(
          n_estimators=100, max_depth=14, n_jobs=-1, random_state=RANDOM_SEED
      ),
      "XGBoost": lambda: XGBRegressor(
          n_estimators=200, max_depth=7, learning_rate=0.08,
          subsample=0.9, n_jobs=-1, random_state=RANDOM_SEED
      ),
      "LightGBM": lambda: LGBMRegressor(
          n_estimators=250, max_depth=8, num_leaves=31, learning_rate=0.08,
          n_jobs=-1, random_state=RANDOM_SEED, verbose=-1
      ),
      "MLP": lambda: Pipeline([
          ("scaler", StandardScaler()),
          ("mlp", MLPRegressor(
              hidden_layer_sizes=(128, 64), max_iter=250,
              learning_rate_init=0.005, random_state=RANDOM_SEED
          ))
      ])
  }

# ---------------------------------------------------------------------------
# Main Training & Evaluation Pipeline
# ---------------------------------------------------------------------------
def load_case_dataset(case):
  # Check datasets/ folder first, fallback to root
  path1 = os.path.join("datasets", f"{case}.csv")
  path2 = f"{case}.csv"
  filepath = path1 if os.path.exists(path1) else path2
  if not os.path.exists(filepath):
    raise FileNotFoundError(f"Could not locate {filepath}")
  return pd.read_csv(filepath)

def main():
  print("=" * 80)
  print("PART A: ML MODEL COMPARISON, SELECTION & EVALUATION")
  print("=" * 80)

  comparison_records = []
  best_models_per_case = {}

  for case in CASES:
    print(f"\n>>> Processing Case: {case} <<<")
    df = load_case_dataset(case)

    # 70% Train (7,000), 15% Val (1,500), 15% Test (1,500)
    df_train, df_temp = train_test_split(df, test_size=0.30, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=RANDOM_SEED)

    X_train = engineer_features(df_train)
    X_val = engineer_features(df_val)
    X_test = engineer_features(df_test)

    Y_train = df_train[TARGET_COLS]
    Y_val = df_val[TARGET_COLS]
    Y_test = df_test[TARGET_COLS]

    factories = get_model_factories()
    case_results = []

    for name, factory in factories.items():
      t0 = time.time()
      multi_model = MultiTerminalModel(factory)
      multi_model.fit(X_train, Y_train)
      elapsed = time.time() - t0

      # Predict on Validation set
      val_preds = multi_model.predict(X_val)

      # Aggregate metrics across all 4 terminal currents
      r2_list, med_list, p95_list = [], [], []
      for col in TARGET_COLS:
        r2, med, p95 = compute_metrics(Y_val[col].values, val_preds[col].values)
        r2_list.append(r2)
        med_list.append(med)
        p95_list.append(p95)

      # Overall score is the mean across terminal currents
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
          "_model_obj": multi_model,
          "_val_preds": val_preds
      }
      comparison_records.append(record)
      case_results.append(record)
      print(f"  [{name:14s}] Val R2: {avg_r2:.5f} | Median Err: {avg_med:6.2f}% | 95th% Err: {avg_p95:6.2f}% | Time: {elapsed:4.1f}s")

    # Select best model: highest R2 (or lowest median error)
    best_candidate = sorted(case_results, key=lambda x: (-x["Val R2"], x["Val Median Error (%)"]))[0]
    best_name = best_candidate["Model"]
    best_model = best_candidate["_model_obj"]
    print(f"  >> Selected Best Model for {case}: {best_name}")

    # Evaluate Best Model on Test Set (Deliverable 5)
    test_preds = best_model.predict(X_test)
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
        "X_test": X_test,
        "Y_test": Y_test,
        "test_preds": test_preds
    }

    # Save best model to disk
    model_save_path = f"models/{case}_model.joblib"
    joblib.dump(best_model, model_save_path)
    print(f"  >> Saved: {model_save_path}")

  # -------------------------------------------------------------------------
  # Deliverable 4: Model Comparison Table
  # -------------------------------------------------------------------------
  comp_df = pd.DataFrame(comparison_records).drop(columns=["_model_obj", "_val_preds"])
  comp_df.to_csv("deliverable4_model_comparison.csv", index=False)
  print("\n" + "=" * 80)
  print("DELIVERABLE 4: MODEL COMPARISON TABLE (VALIDATION SET)")
  print("=" * 80)
  print(comp_df.to_string(index=False))
  print("\nSaved comparison table to: deliverable4_model_comparison.csv")

  # -------------------------------------------------------------------------
  # Deliverable 5: Test Set Performance & Parity Plots
  # -------------------------------------------------------------------------
  print("\n" + "=" * 80)
  print("DELIVERABLE 5: CHOSEN MODELS TEST SET PERFORMANCE")
  print("=" * 80)
  for case, info in best_models_per_case.items():
    print(f"[{case:8s}] Model: {info['model_name']:12s} | Test R2: {info['test_r2']:.5f} | Test Median Err: {info['test_med']:5.2f}% | Test 95th% Err: {info['test_p95']:5.2f}%")

    # Generate Parity Plot (Predicted vs SPICE) for all 4 terminals on log scale
    Y_true = info["Y_test"]
    Y_pred = info["test_preds"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, col in enumerate(TARGET_COLS):
      ax = axes[idx]
      y_t = np.abs(Y_true[col].values)
      y_p = np.abs(Y_pred[col].values)

      # Filter out exact zeros for plotting
      valid = (y_t > 1e-17) & (y_p > 1e-17)
      ax.scatter(y_t[valid], y_p[valid], alpha=0.35, s=10, color="teal", edgecolors="none")

      # Reference ideal line y = x
      min_val = min(y_t[valid].min(), y_p[valid].min())
      max_val = max(y_t[valid].max(), y_p[valid].max())
      ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Ideal (SPICE = ML)")

      ax.set_xscale("log")
      ax.set_yscale("log")
      ax.set_title(f"Terminal Current |{col.upper()}|", fontsize=11, fontweight="bold")
      ax.set_xlabel("SPICE Reference |I| (A)", fontsize=10)
      ax.set_ylabel("ML Predicted |I| (A)", fontsize=10)
      ax.legend(loc="upper left", frameon=True)

    plt.suptitle(f"{case} Parity Plots (Test Set - {info['model_name']})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    parity_plot_path = f"plots/parity_{case}.png"
    plt.savefig(parity_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {parity_plot_path}")

  print("\n" + "=" * 80)
  print("Part A finished! You are ready for Part B (Gate-Level Validation).")
  print("=" * 80)

if __name__ == "__main__":
  main()