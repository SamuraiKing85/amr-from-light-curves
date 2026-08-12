"""
Traditional ML Baselines
============================
Non-deep-learning baselines for direct comparison with the neural network.
Uses the same train/test splits and evaluation metrics.

Models:
  1. B* Analytical    - Linear regression from B* to log10(A/m)
  2. RF Orbital       - Random Forest on TLE orbital features
  3. GBT Orbital      - Gradient Boosting on TLE orbital features
  4. RF LC Features   - Random Forest on hand-crafted light curve statistics

Usage:
    python baselines.py                               # Interactive menu
    python baselines.py --all                         # Run all baselines
    python baselines.py --data-dir ../data/training   # Custom path
"""

__version__ = "1.0.0"

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import TRAINING_DIR, ensure_dirs
from common.menu import print_header, prompt_choice, print_section, print_summary_table


#  DATA LOADING


TYPE_ENCODING = {"Payload": 0, "Rocket Body": 1, "Debris": 2}
CLASS_NAMES = {0: "Payload", 1: "RocketBody", 2: "Debris"}
BSTAR_COLUMNS = {"bstar_log_scaled", "bstar_scaled", "bstar_trend_scaled"}


def load_metadata(data_dir):
    """Load pipeline metadata."""
    for name in ["normalisation_params.json", "metadata.json"]:
        p = Path(data_dir) / name
        if p.exists():
            with open(p) as f:
                meta = json.load(f)
            if "scaled_feature_columns" in meta:
                return [c + "_scaled" for c in meta["scaled_feature_columns"]]
            else:
                return meta.get("tle_feature_cols", [])
    return []


def load_split(data_dir, split, orbital_cols):
    """Load a data split and extract features/targets."""
    df = pd.read_parquet(Path(data_dir) / f"{split}.parquet")

    # Light curves and masks
    lc_mags = np.stack(df["lightcurve"].values).astype(np.float32)
    lc_masks = np.stack(df["mask"].values).astype(np.float32)

    # Orbital features
    avail = [c for c in orbital_cols if c in df.columns]
    if avail:
        orbital = np.nan_to_num(df[avail].fillna(0).values.astype(np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0)
    else:
        orbital = np.zeros((len(df), 1), dtype=np.float32)

    # Regression target
    if "am_ratio_log" in df.columns:
        am_targets = df["am_ratio_log"].fillna(0).values.astype(np.float32)
        am_valid = df["am_ratio_log"].notna().values
    else:
        am_targets = np.zeros(len(df), dtype=np.float32)
        am_valid = np.zeros(len(df), dtype=bool)

    # Classification target
    if "object_type" in df.columns:
        class_targets = df["object_type"].map(TYPE_ENCODING).fillna(-1).values.astype(np.int64)
        class_valid = (df["object_type"].notna() & df["object_type"].isin(TYPE_ENCODING)).values
    else:
        class_targets = np.full(len(df), -1, dtype=np.int64)
        class_valid = np.zeros(len(df), dtype=bool)

    # B* columns for analytical baseline
    bstar_col = None
    for candidate in ["bstar_log_scaled", "bstar_log", "bstar_scaled"]:
        if candidate in df.columns:
            bstar_col = candidate
            break

    bstar_vals = None
    if bstar_col:
        bstar_vals = pd.to_numeric(df[bstar_col], errors="coerce").fillna(0).values.astype(np.float32)

    return {
        "lc_mags": lc_mags, "lc_masks": lc_masks,
        "orbital": orbital, "orbital_cols": avail,
        "am_targets": am_targets, "am_valid": am_valid,
        "class_targets": class_targets, "class_valid": class_valid,
        "bstar": bstar_vals, "n": len(df),
    }



#  LIGHT CURVE FEATURE EXTRACTION


def extract_lc_features(lc_mags, lc_masks):
    """Extract hand-crafted statistical features from light curves.

    Features per track:
      - mean, std, min, max, amplitude (max-min)
      - skewness, kurtosis
      - median absolute deviation
      - fraction of real (non-masked) points
      - number of peaks (local maxima)
      - dominant FFT frequency and its power
      - second dominant FFT frequency
      - slope (linear trend)
    """
    n_samples = len(lc_mags)
    n_features = 14
    features = np.zeros((n_samples, n_features), dtype=np.float32)
    feature_names = [
        "lc_mean", "lc_std", "lc_min", "lc_max", "lc_amplitude",
        "lc_skewness", "lc_kurtosis", "lc_mad",
        "lc_real_fraction", "lc_n_peaks",
        "lc_fft_freq1", "lc_fft_power1", "lc_fft_freq2",
        "lc_slope",
    ]

    for i in range(n_samples):
        mag = lc_mags[i]
        mask = lc_masks[i]
        real_idx = mask > 0.5
        n_real = real_idx.sum()

        if n_real < 3:
            features[i, 8] = 0.0  # real_fraction
            continue

        real_vals = mag[real_idx]

        # Basic statistics
        features[i, 0] = np.mean(real_vals)           # mean
        features[i, 1] = np.std(real_vals)             # std
        features[i, 2] = np.min(real_vals)             # min
        features[i, 3] = np.max(real_vals)             # max
        features[i, 4] = np.ptp(real_vals)             # amplitude
        features[i, 5] = float(sp_stats.skew(real_vals))   # skewness
        features[i, 6] = float(sp_stats.kurtosis(real_vals))  # kurtosis
        features[i, 7] = np.median(np.abs(real_vals - np.median(real_vals)))  # MAD
        features[i, 8] = n_real / len(mag)             # real fraction

        # Number of peaks (local maxima in real data)
        if n_real > 2:
            diffs = np.diff(real_vals)
            sign_changes = np.diff(np.sign(diffs))
            features[i, 9] = (sign_changes < 0).sum()

        # FFT features (on full sequence including padding)
        fft_vals = np.abs(np.fft.rfft(mag))
        fft_vals[0] = 0  # Remove DC component
        if len(fft_vals) > 2:
            freqs = np.fft.rfftfreq(len(mag))
            sorted_idx = np.argsort(fft_vals)[::-1]
            features[i, 10] = freqs[sorted_idx[0]]    # dominant frequency
            features[i, 11] = fft_vals[sorted_idx[0]]  # dominant power
            if len(sorted_idx) > 1:
                features[i, 12] = freqs[sorted_idx[1]]  # second frequency

        # Linear slope
        if n_real > 1:
            x = np.where(real_idx)[0].astype(np.float32)
            slope, _, _, _, _ = sp_stats.linregress(x, real_vals)
            features[i, 13] = slope

    return features, feature_names



#  METRICS (same as evaluate.py)


def compute_regression_metrics(pred, target, valid):
    mask = valid & np.isfinite(pred) & np.isfinite(target)
    if mask.sum() == 0:
        return {"n_samples": 0, "rmse_log": np.nan, "mae_log": np.nan, "r2": np.nan}
    p, t = pred[mask], target[mask]
    res = p - t
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "n_samples": int(mask.sum()),
        "rmse_log": round(np.sqrt(np.mean(res**2)), 4),
        "mae_log": round(np.mean(np.abs(res)), 4),
        "r2": round(r2, 4),
        "residual_mean": round(float(np.mean(res)), 4),
        "residual_std": round(float(np.std(res)), 4),
    }


def compute_classification_metrics(pred, target, valid, n_classes=3):
    mask = valid & (target >= 0) & (target < n_classes)
    if mask.sum() == 0:
        return {"n_samples": 0, "accuracy": np.nan, "macro_f1": np.nan}
    p, t = pred[mask], target[mask]
    accuracy = float((p == t).mean())
    per_class = {}
    f1s = []
    for c in range(n_classes):
        tp = ((p == c) & (t == c)).sum()
        fp = ((p == c) & (t != c)).sum()
        fn = ((p != c) & (t == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        name = CLASS_NAMES.get(c, f"cls{c}")
        per_class[name] = {"precision": round(prec, 4), "recall": round(rec, 4),
                           "f1": round(f1, 4), "support": int((t == c).sum())}
        if (t == c).sum() > 0:
            f1s.append(f1)
    return {
        "n_samples": int(mask.sum()),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "per_class": per_class,
    }



#  BASELINE 1: B* ANALYTICAL


def run_bstar_baseline(train_data, test_data, output_dir):
    """Linear regression from B* to log10(A/m).

    This is the 'why bother with ML?' baseline. B* is a drag coefficient
    derived from orbital fitting that has a known physical relationship
    to area-to-mass ratio.
    """
    print_section("BASELINE: B* Analytical (Linear Regression)")

    if train_data["bstar"] is None:
        print("  [SKIP] No B* column found in data\n")
        return None

    from sklearn.linear_model import LinearRegression

    # Train
    tr_mask = train_data["am_valid"]
    X_train = train_data["bstar"][tr_mask].reshape(-1, 1)
    y_train = train_data["am_targets"][tr_mask]

    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"  Fitted: A/m = {model.coef_[0]:.4f} * B* + {model.intercept_:.4f}")

    # Test
    te_mask = test_data["am_valid"]
    X_test = test_data["bstar"][te_mask].reshape(-1, 1)
    y_test = test_data["am_targets"][te_mask]

    pred_all = model.predict(test_data["bstar"].reshape(-1, 1))
    reg = compute_regression_metrics(pred_all, test_data["am_targets"], test_data["am_valid"])

    print(f"  Test R²:   {reg['r2']:.4f}")
    print(f"  Test RMSE: {reg['rmse_log']:.4f}")
    print(f"  Test MAE:  {reg['mae_log']:.4f}")
    print(f"  Samples:   {reg['n_samples']:,}\n")

    result = {"model": "bstar_analytical", "type": "Linear Regression on B*",
              "regression": reg, "classification": None}

    save_result(result, output_dir / "bstar_analytical.json")
    return result



#  BASELINE 2: RANDOM FOREST ON ORBITAL FEATURES


def run_rf_orbital(train_data, test_data, output_dir):
    """Random Forest on TLE orbital features (regression + classification)."""
    print_section("BASELINE: Random Forest (Orbital Features)")

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    X_train = train_data["orbital"]
    X_test = test_data["orbital"]
    print(f"  Features: {X_train.shape[1]} orbital columns")
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    # Regression
    print("\n  Training regressor (n_estimators=200)...")
    t0 = time.time()
    tr_mask = train_data["am_valid"]
    reg_model = RandomForestRegressor(n_estimators=200, max_depth=20,
                                       min_samples_leaf=10, n_jobs=-1,
                                       random_state=42)
    reg_model.fit(X_train[tr_mask], train_data["am_targets"][tr_mask])
    print(f"  Trained in {time.time()-t0:.1f}s")

    reg_pred = reg_model.predict(X_test)
    reg = compute_regression_metrics(reg_pred, test_data["am_targets"], test_data["am_valid"])
    print(f"  Regression - R²: {reg['r2']:.4f}  RMSE: {reg['rmse_log']:.4f}  MAE: {reg['mae_log']:.4f}")

    # Classification
    print("\n  Training classifier (n_estimators=200)...")
    t0 = time.time()
    tr_cls_mask = train_data["class_valid"]
    cls_model = RandomForestClassifier(n_estimators=200, max_depth=20,
                                        min_samples_leaf=10, n_jobs=-1,
                                        random_state=42, class_weight="balanced")
    cls_model.fit(X_train[tr_cls_mask], train_data["class_targets"][tr_cls_mask])
    print(f"  Trained in {time.time()-t0:.1f}s")

    cls_pred = cls_model.predict(X_test)
    cls = compute_classification_metrics(cls_pred, test_data["class_targets"],
                                          test_data["class_valid"])
    print(f"  Classification - Acc: {cls['accuracy']:.4f}  F1: {cls['macro_f1']:.4f}")

    # Feature importance
    feat_imp = dict(zip(train_data["orbital_cols"],
                        reg_model.feature_importances_.tolist()))
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    print(f"\n  Top 5 features:")
    for name, imp in list(feat_imp.items())[:5]:
        print(f"    {name:<30s} {imp:.4f}")
    print()

    result = {"model": "rf_orbital", "type": "Random Forest on Orbital Features",
              "regression": reg, "classification": cls,
              "feature_importance": feat_imp}

    save_result(result, output_dir / "rf_orbital.json")
    return result



#  BASELINE 3: GRADIENT BOOSTING ON ORBITAL FEATURES


def run_gbt_orbital(train_data, test_data, output_dir):
    """Gradient Boosting on TLE orbital features."""
    print_section("BASELINE: Gradient Boosting (Orbital Features)")

    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

    X_train = train_data["orbital"]
    X_test = test_data["orbital"]
    print(f"  Features: {X_train.shape[1]} orbital columns")

    # Regression
    print("\n  Training regressor (n_estimators=300)...")
    t0 = time.time()
    tr_mask = train_data["am_valid"]
    reg_model = GradientBoostingRegressor(n_estimators=300, max_depth=6,
                                           learning_rate=0.1, subsample=0.8,
                                           random_state=42)
    reg_model.fit(X_train[tr_mask], train_data["am_targets"][tr_mask])
    print(f"  Trained in {time.time()-t0:.1f}s")

    reg_pred = reg_model.predict(X_test)
    reg = compute_regression_metrics(reg_pred, test_data["am_targets"], test_data["am_valid"])
    print(f"  Regression - R²: {reg['r2']:.4f}  RMSE: {reg['rmse_log']:.4f}  MAE: {reg['mae_log']:.4f}")

    # Classification
    print("\n  Training classifier (n_estimators=300)...")
    t0 = time.time()
    tr_cls_mask = train_data["class_valid"]
    cls_model = GradientBoostingClassifier(n_estimators=300, max_depth=6,
                                            learning_rate=0.1, subsample=0.8,
                                            random_state=42)
    cls_model.fit(X_train[tr_cls_mask], train_data["class_targets"][tr_cls_mask])
    print(f"  Trained in {time.time()-t0:.1f}s")

    cls_pred = cls_model.predict(X_test)
    cls = compute_classification_metrics(cls_pred, test_data["class_targets"],
                                          test_data["class_valid"])
    print(f"  Classification - Acc: {cls['accuracy']:.4f}  F1: {cls['macro_f1']:.4f}")

    # Feature importance
    feat_imp = dict(zip(train_data["orbital_cols"],
                        reg_model.feature_importances_.tolist()))
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    print(f"\n  Top 5 features:")
    for name, imp in list(feat_imp.items())[:5]:
        print(f"    {name:<30s} {imp:.4f}")
    print()

    result = {"model": "gbt_orbital", "type": "Gradient Boosting on Orbital Features",
              "regression": reg, "classification": cls,
              "feature_importance": feat_imp}

    save_result(result, output_dir / "gbt_orbital.json")
    return result



#  BASELINE 4: RANDOM FOREST ON LC FEATURES


def run_rf_lc_features(train_data, test_data, output_dir):
    """Random Forest on hand-crafted light curve statistical features.

    Tests whether the CNN learns anything beyond simple summary statistics.
    """
    print_section("BASELINE: Random Forest (LC Statistical Features)")

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # Extract features
    print("  Extracting light curve features (train)...")
    t0 = time.time()
    X_train, feat_names = extract_lc_features(train_data["lc_mags"], train_data["lc_masks"])
    print(f"  Extracted {len(feat_names)} features in {time.time()-t0:.1f}s")

    print("  Extracting light curve features (test)...")
    X_test, _ = extract_lc_features(test_data["lc_mags"], test_data["lc_masks"])

    print(f"  Features: {', '.join(feat_names)}")

    # Handle NaN/inf from feature extraction
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Regression
    print("\n  Training regressor (n_estimators=200)...")
    t0 = time.time()
    tr_mask = train_data["am_valid"]
    reg_model = RandomForestRegressor(n_estimators=200, max_depth=20,
                                       min_samples_leaf=10, n_jobs=-1,
                                       random_state=42)
    reg_model.fit(X_train[tr_mask], train_data["am_targets"][tr_mask])
    print(f"  Trained in {time.time()-t0:.1f}s")

    reg_pred = reg_model.predict(X_test)
    reg = compute_regression_metrics(reg_pred, test_data["am_targets"], test_data["am_valid"])
    print(f"  Regression - R²: {reg['r2']:.4f}  RMSE: {reg['rmse_log']:.4f}  MAE: {reg['mae_log']:.4f}")

    # Classification
    print("\n  Training classifier (n_estimators=200)...")
    t0 = time.time()
    tr_cls_mask = train_data["class_valid"]
    cls_model = RandomForestClassifier(n_estimators=200, max_depth=20,
                                        min_samples_leaf=10, n_jobs=-1,
                                        random_state=42, class_weight="balanced")
    cls_model.fit(X_train[tr_cls_mask], train_data["class_targets"][tr_cls_mask])
    print(f"  Trained in {time.time()-t0:.1f}s")

    cls_pred = cls_model.predict(X_test)
    cls = compute_classification_metrics(cls_pred, test_data["class_targets"],
                                          test_data["class_valid"])
    print(f"  Classification - Acc: {cls['accuracy']:.4f}  F1: {cls['macro_f1']:.4f}")

    # Feature importance
    feat_imp = dict(zip(feat_names, reg_model.feature_importances_.tolist()))
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    print(f"\n  Feature importances:")
    for name, imp in feat_imp.items():
        print(f"    {name:<20s} {imp:.4f}")
    print()

    result = {"model": "rf_lc_features", "type": "Random Forest on LC Statistics",
              "regression": reg, "classification": cls,
              "feature_importance": feat_imp,
              "feature_names": feat_names}

    save_result(result, output_dir / "rf_lc_features.json")
    return result



#  BASELINE 5: RF COMBINED (ORBITAL + LC FEATURES)


def run_rf_combined(train_data, test_data, output_dir):
    """Random Forest on orbital features + hand-crafted LC features combined.

    The non-deep-learning equivalent of the fusion model.
    """
    print_section("BASELINE: Random Forest (Orbital + LC Features Combined)")

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # Extract LC features
    print("  Extracting light curve features...")
    t0 = time.time()
    lc_train, lc_names = extract_lc_features(train_data["lc_mags"], train_data["lc_masks"])
    lc_test, _ = extract_lc_features(test_data["lc_mags"], test_data["lc_masks"])
    lc_train = np.nan_to_num(lc_train, nan=0.0, posinf=0.0, neginf=0.0)
    lc_test = np.nan_to_num(lc_test, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  LC features: {len(lc_names)} in {time.time()-t0:.1f}s")

    # Combine
    X_train = np.hstack([train_data["orbital"], lc_train])
    X_test = np.hstack([test_data["orbital"], lc_test])
    all_names = train_data["orbital_cols"] + lc_names
    print(f"  Combined: {X_train.shape[1]} features ({len(train_data['orbital_cols'])} orbital + {len(lc_names)} LC)")

    # Regression
    print("\n  Training regressor (n_estimators=200)...")
    t0 = time.time()
    tr_mask = train_data["am_valid"]
    reg_model = RandomForestRegressor(n_estimators=200, max_depth=20,
                                       min_samples_leaf=10, n_jobs=-1,
                                       random_state=42)
    reg_model.fit(X_train[tr_mask], train_data["am_targets"][tr_mask])
    print(f"  Trained in {time.time()-t0:.1f}s")

    reg_pred = reg_model.predict(X_test)
    reg = compute_regression_metrics(reg_pred, test_data["am_targets"], test_data["am_valid"])
    print(f"  Regression - R²: {reg['r2']:.4f}  RMSE: {reg['rmse_log']:.4f}  MAE: {reg['mae_log']:.4f}")

    # Classification
    print("\n  Training classifier (n_estimators=200)...")
    t0 = time.time()
    tr_cls_mask = train_data["class_valid"]
    cls_model = RandomForestClassifier(n_estimators=200, max_depth=20,
                                        min_samples_leaf=10, n_jobs=-1,
                                        random_state=42, class_weight="balanced")
    cls_model.fit(X_train[tr_cls_mask], train_data["class_targets"][tr_cls_mask])
    print(f"  Trained in {time.time()-t0:.1f}s")

    cls_pred = cls_model.predict(X_test)
    cls = compute_classification_metrics(cls_pred, test_data["class_targets"],
                                          test_data["class_valid"])
    print(f"  Classification - Acc: {cls['accuracy']:.4f}  F1: {cls['macro_f1']:.4f}")

    # Feature importance
    feat_imp = dict(zip(all_names, reg_model.feature_importances_.tolist()))
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    print(f"\n  Top 10 features:")
    for name, imp in list(feat_imp.items())[:10]:
        print(f"    {name:<30s} {imp:.4f}")
    print()

    result = {"model": "rf_combined", "type": "Random Forest (Orbital + LC Features)",
              "regression": reg, "classification": cls,
              "feature_importance": feat_imp}

    save_result(result, output_dir / "rf_combined.json")
    return result



#  OUTPUT


def save_result(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"  Saved: {path}")


def print_comparison_table(results):
    """Print all baseline results alongside each other."""
    print_section("BASELINE COMPARISON")
    print(f"  {'Model':<38s} {'R²':>8s} {'RMSE':>8s} {'MAE':>8s} {'Acc':>8s} {'F1':>8s}")
    print(f"  {'─' * 78}")

    for r in results:
        reg = r.get("regression", {})
        cls = r.get("classification") or {}
        name = r.get("type", r.get("model", "?"))

        def fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "     n/a"
            return f"{v:>8.4f}"

        print(f"  {name:<38s} "
              f"{fmt(reg.get('r2'))} "
              f"{fmt(reg.get('rmse_log'))} "
              f"{fmt(reg.get('mae_log'))} "
              f"{fmt(cls.get('accuracy'))} "
              f"{fmt(cls.get('macro_f1'))}")
    print()


def plot_comparison(results, output_dir):
    """Generate comparison bar charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r.get("type", r.get("model", "?"))[:25] for r in results]
        r2s = [r.get("regression", {}).get("r2", 0) or 0 for r in results]
        accs = [r.get("classification", {}).get("accuracy", 0) or 0
                if r.get("classification") else 0 for r in results]
        f1s = [r.get("classification", {}).get("macro_f1", 0) or 0
               if r.get("classification") else 0 for r in results]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        x = range(len(names))

        for ax, vals, title, colour in [
            (axes[0], r2s, "R\u00b2 (A/m Regression)", "steelblue"),
            (axes[1], accs, "Accuracy (Classification)", "darkorange"),
            (axes[2], f1s, "Macro F1 (Classification)", "seagreen"),
        ]:
            bars = ax.bar(x, vals, color=colour, alpha=0.8)
            ax.set_title(title)
            ax.set_ylabel("Score")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis="y")
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle("Traditional ML Baselines", fontsize=13)
        fig.tight_layout()
        plot_path = output_dir / "baseline_comparison.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Comparison plot: {plot_path}")
    except Exception as e:
        print(f"  [WARN] Could not generate plot: {e}")



#  MAIN


def run_all_baselines(data_dir):
    """Run all baseline models and print comparison."""
    data_dir = Path(data_dir)
    output_dir = data_dir / "models" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    orbital_cols = load_metadata(data_dir)

    print(f"  Loading training data...")
    t0 = time.time()
    train_data = load_split(data_dir, "train", orbital_cols)
    print(f"  Loading test data...")
    test_data = load_split(data_dir, "test", orbital_cols)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Train: {train_data['n']:,}  Test: {test_data['n']:,}")
    print(f"  Orbital features: {len(train_data['orbital_cols'])}")
    print(f"  B* available: {train_data['bstar'] is not None}\n")

    results = []

    # 1. B* analytical
    r = run_bstar_baseline(train_data, test_data, output_dir)
    if r: results.append(r)

    # 2. RF on orbital
    r = run_rf_orbital(train_data, test_data, output_dir)
    if r: results.append(r)

    # 3. GBT on orbital
    r = run_gbt_orbital(train_data, test_data, output_dir)
    if r: results.append(r)

    # 4. RF on LC features
    r = run_rf_lc_features(train_data, test_data, output_dir)
    if r: results.append(r)

    # 5. RF combined
    r = run_rf_combined(train_data, test_data, output_dir)
    if r: results.append(r)

    # Comparison
    if results:
        print_comparison_table(results)
        plot_comparison(results, output_dir)

        # Save combined results
        save_result({"baselines": [
            {k: v for k, v in r.items() if k != "feature_importance"}
            for r in results
        ]}, output_dir / "all_baselines.json")

    return results


def run_interactive(data_dir):
    print_header("Traditional ML Baselines", __version__,
                 "Non-deep-learning models for comparison")

    data_dir = Path(data_dir)
    orbital_cols = load_metadata(data_dir)

    print(f"  Loading data...")
    train_data = load_split(data_dir, "train", orbital_cols)
    test_data = load_split(data_dir, "test", orbital_cols)
    print(f"  Train: {train_data['n']:,}  Test: {test_data['n']:,}\n")

    output_dir = data_dir / "models" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        choice = prompt_choice("Select baseline to run:", [
            ("bstar",    "B* analytical (linear regression)"),
            ("rf_orb",   "Random Forest (orbital features)"),
            ("gbt_orb",  "Gradient Boosting (orbital features)"),
            ("rf_lc",    "Random Forest (LC statistical features)"),
            ("rf_combo", "Random Forest (orbital + LC combined)"),
            ("all",      "Run ALL baselines"),
            ("quit",     "Quit"),
        ])

        if choice in (None, "quit"):
            print("\n  Goodbye!\n")
            break

        results = []
        if choice == "all":
            results = run_all_baselines(data_dir)
        elif choice == "bstar":
            r = run_bstar_baseline(train_data, test_data, output_dir)
            if r: results.append(r)
        elif choice == "rf_orb":
            r = run_rf_orbital(train_data, test_data, output_dir)
            if r: results.append(r)
        elif choice == "gbt_orb":
            r = run_gbt_orbital(train_data, test_data, output_dir)
            if r: results.append(r)
        elif choice == "rf_lc":
            r = run_rf_lc_features(train_data, test_data, output_dir)
            if r: results.append(r)
        elif choice == "rf_combo":
            r = run_rf_combined(train_data, test_data, output_dir)
            if r: results.append(r)


def main():
    parser = argparse.ArgumentParser(description="Space Debris ML - Traditional Baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Examples:
  python baselines.py                               # Interactive menu
  python baselines.py --all                         # Run all baselines
  python baselines.py --all --data-dir path/to/data # Custom data path""")

    parser.add_argument("--all", action="store_true", help="Run all baselines")
    parser.add_argument("--data-dir", type=str, default=None)

    args = parser.parse_args()
    ensure_dirs()
    data_dir = Path(args.data_dir) if args.data_dir else TRAINING_DIR

    if not (data_dir / "train.parquet").exists():
        print(f"  [ERROR] No training data at {data_dir}")
        print(f"  Run the normalisation pipeline first.")
        sys.exit(1)

    try:
        if args.all:
            print_header("Traditional ML Baselines", __version__)
            run_all_baselines(data_dir)
        else:
            run_interactive(data_dir)
    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        sys.exit(2)


if __name__ == "__main__":
    main()
