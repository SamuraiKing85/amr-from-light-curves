"""
Results Analysis & Dissertation Support
============================================
Comprehensive, modular analysis of all trained models and baselines.
Each analyser is independently callable and returns structured findings.

Analyses:
  1. Dataset statistics (class balance, source distribution)
  2. Core ablation (modality contribution, fusion benefit)
  3. B* dependency investigation
  4. Cross-source generalisation matrix
  5. Deep learning vs traditional ML baselines
  6. Training dynamics (convergence, overfitting, stability)
  7. Per-class performance deep dive
  8. XAI synthesis (Grad-CAM patterns, feature importance consensus)
  9. Model efficiency (parameters vs performance)
  10. Publication figures (300 dpi PNGs)
  11. Structured Markdown report with numbered findings

Usage:
    python analysis.py --all                         # Full analysis
    python analysis.py --all --data-dir path/to/data
    python analysis.py                               # Interactive menu
"""

__version__ = "2.0.0"

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import TRAINING_DIR, ensure_dirs
from common.menu import print_header, prompt_choice, print_section

CLASS_NAMES = {0: "Payload", 1: "RocketBody", 2: "Debris"}
# evaluate.py uses "Rocket Body" (with space), normalise to match
CLASS_NAME_ALIASES = {"Rocket Body": "RocketBody", "rocket body": "RocketBody"}



#  UTILITIES


def safe_get(d, *keys, default=None):
    """Safely traverse nested dicts."""
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


def fmt(v, dp=4):
    """Format a numeric value or return 'n/a'."""
    if v is None:
        return "   n/a"
    return f"{v:.{dp}f}"


def normalise_class_name(name):
    """Normalise class name variants (e.g. 'Rocket Body' -> 'RocketBody')."""
    return CLASS_NAME_ALIASES.get(name, name)


def normalise_per_class(per_class_dict):
    """Normalise all keys in a per_class metrics dict."""
    if not per_class_dict:
        return per_class_dict
    return {normalise_class_name(k): v for k, v in per_class_dict.items()}


def pct_change(new, old):
    """Compute percentage change from old to new."""
    if old is None or old == 0:
        return None
    return ((new - old) / abs(old)) * 100


def save_json(data, path):
    """Save dict as JSON with numpy type handling."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer))
                  else o.tolist() if isinstance(o, np.ndarray) else str(o))



#  RESULTS LOADER


class ResultsLoader:
    """Loads all evaluation results, training histories, configs, and XAI outputs."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.results = OrderedDict()
        self.histories = {}
        self.configs = {}
        self.xai = {}

    def load(self):
        if not self.models_dir.exists():
            print(f"  [ERROR] No models at {self.models_dir}")
            return self

        # Neural network models
        for d in sorted(self.models_dir.iterdir()):
            if not d.is_dir() or d.name == "baselines":
                continue

            # Experiment config
            cfg_path = d / "experiment_config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    self.configs[d.name] = json.load(f)

            # Training history
            hist_path = d / "training_history.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    self.histories[d.name] = json.load(f)

            # XAI outputs
            xai_dir = d / "xai"
            if xai_dir.exists():
                xai_data = {}
                for xai_file in xai_dir.glob("*.json"):
                    with open(xai_file) as f:
                        xai_data[xai_file.stem] = json.load(f)
                if xai_data:
                    self.xai[d.name] = xai_data

            # Test results (may have multiple: test_results.json, test_results_mmt9.json, etc.)
            for rfile in sorted(d.glob("test_results*.json")):
                with open(rfile) as f:
                    r = json.load(f)
                fname = rfile.stem
                source = "all" if fname == "test_results" else fname.replace("test_results_", "")
                label = d.name if source == "all" else f"{d.name} [{source}]"
                r["_label"] = label
                r["_dir"] = d.name
                r["_category"] = "neural_network"
                r["_source"] = source
                self.results[label] = r

        # Baselines
        bl_dir = self.models_dir / "baselines"
        if bl_dir.exists():
            for rfile in sorted(bl_dir.glob("*.json")):
                if rfile.name == "all_baselines.json":
                    continue
                with open(rfile) as f:
                    r = json.load(f)
                label = r.get("type", rfile.stem)
                r["_label"] = label
                r["_dir"] = "baselines"
                r["_category"] = "baseline"
                r["_source"] = "all"
                self.results[label] = r

        nn = sum(1 for r in self.results.values() if r["_category"] == "neural_network")
        bl = sum(1 for r in self.results.values() if r["_category"] == "baseline")
        print(f"  Loaded: {nn} NN results, {bl} baselines, "
              f"{len(self.histories)} histories, {len(self.xai)} XAI outputs")
        return self

    def nn_results(self, source="all"):
        return {k: v for k, v in self.results.items()
                if v["_category"] == "neural_network" and v["_source"] == source}

    def baseline_results(self):
        return {k: v for k, v in self.results.items() if v["_category"] == "baseline"}

    def get(self, name):
        return self.results.get(name)



#  1. DATASET STATISTICS


class DatasetAnalyser:
    """Analyse training data composition and class balance."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def run(self, output_dir):
        print_section("1. DATASET STATISTICS")
        findings = []

        for split in ["train", "val", "test"]:
            p = self.data_dir / f"{split}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p, columns=["object_type", "source", "am_ratio_log"])

            n = len(df)
            print(f"  {split.upper()}: {n:,} samples")

            # Class distribution
            if "object_type" in df.columns:
                counts = df["object_type"].value_counts()
                print(f"    Classes:")
                for cls, count in counts.items():
                    pct = count / n * 100
                    print(f"      {cls:<20s} {count:>8,} ({pct:>5.1f}%)")
                    if split == "train" and cls == "Payload":
                        findings.append(f"Training set class distribution: {pct:.1f}% Payload "
                                      f"({count:,} of {n:,} samples). The model predominantly "
                                      f"learns to characterise intact satellites.")

            # Source distribution
            if "source" in df.columns:
                src_counts = df["source"].value_counts()
                print(f"    Sources:")
                for src, count in src_counts.items():
                    pct = count / n * 100
                    print(f"      {src:<20s} {count:>8,} ({pct:>5.1f}%)")

            # A/m coverage
            if "am_ratio_log" in df.columns:
                valid = df["am_ratio_log"].notna().sum()
                pct = valid / n * 100
                print(f"    A/m labels: {valid:,} ({pct:.1f}%)")
            print()

        return findings



#  2. CORE ABLATION


class AblationAnalyser:
    """Analyse contribution of each modality and fusion benefit."""

    def run(self, loader, output_dir):
        print_section("2. ABLATION ANALYSIS")
        findings = []

        fusion = loader.get("fusion")
        lc_only = loader.get("lc_only")
        orbital = loader.get("orbital_only")

        if not (fusion and lc_only and orbital):
            print("  [SKIP] Need fusion, lc_only, and orbital_only results.\n")
            return findings

        # Extract key metrics
        models = {"Fusion": fusion, "LC-only": lc_only, "Orbital-only": orbital}
        metrics = {}
        for name, r in models.items():
            metrics[name] = {
                "r2": safe_get(r, "regression", "r2", default=0),
                "rmse": safe_get(r, "regression", "rmse_log", default=0),
                "mae": safe_get(r, "regression", "mae_log", default=0),
                "acc": safe_get(r, "classification", "accuracy", default=0),
                "f1": safe_get(r, "classification", "macro_f1", default=0),
            }

        # Print comparison
        header = f"  {'Model':<20s} {'R²':>8s} {'RMSE':>8s} {'MAE':>8s} {'Acc':>8s} {'F1':>8s}"
        print(header)
        print(f"  {'─' * 60}")
        for name, m in metrics.items():
            print(f"  {name:<20s} {fmt(m['r2']):>8s} {fmt(m['rmse']):>8s} "
                  f"{fmt(m['mae']):>8s} {fmt(m['acc']):>8s} {fmt(m['f1']):>8s}")

        # Fusion benefit over best single modality
        f_r2 = metrics["Fusion"]["r2"]
        best_single_name = "Orbital-only" if metrics["Orbital-only"]["r2"] > metrics["LC-only"]["r2"] else "LC-only"
        best_single_r2 = metrics[best_single_name]["r2"]
        gain = pct_change(f_r2, best_single_r2)

        print(f"\n  Fusion benefit over {best_single_name}: {gain:+.1f}% R²")
        findings.append(
            f"Multi-modal fusion (R²={f_r2:.4f}) provides a {gain:+.1f}% improvement "
            f"over the best single modality ({best_single_name}, R²={best_single_r2:.4f})."
        )

        # Modality dominance
        o_r2, l_r2 = metrics["Orbital-only"]["r2"], metrics["LC-only"]["r2"]
        gap = o_r2 - l_r2
        if abs(gap) > 0.01:
            dominant = "Orbital features" if gap > 0 else "Light curves"
            weaker = "light curves" if gap > 0 else "orbital features"
            findings.append(
                f"{dominant} (R²={max(o_r2, l_r2):.4f}) substantially outperform "
                f"{weaker} (R²={min(o_r2, l_r2):.4f}) for A/m regression, "
                f"with an R² difference of {abs(gap):.4f}."
            )
        print()

        # Save ablation table
        save_json(metrics, output_dir / "ablation_metrics.json")
        return findings



#  3. B* DEPENDENCY


class BStarAnalyser:
    """Investigate model dependency on B* drag coefficient."""

    def run(self, loader, output_dir):
        print_section("3. B* DEPENDENCY INVESTIGATION")
        findings = []

        orbital = loader.get("orbital_only")
        no_bstar = loader.get("orbital_no_bstar")

        if not (orbital and no_bstar):
            print("  [SKIP] Need orbital_only and orbital_no_bstar results.\n")
            return findings

        o_r2 = safe_get(orbital, "regression", "r2", default=0)
        nb_r2 = safe_get(no_bstar, "regression", "r2", default=0)
        o_f1 = safe_get(orbital, "classification", "macro_f1", default=0)
        nb_f1 = safe_get(no_bstar, "classification", "macro_f1", default=0)

        r2_drop = pct_change(nb_r2, o_r2)
        f1_drop = pct_change(nb_f1, o_f1)

        print(f"  {'Metric':<25s} {'With B*':>10s} {'Without B*':>12s} {'Change':>10s}")
        print(f"  {'─' * 60}")
        print(f"  {'R² (regression)':<25s} {fmt(o_r2):>10s} {fmt(nb_r2):>12s} {r2_drop:>+9.1f}%")
        print(f"  {'Macro F1 (class.)':<25s} {fmt(o_f1):>10s} {fmt(nb_f1):>12s} {f1_drop:>+9.1f}%")
        print()

        if r2_drop is not None:
            severity = abs(r2_drop)
            if severity > 50:
                interpretation = (
                    f"B* is the dominant predictor, with its removal causing a "
                    f"{severity:.1f}% R² collapse. This is expected: B* has a direct "
                    f"physical relationship to A/m through atmospheric drag modelling. "
                    f"However, this dependency means the orbital branch primarily "
                    f"recovers information already available from TLE fitting."
                )
            elif severity > 20:
                interpretation = (
                    f"B* contributes meaningfully ({severity:.1f}% R² drop) but other "
                    f"orbital features partially compensate. The model learns "
                    f"complementary A/m signals from orbital geometry (inclination, "
                    f"eccentricity, mean motion)."
                )
            else:
                interpretation = (
                    f"B* removal has limited impact ({severity:.1f}% R² drop), suggesting "
                    f"orbital geometry features independently encode sufficient A/m information."
                )
            findings.append(interpretation)

        save_json({"with_bstar": {"r2": o_r2, "f1": o_f1},
                   "without_bstar": {"r2": nb_r2, "f1": nb_f1},
                   "r2_change_pct": r2_drop, "f1_change_pct": f1_drop},
                  output_dir / "bstar_investigation.json")
        return findings



#  4. CROSS-SOURCE GENERALISATION


class CrossSourceAnalyser:
    """Build and analyse the cross-source generalisation matrix."""

    def run(self, loader, output_dir):
        print_section("4. CROSS-SOURCE GENERALISATION")
        findings = []

        # Collect LC cross-source results
        lc_entries = []
        orb_entries = []
        for label, r in loader.results.items():
            d = r.get("_dir", "")
            src = r.get("_source", "all")
            if src == "all":
                continue
            train_src = "MMT9" if "mmt9" in d else ("SDLCD" if "sdlcd" in d else None)
            if not train_src:
                continue
            entry = {"train": train_src, "test": src.upper(), "r2": safe_get(r, "regression", "r2"),
                     "rmse": safe_get(r, "regression", "rmse_log"),
                     "acc": safe_get(r, "classification", "accuracy"),
                     "f1": safe_get(r, "classification", "macro_f1")}
            if "lc_only" in d:
                lc_entries.append(entry)
            elif "orbital" in d:
                orb_entries.append(entry)

        if not lc_entries:
            print("  [SKIP] No cross-source results found.\n")
            return findings

        # Print LC matrix
        print(f"  LC-only Cross-Source Matrix:")
        print(f"  {'':>15s} ", end="")
        for metric in ["R²", "RMSE", "Acc", "F1"]:
            print(f"{'Test MMT9':>10s} {'Test SDLCD':>11s}  ", end="")
        print()
        print(f"  {'─' * 100}")

        for train_src in ["MMT9", "SDLCD"]:
            print(f"  Train {train_src:<8s} ", end="")
            for metric_key in ["r2", "rmse", "acc", "f1"]:
                for test_src in ["MMT9", "SDLCD"]:
                    match = [e for e in lc_entries if e["train"] == train_src and e["test"] == test_src]
                    val = match[0][metric_key] if match else None
                    marker = " *" if train_src == test_src else "  "
                    print(f"{fmt(val):>10s}{marker}", end="")
            print()
        print(f"  (* = within-source)\n")

        # Compute degradation statistics
        within = [e for e in lc_entries if e["train"] == e["test"]]
        cross = [e for e in lc_entries if e["train"] != e["test"]]

        if within and cross:
            within_r2 = np.mean([e["r2"] for e in within if e["r2"] is not None])
            cross_r2 = np.mean([e["r2"] for e in cross if e["r2"] is not None])
            drop = pct_change(cross_r2, within_r2) or 0

            print(f"  LC R² degradation:")
            print(f"    Within-source average: {within_r2:.4f}")
            print(f"    Cross-source average:  {cross_r2:.4f}")
            print(f"    Degradation:           {abs(drop):.1f}%\n")

            if abs(drop) > 50:
                findings.append(
                    f"LC models show substantial {abs(drop):.1f}% R² degradation cross-source "
                    f"(within={within_r2:.4f}, cross={cross_r2:.4f}). This indicates the CNN "
                    f"learns telescope-specific features, though the confound between telescope "
                    f"source and orbit regime (MMT-9=LEO, SDLCD=GEO) makes definitive "
                    f"attribution difficult."
                )
            elif abs(drop) > 20:
                findings.append(
                    f"LC models show moderate {abs(drop):.1f}% R² degradation cross-source "
                    f"(within={within_r2:.4f}, cross={cross_r2:.4f}). Partial generalisation "
                    f"is achieved but telescope-specific learning is present."
                )
            else:
                findings.append(
                    f"LC models generalise well with only {abs(drop):.1f}% R² degradation "
                    f"cross-source (within={within_r2:.4f}, cross={cross_r2:.4f}), providing "
                    f"evidence that the CNN learns physical properties."
                )

        # Orbital control
        if orb_entries:
            orb_within = [e for e in orb_entries if e["train"] == e["test"]]
            orb_cross = [e for e in orb_entries if e["train"] != e["test"]]
            if orb_within and orb_cross:
                ow_r2 = np.mean([e["r2"] for e in orb_within if e["r2"] is not None])
                oc_r2 = np.mean([e["r2"] for e in orb_cross if e["r2"] is not None])
                orb_drop = pct_change(oc_r2, ow_r2) or 0
                print(f"  Orbital control:")
                print(f"    Within: {ow_r2:.4f}  Cross: {oc_r2:.4f}  Drop: {abs(orb_drop):.1f}%\n")
                findings.append(
                    f"Orbital-only control shows {abs(orb_drop):.1f}% cross-source degradation, "
                    f"{'confirming the LC branch specifically drives the cross-source performance gap.' if abs(drop) > abs(orb_drop) + 5 else 'suggesting orbit regime differences partially explain the cross-source gap.'}"
                )

        save_json({"lc": lc_entries, "orbital": orb_entries},
                  output_dir / "cross_source_matrix.json")
        return findings



#  5. DEEP LEARNING vs BASELINES


class BaselineComparer:
    """Compare neural network models against traditional ML baselines."""

    def run(self, loader, output_dir):
        print_section("5. DEEP LEARNING vs TRADITIONAL ML")
        findings = []
        baselines = loader.baseline_results()

        if not baselines:
            print("  [SKIP] No baseline results. Run baselines.py first.\n")
            return findings

        nn_core = loader.nn_results(source="all")

        # Build unified comparison
        all_models = OrderedDict()
        for k, v in nn_core.items():
            all_models[k] = {"r2": safe_get(v, "regression", "r2"),
                             "f1": safe_get(v, "classification", "macro_f1"),
                             "cat": "NN"}
        for k, v in baselines.items():
            all_models[k] = {"r2": safe_get(v, "regression", "r2"),
                             "f1": safe_get(v, "classification", "macro_f1"),
                             "cat": "Baseline"}

        # Sort by R²
        sorted_models = sorted(all_models.items(), key=lambda x: x[1]["r2"] or -1, reverse=True)

        print(f"  {'Rank':>4s}  {'Model':<40s} {'R²':>8s} {'F1':>8s} {'Type':>10s}")
        print(f"  {'─' * 76}")
        for rank, (name, m) in enumerate(sorted_models, 1):
            print(f"  {rank:>4d}  {name[:40]:<40s} {fmt(m['r2']):>8s} "
                  f"{fmt(m['f1']):>8s} {m['cat']:>10s}")
        print()

        # Key comparisons
        comparisons = [
            ("orbital_only", "Random Forest on Orbital Features",
             "orbital features", "tabular"),
            ("orbital_only", "Gradient Boosting on Orbital Features",
             "orbital features", "tabular"),
            ("lc_only", "Random Forest on LC Statistics",
             "light curves", "temporal"),
            ("fusion", "Random Forest (Orbital + LC Features)",
             "multi-modal", "fusion"),
        ]

        for nn_key, bl_key, domain, comp_type in comparisons:
            nn_r = nn_core.get(nn_key)
            bl_r = baselines.get(bl_key)
            if not (nn_r and bl_r):
                continue
            nn_r2 = safe_get(nn_r, "regression", "r2", default=0)
            bl_r2 = safe_get(bl_r, "regression", "r2", default=0)
            diff = nn_r2 - bl_r2

            if diff > 0.01:
                findings.append(
                    f"On {domain}: neural network (R²={nn_r2:.4f}) outperforms "
                    f"{bl_key} (R²={bl_r2:.4f}) by {diff:.4f} R²."
                )
            elif diff < -0.01:
                findings.append(
                    f"On {domain}: {bl_key} (R²={bl_r2:.4f}) outperforms the "
                    f"neural network (R²={nn_r2:.4f}) by {abs(diff):.4f} R². "
                    + ("This is consistent with known results that tree-based methods "
                       "often outperform neural networks on tabular data."
                       if comp_type == "tabular" else "")
                )
            else:
                findings.append(
                    f"On {domain}: neural network (R²={nn_r2:.4f}) and "
                    f"{bl_key} (R²={bl_r2:.4f}) perform comparably."
                )

        # B* analytical comparison
        bstar = baselines.get("Linear Regression on B*")
        fusion = nn_core.get("fusion")
        if bstar and fusion:
            b_r2 = safe_get(bstar, "regression", "r2", default=0)
            f_r2 = safe_get(fusion, "regression", "r2", default=0)
            gain = pct_change(f_r2, b_r2)
            if gain is not None:
                findings.append(
                    f"The fusion model (R²={f_r2:.4f}) improves {gain:+.1f}% over the "
                    f"B* analytical baseline (R²={b_r2:.4f}), demonstrating ML extracts "
                    f"information beyond the known B*-to-A/m physical relationship."
                )

        save_json({k: v for k, v in all_models.items()},
                  output_dir / "dl_vs_baselines.json")
        return findings



#  6. TRAINING DYNAMICS


class TrainingDynamicsAnalyser:
    """Analyse convergence speed, overfitting, and training stability."""

    def run(self, loader, output_dir):
        print_section("6. TRAINING DYNAMICS")
        findings = []

        if not loader.histories:
            print("  [SKIP] No training histories found.\n")
            return findings

        rows = []
        print(f"  {'Model':<24s} {'Epochs':>7s} {'Best Ep':>8s} {'Best R²':>8s} "
              f"{'Final R²':>9s} {'Overfit':>8s} {'Avg s/ep':>9s}")
        print(f"  {'─' * 82}")

        for name, hist in loader.histories.items():
            n_epochs = len(hist.get("val_loss", []))
            if n_epochs == 0:
                continue

            best_epoch = hist.get("best_epoch", "?")
            val_r2 = hist.get("val_r2", [])
            best_r2 = max(val_r2) if val_r2 else 0
            final_r2 = val_r2[-1] if val_r2 else 0

            # Overfitting: gap between train and val loss at end
            train_loss = hist.get("train_loss", [])
            val_loss = hist.get("val_loss", [])
            if train_loss and val_loss and train_loss[-1] is not None:
                overfit_gap = val_loss[-1] - train_loss[-1]
            else:
                overfit_gap = None

            # Average time per epoch
            times = [t for t in hist.get("epoch_time", []) if t is not None]
            avg_time = np.mean(times) if times else None

            rows.append({
                "model": name, "epochs": n_epochs, "best_epoch": best_epoch,
                "best_r2": best_r2, "final_r2": final_r2,
                "overfit_gap": overfit_gap, "avg_time": avg_time,
                "total_time_h": sum(times) / 3600 if times else None,
            })

            of_str = f"{overfit_gap:.4f}" if overfit_gap is not None else "  n/a"
            time_str = f"{avg_time:.0f}s" if avg_time is not None else "  n/a"
            print(f"  {name:<24s} {n_epochs:>7d} {str(best_epoch):>8s} {best_r2:>8.4f} "
                  f"{final_r2:>9.4f} {of_str:>8s} {time_str:>9s}")

        print()

        if rows:
            # Total compute time
            total_h = sum(r["total_time_h"] for r in rows if r["total_time_h"] is not None)
            total_epochs = sum(r["epochs"] for r in rows)
            findings.append(
                f"Training required {total_epochs} total epochs across {len(rows)} experiments, "
                f"completing in approximately {total_h:.1f} compute-hours."
            )

            # Early stopping analysis
            stopped_early = [r for r in rows if r["epochs"] < 100]
            if stopped_early:
                avg_stop = np.mean([r["epochs"] for r in stopped_early])
                findings.append(
                    f"{len(stopped_early)} of {len(rows)} experiments triggered early stopping "
                    f"(patience=15), converging at an average of {avg_stop:.0f} epochs."
                )

            # Overfitting assessment
            overfit_rows = [r for r in rows if r["overfit_gap"] is not None]
            if overfit_rows:
                max_gap = max(r["overfit_gap"] for r in overfit_rows)
                min_gap = min(r["overfit_gap"] for r in overfit_rows)
                if max_gap < 0.05:
                    findings.append("Minimal overfitting observed across all models "
                                  f"(max train-val gap: {max_gap:.4f}).")

        save_json(rows, output_dir / "training_dynamics.json")
        return findings



#  7. PER-CLASS PERFORMANCE


class ClassAnalyser:
    """Deep dive into per-class performance and misclassification patterns."""

    def run(self, loader, output_dir):
        print_section("7. PER-CLASS PERFORMANCE")
        findings = []
        rows = []

        for label, r in loader.results.items():
            if r.get("_source") != "all":
                continue
            per_class = normalise_per_class(safe_get(r, "classification", "per_class"))
            if not per_class:
                continue
            for cls_name, metrics in per_class.items():
                if isinstance(metrics, dict):
                    rows.append({
                        "Model": label, "Category": r["_category"],
                        "Class": cls_name,
                        "Precision": metrics.get("precision"),
                        "Recall": metrics.get("recall"),
                        "F1": metrics.get("f1"),
                        "Support": metrics.get("support"),
                    })

        if not rows:
            print("  [SKIP] No per-class data available.\n")
            return findings

        df = pd.DataFrame(rows)

        for cls in CLASS_NAMES.values():
            cls_df = df[df["Class"] == cls].sort_values("F1", ascending=False)
            if cls_df.empty:
                continue
            print(f"\n  {cls}:")
            print(f"  {'Model':<40s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'N':>8s}")
            print(f"  {'─' * 72}")
            for _, row in cls_df.iterrows():
                print(f"  {row['Model'][:40]:<40s} {fmt(row['Precision']):>8s} "
                      f"{fmt(row['Recall']):>8s} {fmt(row['F1']):>8s} "
                      f"{int(row['Support']) if row['Support'] else 'n/a':>8}")

        # Debris findings
        debris = df[df["Class"] == "Debris"]
        if not debris.empty:
            best = debris.loc[debris["F1"].idxmax()]
            findings.append(
                f"Debris classification: best F1={best['F1']:.4f} ({best['Model']}). "
                f"All models struggle with debris (support={int(best['Support'])} test samples), "
                f"reflecting the severe training set imbalance."
            )

        # Payload vs RocketBody confusion
        payload = df[df["Class"] == "Payload"]
        rb = df[df["Class"] == "RocketBody"]
        if not payload.empty and not rb.empty:
            best_payload_f1 = payload["F1"].max()
            best_rb_f1 = rb["F1"].max()
            if best_rb_f1 < best_payload_f1 - 0.1:
                findings.append(
                    f"Rocket body classification (best F1={best_rb_f1:.4f}) consistently "
                    f"underperforms payload classification (best F1={best_payload_f1:.4f}), "
                    f"likely due to lower rocket body representation in the training data."
                )
        print()

        csv_path = output_dir / "per_class_breakdown.csv"
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"  Saved: {csv_path}\n")
        return findings



#  8. XAI SYNTHESIS


class XAIAnalyser:
    """Synthesise Grad-CAM and permutation importance across models."""

    def run(self, loader, output_dir):
        print_section("8. XAI SYNTHESIS")
        findings = []

        # ── Grad-CAM analysis ──
        gradcam_models = {}
        for name, xai in loader.xai.items():
            gc = xai.get("gradcam_summary")
            if gc:
                gradcam_models[name] = gc

        if gradcam_models:
            print(f"  Grad-CAM attention patterns:")
            print(f"  {'Model':<24s} {'Peak pos':>10s} {'50% energy':>12s} {'Concentration':>14s}")
            print(f"  {'─' * 64}")
            for name, gc in gradcam_models.items():
                peak = gc.get("avg_cam_peak_position", "?")
                frac = gc.get("attention_50pct_fraction", 0)
                points = gc.get("attention_50pct_points", "?")
                print(f"  {name:<24s} {str(peak):>10s} {str(points):>10s} pts {frac*100:>10.0f}%")

            # Compare attention concentration across models
            concentrations = {n: gc.get("attention_50pct_fraction", 1)
                            for n, gc in gradcam_models.items()}
            most_focused = min(concentrations, key=concentrations.get)
            least_focused = max(concentrations, key=concentrations.get)

            if concentrations[most_focused] < 0.5:
                findings.append(
                    f"Grad-CAM reveals {most_focused} concentrates 50% of attention on "
                    f"just {concentrations[most_focused]*100:.0f}% of the light curve, "
                    f"indicating the model focuses on specific temporal features rather "
                    f"than using the full sequence uniformly."
                )
            print()

        # ── Permutation importance synthesis ──
        all_importances = {}
        for name, xai in loader.xai.items():
            pi = xai.get("permutation_importance", {}).get("features")
            if pi:
                all_importances[name] = pi

        # Also load baseline feature importances
        bl_dir = loader.models_dir / "baselines"
        if bl_dir.exists():
            for rfile in bl_dir.glob("*.json"):
                if rfile.name == "all_baselines.json":
                    continue
                with open(rfile) as f:
                    r = json.load(f)
                if "feature_importance" in r:
                    all_importances[f"BL:{r.get('model', rfile.stem)}"] = r["feature_importance"]

        if all_importances:
            print(f"  Feature importance consensus ({len(all_importances)} sources):")

            # Build consensus ranking
            rankings = {}
            for source, imp_dict in all_importances.items():
                sorted_feats = sorted(imp_dict.keys(),
                                     key=lambda k: (imp_dict[k].get("importance", imp_dict[k])
                                                     if isinstance(imp_dict[k], dict)
                                                     else imp_dict[k]),
                                     reverse=True)
                for rank, feat in enumerate(sorted_feats):
                    clean = feat.replace("_scaled", "")
                    if clean not in rankings:
                        rankings[clean] = []
                    rankings[clean].append(rank + 1)

            consensus = sorted(rankings.items(), key=lambda x: np.mean(x[1]))
            print(f"  {'Feature':<30s} {'Avg Rank':>10s} {'Sources':>10s}")
            print(f"  {'─' * 54}")
            for feat, ranks in consensus[:10]:
                bstar_marker = " ***" if "bstar" in feat.lower() else ""
                print(f"  {feat:<30s} {np.mean(ranks):>10.1f} {len(ranks):>10d}{bstar_marker}")

            top5 = [f[0] for f in consensus[:5]]
            bstar_in_top = sum(1 for f in top5 if "bstar" in f.lower())
            if bstar_in_top > 0:
                findings.append(
                    f"B*-related features appear {bstar_in_top} time(s) in the top 5 "
                    f"consensus ranking, confirming the B* ablation results."
                )
            findings.append(
                f"Consensus top 5 features across all models: {', '.join(top5)}."
            )
            print()

        save_json({"gradcam": gradcam_models,
                   "consensus_ranking": consensus[:15] if all_importances else []},
                  output_dir / "xai_synthesis.json")
        return findings



#  9. MODEL EFFICIENCY


class EfficiencyAnalyser:
    """Compare model complexity vs performance."""

    def run(self, loader, output_dir):
        print_section("9. MODEL EFFICIENCY")
        findings = []
        rows = []

        for name, cfg in loader.configs.items():
            r = loader.get(name)
            if not r:
                continue
            hist = loader.histories.get(name, {})
            times = [t for t in hist.get("epoch_time", []) if t is not None]

            params = cfg.get("config", {}).get("model_params")
            # Try to get params from train log
            train_samples = cfg.get("train_samples", 0)

            rows.append({
                "model": name,
                "type": cfg.get("model_type", "?"),
                "orbital_dim": cfg.get("orbital_dim", "?"),
                "train_samples": train_samples,
                "epochs": len(hist.get("val_loss", [])),
                "total_time_h": sum(times) / 3600 if times else None,
                "r2": safe_get(r, "regression", "r2"),
                "f1": safe_get(r, "classification", "macro_f1"),
            })

        if not rows:
            print("  [SKIP] No experiment configs found.\n")
            return findings

        print(f"  {'Model':<24s} {'Type':<15s} {'Samples':>10s} {'Epochs':>7s} "
              f"{'Time':>7s} {'R²':>8s}")
        print(f"  {'─' * 78}")
        for row in rows:
            t = f"{row['total_time_h']:.1f}h" if row["total_time_h"] else "  n/a"
            print(f"  {row['model']:<24s} {row['type']:<15s} "
                  f"{row['train_samples']:>10,} {row['epochs']:>7d} "
                  f"{t:>7s} {fmt(row['r2']):>8s}")

        # Find most efficient model (best R² per compute-hour)
        with_time = [r for r in rows if r["total_time_h"] and r["r2"]]
        if with_time:
            for r in with_time:
                r["r2_per_hour"] = r["r2"] / r["total_time_h"] if r["total_time_h"] > 0 else 0
            best_eff = max(with_time, key=lambda x: x["r2_per_hour"])
            findings.append(
                f"Most efficient model: {best_eff['model']} achieves R²={best_eff['r2']:.4f} "
                f"in {best_eff['total_time_h']:.1f} hours ({best_eff['r2_per_hour']:.3f} R²/hour)."
            )

        print()
        save_json(rows, output_dir / "model_efficiency.json")
        return findings



#  10. PREDICTION ANALYSIS (scatter plots + residuals)


class PredictionAnalyser:
    """Predicted vs actual scatter plots and residual distribution analysis."""

    def _load_predictions(self, models_dir, model_name):
        """Load saved .npz predictions for a model."""
        d = models_dir / model_name
        for npz_name in ["test_results.npz"]:
            p = d / npz_name
            if p.exists():
                return dict(np.load(p, allow_pickle=True))
        return None

    def run(self, loader, output_dir):
        print_section("10. PREDICTION ANALYSIS")
        findings = []

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scipy import stats as sp_stats
        except ImportError:
            print("  [SKIP] matplotlib/scipy not available.\n")
            return findings

        # Analyse core models
        core_models = ["fusion", "lc_only", "orbital_only"]
        available = []
        for name in core_models:
            preds = self._load_predictions(loader.models_dir, name)
            if preds is not None:
                available.append((name, preds))

        if not available:
            print("  [SKIP] No prediction .npz files found.")
            print("  Re-run: python evaluate.py --full\n")
            return findings

        # ── Scatter plots (predicted vs actual) ──
        n_models = len(available)
        fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        for ax, (name, preds) in zip(axes, available):
            am_p = preds["am_pred"]
            am_t = preds["am_target"]
            am_v = preds["am_valid"].astype(bool)
            mask = am_v & np.isfinite(am_p) & np.isfinite(am_t)

            p, t = am_p[mask], am_t[mask]

            # Colour by object type if available
            if "class_target" in preds:
                cls = preds["class_target"][mask]
                colours = {0: "#2196F3", 1: "#4CAF50", 2: "#F44336"}
                for c, colour in colours.items():
                    cmask = cls == c
                    if cmask.sum() > 0:
                        ax.scatter(t[cmask], p[cmask], s=2, alpha=0.3, c=colour,
                                  label=CLASS_NAMES.get(c, f"C{c}"), rasterized=True)
                ax.legend(fontsize=8, markerscale=4)
            else:
                ax.scatter(t, p, s=2, alpha=0.2, c="steelblue", rasterized=True)

            # Perfect prediction line
            lims = [min(t.min(), p.min()), max(t.max(), p.max())]
            ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect")
            ax.set_xlabel("Actual log\u2081\u2080(A/m)")
            ax.set_ylabel("Predicted log\u2081\u2080(A/m)")
            r2 = safe_get(loader.get(name), "regression", "r2", default=0)
            ax.set_title(f"{name} (R\u00b2={r2:.3f})")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)

        fig.suptitle("Predicted vs Actual A/m Ratio", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_scatter_predicted_vs_actual.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  fig_scatter_predicted_vs_actual.png")

        # ── Residual distributions ──
        fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4))
        if n_models == 1:
            axes = [axes]

        for ax, (name, preds) in zip(axes, available):
            am_p = preds["am_pred"]
            am_t = preds["am_target"]
            am_v = preds["am_valid"].astype(bool)
            mask = am_v & np.isfinite(am_p) & np.isfinite(am_t)
            residuals = am_p[mask] - am_t[mask]

            ax.hist(residuals, bins=80, density=True, alpha=0.7, color="steelblue",
                   edgecolor="white", linewidth=0.5)

            # Fit normal distribution overlay
            mu, sigma = residuals.mean(), residuals.std()
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
            ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), "r-", linewidth=2,
                   label=f"N({mu:.3f}, {sigma:.3f})")

            # Normality test
            if len(residuals) > 5000:
                stat, p_val = sp_stats.normaltest(residuals[:5000])
            else:
                stat, p_val = sp_stats.normaltest(residuals)

            ax.set_xlabel("Residual (predicted - actual)")
            ax.set_ylabel("Density")
            ax.set_title(f"{name}\n\u03bc={mu:.4f}, \u03c3={sigma:.4f}")
            ax.legend(fontsize=8)
            ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.2)

            # Check for systematic bias
            if abs(mu) > 0.05:
                findings.append(
                    f"{name}: systematic bias detected (mean residual={mu:.4f}). "
                    f"The model {'overestimates' if mu > 0 else 'underestimates'} "
                    f"A/m on average."
                )

        fig.suptitle("Residual Distributions", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_residual_distributions.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  fig_residual_distributions.png")

        # ── Confusion matrices ──
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
        if n_models == 1:
            axes = [axes]

        class_labels = list(CLASS_NAMES.values())
        for ax, (name, preds) in zip(axes, available):
            cls_p = preds["class_pred"]
            cls_t = preds["class_target"]
            cls_v = preds["class_valid"].astype(bool)
            cmask = cls_v & (cls_t >= 0) & (cls_t < 3)
            p, t = cls_p[cmask], cls_t[cmask]

            cm = np.zeros((3, 3), dtype=int)
            for i in range(len(p)):
                if 0 <= t[i] < 3 and 0 <= p[i] < 3:
                    cm[t[i], p[i]] += 1

            # Normalise by row (recall-oriented)
            cm_norm = cm.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums > 0)

            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(class_labels, fontsize=9)
            ax.set_yticklabels(class_labels, fontsize=9)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{name}")
            for i in range(3):
                for j in range(3):
                    txt = f"{cm[i,j]}\n({cm_norm[i,j]:.0%})"
                    colour = "white" if cm_norm[i,j] > 0.5 else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=colour)

        fig.suptitle("Confusion Matrices (normalised by true class)", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_confusion_matrices.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  fig_confusion_matrices.png\n")

        return findings



#  11. ORBIT REGIME ANALYSIS


class OrbitRegimeAnalyser:
    """Break down performance by orbit regime using TLE-derived orbital parameters.

    Classifies objects by perigee altitude using standard definitions:
      LEO:  perigee < 2,000 km
      MEO:  2,000 km <= perigee < 35,586 km
      GEO:  35,586 km <= perigee < 35,986 km
      HEO:  apogee > 35,986 km and perigee < 2,000 km (highly eccentric)

    This is more accurate than using telescope source as a proxy, since
    MMT-9 includes some MEO/HEO objects and SDLCD includes non-GEO targets.
    """

    REGIMES = [
        ("LEO",  lambda p, a: p < 2000),
        ("MEO",  lambda p, a: 2000 <= p < 35586 and a < 35986),
        ("GEO",  lambda p, a: 35586 <= p < 35986),
        ("HEO",  lambda p, a: a > 35986 and p < 2000),
        ("Other", lambda p, a: True),  # Catch-all for anything else
    ]

    def _load_predictions(self, models_dir, model_name):
        p = models_dir / model_name / "test_results.npz"
        return dict(np.load(p, allow_pickle=True)) if p.exists() else None

    def _classify_regime(self, perigee, apogee):
        """Assign orbit regime based on perigee and apogee altitudes."""
        for name, check in self.REGIMES:
            if check(perigee, apogee):
                return name
        return "Other"

    def _compute_metrics(self, am_p, am_t, am_v, cls_p, cls_t, cls_v, mask):
        """Compute regression + classification metrics for a subset."""
        result = {"n": int(mask.sum())}

        # Regression
        reg_mask = mask & am_v & np.isfinite(am_p) & np.isfinite(am_t)
        if reg_mask.sum() > 0:
            res = am_p[reg_mask] - am_t[reg_mask]
            ss_res = np.sum(res**2)
            ss_tot = np.sum((am_t[reg_mask] - am_t[reg_mask].mean())**2)
            result["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0
            result["rmse"] = float(np.sqrt(np.mean(res**2)))
            result["mae"] = float(np.mean(np.abs(res)))
        else:
            result["r2"], result["rmse"], result["mae"] = None, None, None

        # Classification
        cls_mask = mask & cls_v & (cls_t >= 0) & (cls_t < 3)
        if cls_mask.sum() > 0:
            cp, ct = cls_p[cls_mask], cls_t[cls_mask]
            result["acc"] = float((cp == ct).mean())
            f1s = []
            for c in range(3):
                tp = ((cp == c) & (ct == c)).sum()
                fp = ((cp == c) & (ct != c)).sum()
                fn = ((cp != c) & (ct == c)).sum()
                pr = tp / (tp + fp) if (tp + fp) > 0 else 0
                re = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1s.append(2 * pr * re / (pr + re) if (pr + re) > 0 else 0)
            result["f1"] = float(np.mean([f for f, c in zip(f1s, range(3))
                                          if (ct == c).sum() > 0]))
        else:
            result["acc"], result["f1"] = None, None

        return result

    def run(self, loader, output_dir):
        print_section("11. PERFORMANCE BY ORBIT REGIME")
        findings = []

        # Load test data for orbital parameters
        test_path = loader.data_dir / "test.parquet"
        if not test_path.exists():
            print("  [SKIP] No test.parquet found.\n")
            return findings

        # Read orbital parameters and source
        cols_to_read = ["perigee_km", "apogee_km"]
        optional_cols = ["perigee_km_scaled", "apogee_km_scaled", "source"]
        test_df = pd.read_parquet(test_path)

        # Find perigee/apogee columns (could be raw or scaled)
        perigee_col = None
        apogee_col = None
        for candidate in ["perigee_km", "perigee_km_scaled"]:
            if candidate in test_df.columns:
                perigee_col = candidate
                break
        for candidate in ["apogee_km", "apogee_km_scaled"]:
            if candidate in test_df.columns:
                apogee_col = candidate
                break

        if perigee_col is None or apogee_col is None:
            print("  [SKIP] No perigee/apogee columns in test data.\n")
            return findings

        # If using scaled values, we need to unscale them
        # Check for normalisation params
        is_scaled = perigee_col.endswith("_scaled")
        perigee_vals = test_df[perigee_col].values.astype(np.float64)
        apogee_vals = test_df[apogee_col].values.astype(np.float64)

        if is_scaled:
            # Try to load scaling parameters to unscale
            meta_path = loader.data_dir / "normalisation_params.json"
            unscaled = False
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                scaling = meta.get("global_scaling", meta.get("feature_scaling", {}))
                means = scaling.get("means", {})
                stds = scaling.get("stds", {})
                p_key = perigee_col.replace("_scaled", "")
                a_key = apogee_col.replace("_scaled", "")
                if p_key in means and a_key in means:
                    perigee_vals = perigee_vals * stds[p_key] + means[p_key]
                    apogee_vals = apogee_vals * stds[a_key] + means[a_key]
                    unscaled = True
            if not unscaled:
                # Check for raw columns as fallback
                for raw in ["perigee_km", "apogee_km"]:
                    if raw in test_df.columns and raw != perigee_col:
                        perigee_vals = test_df["perigee_km"].values.astype(np.float64)
                        apogee_vals = test_df["apogee_km"].values.astype(np.float64)
                        break
                else:
                    print("  [WARN] Using scaled orbital values - regime thresholds may be inaccurate")

        # Assign regimes
        regimes = np.array([
            self._classify_regime(p, a) for p, a in zip(perigee_vals, apogee_vals)
        ])

        # Load predictions from best model
        model_name = "fusion"
        preds = self._load_predictions(loader.models_dir, model_name)
        if preds is None:
            for fallback in ["orbital_only", "lc_only"]:
                preds = self._load_predictions(loader.models_dir, fallback)
                if preds is not None:
                    model_name = fallback
                    break
        if preds is None:
            print("  [SKIP] No prediction .npz files found.\n")
            return findings

        am_p = preds["am_pred"]
        am_t = preds["am_target"]
        am_v = preds["am_valid"].astype(bool)
        cls_p = preds["class_pred"]
        cls_t = preds["class_target"]
        cls_v = preds["class_valid"].astype(bool)

        # Verify alignment
        if len(regimes) != len(am_p):
            print(f"  [WARN] Size mismatch: test={len(regimes)}, preds={len(am_p)}")
            print("  Re-run evaluate.py to regenerate predictions.\n")
            return findings

        # Report regime distribution
        unique, counts = np.unique(regimes, return_counts=True)
        print(f"  Orbit regime distribution (from TLE perigee/apogee):")
        for regime, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
            pct = count / len(regimes) * 100
            print(f"    {regime:<8s} {count:>8,} ({pct:.1f}%)")
        print()

        # Source cross-reference (if available)
        if "source" in test_df.columns:
            print(f"  Regime vs telescope source:")
            print(f"  {'':>8s} ", end="")
            sources_unique = sorted(test_df["source"].unique())
            for src in sources_unique:
                print(f"{src:>8s} ", end="")
            print()
            print(f"  {'─' * (10 + 9 * len(sources_unique))}")
            for regime in sorted(unique, key=lambda r: -np.sum(regimes == r)):
                print(f"  {regime:<8s} ", end="")
                for src in sources_unique:
                    src_mask = (test_df["source"].values == src)
                    count = np.sum((regimes == regime) & src_mask)
                    print(f"{count:>8,} ", end="")
                print()
            print()

        # Compute per-regime metrics
        print(f"  Performance by orbit regime (model: {model_name}):")
        print(f"  {'Regime':<8s} {'N':>8s} {'N(A/m)':>8s} {'R²':>8s} {'RMSE':>8s} "
              f"{'MAE':>8s} {'Acc':>8s} {'F1':>8s}")
        print(f"  {'─' * 68}")

        regime_metrics = {}
        for regime in sorted(unique, key=lambda r: -np.sum(regimes == r)):
            mask = regimes == regime
            if mask.sum() < 5:
                continue

            m = self._compute_metrics(am_p, am_t, am_v, cls_p, cls_t, cls_v, mask)
            n_am = int((mask & am_v).sum())
            regime_metrics[regime] = m

            print(f"  {regime:<8s} {m['n']:>8,} {n_am:>8,} {fmt(m['r2']):>8s} "
                  f"{fmt(m['rmse']):>8s} {fmt(m['mae']):>8s} "
                  f"{fmt(m['acc']):>8s} {fmt(m['f1']):>8s}")

        # Generate findings
        if "LEO" in regime_metrics and len(regime_metrics) > 1:
            leo = regime_metrics["LEO"]
            other_regimes = {k: v for k, v in regime_metrics.items() if k != "LEO" and v["r2"] is not None}

            if leo["r2"] is not None and other_regimes:
                best_other = max(other_regimes.items(), key=lambda x: x[1]["n"])
                other_name, other_m = best_other

                if other_m["r2"] is not None:
                    diff = leo["r2"] - other_m["r2"]
                    findings.append(
                        f"Performance varies by orbit regime: LEO R²={leo['r2']:.4f} "
                        f"(N={leo['n']:,}) vs {other_name} R²={other_m['r2']:.4f} "
                        f"(N={other_m['n']:,}). "
                        + (f"The {abs(diff):.4f} R² gap likely reflects both the larger "
                           f"LEO training sample ({leo['n']:,} vs {other_m['n']:,} test samples) "
                           f"and physical differences in A/m distributions between orbit regimes."
                           if diff > 0 else
                           f"{other_name} objects achieve higher R² despite fewer samples, "
                           f"suggesting more predictable A/m characteristics.")
                    )

            # Report regime composition
            regime_counts = {r: m["n"] for r, m in regime_metrics.items()}
            total = sum(regime_counts.values())
            composition = ", ".join(f"{r}={n:,} ({n/total*100:.1f}%)"
                                   for r, n in sorted(regime_counts.items(), key=lambda x: -x[1]))
            findings.append(
                f"Test set orbit regime composition (from TLE parameters): {composition}."
            )

        print()
        save_json(regime_metrics, output_dir / "orbit_regime_metrics.json")
        return findings



#  12. BOOTSTRAP CONFIDENCE INTERVALS


class BootstrapAnalyser:
    """Compute bootstrap confidence intervals for key metrics."""

    def _load_predictions(self, models_dir, model_name):
        p = models_dir / model_name / "test_results.npz"
        return dict(np.load(p, allow_pickle=True)) if p.exists() else None

    def _bootstrap_r2(self, pred, target, valid, n_boot=1000):
        mask = valid & np.isfinite(pred) & np.isfinite(target)
        p, t = pred[mask], target[mask]
        n = len(p)
        if n < 10:
            return None, None, None

        rng = np.random.RandomState(42)
        r2s = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            res = p[idx] - t[idx]
            ss_res = np.sum(res**2)
            ss_tot = np.sum((t[idx] - t[idx].mean())**2)
            r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0)

        r2s = np.array(r2s)
        return float(np.mean(r2s)), float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))

    def run(self, loader, output_dir):
        print_section("12. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
        findings = []

        core_models = ["fusion", "lc_only", "orbital_only", "orbital_no_bstar"]
        ci_results = {}

        print(f"  {'Model':<24s} {'R² mean':>10s} {'95% CI':>20s} {'Width':>8s}")
        print(f"  {'─' * 66}")

        for name in core_models:
            preds = self._load_predictions(loader.models_dir, name)
            if preds is None:
                continue

            mean, lo, hi = self._bootstrap_r2(
                preds["am_pred"], preds["am_target"],
                preds["am_valid"].astype(bool), n_boot=2000
            )
            if mean is None:
                continue

            width = hi - lo
            ci_results[name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "width": width}
            print(f"  {name:<24s} {mean:>10.4f} [{lo:.4f}, {hi:.4f}] {width:>8.4f}")

        # Statistical significance tests
        if "fusion" in ci_results and "orbital_only" in ci_results:
            f_lo = ci_results["fusion"]["ci_lo"]
            o_hi = ci_results["orbital_only"]["ci_hi"]
            if f_lo > o_hi:
                findings.append(
                    "The fusion model's R² confidence interval does not overlap with "
                    "orbital-only, indicating the improvement is statistically significant "
                    "at the 95% level."
                )
            else:
                findings.append(
                    "The fusion and orbital-only R² confidence intervals overlap, "
                    "so the fusion improvement may not be statistically significant."
                )

        if "orbital_only" in ci_results and "orbital_no_bstar" in ci_results:
            o_lo = ci_results["orbital_only"]["ci_lo"]
            nb_hi = ci_results["orbital_no_bstar"]["ci_hi"]
            if o_lo > nb_hi:
                findings.append(
                    "B* removal causes a statistically significant R² drop "
                    "(non-overlapping 95% confidence intervals)."
                )

        print()
        save_json(ci_results, output_dir / "bootstrap_confidence_intervals.json")
        return findings



#  13. A/M DISTRIBUTION & ERROR ANALYSIS


class AmDistributionAnalyser:
    """Analyse A/m predictions by class and errors by A/m magnitude."""

    def _load_predictions(self, models_dir, model_name):
        p = models_dir / model_name / "test_results.npz"
        return dict(np.load(p, allow_pickle=True)) if p.exists() else None

    def run(self, loader, output_dir):
        print_section("13. A/M DISTRIBUTION & ERROR ANALYSIS")
        findings = []

        preds = self._load_predictions(loader.models_dir, "fusion")
        if preds is None:
            for name in ["orbital_only", "lc_only"]:
                preds = self._load_predictions(loader.models_dir, name)
                if preds is not None:
                    break
        if preds is None:
            print("  [SKIP] No predictions found.\n")
            return findings

        am_p = preds["am_pred"]
        am_t = preds["am_target"]
        am_v = preds["am_valid"].astype(bool)
        cls_t = preds["class_target"]
        cls_v = preds["class_valid"].astype(bool)
        mask = am_v & np.isfinite(am_p) & np.isfinite(am_t)

        # ── A/m distribution by predicted class ──
        cls_p = preds["class_pred"]
        print(f"  Predicted A/m distribution by class (linear m\u00b2/kg):")
        print(f"  {'Class':<15s} {'N':>6s} {'Median':>10s} {'Mean':>10s} {'Std':>10s} {'Range':>20s}")
        print(f"  {'─' * 75}")

        for c in range(3):
            cmask = mask & (cls_p == c)
            if cmask.sum() == 0:
                continue
            am_linear = 10.0 ** am_p[cmask]
            name = CLASS_NAMES.get(c, f"C{c}")
            print(f"  {name:<15s} {cmask.sum():>6d} {np.median(am_linear):>10.4f} "
                  f"{np.mean(am_linear):>10.4f} {np.std(am_linear):>10.4f} "
                  f"[{np.min(am_linear):.4f}, {np.max(am_linear):.4f}]")

        # Physical plausibility check
        payload_pred = 10.0 ** am_p[mask & (cls_p == 0)]
        if len(payload_pred) > 0:
            median_payload = np.median(payload_pred)
            if 0.001 < median_payload < 0.1:
                findings.append(
                    f"Predicted payload A/m (median={median_payload:.4f} m\u00b2/kg) falls "
                    f"within the expected physical range (0.005-0.03 m\u00b2/kg for typical "
                    f"satellites), indicating physically plausible predictions."
                )

        # ── Error by A/m magnitude ──
        print(f"\n  Error by A/m magnitude (log\u2081\u2080 bins):")
        residuals = am_p[mask] - am_t[mask]
        targets = am_t[mask]

        # Bin by target magnitude
        bin_edges = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2]
        print(f"  {'Bin':<20s} {'N':>6s} {'RMSE':>8s} {'MAE':>8s} {'Bias':>8s}")
        print(f"  {'─' * 55}")

        bin_metrics = []
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i+1]
            bmask = (targets >= lo) & (targets < hi)
            n = bmask.sum()
            if n < 5:
                continue
            res = residuals[bmask]
            rmse = np.sqrt(np.mean(res**2))
            mae = np.mean(np.abs(res))
            bias = np.mean(res)
            label = f"[{lo:.1f}, {hi:.1f})"
            print(f"  {label:<20s} {n:>6d} {rmse:>8.4f} {mae:>8.4f} {bias:>+8.4f}")
            bin_metrics.append({"bin": label, "n": int(n), "rmse": rmse,
                               "mae": mae, "bias": bias})

        # HAMR analysis (high A/m objects: log10 > 0, i.e. A/m > 1 m²/kg)
        hamr_mask = mask & (am_t > 0)
        if hamr_mask.sum() > 10:
            hamr_res = am_p[hamr_mask] - am_t[hamr_mask]
            hamr_rmse = np.sqrt(np.mean(hamr_res**2))
            normal_mask = mask & (am_t <= 0)
            normal_res = am_p[normal_mask] - am_t[normal_mask]
            normal_rmse = np.sqrt(np.mean(normal_res**2))
            findings.append(
                f"HAMR objects (A/m > 1 m\u00b2/kg, N={hamr_mask.sum()}) show "
                f"RMSE={hamr_rmse:.4f} vs {normal_rmse:.4f} for normal objects. "
                + ("Higher errors on HAMR objects are expected as these are predominantly "
                   "debris fragments with limited training representation."
                   if hamr_rmse > normal_rmse * 1.2 else
                   "Comparable performance suggests the model generalises across "
                   "the A/m range.")
            )
        print()

        # ── Generate figures ──
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # A/m distribution by predicted class
            fig, ax = plt.subplots(figsize=(10, 5))
            colours = {0: "#2196F3", 1: "#4CAF50", 2: "#F44336"}
            for c in range(3):
                cmask = mask & (cls_p == c)
                if cmask.sum() > 0:
                    ax.hist(am_p[cmask], bins=60, alpha=0.5, color=colours[c],
                           label=f"{CLASS_NAMES[c]} (N={cmask.sum():,})",
                           density=True)
            ax.set_xlabel("Predicted log\u2081\u2080(A/m)")
            ax.set_ylabel("Density")
            ax.set_title("Predicted A/m Distribution by Object Class")
            ax.legend()
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(output_dir / "fig_am_distribution_by_class.png",
                       dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  fig_am_distribution_by_class.png")

            # Error by magnitude
            if bin_metrics:
                fig, ax = plt.subplots(figsize=(10, 5))
                bins = [b["bin"] for b in bin_metrics]
                rmses = [b["rmse"] for b in bin_metrics]
                biases = [b["bias"] for b in bin_metrics]
                x = range(len(bins))
                ax.bar(x, rmses, alpha=0.7, color="steelblue", label="RMSE")
                ax.plot(x, biases, "ro-", markersize=6, label="Bias")
                ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(bins, rotation=30, fontsize=9)
                ax.set_xlabel("True log\u2081\u2080(A/m) bin")
                ax.set_ylabel("Error (log\u2081\u2080 scale)")
                ax.set_title("Prediction Error by A/m Magnitude")
                ax.legend()
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig(output_dir / "fig_error_by_am_magnitude.png",
                           dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"  fig_error_by_am_magnitude.png")

        except Exception as e:
            print(f"  [WARN] Figure generation failed: {e}")

        print()
        save_json({"bin_metrics": bin_metrics}, output_dir / "am_error_analysis.json")
        return findings



#  14. PUBLICATION FIGURES


class FigureGenerator:
    """Generate all publication-quality figures (300 dpi)."""

    def run(self, loader, output_dir):
        print_section("14. PUBLICATION FIGURES")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
        except ImportError:
            print("  [SKIP] matplotlib not available.\n")
            return

        nn = loader.nn_results(source="all")
        bl = loader.baseline_results()

        # ── Fig 1: Full model comparison ──
        all_models = OrderedDict()
        for k, v in nn.items():
            all_models[k] = (safe_get(v, "regression", "r2", default=0),
                             safe_get(v, "classification", "macro_f1", default=0), "NN")
        for k, v in bl.items():
            all_models[k] = (safe_get(v, "regression", "r2", default=0),
                             safe_get(v, "classification", "macro_f1", default=0), "BL")

        if all_models:
            sorted_m = sorted(all_models.items(), key=lambda x: x[1][0], reverse=True)
            names = [x[0][:32] for x in sorted_m]
            r2s = [x[1][0] for x in sorted_m]
            f1s = [x[1][1] for x in sorted_m]
            cats = [x[1][2] for x in sorted_m]
            colors = ["#2196F3" if c == "NN" else "#FF7043" for c in cats]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(5, len(names) * 0.4)))
            y = range(len(names))

            for ax, vals, title, xlabel in [(ax1, r2s, "A/m Regression", "R\u00b2"),
                                             (ax2, f1s, "Classification", "Macro F1")]:
                bars = ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
                ax.set_yticks(y)
                ax.set_yticklabels(names, fontsize=8)
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_title(title, fontsize=12)
                ax.invert_yaxis()
                ax.set_xlim(0, max(vals) * 1.15 if vals else 1)
                ax.grid(True, alpha=0.2, axis="x")
                for bar, val in zip(bars, vals):
                    if val > 0.01:
                        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                                f"{val:.3f}", va="center", fontsize=7)

            legend = [Patch(color="#2196F3", label="Neural Network"),
                     Patch(color="#FF7043", label="Traditional ML")]
            ax1.legend(handles=legend, loc="lower right", fontsize=9)
            fig.tight_layout()
            fig.savefig(output_dir / "fig_model_comparison.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  fig_model_comparison.png")

        # ── Fig 2: Ablation ──
        fusion = loader.get("fusion")
        lc_only = loader.get("lc_only")
        orbital = loader.get("orbital_only")
        no_bstar = loader.get("orbital_no_bstar")

        if fusion and lc_only and orbital:
            models = OrderedDict([
                ("Fusion", fusion), ("Orbital", orbital),
                ("LC-only", lc_only)])
            if no_bstar:
                models["Orbital\n(no B*)"] = no_bstar

            names = list(models.keys())
            r2s = [safe_get(v, "regression", "r2", default=0) for v in models.values()]
            f1s = [safe_get(v, "classification", "macro_f1", default=0) for v in models.values()]
            colours = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"][:len(names)]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            x = range(len(names))
            for ax, vals, title, ylabel in [(ax1, r2s, "A/m Regression", "R\u00b2"),
                                             (ax2, f1s, "Classification", "Macro F1")]:
                bars = ax.bar(x, vals, color=colours, alpha=0.85, edgecolor="white")
                ax.set_xticks(x)
                ax.set_xticklabels(names, fontsize=10)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.set_ylim(0, max(vals) * 1.25 if vals else 1)
                for i, v in enumerate(vals):
                    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
            fig.suptitle("Ablation Study", fontsize=13)
            fig.tight_layout()
            fig.savefig(output_dir / "fig_ablation.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  fig_ablation.png")

        # ── Fig 3: Per-class F1 heatmap ──
        per_class_data = {}
        for label, r in loader.results.items():
            if r.get("_source") != "all":
                continue
            pc = normalise_per_class(safe_get(r, "classification", "per_class"))
            if pc:
                per_class_data[label] = {cls: (m.get("f1") if isinstance(m, dict) else None)
                                         for cls, m in pc.items()}

        if per_class_data:
            model_names = list(per_class_data.keys())
            class_names = list(CLASS_NAMES.values())
            matrix = np.zeros((len(model_names), len(class_names)))
            for i, model in enumerate(model_names):
                for j, cls in enumerate(class_names):
                    matrix[i, j] = per_class_data[model].get(cls) or 0

            fig, ax = plt.subplots(figsize=(8, max(4, len(model_names) * 0.45)))
            im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_names, fontsize=10)
            ax.set_yticks(range(len(model_names)))
            ax.set_yticklabels([n[:35] for n in model_names], fontsize=8)
            for i in range(len(model_names)):
                for j in range(len(class_names)):
                    ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                           fontsize=8, color="white" if matrix[i,j] > 0.5 else "black")
            fig.colorbar(im, label="F1 Score")
            ax.set_title("Per-Class F1 Scores Across Models")
            fig.tight_layout()
            fig.savefig(output_dir / "fig_per_class_heatmap.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  fig_per_class_heatmap.png")

        print(f"\n  All figures saved to: {output_dir}\n")



#  11. REPORT GENERATOR


class ReportGenerator:
    """Generate structured Markdown analysis report."""

    def generate(self, loader, all_findings, output_dir):
        print_section("11. ANALYSIS REPORT")

        lines = []
        lines.append("# Results Analysis Report")
        lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M}")
        lines.append(f"Data directory: `{loader.data_dir}`\n")

        # Numbered key findings
        flat_findings = []
        for section_name, section_findings in all_findings.items():
            flat_findings.extend(section_findings)

        lines.append("## Executive Summary\n")
        lines.append(f"This analysis covers {len(loader.results)} model evaluations "
                    f"({sum(1 for r in loader.results.values() if r['_category'] == 'neural_network')} "
                    f"neural network, {sum(1 for r in loader.results.values() if r['_category'] == 'baseline')} "
                    f"traditional ML baselines) with {len(flat_findings)} key findings.\n")

        lines.append("## Key Findings\n")
        for i, finding in enumerate(flat_findings, 1):
            lines.append(f"{i}. {finding}\n")

        # Full comparison table
        lines.append("\n## Model Comparison Table\n")
        lines.append("| Model | Category | R\u00b2 | RMSE | MAE | Accuracy | Macro F1 |")
        lines.append("|-------|----------|-----|------|-----|----------|----------|")

        sorted_results = sorted(
            [(k, v) for k, v in loader.results.items() if v.get("_source") == "all"],
            key=lambda x: safe_get(x[1], "regression", "r2", default=-1),
            reverse=True)
        for label, r in sorted_results:
            cat = r.get("_category", "?")
            vals = [safe_get(r, "regression", "r2"),
                    safe_get(r, "regression", "rmse_log"),
                    safe_get(r, "regression", "mae_log"),
                    safe_get(r, "classification", "accuracy"),
                    safe_get(r, "classification", "macro_f1")]
            vals_str = " | ".join(fmt(v) for v in vals)
            lines.append(f"| {label} | {cat} | {vals_str} |")

        # Per-section details
        section_titles = {
            "dataset": "Dataset Statistics",
            "ablation": "Ablation Analysis",
            "bstar": "B* Dependency Investigation",
            "cross_source": "Cross-Source Generalisation",
            "dl_vs_baselines": "Deep Learning vs Traditional ML",
            "training_dynamics": "Training Dynamics",
            "per_class": "Per-Class Performance",
            "xai": "XAI Synthesis",
            "efficiency": "Model Efficiency",
            "predictions": "Prediction Analysis",
            "orbit_regime": "Performance by Orbit Regime",
            "bootstrap": "Statistical Significance",
            "am_distribution": "A/m Distribution and Error Analysis",
        }

        for key, title in section_titles.items():
            section_findings = all_findings.get(key, [])
            if section_findings:
                lines.append(f"\n## {title}\n")
                for finding in section_findings:
                    lines.append(f"- {finding}\n")

        # Limitations
        lines.append("\n## Limitations\n")
        limitations = [
            "The training set comprises approximately 90% payloads (intact satellites). "
            "Model performance on debris fragments, the operationally most important category, "
            "is limited by severe class imbalance.",

            "Cross-source generalisation testing is confounded by orbit regime: MMT-9 observes "
            "predominantly LEO objects whilst SDLCD observes predominantly GEO. Performance "
            "degradation may reflect physical differences between orbit regimes rather than "
            "telescope-specific biases.",

            "DISCOS ground truth A/m values derive from physical characterisation (launch records, "
            "manufacturer specifications, radar sizing) and are independent from the B* drag "
            "coefficient used as a model input. However, ground truth quality varies by object, "
            "with older or less well-documented objects having less reliable A/m estimates.",

            "The model is trained on objects with known identities in the DISCOS catalogue. "
            "Operational utility for truly uncharacterised objects (e.g., newly detected debris) "
            "remains unvalidated.",
        ]
        for lim in limitations:
            lines.append(f"- {lim}\n")

        # Recommendations
        lines.append("\n## Recommendations for Future Work\n")
        recommendations = [
            "Address class imbalance through targeted debris data collection or "
            "synthetic data augmentation.",
            "Validate predictions on objects outside the DISCOS catalogue using "
            "independent radar or photometric measurements.",
            "Investigate attention-based architectures (transformers) for the light "
            "curve branch, following recent success in the RoBo6 benchmark.",
            "Add uncertainty quantification (MC Dropout or ensemble methods) to "
            "provide confidence intervals alongside point predictions.",
        ]
        for rec in recommendations:
            lines.append(f"- {rec}\n")

        # Write
        report_text = "\n".join(lines)
        report_path = output_dir / "analysis_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"  Report: {report_path}")
        print(f"  Findings: {len(flat_findings)}")
        print(f"  Length: {len(lines)} lines\n")
        return report_path



#  ORCHESTRATOR


def run_full_analysis(data_dir, output_suffix=None, exclude_pattern=None):
    """Run all analyses and generate the complete report."""
    data_dir = Path(data_dir)
    dir_name = "analysis" if not output_suffix else f"analysis_{output_suffix}"
    output_dir = data_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load everything
    loader = ResultsLoader(data_dir).load()
    if not loader.results:
        print("  [ERROR] No results found. Run training, evaluation, and baselines first.\n")
        return

    # Filter out models matching exclude pattern
    if exclude_pattern:
        before = len(loader.results)
        loader.results = {k: v for k, v in loader.results.items()
                          if exclude_pattern not in k}
        after = len(loader.results)
        if before != after:
            print(f"  Excluded {before - after} models matching '{exclude_pattern}' "
                  f"({after} remaining)\n")

    all_findings = OrderedDict()

    # Run each analyser
    all_findings["dataset"] = DatasetAnalyser(data_dir).run(output_dir)
    all_findings["ablation"] = AblationAnalyser().run(loader, output_dir)
    all_findings["bstar"] = BStarAnalyser().run(loader, output_dir)
    all_findings["cross_source"] = CrossSourceAnalyser().run(loader, output_dir)
    all_findings["dl_vs_baselines"] = BaselineComparer().run(loader, output_dir)
    all_findings["training_dynamics"] = TrainingDynamicsAnalyser().run(loader, output_dir)
    all_findings["per_class"] = ClassAnalyser().run(loader, output_dir)
    all_findings["xai"] = XAIAnalyser().run(loader, output_dir)
    all_findings["efficiency"] = EfficiencyAnalyser().run(loader, output_dir)
    all_findings["predictions"] = PredictionAnalyser().run(loader, output_dir)
    all_findings["orbit_regime"] = OrbitRegimeAnalyser().run(loader, output_dir)
    all_findings["bootstrap"] = BootstrapAnalyser().run(loader, output_dir)
    all_findings["am_distribution"] = AmDistributionAnalyser().run(loader, output_dir)
    FigureGenerator().run(loader, output_dir)
    report_path = ReportGenerator().generate(loader, all_findings, output_dir)

    # Final summary
    total = sum(len(f) for f in all_findings.values())
    print_section("ANALYSIS COMPLETE")
    print(f"  Output:   {output_dir}")
    print(f"  Report:   {report_path}")
    print(f"  Findings: {total}")
    print(f"  Figures:  {len(list(output_dir.glob('fig_*.png')))} PNGs (300 dpi)")
    print(f"  Tables:   {len(list(output_dir.glob('*.csv')))} CSVs")
    print(f"  Data:     {len(list(output_dir.glob('*.json')))} JSONs\n")



#  MENU / CLI


def run_interactive(data_dir):
    print_header("Results Analysis", __version__, "Comprehensive model comparison and insights")
    data_dir = Path(data_dir)
    output_dir = data_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ResultsLoader(data_dir).load()
    if not loader.results:
        print("  No results found.\n")
        return

    while True:
        choice = prompt_choice("Select analysis:", [
            ("full",       "Full analysis (all sections + report)"),
            ("comparison", "Model comparison table"),
            ("ablation",   "Ablation analysis"),
            ("bstar",      "B* investigation"),
            ("cross",      "Cross-source generalisation"),
            ("baselines",  "DL vs traditional ML"),
            ("dynamics",   "Training dynamics"),
            ("classes",    "Per-class breakdown"),
            ("xai",        "XAI synthesis"),
            ("efficiency", "Model efficiency"),
            ("scatter",    "Scatter plots + residuals + confusion matrices"),
            ("orbit",      "Performance by orbit regime"),
            ("bootstrap",  "Bootstrap confidence intervals"),
            ("am",         "A/m distribution + error analysis"),
            ("figures",    "Generate comparison figures only"),
            ("quit",       "Quit"),
        ])

        if choice in (None, "quit"):
            print("\n  Goodbye!\n")
            break
        elif choice == "full":
            run_full_analysis(data_dir)
        else:
            analysers = {
                "comparison": lambda: print("  Use 'full' for the comparison table."),
                "ablation":   lambda: AblationAnalyser().run(loader, output_dir),
                "bstar":      lambda: BStarAnalyser().run(loader, output_dir),
                "cross":      lambda: CrossSourceAnalyser().run(loader, output_dir),
                "baselines":  lambda: BaselineComparer().run(loader, output_dir),
                "dynamics":   lambda: TrainingDynamicsAnalyser().run(loader, output_dir),
                "classes":    lambda: ClassAnalyser().run(loader, output_dir),
                "xai":        lambda: XAIAnalyser().run(loader, output_dir),
                "efficiency": lambda: EfficiencyAnalyser().run(loader, output_dir),
                "scatter":    lambda: PredictionAnalyser().run(loader, output_dir),
                "orbit":      lambda: OrbitRegimeAnalyser().run(loader, output_dir),
                "bootstrap":  lambda: BootstrapAnalyser().run(loader, output_dir),
                "am":         lambda: AmDistributionAnalyser().run(loader, output_dir),
                "figures":    lambda: FigureGenerator().run(loader, output_dir),
            }
            analysers[choice]()


def main():
    parser = argparse.ArgumentParser(description="Space Debris ML - Results Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Examples:
  python analysis.py --all
  python analysis.py --all --data-dir path/to/training_1024
  python analysis.py --all --suffix baseline --exclude _aug
  python analysis.py --all --suffix aug""")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None,
                        help="Append suffix to output dir (e.g. 'baseline' -> analysis_baseline/)")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Exclude models containing this string (e.g. '_aug')")
    args = parser.parse_args()
    ensure_dirs()
    data_dir = Path(args.data_dir) if args.data_dir else TRAINING_DIR

    try:
        if args.all:
            print_header("Results Analysis", __version__)
            if args.suffix:
                print(f"  Output suffix: {args.suffix}")
            if args.exclude:
                print(f"  Excluding models matching: {args.exclude}")
            run_full_analysis(data_dir, output_suffix=args.suffix,
                              exclude_pattern=args.exclude)
        else:
            run_interactive(data_dir)
    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        sys.exit(2)

if __name__ == "__main__":
    main()
