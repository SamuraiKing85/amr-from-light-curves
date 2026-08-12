"""
Model Evaluation Script
==========================
Evaluates trained models on the test set with comprehensive metrics.

Computes:
  - A/m regression: RMSE, MAE, R², residual analysis
  - Classification: accuracy, per-class precision/recall/F1, confusion matrix
  - Core ablation comparison (fusion vs lc_only vs orbital_only)
  - B* investigation (orbital_only vs orbital_no_bstar)
  - Cross-source generalisation matrix

Usage:
    python evaluate.py                               # Interactive menu
    python evaluate.py --model fusion                # Evaluate fusion model
    python evaluate.py --ablation                    # Core ablation comparison
    python evaluate.py --cross-source                # Cross-source matrix
    python evaluate.py --full                        # Everything
"""

__version__ = "3.0.0"

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import TRAINING_DIR, ensure_dirs
from common.menu import (
    print_header, prompt_choice, print_section, print_summary_table,
    print_status_bar,
)
from common.device import (
    get_device, print_device_info, move_batch_to_device,
    add_device_args, device_from_args,
)
from common.logging_setup import setup_logging

from models.fusion_model import create_model
from train import SpaceObjectDataset, load_metadata


#  SETUP


logger = setup_logging("evaluate")

CLASS_NAMES = {0: "Payload", 1: "RocketBody", 2: "Debris", 3: "Unknown"}



#  INFERENCE


@torch.no_grad()
def predict(model, dataloader, device) -> dict:
    """Run inference on a dataset."""
    model.eval()
    all_am_pred, all_am_target, all_am_valid = [], [], []
    all_class_pred, all_class_target, all_class_valid = [], [], []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        am_pred, class_pred = model(
            batch["lc_magnitudes"], batch["lc_masks"], batch["orbital_features"]
        )

        all_am_pred.append(am_pred.squeeze(-1).cpu().numpy())
        all_am_target.append(batch["am_target"].cpu().numpy())
        all_am_valid.append(batch["am_valid"].cpu().numpy())

        all_class_pred.append(class_pred.argmax(dim=1).cpu().numpy())
        all_class_target.append(batch["class_target"].cpu().numpy())
        all_class_valid.append(batch["class_valid"].cpu().numpy())

    return {
        "am_pred": np.concatenate(all_am_pred),
        "am_target": np.concatenate(all_am_target),
        "am_valid": np.concatenate(all_am_valid).astype(bool),
        "class_pred": np.concatenate(all_class_pred),
        "class_target": np.concatenate(all_class_target),
        "class_valid": np.concatenate(all_class_valid).astype(bool),
    }



#  REGRESSION METRICS


def compute_regression_metrics(pred, target, valid) -> dict:
    """Compute A/m regression metrics on valid samples."""
    mask = valid & np.isfinite(pred) & np.isfinite(target)
    if mask.sum() == 0:
        return {"n_samples": 0, "rmse_log": np.nan, "mae_log": np.nan, "r2": np.nan}

    p, t = pred[mask], target[mask]
    n = len(p)

    residuals = p - t
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    pred_am = 10.0 ** p
    target_am = 10.0 ** t
    am_residuals = pred_am - target_am
    rmse_linear = np.sqrt(np.mean(am_residuals ** 2))
    mae_linear = np.mean(np.abs(am_residuals))
    mape = np.median(np.abs(am_residuals) / np.maximum(target_am, 1e-10)) * 100

    return {
        "n_samples": n,
        "rmse_log": round(rmse, 4),
        "mae_log": round(mae, 4),
        "r2": round(r2, 4),
        "rmse_linear": round(rmse_linear, 6),
        "mae_linear": round(mae_linear, 6),
        "median_pct_error": round(mape, 2),
        "residual_mean": round(np.mean(residuals), 4),
        "residual_std": round(np.std(residuals), 4),
    }



#  CLASSIFICATION METRICS


def compute_classification_metrics(pred, target, valid, n_classes=3) -> dict:
    """Compute classification metrics on valid samples."""
    mask = valid & (target >= 0) & (target < n_classes)
    if mask.sum() == 0:
        return {"n_samples": 0, "accuracy": np.nan}

    p, t = pred[mask], target[mask]
    n = len(p)

    accuracy = (p == t).mean()

    per_class = {}
    for cls_id in range(n_classes):
        cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
        tp = ((p == cls_id) & (t == cls_id)).sum()
        fp = ((p == cls_id) & (t != cls_id)).sum()
        fn = ((p != cls_id) & (t == cls_id)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = (t == cls_id).sum()

        per_class[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support),
        }

    f1s = [v["f1"] for v in per_class.values() if v["support"] > 0]
    macro_f1 = np.mean(f1s) if f1s else 0.0

    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(n):
        if 0 <= t[i] < n_classes and 0 <= p[i] < n_classes:
            confusion[t[i], p[i]] += 1

    return {
        "n_samples": n,
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }



#  DISPLAY


def print_regression_results(metrics: dict, model_name: str = ""):
    title = f"A/M REGRESSION -- {model_name}" if model_name else "A/M REGRESSION"
    print_section(title)
    if metrics.get("n_samples", 0) == 0:
        print("  No valid regression samples.\n")
        return
    print_summary_table([
        ("Samples:",              f"{metrics['n_samples']:,}"),
        ("RMSE (log10 A/m):",     f"{metrics['rmse_log']:.4f}"),
        ("MAE (log10 A/m):",      f"{metrics['mae_log']:.4f}"),
        ("R\u00b2:",              f"{metrics['r2']:.4f}"),
        ("RMSE (linear A/m):",    f"{metrics['rmse_linear']:.6f}"),
        ("MAE (linear A/m):",     f"{metrics['mae_linear']:.6f}"),
        ("Median % error:",       f"{metrics['median_pct_error']:.1f}%"),
        ("Residual bias:",        f"{metrics['residual_mean']:.4f}"),
    ])
    print()


def print_classification_results(metrics: dict, model_name: str = ""):
    title = f"CLASSIFICATION -- {model_name}" if model_name else "CLASSIFICATION"
    print_section(title)
    if metrics.get("n_samples", 0) == 0:
        print("  No valid classification samples.\n")
        return
    print_summary_table([
        ("Samples:",   f"{metrics['n_samples']:,}"),
        ("Accuracy:",  f"{metrics['accuracy']:.4f}"),
        ("Macro F1:",  f"{metrics['macro_f1']:.4f}"),
    ])

    print(f"\n  {'Class':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print(f"  {'─' * 60}")
    for cls_name, vals in metrics["per_class"].items():
        print(f"  {cls_name:<20s} {vals['precision']:>10.4f} {vals['recall']:>10.4f} "
              f"{vals['f1']:>10.4f} {vals['support']:>10d}")

    cm = np.array(metrics["confusion_matrix"])
    n = cm.shape[0]
    if n > 0:
        labels = [CLASS_NAMES.get(i, f"C{i}")[:8] for i in range(n)]
        print(f"\n  Confusion matrix (rows=true, cols=pred):")
        print(f"  {'':>12s}  " + "  ".join(f"{l:>8s}" for l in labels))
        for i in range(n):
            row_label = CLASS_NAMES.get(i, f"C{i}")[:12]
            print(f"  {row_label:>12s}  " + "  ".join(f"{cm[i,j]:>8d}" for j in range(n)))
    print()



#  SINGLE MODEL EVALUATION


def evaluate_model(model_type: str, model_dir: Path, data_dir: Path,
                   device: torch.device, source_filter: str = None,
                   exclude_bstar: bool = False, log_fn=print) -> dict:
    """Evaluate a single trained model on a test set.

    Args:
        model_type:     Architecture name (fusion/lc_only/orbital_only).
        model_dir:      Directory containing best_model.pt or final_model.pt.
        data_dir:       Directory containing test.parquet + metadata.
        device:         Compute device.
        source_filter:  If set, evaluate only on this source's test data.
        exclude_bstar:  If True, exclude B* features from orbital input.
        log_fn:         Logging function.

    Returns:
        Dict with model_type, regression metrics, classification metrics.
    """
    orbital_feature_cols, lc_length, n_classes = load_metadata(data_dir)
    if orbital_feature_cols is None:
        log_fn("  [ERROR] No metadata found.")
        return {}

    # Load test dataset with optional source filtering
    try:
        test_ds = SpaceObjectDataset(data_dir, "test", orbital_feature_cols,
                                     source_filter=source_filter,
                                     exclude_bstar=exclude_bstar)
    except ValueError as e:
        log_fn(f"  [WARN] {e}")
        return {}

    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    source_label = f" [{source_filter}]" if source_filter else ""
    log_fn(f"  Test set: {len(test_ds):,} samples{source_label}")

    # Load model checkpoint
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "final_model.pt"
    if not checkpoint_path.exists():
        log_fn(f"  [ERROR] No model checkpoint found in {model_dir}")
        return {}

    # Read experiment config to get correct orbital_dim
    exp_config_path = model_dir / "experiment_config.json"
    if exp_config_path.exists():
        with open(exp_config_path) as f:
            exp_config = json.load(f)
        orbital_dim = exp_config.get("orbital_dim", test_ds.orbital_features.shape[1])
    else:
        orbital_dim = test_ds.orbital_features.shape[1]

    model = create_model(
        model_type,
        lc_length=lc_length,
        orbital_dim=orbital_dim,
        n_classes=n_classes,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    log_fn(f"  Loaded: {checkpoint_path.name} (epoch {ckpt.get('epoch', '?')})")

    # Run inference
    predictions = predict(model, test_loader, device)

    # Add metadata from the test dataset for downstream analysis
    if hasattr(test_ds, "df"):
        if "source" in test_ds.df.columns:
            predictions["source"] = test_ds.df["source"].values
        if "object_type" in test_ds.df.columns:
            predictions["object_type"] = test_ds.df["object_type"].values
        if "norad_id" in test_ds.df.columns:
            predictions["norad_id"] = test_ds.df["norad_id"].values

    # Compute metrics
    reg_metrics = compute_regression_metrics(
        predictions["am_pred"], predictions["am_target"], predictions["am_valid"]
    )
    cls_metrics = compute_classification_metrics(
        predictions["class_pred"], predictions["class_target"],
        predictions["class_valid"], n_classes,
    )

    return {
        "model_type": model_type,
        "regression": reg_metrics,
        "classification": cls_metrics,
        "predictions": predictions,
    }


def save_results(result: dict, save_path: Path):
    """Save evaluation results to JSON + raw predictions as .npz."""
    # JSON metrics
    saveable = {
        "model_type": result["model_type"],
        "regression": result["regression"],
        "classification": {
            k: v for k, v in result["classification"].items()
            if k != "confusion_matrix"
        },
        "confusion_matrix": result["classification"].get("confusion_matrix"),
    }
    with open(save_path, "w") as f:
        json.dump(saveable, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    # Raw predictions as .npz (for analysis.py scatter plots, residuals, etc.)
    predictions = result.get("predictions")
    if predictions:
        npz_path = save_path.with_suffix(".npz")
        save_dict = {
            "am_pred": predictions["am_pred"],
            "am_target": predictions["am_target"],
            "am_valid": predictions["am_valid"],
            "class_pred": predictions["class_pred"],
            "class_target": predictions["class_target"],
            "class_valid": predictions["class_valid"],
        }
        # Include metadata arrays if available
        for key in ["source", "object_type", "norad_id"]:
            if key in predictions:
                save_dict[key] = np.array(predictions[key], dtype=str)
        np.savez_compressed(npz_path, **save_dict)



#  CORE ABLATION COMPARISON


def run_core_ablation(data_dir: Path, models_dir: Path, device: torch.device,
                      log_fn=print) -> list:
    """Evaluate core models (fusion, lc_only, orbital_only) and compare."""
    results = []

    for model_type in ["fusion", "lc_only", "orbital_only", "orbital_no_bstar"]:
        model_dir = models_dir / (model_type if model_type != "orbital_no_bstar"
                                  else "orbital_no_bstar")
        if not model_dir.exists():
            continue

        # Determine if this experiment used B* exclusion
        exclude_bstar = model_type == "orbital_no_bstar"
        actual_arch = "orbital_only" if exclude_bstar else model_type

        log_fn(f"\n  Evaluating {model_type}...")
        result = evaluate_model(actual_arch, model_dir, data_dir, device,
                                exclude_bstar=exclude_bstar, log_fn=log_fn)
        if result:
            result["experiment_name"] = model_type
            results.append(result)
            save_results(result, model_dir / "test_results.json")

    _print_comparison_table(results, "CORE ABLATION COMPARISON")
    return results



#  CROSS-SOURCE EVALUATION


def run_cross_source(data_dir: Path, models_dir: Path, device: torch.device,
                     log_fn=print) -> list:
    """Evaluate cross-source generalisation for LC and orbital models.

    For each source-filtered model, evaluates on:
      - Same source test set (within-source baseline)
      - Other source test set (cross-source generalisation)
    """
    results = []

    # (model_dir_name, architecture, train_source, test_source, label)
    evaluations = [
        # LC within-source baselines
        ("lc_only_mmt9",   "lc_only",      "MMT9",  "MMT9",  "LC [MMT9->MMT9]"),
        ("lc_only_sdlcd",  "lc_only",      "SDLCD", "SDLCD", "LC [SDLCD->SDLCD]"),
        # LC cross-source
        ("lc_only_mmt9",   "lc_only",      "MMT9",  "SDLCD", "LC [MMT9->SDLCD]"),
        ("lc_only_sdlcd",  "lc_only",      "SDLCD", "MMT9",  "LC [SDLCD->MMT9]"),
        # Orbital within-source baselines
        ("orbital_only_mmt9",  "orbital_only", "MMT9",  "MMT9",  "Orbital [MMT9->MMT9]"),
        ("orbital_only_sdlcd", "orbital_only", "SDLCD", "SDLCD", "Orbital [SDLCD->SDLCD]"),
        # Orbital cross-source
        ("orbital_only_mmt9",  "orbital_only", "MMT9",  "SDLCD", "Orbital [MMT9->SDLCD]"),
        ("orbital_only_sdlcd", "orbital_only", "SDLCD", "MMT9",  "Orbital [SDLCD->MMT9]"),
    ]

    for dir_name, arch, train_src, test_src, label in evaluations:
        model_dir = models_dir / dir_name
        if not model_dir.exists():
            log_fn(f"  [SKIP] {label}: model not found at {model_dir}")
            continue

        log_fn(f"\n  {label}...")
        result = evaluate_model(arch, model_dir, data_dir, device,
                                source_filter=test_src, log_fn=log_fn)
        if result:
            result["experiment_name"] = label
            result["train_source"] = train_src
            result["test_source"] = test_src
            results.append(result)

            # Save with source-specific filename
            save_path = model_dir / f"test_results_{test_src.lower()}.json"
            save_results(result, save_path)

    _print_comparison_table(results, "CROSS-SOURCE GENERALISATION")

    # Print the key insight
    if len(results) >= 4:
        print("  KEY QUESTION: Does LC performance drop cross-source?")
        lc_within = [r for r in results if "LC" in r["experiment_name"]
                     and r.get("train_source") == r.get("test_source")]
        lc_cross = [r for r in results if "LC" in r["experiment_name"]
                    and r.get("train_source") != r.get("test_source")]
        orb_cross = [r for r in results if "Orbital" in r["experiment_name"]]

        if lc_within and lc_cross:
            avg_within = np.mean([r["regression"]["r2"] for r in lc_within
                                  if r["regression"].get("r2") is not None
                                  and not np.isnan(r["regression"]["r2"])])
            avg_cross = np.mean([r["regression"]["r2"] for r in lc_cross
                                 if r["regression"].get("r2") is not None
                                 and not np.isnan(r["regression"]["r2"])])
            print(f"  LC within-source avg R2: {avg_within:.4f}")
            print(f"  LC cross-source avg R2:  {avg_cross:.4f}")

            if avg_within > 0 and avg_cross > 0:
                drop = (1 - avg_cross / avg_within) * 100
                print(f"  Cross-source R2 drop:    {drop:.1f}%")
        print()

    return results



#  COMPARISON TABLE


def _print_comparison_table(results: list, title: str):
    """Print a comparison table for a list of evaluation results."""
    if len(results) < 2:
        return

    print_section(title)
    print(f"  {'Experiment':<24s} {'N':>7s} {'RMSE':>8s} {'MAE':>8s} "
          f"{'R\u00b2':>8s} {'Acc':>8s} {'F1':>8s}")
    print(f"  {'─' * 73}")

    for r in results:
        name = r.get("experiment_name", r["model_type"])
        reg = r["regression"]
        cls = r["classification"]
        n = reg.get("n_samples", 0)
        rmse = f"{reg['rmse_log']:.4f}" if not np.isnan(reg.get('rmse_log', np.nan)) else "   n/a"
        mae = f"{reg['mae_log']:.4f}" if not np.isnan(reg.get('mae_log', np.nan)) else "   n/a"
        r2 = f"{reg['r2']:.4f}" if not np.isnan(reg.get('r2', np.nan)) else "   n/a"
        acc = f"{cls['accuracy']:.4f}" if not np.isnan(cls.get('accuracy', np.nan)) else "   n/a"
        f1 = f"{cls['macro_f1']:.4f}" if not np.isnan(cls.get('macro_f1', np.nan)) else "   n/a"
        print(f"  {name:<24s} {n:>7,} {rmse:>8s} {mae:>8s} "
              f"{r2:>8s} {acc:>8s} {f1:>8s}")
    print()



#  INTERACTIVE MENU


def run_interactive(data_dir: Path, models_dir: Path):
    """Run the interactive menu."""
    print_header("Model Evaluation", __version__,
                 "Evaluate trained models on test data")

    device = get_device()
    print_device_info(device)

    while True:
        # Auto-detect available models
        available = []
        for dirname in sorted(models_dir.iterdir()) if models_dir.exists() else []:
            if dirname.is_dir() and (
                (dirname / "best_model.pt").exists() or
                (dirname / "final_model.pt").exists()
            ):
                available.append(dirname.name)

        options = []

        if available:
            # Individual model evaluation
            for name in available:
                options.append((name, f"Evaluate: {name}"))

            if len(available) >= 2:
                options.append(("core_ablation", "Core ablation comparison"))

            # Cross-source if those models exist
            cross_models = {"lc_only_mmt9", "lc_only_sdlcd"}
            if cross_models.issubset(set(available)):
                options.append(("cross_source", "Cross-source generalisation analysis"))

            if len(available) >= 4:
                options.append(("full", "Full evaluation (all models + cross-source)"))

            options.append(("xai", "Explainable AI (Grad-CAM + feature importance)"))

        options.append(("quit", "Quit"))

        if not available:
            print("  No trained models found. Run 4_train/train.py first.\n")
            break

        choice = prompt_choice("What would you like to evaluate?", options)

        if choice is None or choice == "quit":
            print("\n  Goodbye!\n")
            break

        elif choice == "core_ablation":
            run_core_ablation(data_dir, models_dir, device)

        elif choice == "cross_source":
            run_cross_source(data_dir, models_dir, device)

        elif choice == "full":
            print_section("FULL EVALUATION")
            run_core_ablation(data_dir, models_dir, device)
            run_cross_source(data_dir, models_dir, device)

        elif choice == "xai":
            try:
                from explain import explain_all_models
                explain_all_models(data_dir, models_dir, device)
            except ImportError as e:
                print(f"  [ERROR] Could not load explain module: {e}\n")

        elif choice in available:
            # Determine architecture from experiment_config.json
            model_dir = models_dir / choice
            exp_cfg_path = model_dir / "experiment_config.json"
            if exp_cfg_path.exists():
                with open(exp_cfg_path) as f:
                    exp_cfg = json.load(f)
                arch = exp_cfg.get("model_type", choice.split("_")[0])
                src = exp_cfg.get("source_filter")
                no_bstar = exp_cfg.get("exclude_bstar", False)
            else:
                arch = choice if choice in ["fusion", "lc_only", "orbital_only"] else "orbital_only"
                src = None
                no_bstar = "no_bstar" in choice

            result = evaluate_model(arch, model_dir, data_dir, device,
                                    source_filter=src, exclude_bstar=no_bstar)
            if result:
                print_regression_results(result["regression"], choice)
                print_classification_results(result["classification"], choice)
                save_results(result, model_dir / "test_results.json")
                print(f"  Results saved: {model_dir / 'test_results.json'}\n")



#  CLI MODE


def run_discovered_models(data_dir: Path, models_dir: Path, device: torch.device,
                          log_fn=print) -> list:
    """Auto-discover and evaluate all model directories.

    Finds any subdirectory with experiment_config.json + a checkpoint,
    reads the config to determine architecture and settings, and evaluates.
    This handles _aug suffixed models and any other variants.
    """
    results = []
    if not models_dir.exists():
        log_fn(f"  [SKIP] Models directory not found: {models_dir}")
        return results

    # Find all model directories with a config and checkpoint
    discovered = []
    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
        config_path = subdir / "experiment_config.json"
        has_checkpoint = (subdir / "best_model.pt").exists() or (subdir / "final_model.pt").exists()
        has_results = (subdir / "test_results.json").exists()

        if config_path.exists() and has_checkpoint and not has_results:
            with open(config_path) as f:
                cfg = json.load(f)
            discovered.append((subdir.name, cfg))

    if not discovered:
        return results

    log_fn(f"\n  Auto-discovered {len(discovered)} unevaluated model(s)")

    for dir_name, cfg in discovered:
        model_type = cfg.get("model_type", "fusion")
        source_filter = cfg.get("source_filter")
        exclude_bstar = cfg.get("exclude_bstar", False)

        log_fn(f"\n  Evaluating {dir_name} (arch={model_type}, "
               f"src={source_filter}, bstar={not exclude_bstar})...")

        model_dir = models_dir / dir_name
        result = evaluate_model(model_type, model_dir, data_dir, device,
                                source_filter=source_filter,
                                exclude_bstar=exclude_bstar, log_fn=log_fn)
        if result:
            result["experiment_name"] = dir_name
            results.append(result)
            save_results(result, model_dir / "test_results.json")

    if results:
        _print_comparison_table(results, "AUTO-DISCOVERED MODELS")
    return results


def run_cli(args, data_dir: Path, models_dir: Path):
    """Run in CLI mode."""
    print_header("Model Evaluation", __version__)

    device = device_from_args(args)
    print_device_info(device)

    if args.full:
        run_core_ablation(data_dir, models_dir, device)
        run_cross_source(data_dir, models_dir, device)
        # Auto-discover any remaining models (e.g. _aug variants)
        run_discovered_models(data_dir, models_dir, device)
    elif args.ablation:
        run_core_ablation(data_dir, models_dir, device)
    elif args.cross_source:
        run_cross_source(data_dir, models_dir, device)
    elif args.model:
        model_dir = models_dir / args.model
        if not model_dir.exists():
            print(f"  [ERROR] No trained model at {model_dir}")
            sys.exit(1)

        result = evaluate_model(args.model, model_dir, data_dir, device,
                                source_filter=args.source)
        if result:
            print_regression_results(result["regression"], args.model)
            print_classification_results(result["classification"], args.model)



#  MAIN


def main():
    parser = argparse.ArgumentParser(
        description="Space Debris ML - Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py                           # Interactive menu
  python evaluate.py --model fusion            # Evaluate fusion model
  python evaluate.py --ablation                # Core ablation comparison
  python evaluate.py --cross-source            # Cross-source matrix
  python evaluate.py --full                    # Everything
        """,
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Model directory name to evaluate")
    parser.add_argument("--source", type=str, default=None,
                        choices=["MMT9", "SDLCD"],
                        help="Evaluate only on this source's test data")
    parser.add_argument("--ablation", action="store_true",
                        help="Run core ablation comparison")
    parser.add_argument("--cross-source", action="store_true",
                        help="Run cross-source generalisation analysis")
    parser.add_argument("--full", action="store_true",
                        help="Run all evaluations")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    add_device_args(parser)

    args = parser.parse_args()

    ensure_dirs()
    data_dir = Path(args.data_dir) if args.data_dir else TRAINING_DIR
    models_dir = Path(args.models_dir) if args.models_dir else TRAINING_DIR / "models"

    has_action = args.model or args.ablation or args.cross_source or args.full

    try:
        if has_action:
            run_cli(args, data_dir, models_dir)
        else:
            run_interactive(data_dir, models_dir)
    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        sys.exit(2)


if __name__ == "__main__":
    main()
