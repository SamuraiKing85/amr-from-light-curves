"""
Augmentation Comparison Report
==================================
Compares non-augmented vs augmented model results side by side.
Produces a comparison table, per-class breakdown, and summary figures.

Usage:
    python compare_augmentation.py
    python compare_augmentation.py --data-dir path/to/training
    python compare_augmentation.py --suffix aug --data-dir path/to/training
"""

__version__ = "1.0.0"

import sys
import json
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import TRAINING_DIR, ensure_dirs
from common.menu import print_header, print_section

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Models to compare (base_name, description)
CORE_MODELS = [
    ("fusion",         "Fusion (LC + Orbital)"),
    ("lc_only",        "Light curve only"),
    ("orbital_only",   "Orbital only"),
    ("orbital_no_bstar", "Orbital (no B*)"),
]

CROSS_SOURCE_MODELS = [
    ("lc_only_mmt9",       "LC [MMT-9]"),
    ("lc_only_sdlcd",      "LC [SDLCD]"),
    ("orbital_only_mmt9",  "Orbital [MMT-9]"),
    ("orbital_only_sdlcd", "Orbital [SDLCD]"),
]

CLASS_NAMES = ["Payload", "RocketBody", "Debris"]

# Normalise "Rocket Body" -> "RocketBody"
CLASS_ALIASES = {"Rocket Body": "RocketBody"}


def load_result(models_dir, model_name):
    """Load test_results.json for a model."""
    path = models_dir / model_name / "test_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_config(models_dir, model_name):
    """Load experiment_config.json for a model."""
    path = models_dir / model_name / "experiment_config.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def get_metric(result, *keys):
    """Safely traverse nested dict."""
    val = result
    for k in keys:
        if val is None or not isinstance(val, dict):
            return None
        val = val.get(k)
    return val


def normalise_per_class(pc):
    """Normalise class name keys."""
    if not pc:
        return {}
    return {CLASS_ALIASES.get(k, k): v for k, v in pc.items()}


def fmt(val, dp=4):
    if val is None:
        return "     n/a"
    return f"{val:>{dp + 4}.{dp}f}"


def fmt_delta(val, dp=4, higher_is_better=True):
    """Format a delta value with +/- and colour hint."""
    if val is None:
        return "     n/a"
    sign = "+" if val > 0 else ""
    indicator = ""
    if abs(val) > 0.001:
        if (val > 0 and higher_is_better) or (val < 0 and not higher_is_better):
            indicator = " ^"  # improved
        else:
            indicator = " v"  # worsened
    return f"{sign}{val:.{dp}f}{indicator}"


def compare_pair(base_result, aug_result):
    """Extract comparable metrics from a pair of results."""
    if base_result is None or aug_result is None:
        return None

    row = OrderedDict()

    # Regression
    for metric, higher_better in [("r2", True), ("rmse", False), ("mae", False)]:
        b = get_metric(base_result, "regression", metric)
        a = get_metric(aug_result, "regression", metric)
        row[metric] = {"base": b, "aug": a,
                       "delta": (a - b) if (a is not None and b is not None) else None,
                       "higher_better": higher_better}

    # Classification
    for metric, higher_better in [("accuracy", True), ("macro_f1", True)]:
        b = get_metric(base_result, "classification", metric)
        a = get_metric(aug_result, "classification", metric)
        row[metric] = {"base": b, "aug": a,
                       "delta": (a - b) if (a is not None and b is not None) else None,
                       "higher_better": higher_better}

    # Per-class F1
    base_pc = normalise_per_class(get_metric(base_result, "classification", "per_class") or {})
    aug_pc = normalise_per_class(get_metric(aug_result, "classification", "per_class") or {})

    for cls in CLASS_NAMES:
        b_f1 = get_metric(base_pc, cls, "f1")
        a_f1 = get_metric(aug_pc, cls, "f1")
        row[f"f1_{cls}"] = {"base": b_f1, "aug": a_f1,
                            "delta": (a_f1 - b_f1) if (a_f1 is not None and b_f1 is not None) else None,
                            "higher_better": True}

    # Per-class precision and recall for debris specifically
    for cls in ["Debris"]:
        for sub_metric in ["precision", "recall"]:
            b_val = get_metric(base_pc, cls, sub_metric)
            a_val = get_metric(aug_pc, cls, sub_metric)
            row[f"{sub_metric}_{cls}"] = {
                "base": b_val, "aug": a_val,
                "delta": (a_val - b_val) if (a_val is not None and b_val is not None) else None,
                "higher_better": True}

    return row


def print_comparison_table(comparisons, models_dir, suffix):
    """Print the main comparison table."""
    print_section("AUGMENTATION COMPARISON: CORE METRICS")

    # Show config differences
    sample_base = load_config(models_dir, "fusion")
    sample_aug = load_config(models_dir, f"fusion_{suffix}")
    if sample_base or sample_aug:
        print(f"  Baseline:    augmentation={sample_base.get('augmentation', False)}, "
              f"weighted_sampler={sample_base.get('weighted_sampler', False)}")
        print(f"  Augmented:   augmentation={sample_aug.get('augmentation', False)}, "
              f"weighted_sampler={sample_aug.get('weighted_sampler', False)}")
        print()

    # Main metrics table
    metrics = [("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE"),
               ("accuracy", "Acc"), ("macro_f1", "Macro F1")]

    header = f"  {'Model':<24s}"
    for _, label in metrics:
        header += f" {'Base':>8s} {'Aug':>8s} {'Delta':>10s}"
    print(header)
    print(f"  {'─' * (24 + len(metrics) * 28)}")

    for name, desc in CORE_MODELS:
        if name not in comparisons:
            continue
        row = comparisons[name]
        line = f"  {desc:<24s}"
        for key, _ in metrics:
            m = row[key]
            line += f" {fmt(m['base']):>8s} {fmt(m['aug']):>8s} {fmt_delta(m['delta'], higher_is_better=m['higher_better']):>10s}"
        print(line)

    print()


def print_per_class_table(comparisons):
    """Print per-class F1 comparison focused on debris improvement."""
    print_section("PER-CLASS F1 COMPARISON")

    header = f"  {'Model':<24s}"
    for cls in CLASS_NAMES:
        header += f" {'Base':>8s} {'Aug':>8s} {'Delta':>10s}"
    print(f"  {'':>24s}", end="")
    for cls in CLASS_NAMES:
        print(f" {'─── ' + cls + ' ───':^28s}", end="")
    print()
    print(header)
    print(f"  {'─' * (24 + len(CLASS_NAMES) * 28)}")

    for name, desc in CORE_MODELS:
        if name not in comparisons:
            continue
        row = comparisons[name]
        line = f"  {desc:<24s}"
        for cls in CLASS_NAMES:
            m = row[f"f1_{cls}"]
            line += f" {fmt(m['base']):>8s} {fmt(m['aug']):>8s} {fmt_delta(m['delta']):>10s}"
        print(line)

    print()


def print_debris_detail(comparisons):
    """Print detailed debris precision/recall comparison."""
    print_section("DEBRIS CLASSIFICATION DETAIL")

    print(f"  {'Model':<24s} {'Prec(B)':>8s} {'Prec(A)':>8s} {'Delta':>10s}"
          f" {'Rec(B)':>8s} {'Rec(A)':>8s} {'Delta':>10s}"
          f" {'F1(B)':>8s} {'F1(A)':>8s} {'Delta':>10s}")
    print(f"  {'─' * 108}")

    for name, desc in CORE_MODELS:
        if name not in comparisons:
            continue
        row = comparisons[name]
        p = row.get("precision_Debris", {})
        r = row.get("recall_Debris", {})
        f = row.get("f1_Debris", {})
        print(f"  {desc:<24s}"
              f" {fmt(p.get('base')):>8s} {fmt(p.get('aug')):>8s} {fmt_delta(p.get('delta')):>10s}"
              f" {fmt(r.get('base')):>8s} {fmt(r.get('aug')):>8s} {fmt_delta(r.get('delta')):>10s}"
              f" {fmt(f.get('base')):>8s} {fmt(f.get('aug')):>8s} {fmt_delta(f.get('delta')):>10s}")

    print()


def print_summary(comparisons):
    """Print high-level summary of augmentation effect."""
    print_section("SUMMARY")

    findings = []

    # Core model R² changes
    for name, desc in CORE_MODELS:
        if name not in comparisons:
            continue
        r2 = comparisons[name]["r2"]
        f1 = comparisons[name]["macro_f1"]
        debris_f1 = comparisons[name]["f1_Debris"]

        if r2["delta"] is not None:
            direction = "improved" if r2["delta"] > 0 else "decreased"
            findings.append(
                f"{desc}: R² {direction} by {abs(r2['delta']):.4f} "
                f"({r2['base']:.4f} -> {r2['aug']:.4f})")

        if debris_f1["delta"] is not None and abs(debris_f1["delta"]) > 0.01:
            direction = "improved" if debris_f1["delta"] > 0 else "decreased"
            findings.append(
                f"{desc}: Debris F1 {direction} by {abs(debris_f1['delta']):.4f} "
                f"({debris_f1['base']:.4f} -> {debris_f1['aug']:.4f})")

    for i, f in enumerate(findings, 1):
        print(f"  {i}. {f}")

    print()


def generate_comparison_figure(comparisons, output_dir):
    """Generate side-by-side bar chart comparing base vs augmented."""
    if not HAS_MPL:
        print("  [SKIP] matplotlib not available\n")
        return

    models = [(name, desc) for name, desc in CORE_MODELS if name in comparisons]
    if not models:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Augmentation Effect: Baseline vs Augmented", fontsize=14, fontweight="bold")

    x = np.arange(len(models))
    width = 0.35
    labels = [desc for _, desc in models]

    # R squared comparison
    ax = axes[0]
    base_vals = [comparisons[n]["r2"]["base"] or 0 for n, _ in models]
    aug_vals = [comparisons[n]["r2"]["aug"] or 0 for n, _ in models]
    bars1 = ax.bar(x - width/2, base_vals, width, label="Baseline", color="#5B9BD5")
    bars2 = ax.bar(x + width/2, aug_vals, width, label="Augmented", color="#ED7D31")
    ax.set_ylabel("R²")
    ax.set_title("A/m Regression")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars1, base_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, aug_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Macro F1 comparison
    ax = axes[1]
    base_vals = [comparisons[n]["macro_f1"]["base"] or 0 for n, _ in models]
    aug_vals = [comparisons[n]["macro_f1"]["aug"] or 0 for n, _ in models]
    bars1 = ax.bar(x - width/2, base_vals, width, label="Baseline", color="#5B9BD5")
    bars2 = ax.bar(x + width/2, aug_vals, width, label="Augmented", color="#ED7D31")
    ax.set_ylabel("Macro F1")
    ax.set_title("Classification (Overall)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars1, base_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, aug_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Per-class F1 (grouped by class)
    ax = axes[2]
    x_cls = np.arange(len(CLASS_NAMES))
    # Use fusion model for per-class comparison
    fusion_name = "fusion" if "fusion" in comparisons else list(comparisons.keys())[0]
    base_f1s = [comparisons[fusion_name][f"f1_{c}"]["base"] or 0 for c in CLASS_NAMES]
    aug_f1s = [comparisons[fusion_name][f"f1_{c}"]["aug"] or 0 for c in CLASS_NAMES]
    bars1 = ax.bar(x_cls - width/2, base_f1s, width, label="Baseline", color="#5B9BD5")
    bars2 = ax.bar(x_cls + width/2, aug_f1s, width, label="Augmented", color="#ED7D31")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 (Fusion Model)")
    ax.set_xticks(x_cls)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.legend()
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars1, base_f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, aug_f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig_path = output_dir / "fig_augmentation_comparison.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}\n")


def save_comparison_json(comparisons, output_dir):
    """Save comparison data as JSON for the dissertation."""
    data = {}
    for name, row in comparisons.items():
        data[name] = {}
        for metric, vals in row.items():
            data[name][metric] = {
                "baseline": vals["base"],
                "augmented": vals["aug"],
                "delta": vals["delta"],
            }

    path = output_dir / "augmentation_comparison.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else str(x))
    print(f"  Saved: {path}")


def save_comparison_report(comparisons, output_dir, suffix):
    """Save a markdown comparison report."""
    lines = [
        "# Augmentation Comparison Report",
        f"Comparing baseline models vs augmented (suffix: {suffix})",
        "",
        "## Core Metrics",
        "",
        "| Model | R² (base) | R² (aug) | Delta | Acc (base) | Acc (aug) | Delta | F1 (base) | F1 (aug) | Delta |",
        "|-------|-----------|----------|-------|------------|----------|-------|-----------|----------|-------|",
    ]

    for name, desc in CORE_MODELS:
        if name not in comparisons:
            continue
        r = comparisons[name]
        r2b, r2a = r["r2"]["base"], r["r2"]["aug"]
        ab, aa = r["accuracy"]["base"], r["accuracy"]["aug"]
        fb, fa = r["macro_f1"]["base"], r["macro_f1"]["aug"]
        r2d = r["r2"]["delta"]
        ad = r["accuracy"]["delta"]
        fd = r["macro_f1"]["delta"]
        lines.append(
            f"| {desc} | {r2b:.4f} | {r2a:.4f} | {r2d:+.4f} | "
            f"{ab:.4f} | {aa:.4f} | {ad:+.4f} | "
            f"{fb:.4f} | {fa:.4f} | {fd:+.4f} |"
        )

    lines += [
        "",
        "## Per-Class F1",
        "",
        "| Model | Payload (base) | Payload (aug) | RB (base) | RB (aug) | Debris (base) | Debris (aug) | Debris Delta |",
        "|-------|----------------|---------------|-----------|----------|---------------|--------------|--------------|",
    ]

    for name, desc in CORE_MODELS:
        if name not in comparisons:
            continue
        r = comparisons[name]
        vals = []
        for cls in CLASS_NAMES:
            m = r[f"f1_{cls}"]
            vals.extend([m["base"], m["aug"]])
        debris_delta = r["f1_Debris"]["delta"]
        lines.append(
            f"| {desc} | {vals[0]:.4f} | {vals[1]:.4f} | "
            f"{vals[2]:.4f} | {vals[3]:.4f} | "
            f"{vals[4]:.4f} | {vals[5]:.4f} | {debris_delta:+.4f} |"
        )

    lines += [
        "",
        "## Debris Detail (Precision / Recall / F1)",
        "",
        "| Model | Prec (base) | Prec (aug) | Rec (base) | Rec (aug) | F1 (base) | F1 (aug) |",
        "|-------|-------------|------------|------------|----------|-----------|----------|",
    ]

    for name, desc in CORE_MODELS:
        if name not in comparisons:
            continue
        r = comparisons[name]
        pb = r.get("precision_Debris", {}).get("base")
        pa = r.get("precision_Debris", {}).get("aug")
        rb = r.get("recall_Debris", {}).get("base")
        ra = r.get("recall_Debris", {}).get("aug")
        fb = r["f1_Debris"]["base"]
        fa = r["f1_Debris"]["aug"]
        lines.append(
            f"| {desc} | {pb:.4f} | {pa:.4f} | "
            f"{rb:.4f} | {ra:.4f} | {fb:.4f} | {fa:.4f} |"
        )

    path = output_dir / "augmentation_comparison_report.md"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare non-augmented vs augmented model results",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Examples:
  python compare_augmentation.py
  python compare_augmentation.py --suffix aug
  python compare_augmentation.py --data-dir path/to/training""")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="aug",
                        help="Suffix used for augmented models (default: 'aug')")
    args = parser.parse_args()
    ensure_dirs()
    data_dir = Path(args.data_dir) if args.data_dir else TRAINING_DIR
    models_dir = data_dir / "models"
    output_dir = data_dir / "analysis_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix

    print_header("Augmentation Comparison", __version__,
                 "Comparing baseline vs augmented models")

    # Load and compare all model pairs
    all_models = CORE_MODELS + CROSS_SOURCE_MODELS
    comparisons = OrderedDict()
    found = 0
    missing = []

    for name, desc in all_models:
        aug_name = f"{name}_{suffix}"
        base = load_result(models_dir, name)
        aug = load_result(models_dir, aug_name)

        if base is None and aug is None:
            continue
        if base is None:
            missing.append(f"{name} (baseline missing)")
            continue
        if aug is None:
            missing.append(f"{aug_name} (augmented missing)")
            continue

        row = compare_pair(base, aug)
        if row:
            comparisons[name] = row
            found += 1

    if found == 0:
        print(f"\n  [ERROR] No model pairs found.")
        print(f"  Expected baseline models in: {models_dir}")
        print(f"  Expected augmented models with suffix '_{suffix}'\n")
        return

    print(f"  Found {found} model pairs to compare")
    if missing:
        print(f"  Missing: {', '.join(missing)}")
    print()

    # Print tables
    print_comparison_table(comparisons, models_dir, suffix)
    print_per_class_table(comparisons)
    print_debris_detail(comparisons)
    print_summary(comparisons)

    # Generate outputs
    print_section("OUTPUTS")
    generate_comparison_figure(comparisons, output_dir)
    save_comparison_json(comparisons, output_dir)
    save_comparison_report(comparisons, output_dir, suffix)

    print(f"\n  All outputs saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
