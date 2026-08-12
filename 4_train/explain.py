"""
Explainable AI for Space Debris Models
==========================================
Post-hoc explanation methods applied to already-trained models.
No additional training required.

Methods:
  - 1D Grad-CAM:             Highlights which temporal regions of a light curve
                              the model focuses on for its A/m prediction.
  - Permutation importance:   Ranks orbital features by their contribution
                              to model performance.

Outputs saved as PNG figures + JSON summaries in each model's directory.

Usage:
    python explain.py                               # Interactive menu
    python explain.py --model fusion --gradcam       # Grad-CAM for fusion model
    python explain.py --model orbital_only --permutation  # Feature importance
    python explain.py --all                          # Everything for all models
"""

__version__ = "1.0.0"

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
)
from common.device import (
    get_device, print_device_info, move_batch_to_device,
    add_device_args, device_from_args,
)
from common.logging_setup import setup_logging

from models.fusion_model import create_model
from train import SpaceObjectDataset, load_metadata

logger = setup_logging("explain")



#  1D GRAD-CAM


class GradCAM1D:
    """Grad-CAM adapted for 1D convolutional light curve models.

    Hooks into the last convolutional layer (before global pooling)
    to produce a temporal heatmap showing which parts of the light
    curve the model attends to.

    Works with FusionModel, LightCurveOnlyModel, or any model
    that has a .lc_branch.conv attribute.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.activations = None
        self.gradients = None

        # Find the target layer: last ReLU before AdaptiveAvgPool1d
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "lc_branch"):
            # conv[10] = last ReLU before AdaptiveAvgPool1d(1)
            self.target_layer = model.lc_branch.conv[10]
        else:
            raise ValueError("Model has no lc_branch. Grad-CAM needs a CNN.")

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, lc_magnitudes, lc_masks, orbital_features,
                 target="am") -> np.ndarray:
        """Generate Grad-CAM heatmaps for a batch.

        Args:
            lc_magnitudes:    (batch, lc_length)
            lc_masks:         (batch, lc_length)
            orbital_features: (batch, orbital_dim)
            target:           'am' for regression, or class index for classification

        Returns:
            (batch, lc_length) heatmap array, values in [0, 1].
        """
        self.model.eval()

        # Enable gradients for this pass
        lc_magnitudes = lc_magnitudes.requires_grad_(False)

        am_pred, class_pred = self.model(lc_magnitudes, lc_masks, orbital_features)

        # Select target for backpropagation
        if target == "am":
            score = am_pred.squeeze(-1).sum()
        else:
            score = class_pred[:, int(target)].sum()

        self.model.zero_grad()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            return np.zeros((lc_magnitudes.shape[0], lc_magnitudes.shape[1]))

        # Grad-CAM: weight activations by global-averaged gradients
        # activations: (batch, channels, spatial)
        # gradients:   (batch, channels, spatial)
        weights = self.gradients.mean(dim=2, keepdim=True)  # (batch, channels, 1)
        cam = (weights * self.activations).sum(dim=1)       # (batch, spatial)
        cam = torch.relu(cam)                                # Only positive contributions

        # Normalise per sample
        cam_min = cam.min(dim=1, keepdim=True).values
        cam_max = cam.max(dim=1, keepdim=True).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Upsample to original light curve length
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1), size=lc_magnitudes.shape[1], mode="linear",
            align_corners=False,
        ).squeeze(1)

        return cam.cpu().numpy()


def run_gradcam(model, test_loader, device, output_dir, n_examples=8,
                log_fn=print) -> dict:
    """Run Grad-CAM analysis on the test set.

    Produces:
      - Average heatmap across all test samples
      - Individual example plots for the most/least attended samples
      - JSON summary with statistics

    Returns:
        Dict with summary statistics.
    """
    gradcam = GradCAM1D(model)

    all_cams = []
    all_mags = []
    all_masks = []
    all_am_valid = []

    log_fn("  Computing Grad-CAM heatmaps...")
    for batch in test_loader:
        batch = move_batch_to_device(batch, device)

        cam = gradcam.generate(
            batch["lc_magnitudes"], batch["lc_masks"],
            batch["orbital_features"], target="am",
        )
        all_cams.append(cam)
        all_mags.append(batch["lc_magnitudes"].cpu().numpy())
        all_masks.append(batch["lc_masks"].cpu().numpy())
        all_am_valid.append(batch["am_valid"].cpu().numpy())

    all_cams = np.concatenate(all_cams, axis=0)
    all_mags = np.concatenate(all_mags, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_am_valid = np.concatenate(all_am_valid, axis=0)

    # Filter to samples with valid A/m targets
    valid_idx = np.where(all_am_valid > 0)[0]
    if len(valid_idx) > 0:
        cams_valid = all_cams[valid_idx]
        mags_valid = all_mags[valid_idx]
        masks_valid = all_masks[valid_idx]
    else:
        cams_valid = all_cams
        mags_valid = all_mags
        masks_valid = all_masks

    # Average heatmap
    avg_cam = cams_valid.mean(axis=0)

    # Find most/least attended samples (by peak attention)
    peak_attention = cams_valid.max(axis=1)
    most_attended_idx = np.argsort(peak_attention)[-n_examples:][::-1]
    least_attended_idx = np.argsort(peak_attention)[:n_examples]

    # Statistics
    summary = {
        "total_samples": int(len(all_cams)),
        "valid_am_samples": int(len(valid_idx)),
        "avg_cam_mean": float(avg_cam.mean()),
        "avg_cam_std": float(avg_cam.std()),
        "avg_cam_peak_position": int(np.argmax(avg_cam)),
        "avg_cam_peak_value": float(avg_cam.max()),
    }

    # Find attention concentration: what fraction of the LC gets >50% of attention
    sorted_avg = np.sort(avg_cam)[::-1]
    cumsum = np.cumsum(sorted_avg) / sorted_avg.sum()
    attention_50pct = int(np.searchsorted(cumsum, 0.5)) + 1
    summary["attention_50pct_points"] = attention_50pct
    summary["attention_50pct_fraction"] = round(attention_50pct / len(avg_cam), 3)

    # Save JSON
    with open(output_dir / "gradcam_summary.json", "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    # Save raw average heatmap
    np.save(output_dir / "gradcam_avg_heatmap.npy", avg_cam)

    # Generate plots
    log_fn(f"  Generating Grad-CAM visualisations...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Average heatmap
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.fill_between(range(len(avg_cam)), avg_cam, alpha=0.3, color="red")
        ax.plot(avg_cam, color="red", linewidth=1.5, label="Average Grad-CAM")
        ax.set_xlabel("Resampled time index")
        ax.set_ylabel("Attention")
        ax.set_title("Average Grad-CAM Heatmap (A/m regression target)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "gradcam_average.png", dpi=150)
        plt.close(fig)

        # 2. Example light curves with Grad-CAM overlay
        n_show = min(n_examples, len(most_attended_idx))
        fig, axes = plt.subplots(n_show, 1, figsize=(12, 2.5 * n_show))
        if n_show == 1:
            axes = [axes]

        for i, idx in enumerate(most_attended_idx[:n_show]):
            ax = axes[i]
            lc = mags_valid[idx]
            mask = masks_valid[idx]
            cam = cams_valid[idx]

            # Plot light curve
            ax.plot(lc, color="steelblue", linewidth=0.8, alpha=0.8)

            # Overlay Grad-CAM as coloured background
            ax2 = ax.twinx()
            ax2.fill_between(range(len(cam)), cam, alpha=0.3, color="red")
            ax2.set_ylim(0, 1.5)
            ax2.set_ylabel("Attention", color="red", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="red", labelsize=7)

            # Mark masked regions
            masked_regions = np.where(mask < 0.5)[0]
            for mr in masked_regions:
                ax.axvspan(mr - 0.5, mr + 0.5, alpha=0.05, color="grey")

            ax.set_ylabel("Normalised mag", fontsize=8)
            ax.set_title(f"Sample {idx} (peak attention: {cam.max():.2f})", fontsize=9)

        axes[-1].set_xlabel("Resampled time index")
        fig.suptitle("Grad-CAM: Most Attended Light Curves", fontsize=11, y=1.01)
        fig.tight_layout()
        fig.savefig(output_dir / "gradcam_examples.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        log_fn(f"  Saved: gradcam_average.png, gradcam_examples.png")

    except ImportError:
        log_fn("  [WARN] matplotlib not available, skipping plots")

    log_fn(f"  Attention concentrated in {summary['attention_50pct_fraction']*100:.0f}% "
           f"of the light curve (50% energy)")

    return summary



#  PERMUTATION IMPORTANCE


def compute_permutation_importance(model, test_ds, device, n_repeats=5,
                                   log_fn=print) -> dict:
    """Compute permutation importance for orbital features.

    For each feature, shuffles its values across the test set and
    measures the increase in regression loss. Features whose shuffling
    causes the largest loss increase are most important.

    Args:
        model:     Trained model.
        test_ds:   SpaceObjectDataset instance.
        device:    Compute device.
        n_repeats: Number of shuffle repetitions per feature.

    Returns:
        Dict mapping feature names to importance scores.
    """
    model.eval()
    n_features = test_ds.orbital_features.shape[1]

    # Get baseline loss
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    baseline_loss = _compute_am_loss(model, loader, device)
    log_fn(f"  Baseline A/m loss: {baseline_loss:.4f}")

    # Read feature names from experiment config or metadata
    feature_names = _get_orbital_feature_names(test_ds, n_features)

    importances = {}
    log_fn(f"  Computing importance for {n_features} features "
           f"({n_repeats} repeats each)...")

    for feat_idx in range(n_features):
        losses = []
        for _ in range(n_repeats):
            # Create shuffled copy
            original = test_ds.orbital_features[:, feat_idx].copy()
            np.random.shuffle(test_ds.orbital_features[:, feat_idx])

            loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
            shuffled_loss = _compute_am_loss(model, loader, device)
            losses.append(shuffled_loss)

            # Restore original
            test_ds.orbital_features[:, feat_idx] = original

        mean_loss = np.mean(losses)
        importance = mean_loss - baseline_loss
        name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
        importances[name] = {
            "importance": round(float(importance), 6),
            "mean_shuffled_loss": round(float(mean_loss), 6),
            "std": round(float(np.std(losses)), 6),
        }

        log_fn(f"    {name:<30s}  +{importance:.4f}")

    # Sort by importance
    importances = dict(sorted(importances.items(),
                              key=lambda x: x[1]["importance"], reverse=True))

    return {"baseline_loss": float(baseline_loss), "features": importances}


@torch.no_grad()
def _compute_am_loss(model, loader, device) -> float:
    """Compute mean A/m MSE loss on valid samples."""
    model.eval()
    total_loss = 0.0
    total_valid = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        am_pred, _ = model(
            batch["lc_magnitudes"], batch["lc_masks"], batch["orbital_features"]
        )
        am_valid = batch["am_valid"]
        if am_valid.sum() > 0:
            residuals = (am_pred.squeeze(-1) - batch["am_target"]) ** 2
            total_loss += (residuals * am_valid).sum().item()
            total_valid += am_valid.sum().item()

    return total_loss / max(total_valid, 1)


def _get_orbital_feature_names(test_ds, n_features) -> list:
    """Try to get feature names from the dataset."""
    # Best: use stored column names from dataset construction
    if hasattr(test_ds, "orbital_col_names"):
        names = [c.replace("_scaled", "") for c in test_ds.orbital_col_names]
        if len(names) == n_features:
            return names
    # Fallback: scan DataFrame columns
    if hasattr(test_ds, "df"):
        scaled_cols = [c for c in test_ds.df.columns if c.endswith("_scaled")]
        if len(scaled_cols) == n_features:
            return [c.replace("_scaled", "") for c in scaled_cols]
    return [f"feature_{i}" for i in range(n_features)]


def run_permutation_importance(model, test_ds, device, output_dir,
                               n_repeats=5, log_fn=print) -> dict:
    """Run permutation importance and save results + plot."""
    results = compute_permutation_importance(model, test_ds, device,
                                             n_repeats=n_repeats, log_fn=log_fn)

    # Save JSON
    with open(output_dir / "permutation_importance.json", "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    # Generate bar chart
    log_fn(f"\n  Generating importance plot...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        features = results["features"]
        names = list(features.keys())
        scores = [features[n]["importance"] for n in names]
        stds = [features[n]["std"] for n in names]

        # Trim long names for readability
        short_names = [n.replace("_scaled", "").replace("_", " ")[:20] for n in names]

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
        y_pos = range(len(names))
        bars = ax.barh(y_pos, scores, xerr=stds, align="center",
                       color="steelblue", alpha=0.8, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Loss increase when shuffled (higher = more important)")
        ax.set_title("Orbital Feature Permutation Importance")

        # Highlight B* if present
        for i, name in enumerate(names):
            if "bstar" in name.lower():
                bars[i].set_color("indianred")

        fig.tight_layout()
        fig.savefig(output_dir / "permutation_importance.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        log_fn(f"  Saved: permutation_importance.png")

    except ImportError:
        log_fn("  [WARN] matplotlib not available, skipping plot")

    return results



#  MODEL LOADING HELPER


def load_model_for_explanation(model_dir: Path, data_dir: Path,
                                device: torch.device) -> tuple:
    """Load a trained model and its test dataset for XAI analysis.

    Returns:
        (model, test_ds, model_type, experiment_info) or None on failure.
    """
    # Read experiment config
    exp_cfg_path = model_dir / "experiment_config.json"
    if exp_cfg_path.exists():
        with open(exp_cfg_path) as f:
            exp_cfg = json.load(f)
        model_type = exp_cfg.get("model_type", "fusion")
        source_filter = exp_cfg.get("source_filter")
        exclude_bstar = exp_cfg.get("exclude_bstar", False)
        orbital_dim = exp_cfg.get("orbital_dim")
    else:
        model_type = model_dir.name.split("_")[0]
        if model_type not in ["fusion", "lc", "orbital"]:
            model_type = "fusion"
        if "lc_only" in model_dir.name:
            model_type = "lc_only"
        elif "orbital" in model_dir.name:
            model_type = "orbital_only"
        source_filter = None
        exclude_bstar = "no_bstar" in model_dir.name
        orbital_dim = None

    # Load metadata
    orbital_feature_cols, lc_length, n_classes = load_metadata(data_dir)
    if orbital_feature_cols is None:
        print(f"  [ERROR] No metadata found in {data_dir}")
        return None

    # Load test dataset
    try:
        test_ds = SpaceObjectDataset(data_dir, "test", orbital_feature_cols,
                                     source_filter=source_filter,
                                     exclude_bstar=exclude_bstar)
    except ValueError as e:
        print(f"  [ERROR] {e}")
        return None

    # Load model
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "final_model.pt"
    if not checkpoint_path.exists():
        print(f"  [ERROR] No checkpoint in {model_dir}")
        return None

    actual_orbital_dim = orbital_dim or test_ds.orbital_features.shape[1]
    model = create_model(
        model_type,
        lc_length=lc_length,
        orbital_dim=actual_orbital_dim,
        n_classes=n_classes,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, test_ds, model_type, exp_cfg if exp_cfg_path.exists() else {}



#  ORCHESTRATION


def explain_model(model_dir: Path, data_dir: Path, device: torch.device,
                  run_gc: bool = True, run_pi: bool = True,
                  log_fn=print) -> dict:
    """Run all applicable XAI methods on a single model.

    Args:
        model_dir: Directory with trained model.
        data_dir:  Directory with test data.
        device:    Compute device.
        run_gc:    Run Grad-CAM (only for models with LC branch).
        run_pi:    Run permutation importance (only for models with orbital branch).

    Returns:
        Dict with all results.
    """
    result_tuple = load_model_for_explanation(model_dir, data_dir, device)
    if result_tuple is None:
        return {}

    model, test_ds, model_type, exp_cfg = result_tuple
    results = {"model_type": model_type, "model_dir": str(model_dir)}

    xai_dir = model_dir / "xai"
    xai_dir.mkdir(exist_ok=True)

    has_lc = hasattr(model, "lc_branch")
    has_orbital = hasattr(model, "orbital_branch")

    # Grad-CAM for models with light curve branch
    if run_gc and has_lc:
        print_section(f"GRAD-CAM: {model_dir.name}")
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
        gc_results = run_gradcam(model, test_loader, device, xai_dir, log_fn=log_fn)
        results["gradcam"] = gc_results
        log_fn("")
    elif run_gc and not has_lc:
        log_fn(f"  [SKIP] Grad-CAM: {model_dir.name} has no light curve branch\n")

    # Permutation importance for models with orbital branch
    if run_pi and has_orbital:
        print_section(f"PERMUTATION IMPORTANCE: {model_dir.name}")
        pi_results = run_permutation_importance(model, test_ds, device, xai_dir,
                                                log_fn=log_fn)
        results["permutation_importance"] = pi_results
        log_fn("")
    elif run_pi and not has_orbital:
        log_fn(f"  [SKIP] Permutation importance: {model_dir.name} has no orbital branch\n")

    return results


def explain_all_models(data_dir: Path, models_dir: Path,
                       device: torch.device, log_fn=print):
    """Run XAI analysis on all trained models."""
    if not models_dir.exists():
        log_fn("  No models directory found.")
        return

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if not ((model_dir / "best_model.pt").exists() or
                (model_dir / "final_model.pt").exists()):
            continue

        log_fn(f"\n{'='*60}")
        log_fn(f"  Explaining: {model_dir.name}")
        log_fn(f"{'='*60}")
        explain_model(model_dir, data_dir, device, log_fn=log_fn)



#  INTERACTIVE MENU


def run_interactive(data_dir: Path, models_dir: Path):
    """Run the interactive menu."""
    print_header("Explainable AI", __version__,
                 "Grad-CAM and feature importance analysis")

    device = get_device()
    print_device_info(device)

    while True:
        # Detect available models
        available = []
        if models_dir.exists():
            for d in sorted(models_dir.iterdir()):
                if d.is_dir() and ((d / "best_model.pt").exists() or
                                   (d / "final_model.pt").exists()):
                    available.append(d.name)

        if not available:
            print("  No trained models found. Run train.py first.\n")
            break

        options = []
        for name in available:
            options.append((name, f"Explain: {name}"))

        if len(available) >= 2:
            options.append(("all", "Explain all models"))

        options.append(("quit", "Quit"))

        choice = prompt_choice("Select a model to explain:", options)

        if choice is None or choice == "quit":
            print("\n  Goodbye!\n")
            break

        elif choice == "all":
            explain_all_models(data_dir, models_dir, device)

        elif choice in available:
            model_dir = models_dir / choice
            explain_model(model_dir, data_dir, device)



#  CLI MODE


def run_cli(args, data_dir: Path, models_dir: Path):
    """Run in CLI mode."""
    print_header("Explainable AI", __version__)

    device = device_from_args(args)
    print_device_info(device)

    if args.all:
        explain_all_models(data_dir, models_dir, device)
    elif args.model:
        model_dir = models_dir / args.model
        if not model_dir.exists():
            print(f"  [ERROR] No model at {model_dir}")
            sys.exit(1)
        explain_model(model_dir, data_dir, device,
                      run_gc=args.gradcam or not args.permutation,
                      run_pi=args.permutation or not args.gradcam)



#  MAIN


def main():
    parser = argparse.ArgumentParser(
        description="Space Debris ML - Explainable AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python explain.py                                 # Interactive menu
  python explain.py --model fusion                  # Explain fusion model
  python explain.py --model fusion --gradcam        # Grad-CAM only
  python explain.py --model orbital_only --permutation  # Importance only
  python explain.py --all                           # Explain all models
        """,
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Model directory name to explain")
    parser.add_argument("--gradcam", action="store_true",
                        help="Run Grad-CAM only")
    parser.add_argument("--permutation", action="store_true",
                        help="Run permutation importance only")
    parser.add_argument("--all", action="store_true",
                        help="Explain all trained models")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    add_device_args(parser)

    args = parser.parse_args()

    ensure_dirs()
    data_dir = Path(args.data_dir) if args.data_dir else TRAINING_DIR
    models_dir = Path(args.models_dir) if args.models_dir else TRAINING_DIR / "models"

    has_action = args.model or args.all

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
