"""
Model Training Script
========================
Trains fusion and ablation models with comprehensive metric tracking.

Per-epoch metrics: loss, R², RMSE, MAE, accuracy, macro F1, per-class F1
Train metrics computed every 5 epochs (on eval-mode pass over training data)
tqdm progress bars | optional TensorBoard | matplotlib training curves

Usage:
    python train.py                                   # Interactive menu
    python train.py --run-all                         # Full 8-experiment battery
    python train.py --model fusion --epochs 100
    python train.py --model lc_only --source MMT9
    python train.py --model orbital_only --no-bstar
"""

__version__ = "4.1.0"

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import TRAINING_DIR, LOG_DIR, ensure_dirs
from common.menu import (
    print_header, prompt_choice, print_section, print_summary_table,
    print_warning,
)
from common.device import (
    get_device, print_device_info, move_batch_to_device,
    auto_batch_size, supports_mixed_precision, setup_deterministic,
    add_device_args, device_from_args,
)
from models.fusion_model import create_model, count_parameters

try:
    from augment import LightCurveAugmenter
    HAS_AUGMENT = True
except ImportError:
    HAS_AUGMENT = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

CLASS_NAMES = {0: "Payload", 1: "RocketBody", 2: "Debris"}


#  CONFIGURATION


DEFAULT_CONFIG = {
    "model_type": "fusion", "epochs": 100, "batch_size": 64,
    "learning_rate": 3e-4, "weight_decay": 5e-4,
    "lr_scheduler": "cosine", "lr_step_size": 30, "lr_gamma": 0.5,
    "am_loss_weight": 1.0, "class_loss_weight": 0.5,
    "patience": 20, "min_delta": 1e-4,
    "save_every": 10, "save_best": True,
    "train_metrics_every": 5,   # Compute full train metrics every N epochs
}

BSTAR_COLUMNS = {"bstar_log_scaled", "bstar_scaled", "bstar_trend_scaled"}



#  FILE LOGGER


class FileLogger:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def log(self, msg):
        self.fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")
        self.fh.flush()

    def close(self):
        self.fh.close()



#  DATASET


class SpaceObjectDataset(Dataset):
    TYPE_ENCODING = {"Payload": 0, "Rocket Body": 1, "Debris": 2}

    def __init__(self, data_dir, split, orbital_feature_cols,
                 source_filter=None, exclude_bstar=False, augmenter=None):
        self.df = pd.read_parquet(Path(data_dir) / f"{split}.parquet")
        self.augmenter = augmenter

        if source_filter and "source" in self.df.columns:
            self.df = self.df[self.df["source"] == source_filter].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No data for split='{split}' source='{source_filter}'")

        self.lc_mags = np.stack(self.df["lightcurve"].values).astype(np.float32)
        self.lc_masks = np.stack(self.df["mask"].values).astype(np.float32)

        if exclude_bstar:
            orbital_feature_cols = [c for c in orbital_feature_cols if c not in BSTAR_COLUMNS]
        avail = [c for c in orbital_feature_cols if c in self.df.columns]
        self.orbital_col_names = avail  # Stored for XAI feature name extraction
        if avail:
            self.orbital_features = np.nan_to_num(
                self.df[avail].fillna(0).values.astype(np.float32),
                nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self.orbital_features = np.zeros((len(self.df), 1), dtype=np.float32)

        if "am_ratio_log" in self.df.columns:
            self.am_targets = self.df["am_ratio_log"].fillna(0).values.astype(np.float32)
            self.am_valid = self.df["am_ratio_log"].notna().values.astype(np.float32)
        elif "am_ratio_avg" in self.df.columns:
            am = self.df["am_ratio_avg"]
            log_am = np.log10(am.where(am > 0))
            self.am_targets = log_am.fillna(0).values.astype(np.float32)
            self.am_valid = log_am.notna().values.astype(np.float32)
        else:
            self.am_targets = np.zeros(len(self.df), dtype=np.float32)
            self.am_valid = np.zeros(len(self.df), dtype=np.float32)

        if "object_type" in self.df.columns:
            self.class_targets = self.df["object_type"].map(self.TYPE_ENCODING).fillna(-1).values.astype(np.int64)
            self.class_valid = (self.df["object_type"].notna() & self.df["object_type"].isin(self.TYPE_ENCODING)).values.astype(np.float32)
        else:
            self.class_targets = np.full(len(self.df), -1, dtype=np.int64)
            self.class_valid = np.zeros(len(self.df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mags = self.lc_mags[idx]
        masks = self.lc_masks[idx]

        # Apply augmentation if configured (training only)
        if self.augmenter is not None:
            mags, masks = self.augmenter(mags, masks, self.class_targets[idx])

        return {
            "lc_magnitudes": torch.tensor(mags, dtype=torch.float32),
            "lc_masks": torch.tensor(masks, dtype=torch.float32),
            "orbital_features": torch.tensor(self.orbital_features[idx], dtype=torch.float32),
            "am_target": torch.tensor(self.am_targets[idx], dtype=torch.float32),
            "am_valid": torch.tensor(self.am_valid[idx], dtype=torch.float32),
            "class_target": torch.tensor(self.class_targets[idx], dtype=torch.long),
            "class_valid": torch.tensor(self.class_valid[idx], dtype=torch.float32),
        }



#  LOSS


class MultiTaskLoss(nn.Module):
    def __init__(self, am_weight=1.0, class_weight=0.5, class_weights=None):
        super().__init__()
        self.am_weight, self.class_weight = am_weight, class_weight
        self.mse = nn.MSELoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none", ignore_index=-1)

    def forward(self, am_pred, class_pred, batch):
        am_v = batch["am_valid"]
        am_loss = (self.mse(am_pred.squeeze(-1), batch["am_target"]) * am_v).sum() / am_v.sum() if am_v.sum() > 0 else torch.tensor(0.0, device=am_pred.device)
        cls_v = batch["class_valid"]
        cls_loss = (self.ce(class_pred, batch["class_target"]) * cls_v).sum() / cls_v.sum() if cls_v.sum() > 0 else torch.tensor(0.0, device=am_pred.device)
        return {"total": self.am_weight * am_loss + self.class_weight * cls_loss,
                "am_loss": am_loss, "class_loss": cls_loss}



#  METRICS COMPUTATION


def compute_metrics(am_p, am_t, am_v, cls_p, cls_t, cls_v, n_classes=3):
    """Compute regression + classification metrics from collected predictions.

    Returns dict with: r2, rmse, mae, accuracy, macro_f1, per_class_f1.
    """
    m = {}

    # Regression
    mask = am_v & np.isfinite(am_p) & np.isfinite(am_t)
    if mask.sum() > 0:
        res = am_p[mask] - am_t[mask]
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((am_t[mask] - am_t[mask].mean()) ** 2)
        m["r2"] = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0
        m["rmse"] = round(np.sqrt(np.mean(res ** 2)), 4)
        m["mae"] = round(np.mean(np.abs(res)), 4)
    else:
        m["r2"], m["rmse"], m["mae"] = 0.0, 0.0, 0.0

    # Classification
    cmask = cls_v & (cls_t >= 0) & (cls_t < n_classes)
    if cmask.sum() > 0:
        p, t = cls_p[cmask], cls_t[cmask]
        m["accuracy"] = round(float((p == t).mean()), 4)

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
            per_class[name] = round(f1, 4)
            if (t == c).sum() > 0:
                f1s.append(f1)
        m["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0
        m["per_class_f1"] = per_class
    else:
        m["accuracy"], m["macro_f1"] = 0.0, 0.0
        m["per_class_f1"] = {}

    return m



#  TRAIN / VALIDATE


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train one epoch. Returns loss dict + mean gradient norm."""
    model.train()
    running = {"total": 0.0, "am_loss": 0.0, "class_loss": 0.0, "n": 0}
    grad_norms = []

    pbar = tqdm(loader, desc="    Train", leave=False, unit="bat",
                bar_format="    {l_bar}{bar:30}{r_bar}")
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                am_pred, cls_pred = model(batch["lc_magnitudes"], batch["lc_masks"], batch["orbital_features"])
                losses = criterion(am_pred, cls_pred, batch)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            am_pred, cls_pred = model(batch["lc_magnitudes"], batch["lc_masks"], batch["orbital_features"])
            losses = criterion(am_pred, cls_pred, batch)
            losses["total"].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        grad_norms.append(gn.item() if hasattr(gn, 'item') else float(gn))
        for k in ["total", "am_loss", "class_loss"]:
            running[k] += losses[k].item()
        running["n"] += 1
        pbar.set_postfix(loss=f"{running['total']/running['n']:.4f}")

    n = running["n"]
    result = {k: running[k] / n for k in ["total", "am_loss", "class_loss"]} if n > 0 else running
    result["grad_norm"] = float(np.mean(grad_norms)) if grad_norms else 0.0
    return result


@torch.no_grad()
def evaluate_pass(model, loader, criterion, device, desc="    Val  "):
    """Run eval pass, return loss + all predictions for metric computation."""
    model.eval()
    running = {"total": 0.0, "am_loss": 0.0, "class_loss": 0.0, "n": 0}
    all_am_p, all_am_t, all_am_v = [], [], []
    all_cls_p, all_cls_t, all_cls_v = [], [], []

    pbar = tqdm(loader, desc=desc, leave=False, unit="bat",
                bar_format="    {l_bar}{bar:30}{r_bar}")
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        am_pred, cls_pred = model(batch["lc_magnitudes"], batch["lc_masks"], batch["orbital_features"])
        losses = criterion(am_pred, cls_pred, batch)

        for k in ["total", "am_loss", "class_loss"]:
            running[k] += losses[k].item()
        running["n"] += 1

        all_am_p.append(am_pred.squeeze(-1).cpu().numpy())
        all_am_t.append(batch["am_target"].cpu().numpy())
        all_am_v.append(batch["am_valid"].cpu().numpy())
        all_cls_p.append(cls_pred.argmax(dim=1).cpu().numpy())
        all_cls_t.append(batch["class_target"].cpu().numpy())
        all_cls_v.append(batch["class_valid"].cpu().numpy())

    n = running["n"]
    result = {k: running[k] / n for k in ["total", "am_loss", "class_loss"]} if n > 0 else running
    result["predictions"] = {
        "am_p": np.concatenate(all_am_p), "am_t": np.concatenate(all_am_t),
        "am_v": np.concatenate(all_am_v).astype(bool),
        "cls_p": np.concatenate(all_cls_p), "cls_t": np.concatenate(all_cls_t),
        "cls_v": np.concatenate(all_cls_v).astype(bool),
    }
    return result



#  HELPERS


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, path):
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_loss": val_loss, "config": config},
               path, _use_new_zipfile_serialization=True)

def compute_class_weights(dataset, n_classes, device):
    targets = dataset.class_targets[dataset.class_valid.astype(bool)]
    if len(targets) == 0: return torch.ones(n_classes, device=device)
    counts = np.maximum(np.bincount(targets, minlength=n_classes).astype(np.float32), 1.0)
    w = 1.0 / counts
    return torch.tensor(w / w.sum() * n_classes, dtype=torch.float32, device=device)

def load_metadata(data_dir):
    for name in ["normalisation_params.json", "metadata.json"]:
        p = Path(data_dir) / name
        if p.exists():
            with open(p) as f:
                meta = json.load(f)
            if "scaled_feature_columns" in meta:
                return ([c + "_scaled" for c in meta["scaled_feature_columns"]],
                        meta.get("config", {}).get("resample_length", 1024),
                        len(meta.get("type_encoding", {})) or 3)
            else:
                return (meta.get("tle_feature_cols", []), meta.get("lc_length", 256),
                        len(meta.get("class_map", {})) or 4)
    return None, None, None



#  TRAINING CURVES


def plot_training_curves(history, output_dir, model_name=""):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

        epochs = range(1, len(history["val_loss"]) + 1)
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))

        def _p(ax, yk, title, ylabel="Loss", train_key=None):
            if train_key and train_key in history:
                ax.plot(epochs, history[train_key], label="Train", color="steelblue", alpha=0.8)
            ax.plot(epochs, history[yk], label="Val", color="indianred")
            ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(); ax.grid(True, alpha=0.3)

        _p(axes[0,0], "val_loss", "Total Loss", train_key="train_loss")
        _p(axes[0,1], "val_am", "A/m Loss (MSE)", train_key="train_am")
        _p(axes[1,0], "val_cls", "Classification Loss (CE)", train_key="train_cls")

        # R² (train + val)
        ax = axes[1,1]
        ax.plot(epochs, history["val_r2"], label="Val", color="indianred")
        if "train_r2" in history:
            tr_epochs = [i+1 for i, v in enumerate(history["train_r2"]) if v is not None]
            tr_vals = [v for v in history["train_r2"] if v is not None]
            if tr_vals: ax.plot(tr_epochs, tr_vals, 'o-', label="Train", color="steelblue", markersize=3, alpha=0.8)
        best_r2 = max(history["val_r2"]) if history["val_r2"] else 0
        ax.set_title(f"R\u00b2 (best val: {best_r2:.4f})")
        ax.set_ylabel("R\u00b2"); ax.set_ylim(-0.1, 1.05); ax.legend(); ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="grey", linestyle="--", alpha=0.3)

        # Accuracy (train + val)
        ax = axes[2,0]
        ax.plot(epochs, history["val_accuracy"], label="Val", color="indianred")
        if "train_accuracy" in history:
            tr_e = [i+1 for i, v in enumerate(history["train_accuracy"]) if v is not None]
            tr_v = [v for v in history["train_accuracy"] if v is not None]
            if tr_v: ax.plot(tr_e, tr_v, 'o-', label="Train", color="steelblue", markersize=3, alpha=0.8)
        ax.set_title("Accuracy"); ax.set_ylabel("Accuracy"); ax.set_ylim(-0.05, 1.05)
        ax.legend(); ax.grid(True, alpha=0.3)

        # Per-class F1
        ax = axes[2,1]
        for cls_name in CLASS_NAMES.values():
            key = f"val_f1_{cls_name}"
            if key in history:
                vals = [v if v is not None else np.nan for v in history[key]]
                ax.plot(epochs, vals, label=cls_name, linewidth=1.5)
        ax.set_title("Per-Class F1 (Validation)"); ax.set_ylabel("F1")
        ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

        # Gradient norm
        ax = axes[3,0]
        if "grad_norm" in history:
            ax.plot(epochs, history["grad_norm"], color="teal", alpha=0.7)
        ax.set_title("Gradient Norm"); ax.set_xlabel("Epoch"); ax.set_ylabel("Norm"); ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[3,1]
        ax.plot(epochs, history["lr"], color="seagreen")
        ax.set_title("Learning Rate"); ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.grid(True, alpha=0.3)

        fig.suptitle(f"Training Curves: {model_name}" if model_name else "Training Curves",
                     fontsize=14, y=1.005)
        fig.tight_layout()
        fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        tqdm.write(f"  Training curves: {output_dir / 'training_curves.png'}")
    except Exception as e:
        tqdm.write(f"  [WARN] Plot failed: {e}")



#  MAIN TRAINING FUNCTION


def run_training(config, data_dir, output_dir, device,
                 source_filter=None, exclude_bstar=False,
                 use_augment=False, use_weighted_sampler=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_deterministic(seed=42)

    orbital_cols, lc_length, n_classes = load_metadata(data_dir)
    if orbital_cols is None:
        tqdm.write("  [ERROR] No metadata. Run training_pipeline.py first."); return {}

    # Set up augmenter (training set only)
    augmenter = None
    if use_augment:
        if HAS_AUGMENT:
            augmenter = LightCurveAugmenter()
            tqdm.write("  Augmentation: enabled (jitter + amplitude scaling)")
        else:
            tqdm.write("  [WARN] augment.py not found - augmentation disabled")

    tqdm.write("  Loading datasets...")
    try:
        train_ds = SpaceObjectDataset(data_dir, "train", orbital_cols,
                                       source_filter, exclude_bstar,
                                       augmenter=augmenter)
        val_ds = SpaceObjectDataset(data_dir, "val", orbital_cols,
                                     source_filter, exclude_bstar,
                                     augmenter=None)  # Never augment validation
    except ValueError as e:
        tqdm.write(f"  [ERROR] {e}"); return {}

    bs = config.get("batch_size", 64)
    pin = device.type == "cuda"

    # Weighted sampler for class-balanced batches
    train_sampler = None
    shuffle = True
    if use_weighted_sampler:
        class_targets = train_ds.class_targets
        class_valid = train_ds.class_valid.astype(bool)
        # Compute per-sample weight using sqrt(inverse frequency)
        # Full inverse (1/count) fully equalises classes - too aggressive
        # with only 27K debris tracks. Sqrt partial rebalancing increases
        # minority representation ~3-4x without overwhelming the model.
        valid_targets = class_targets[class_valid]
        counts = np.bincount(valid_targets, minlength=n_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        class_weights = 1.0 / np.sqrt(counts)
        # Assign weight to each sample (invalid class samples get mean weight)
        sample_weights = np.full(len(train_ds), class_weights.mean())
        for i in range(len(train_ds)):
            if class_valid[i] and 0 <= class_targets[i] < n_classes:
                sample_weights[i] = class_weights[class_targets[i]]
        from torch.utils.data import WeightedRandomSampler
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False  # Sampler and shuffle are mutually exclusive
        # Report effective distribution
        normed = class_weights * counts
        normed = normed / normed.sum() * 100
        cls_labels = ["Payload", "RocketBody", "Debris"]
        dist_str = "  ".join(f"{cls_labels[i]}={normed[i]:.0f}%" for i in range(min(n_classes, 3)))
        tqdm.write(f"  Weighted sampler: enabled (sqrt rebalancing: {dist_str})")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=shuffle,
                               sampler=train_sampler, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin)

    src = f" [{source_filter}]" if source_filter else ""
    bst = " [no B*]" if exclude_bstar else ""
    tqdm.write(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Batch: {bs}{src}{bst}")

    orbital_dim = train_ds.orbital_features.shape[1]
    model = create_model(config["model_type"], lc_length=lc_length,
                         orbital_dim=orbital_dim, n_classes=n_classes)
    model.to(device)
    tqdm.write(f"  Model: {config['model_type']} ({count_parameters(model):,} params, orbital_dim={orbital_dim})")

    class_wts = compute_class_weights(train_ds, n_classes, device)
    criterion = MultiTaskLoss(config["am_loss_weight"], config["class_loss_weight"], class_wts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])

    scheduler = None
    if config["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    elif config["lr_scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_step_size"],
                                                     gamma=config["lr_gamma"])

    use_amp = supports_mixed_precision(device) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp: tqdm.write("  Mixed precision: enabled")

    # Logging
    flog = FileLogger(LOG_DIR / f"train_{datetime.now():%Y%m%d}.log")
    flog.log(f"START {config['model_type']} src={source_filter} bstar={not exclude_bstar} "
             f"augment={use_augment} weighted_sampler={use_weighted_sampler} "
             f"n_train={len(train_ds)} n_val={len(val_ds)}")

    tb = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if HAS_TENSORBOARD else None
    if tb: tqdm.write(f"  TensorBoard: {output_dir / 'tensorboard'}")

    # Save experiment config
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump({"model_type": config["model_type"], "source_filter": source_filter,
                    "exclude_bstar": exclude_bstar, "orbital_dim": orbital_dim,
                    "lc_length": lc_length, "n_classes": n_classes,
                    "train_samples": len(train_ds), "val_samples": len(val_ds),
                    "augmentation": use_augment, "weighted_sampler": use_weighted_sampler,
                    "config": config}, f, indent=2, default=str)

    train_metrics_every = config.get("train_metrics_every", 5)

    # History
    history = {
        "train_loss": [], "val_loss": [], "train_am": [], "val_am": [],
        "train_cls": [], "val_cls": [], "lr": [], "grad_norm": [], "epoch_time": [],
        "val_r2": [], "val_rmse": [], "val_mae": [], "val_accuracy": [], "val_macro_f1": [],
        "train_r2": [], "train_accuracy": [],
    }
    for cls_name in CLASS_NAMES.values():
        history[f"val_f1_{cls_name}"] = []

    # ── Training loop ──
    header = (f"  {'Ep':>5s}  {'Val Loss':>8s}  {'R\u00b2':>7s}  {'RMSE':>7s}  "
              f"{'MAE':>7s}  {'Acc':>7s}  {'F1':>7s}  "
              f"{'Time':>5s}  {'ETA':>7s}  {'':>12s}")
    tqdm.write(f"\n{header}\n  {'─' * 90}")

    best_val_loss = float("inf")
    best_val_r2 = float("-inf")
    best_epoch = 0
    patience_ctr = 0
    epoch_times = []

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # Validate (with full metrics)
        val_raw = evaluate_pass(model, val_loader, criterion, device, desc="    Val  ")
        vp = val_raw["predictions"]
        val_m = compute_metrics(vp["am_p"], vp["am_t"], vp["am_v"],
                                vp["cls_p"], vp["cls_t"], vp["cls_v"], n_classes)

        # Train metrics (every N epochs - eval-mode pass on training data)
        if epoch == 1 or epoch % train_metrics_every == 0:
            tr_raw = evaluate_pass(model, train_loader, criterion, device, desc="    TrEval")
            tp = tr_raw["predictions"]
            tr_m = compute_metrics(tp["am_p"], tp["am_t"], tp["am_v"],
                                   tp["cls_p"], tp["cls_t"], tp["cls_v"], n_classes)
            history["train_r2"].append(tr_m["r2"])
            history["train_accuracy"].append(tr_m["accuracy"])
        else:
            history["train_r2"].append(None)
            history["train_accuracy"].append(None)

        if scheduler: scheduler.step()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss["total"])
        history["val_loss"].append(val_raw["total"])
        history["train_am"].append(train_loss["am_loss"])
        history["val_am"].append(val_raw["am_loss"])
        history["train_cls"].append(train_loss["class_loss"])
        history["val_cls"].append(val_raw["class_loss"])
        history["lr"].append(lr)
        history["grad_norm"].append(train_loss["grad_norm"])
        history["epoch_time"].append(round(elapsed, 1))
        history["val_r2"].append(val_m["r2"])
        history["val_rmse"].append(val_m["rmse"])
        history["val_mae"].append(val_m["mae"])
        history["val_accuracy"].append(val_m["accuracy"])
        history["val_macro_f1"].append(val_m["macro_f1"])
        for cls_name in CLASS_NAMES.values():
            history[f"val_f1_{cls_name}"].append(val_m.get("per_class_f1", {}).get(cls_name))

        # TensorBoard
        if tb:
            tb.add_scalars("Loss", {"train": train_loss["total"], "val": val_raw["total"]}, epoch)
            tb.add_scalar("Val/R2", val_m["r2"], epoch)
            tb.add_scalar("Val/RMSE", val_m["rmse"], epoch)
            tb.add_scalar("Val/MAE", val_m["mae"], epoch)
            tb.add_scalar("Val/Accuracy", val_m["accuracy"], epoch)
            tb.add_scalar("Val/Macro_F1", val_m["macro_f1"], epoch)
            for cn, f1v in val_m.get("per_class_f1", {}).items():
                tb.add_scalar(f"Val_F1/{cn}", f1v, epoch)
            tb.add_scalar("GradNorm", train_loss["grad_norm"], epoch)
            tb.add_scalar("LR", lr, epoch)

        # Best model (tracked by R² - more meaningful than total loss,
        # especially with weighted sampling where classification loss
        # on the unweighted validation set can diverge from regression
        # improvement)
        improved = ""
        val_r2 = val_m["r2"]
        if val_r2 > best_val_r2 + config["min_delta"]:
            best_val_r2 = val_r2
            best_val_loss = val_raw["total"]
            best_epoch = epoch
            patience_ctr = 0
            improved = "  BEST"
            if config["save_best"]:
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                                config, output_dir / "best_model.pt")
        else:
            patience_ctr += 1

        # ETA
        eta_s = np.mean(epoch_times) * (config["epochs"] - epoch)
        eta = f"{eta_s/3600:.1f}h" if eta_s >= 3600 else f"{eta_s/60:.0f}m"

        # Epoch summary
        line = (f"  {epoch:>3d}/{config['epochs']:<3d}"
                f"  {val_raw['total']:>8.4f}"
                f"  {val_m['r2']:>7.4f}"
                f"  {val_m['rmse']:>7.4f}"
                f"  {val_m['mae']:>7.4f}"
                f"  {val_m['accuracy']:>7.4f}"
                f"  {val_m['macro_f1']:>7.4f}"
                f"  {elapsed:>4.0f}s"
                f"  {eta:>7s}"
                f"  p={patience_ctr}/{config['patience']}{improved}")
        tqdm.write(line)
        flog.log(line.strip())

        # Per-class detail on train-metric epochs
        if epoch == 1 or epoch % train_metrics_every == 0:
            pcf = val_m.get("per_class_f1", {})
            pcf_str = "  ".join(f"{k}={v:.3f}" for k, v in pcf.items())
            detail = f"         Train: R\u00b2={tr_m['r2']:.3f} Acc={tr_m['accuracy']:.3f} | Val F1: {pcf_str}"
            tqdm.write(detail)
            flog.log(detail.strip())

        # Checkpoints
        if epoch % config["save_every"] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_raw["total"],
                            config, output_dir / f"checkpoint_epoch{epoch}.pt")

        # Early stopping
        if patience_ctr >= config["patience"]:
            msg = f"\n  Early stopping at epoch {epoch} (no improvement for {config['patience']} epochs)"
            tqdm.write(msg); flog.log(msg.strip())
            break

    # Final saves
    save_checkpoint(model, optimizer, scheduler, epoch, val_raw["total"],
                    config, output_dir / "final_model.pt")
    history["best_epoch"] = best_epoch

    def _json_default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, default=_json_default)

    plot_training_curves(history, output_dir, config["model_type"])
    if tb: tb.close()
    flog.log(f"END best_r2={best_val_r2:.4f} best_val={best_val_loss:.4f} best_epoch={best_epoch} total_epochs={epoch}")
    flog.close()

    total_time = sum(epoch_times)
    tqdm.write(f"\n  Best: epoch {best_epoch}, val R²={best_val_r2:.4f}, val loss {best_val_loss:.4f}")
    tqdm.write(f"  Total time: {total_time/3600:.1f}h  |  Saved to: {output_dir}")

    return {"epochs_trained": epoch, "best_val_loss": best_val_loss,
            "best_val_r2": best_val_r2, "best_epoch": best_epoch,
            "final_train_loss": train_loss["total"], "final_val_loss": val_raw["total"],
            "total_time_s": round(total_time, 1)}



#  EXPERIMENT DEFINITIONS


CORE_EXPERIMENTS = [
    ("fusion",            "fusion",       None,    False, "Fusion (LC + Orbital)"),
    ("lc_only",           "lc_only",      None,    False, "Light curve only"),
    ("orbital_only",      "orbital_only", None,    False, "Orbital only"),
]
BSTAR_EXPERIMENT = [
    ("orbital_no_bstar",  "orbital_only", None,    True,  "Orbital only (no B*)"),
]
CROSS_SOURCE_EXPERIMENTS = [
    ("lc_only_mmt9",      "lc_only",      "MMT9",  False, "LC only [MMT-9 data]"),
    ("lc_only_sdlcd",     "lc_only",      "SDLCD", False, "LC only [SDLCD data]"),
    ("orbital_only_mmt9", "orbital_only",  "MMT9",  False, "Orbital only [MMT-9 data]"),
    ("orbital_only_sdlcd","orbital_only",  "SDLCD", False, "Orbital only [SDLCD data]"),
]
ALL_EXPERIMENTS = CORE_EXPERIMENTS + BSTAR_EXPERIMENT + CROSS_SOURCE_EXPERIMENTS


def run_experiment_batch(experiments, config, data_dir, output_dir, device,
                         use_augment=False, use_weighted_sampler=False):
    results, skipped = {}, 0
    for i, (name, mtype, src, no_b, desc) in enumerate(experiments, 1):
        md = Path(output_dir) / name
        if (md / "best_model.pt").exists() or (md / "final_model.pt").exists():
            tqdm.write(f"  [{i}/{len(experiments)}] {desc} -- already trained, skipping")
            skipped += 1; continue
        cfg = dict(config); cfg["model_type"] = mtype
        print_section(f"EXPERIMENT {i}/{len(experiments)}: {desc.upper()}")
        r = run_training(cfg, data_dir, md, device, source_filter=src, exclude_bstar=no_b,
                         use_augment=use_augment, use_weighted_sampler=use_weighted_sampler)
        if r: results[name] = r
        tqdm.write("")
    if skipped:
        tqdm.write(f"  Skipped {skipped} completed. Delete model dir to retrain.\n")
    if len(results) > 1:
        print_section("TRAINING SUMMARY")
        tqdm.write(f"  {'Experiment':<24s} {'Ep':>5s} {'Best Val':>10s} {'R\u00b2':>7s} {'Time':>7s}")
        tqdm.write(f"  {'─' * 56}")
        for n, r in results.items():
            tqdm.write(f"  {n:<24s} {r['epochs_trained']:>5d} {r['best_val_loss']:>10.4f} "
                       f"{'':>7s} {r['total_time_s']/3600:.1f}h")
        tqdm.write("")
    return results



#  MENU / CLI


def run_interactive(data_dir, output_dir):
    print_header("Model Training", __version__, "Train fusion and ablation models")
    device = get_device(); print_device_info(device)
    if HAS_TENSORBOARD:
        tqdm.write("  TensorBoard: run 'tensorboard --logdir data/training/models'\n")
    config = dict(DEFAULT_CONFIG)
    while True:
        ch = prompt_choice("What would you like to train?", [
            ("core","Core models (fusion + lc_only + orbital_only)"), ("fusion","Fusion only"),
            ("lc_only","LC-only"), ("orbital","Orbital-only"), ("no_bstar","Orbital no B*"),
            ("cross","Cross-source (4 runs)"), ("all","ALL 8 experiments"), ("quit","Quit")])
        if ch in (None, "quit"): tqdm.write("\n  Goodbye!\n"); break
        exps = {"core": CORE_EXPERIMENTS, "fusion": [CORE_EXPERIMENTS[0]],
                "lc_only": [CORE_EXPERIMENTS[1]], "orbital": [CORE_EXPERIMENTS[2]],
                "no_bstar": BSTAR_EXPERIMENT, "cross": CROSS_SOURCE_EXPERIMENTS,
                "all": ALL_EXPERIMENTS}
        run_experiment_batch(exps[ch], config, data_dir, output_dir, device)


def main():
    p = argparse.ArgumentParser(description="Space Debris ML -- Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Examples:
  python train.py                                     # Interactive menu
  python train.py --run-all                           # Full 8-experiment battery
  python train.py --run-all --augment --weighted-sampler --suffix aug
  python train.py --model lc_only --source MMT9
  python train.py --model orbital_only --no-bstar""")
    p.add_argument("--model", default="fusion", choices=["fusion","lc_only","orbital_only"])
    p.add_argument("--source", default=None, choices=["MMT9","SDLCD"])
    p.add_argument("--no-bstar", action="store_true")
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--augment", action="store_true",
                   help="Enable LC augmentation for minority classes (jitter + scaling)")
    p.add_argument("--weighted-sampler", action="store_true",
                   help="Enable class-balanced batch sampling")
    p.add_argument("--suffix", type=str, default=None,
                   help="Append suffix to model directory names (e.g. 'aug' -> 'fusion_aug')")
    p.add_argument("--epochs", type=int); p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--data-dir", type=str); p.add_argument("--output-dir", type=str)
    add_device_args(p)
    args = p.parse_args(); ensure_dirs()
    data_dir = Path(args.data_dir) if args.data_dir else TRAINING_DIR
    output_dir = Path(args.output_dir) if args.output_dir else TRAINING_DIR / "models"
    config = dict(DEFAULT_CONFIG)
    if args.epochs: config["epochs"] = args.epochs
    if args.batch_size: config["batch_size"] = args.batch_size
    if args.lr: config["learning_rate"] = args.lr

    # Apply suffix to experiment names if provided
    experiments = ALL_EXPERIMENTS
    if args.suffix:
        experiments = [(f"{name}_{args.suffix}", mt, src, nb, f"{desc} [{args.suffix}]")
                       for name, mt, src, nb, desc in ALL_EXPERIMENTS]

    has_cli = (args.epochs is not None or args.model != "fusion" or args.source
               or args.no_bstar or args.run_all or args.augment or args.weighted_sampler)
    try:
        if args.run_all:
            print_header("Model Training", __version__); device = device_from_args(args); print_device_info(device)
            if args.augment or args.weighted_sampler:
                flags = []
                if args.augment: flags.append("augmentation")
                if args.weighted_sampler: flags.append("weighted sampler")
                tqdm.write(f"  Enabled: {', '.join(flags)}")
                if args.suffix: tqdm.write(f"  Suffix: {args.suffix}\n")
            run_experiment_batch(experiments, config, data_dir, output_dir, device,
                                use_augment=args.augment,
                                use_weighted_sampler=args.weighted_sampler)
        elif has_cli:
            print_header("Model Training", __version__); device = device_from_args(args); print_device_info(device)
            config["model_type"] = args.model
            name = f"{args.model}_{args.source.lower()}" if args.source else (f"{args.model}_no_bstar" if args.no_bstar else args.model)
            if args.suffix: name = f"{name}_{args.suffix}"
            print_section(f"TRAINING: {name.upper()}")
            run_training(config, data_dir, output_dir / name, device,
                         source_filter=args.source, exclude_bstar=args.no_bstar,
                         use_augment=args.augment, use_weighted_sampler=args.weighted_sampler)
        else:
            run_interactive(data_dir, output_dir)
    except KeyboardInterrupt:
        tqdm.write("\n\n  Interrupted. Checkpoint saved if training was in progress."); sys.exit(2)

if __name__ == "__main__":
    main()
