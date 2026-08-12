"""
Model Architecture Definitions
==================================
Late-fusion neural network combining:
  - Branch A: 1D CNN for light curve time series
  - Branch B: MLP for TLE orbital features
  - Fusion head: combined embedding → A/m regression + object classification

Standalone ablation models are provided for systematic comparison.

Usage:
    from models.fusion_model import FusionModel, LightCurveOnlyModel, OrbitalOnlyModel

    model = FusionModel(lc_length=256, orbital_dim=60, n_classes=4)
    am_pred, class_pred = model(lc_magnitudes, lc_masks, orbital_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  LIGHT CURVE BRANCH (1D CNN)
# ═══════════════════════════════════════════════════════════════

class LightCurveBranch(nn.Module):
    """1D Convolutional network for light curve time series.

    Processes a fixed-length magnitude sequence with a binary mask
    indicating real vs padded positions.

    Input:  (batch, lc_length) magnitudes + (batch, lc_length) mask
    Output: (batch, embed_dim) embedding
    """

    def __init__(self, lc_length: int = 256, embed_dim: int = 64):
        super().__init__()
        self.lc_length = lc_length
        self.embed_dim = embed_dim

        # Input: 2 channels (magnitude + mask)
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )

        self.fc = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, magnitudes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            magnitudes: (batch, lc_length) normalised magnitude values
            mask:       (batch, lc_length) binary mask (1=real, 0=padded)
        Returns:
            (batch, embed_dim) embedding
        """
        # Stack as 2-channel input: (batch, 2, lc_length)
        x = torch.stack([magnitudes, mask], dim=1)
        x = self.conv(x)           # (batch, 128, 1)
        x = x.squeeze(-1)          # (batch, 128)
        x = self.fc(x)             # (batch, embed_dim)
        return x


# ═══════════════════════════════════════════════════════════════
#  ORBITAL FEATURE BRANCH (MLP)
# ═══════════════════════════════════════════════════════════════

class OrbitalBranch(nn.Module):
    """MLP for TLE-derived orbital features.

    Input:  (batch, orbital_dim) feature vector
    Output: (batch, embed_dim) embedding
    """

    def __init__(self, orbital_dim: int = 60, embed_dim: int = 64,
                 hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(orbital_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ═══════════════════════════════════════════════════════════════
#  FUSION MODEL (FULL)
# ═══════════════════════════════════════════════════════════════

class FusionModel(nn.Module):
    """Late-fusion model combining light curve CNN and orbital MLP.

    Dual-head output:
      - A/m regression: predicts log10(area-to-mass ratio)
      - Object classification: predicts object type (payload/rocket body/debris/unknown)

    Args:
        lc_length:    Length of resampled light curve sequences.
        orbital_dim:  Number of TLE-derived orbital features.
        n_classes:    Number of classification classes.
        lc_embed_dim: Embedding dimension for LC branch.
        orb_embed_dim: Embedding dimension for orbital branch.
    """

    def __init__(self, lc_length: int = 256, orbital_dim: int = 60,
                 n_classes: int = 4, lc_embed_dim: int = 64,
                 orb_embed_dim: int = 64):
        super().__init__()

        self.lc_branch = LightCurveBranch(lc_length, lc_embed_dim)
        self.orbital_branch = OrbitalBranch(orbital_dim, orb_embed_dim)

        combined_dim = lc_embed_dim + orb_embed_dim

        # Shared fusion layers
        self.fusion_shared = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Regression head: A/m prediction
        self.am_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Classification head: object type
        self.class_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, lc_magnitudes: torch.Tensor, lc_masks: torch.Tensor,
                orbital_features: torch.Tensor) -> tuple:
        """
        Args:
            lc_magnitudes:    (batch, lc_length)
            lc_masks:         (batch, lc_length)
            orbital_features: (batch, orbital_dim)

        Returns:
            am_pred:    (batch, 1) log10(A/m) predictions
            class_pred: (batch, n_classes) class logits
        """
        lc_embed = self.lc_branch(lc_magnitudes, lc_masks)
        orb_embed = self.orbital_branch(orbital_features)

        combined = torch.cat([lc_embed, orb_embed], dim=1)
        shared = self.fusion_shared(combined)

        am_pred = self.am_head(shared)
        class_pred = self.class_head(shared)

        return am_pred, class_pred

    def get_embeddings(self, lc_magnitudes, lc_masks, orbital_features):
        """Return intermediate embeddings for analysis/visualisation."""
        lc_embed = self.lc_branch(lc_magnitudes, lc_masks)
        orb_embed = self.orbital_branch(orbital_features)
        combined = torch.cat([lc_embed, orb_embed], dim=1)
        shared = self.fusion_shared(combined)
        return {"lc_embed": lc_embed, "orb_embed": orb_embed, "fused": shared}


# ═══════════════════════════════════════════════════════════════
#  ABLATION: LIGHT CURVE ONLY
# ═══════════════════════════════════════════════════════════════

class LightCurveOnlyModel(nn.Module):
    """Standalone light curve model for ablation study."""

    def __init__(self, lc_length: int = 256, n_classes: int = 4,
                 embed_dim: int = 64, **kwargs):
        super().__init__()
        self.lc_branch = LightCurveBranch(lc_length, embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.am_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.class_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes))

    def forward(self, lc_magnitudes, lc_masks, orbital_features=None):
        embed = self.lc_branch(lc_magnitudes, lc_masks)
        shared = self.head(embed)
        return self.am_head(shared), self.class_head(shared)


# ═══════════════════════════════════════════════════════════════
#  ABLATION: ORBITAL ONLY
# ═══════════════════════════════════════════════════════════════

class OrbitalOnlyModel(nn.Module):
    """Standalone orbital feature model for ablation study."""

    def __init__(self, orbital_dim: int = 60, n_classes: int = 4,
                 embed_dim: int = 64, **kwargs):
        super().__init__()
        self.orbital_branch = OrbitalBranch(orbital_dim, embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.am_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.class_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes))

    def forward(self, lc_magnitudes=None, lc_masks=None, orbital_features=None):
        embed = self.orbital_branch(orbital_features)
        shared = self.head(embed)
        return self.am_head(shared), self.class_head(shared)


# ═══════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ═══════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "fusion": FusionModel,
    "lc_only": LightCurveOnlyModel,
    "orbital_only": OrbitalOnlyModel,
}


def create_model(model_type: str, **kwargs) -> nn.Module:
    """Create a model by name.

    Args:
        model_type: One of "fusion", "lc_only", "orbital_only".
        **kwargs:   Forwarded to the model constructor.

    Returns:
        nn.Module instance.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
