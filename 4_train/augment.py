"""
Light Curve Augmentation
============================
Physically motivated augmentations for space debris light curves.
Applied on-the-fly during training to minority classes only.

Transforms:
  1. Gaussian jitter  - adds noise scaled to track std, simulates
                         varying atmospheric/detector conditions.
  2. Amplitude scaling - multiplies brightness by a random factor,
                         simulates different viewing geometries
                         and phase angles.

Usage:
    augmenter = LightCurveAugmenter()
    mag_aug, mask_aug = augmenter(magnitudes, mask, class_target)

    # Class-aware: augments Debris always, Rocket Body 50%, Payload never.
    # Returns unmodified copies for classes that aren't augmented.
"""

import numpy as np


class LightCurveAugmenter:
    """On-the-fly light curve augmentation for minority class oversampling.

    Only augments samples from underrepresented classes. Majority class
    samples pass through unmodified to preserve the original signal
    distribution for well-represented categories.

    Args:
        jitter_sigma:    Noise level as fraction of per-track std (default 0.1).
        scale_range:     Min/max amplitude scaling factor (default 0.85-1.15).
        jitter_prob:     Probability of applying jitter (default 0.8).
        scale_prob:      Probability of applying scaling (default 0.5).
        class_config:    Dict mapping class index to augmentation probability.
                         1.0 = always augment, 0.0 = never augment.
                         Default: {0: 0.0, 1: 0.3, 2: 1.0}
                         (Payload=never, RocketBody=30%, Debris=always)
    """

    # Default class indices (matching TYPE_ENCODING in train.py)
    PAYLOAD = 0
    ROCKET_BODY = 1
    DEBRIS = 2

    def __init__(self,
                 jitter_sigma: float = 0.1,
                 scale_range: tuple = (0.85, 1.15),
                 jitter_prob: float = 0.8,
                 scale_prob: float = 0.5,
                 class_config: dict = None):
        self.jitter_sigma = jitter_sigma
        self.scale_lo, self.scale_hi = scale_range
        self.jitter_prob = jitter_prob
        self.scale_prob = scale_prob

        # Per-class probability of applying any augmentation
        self.class_config = class_config or {
            self.PAYLOAD: 0.0,       # Majority class: never augment
            self.ROCKET_BODY: 0.3,   # Moderate minority: augment sometimes
            self.DEBRIS: 1.0,        # Severe minority: always augment
        }

    def should_augment(self, class_target: int) -> bool:
        """Decide whether to augment based on class membership."""
        prob = self.class_config.get(class_target, 0.0)
        if prob <= 0.0:
            return False
        if prob >= 1.0:
            return True
        return np.random.random() < prob

    def apply_jitter(self, magnitudes: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add Gaussian noise scaled to the track's own standard deviation.

        Only adds noise to real (unmasked) data points. Padded regions
        remain zero so the mask channel stays consistent.
        """
        real_mask = mask > 0.5
        if real_mask.sum() < 2:
            return magnitudes

        track_std = magnitudes[real_mask].std()
        if track_std < 1e-8:
            return magnitudes

        noise = np.random.normal(0, self.jitter_sigma * track_std,
                                 size=magnitudes.shape).astype(magnitudes.dtype)
        # Only apply noise to real points
        result = magnitudes.copy()
        result[real_mask] += noise[real_mask]
        return result

    def apply_scaling(self, magnitudes: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Scale brightness by a random factor within the configured range.

        Simulates the same object observed at a slightly different
        phase angle or distance. Only scales real data points.
        """
        real_mask = mask > 0.5
        if real_mask.sum() < 2:
            return magnitudes

        scale = np.random.uniform(self.scale_lo, self.scale_hi)
        result = magnitudes.copy()
        result[real_mask] *= scale
        return result

    def __call__(self, magnitudes: np.ndarray, mask: np.ndarray,
                 class_target: int) -> tuple:
        """Apply augmentation pipeline to a single light curve.

        Args:
            magnitudes:   (lc_length,) normalised magnitude array.
            mask:         (lc_length,) binary mask (1=real, 0=padded).
            class_target: Integer class label (0=Payload, 1=RB, 2=Debris).

        Returns:
            (magnitudes, mask) tuple. Mask is never modified.
            Returns copies - originals are never mutated.
        """
        if not self.should_augment(class_target):
            return magnitudes, mask

        result = magnitudes.copy()

        if np.random.random() < self.jitter_prob:
            result = self.apply_jitter(result, mask)

        if np.random.random() < self.scale_prob:
            result = self.apply_scaling(result, mask)

        return result, mask
