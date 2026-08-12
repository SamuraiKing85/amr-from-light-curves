"""
gui/tab_analysis.py
====================
Tab 5: Interactive ML dataset analysis.
Lets the user select input features and prediction targets,
then shows how many satellites have complete data for that combination.
"""

import tkinter as tk
from tkinter import ttk
import threading
import pandas as pd
import numpy as np
from pathlib import Path
from gui.theme import (T, FONT, FONT_BODY, FONT_SMALL, FONT_SMALL_B,
                       FONT_TINY, FONT_MONO_SM, styled_frame, styled_label,
                       styled_entry, styled_button, styled_lf, styled_cb,
                       styled_text, style_figure, style_axes, plot_color)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# --- Available Features and Targets -------------------------------------------

# (display_name, column_name_or_source, description)
INPUT_FEATURES = {
    "Light Curve Data (MMT-9)": {
        "Has MMT-9 light curves": ("has_lc", "mmt9", "MMT-9 light curve observations exist"),
        "MMT-9 obs (>100)": ("lc_obs_100", "mmt9", "At least 100 MMT-9 observations"),
        "MMT-9 obs (>1000)": ("lc_obs_1000", "mmt9", "At least 1000 MMT-9 observations"),
        "MMT-9 tracks (>3)": ("lc_tracks_3", "mmt9", "At least 3 MMT-9 observation tracks"),
        "MMT-9 magnitude stats": ("lc_mag_stats", "mmt9", "Mean/std StdMag from MMT-9"),
    },
    "Light Curve Data (SDLCD)": {
        "Has SDLCD light curves": ("has_sdlcd_lc", "sdlcd", "SDLCD light curve observations exist"),
        "SDLCD obs (>100)": ("sdlcd_obs_100", "sdlcd", "At least 100 SDLCD observations"),
        "SDLCD obs (>1000)": ("sdlcd_obs_1000", "sdlcd", "At least 1000 SDLCD observations"),
        "SDLCD tracks (>3)": ("sdlcd_tracks_3", "sdlcd", "At least 3 SDLCD observation tracks"),
        "SDLCD magnitude stats": ("sdlcd_mag_stats", "sdlcd", "Mean/std StdMag from SDLCD"),
    },
    "Light Curve Data (Any Source)": {
        "Has any light curves": ("has_any_lc", "any_lc", "Light curves from MMT-9 or SDLCD"),
    },
    "TLE / Orbital Data": {
        "Has TLE history": ("has_tle", "tle", "TLE/GP history records exist"),
        "TLE records (>50)": ("tle_50", "tle", "At least 50 TLE records"),
        "TLE records (>500)": ("tle_500", "tle", "At least 500 TLE records"),
        "B* drag term": ("has_bstar", "tle", "B* drag term available"),
        "Mean motion": ("has_mean_motion", "tle", "Mean motion data available"),
        "Orbital elements": ("has_orbital", "tle", "Inclination, eccentricity, SMA available"),
    },
    "Physical Properties (DISCOS)": {
        "Mass": ("has_mass", "discos", "Mass in kg known"),
        "Cross-section": ("has_xsect", "discos", "Average cross-section area known"),
        "Shape": ("has_shape", "discos", "Shape descriptor available"),
        "Dimensions": ("has_dims", "discos", "Length/height/depth available"),
        "A/m ratio": ("has_am", "discos", "Area-to-mass ratio computable"),
        "Object class": ("has_discos_class", "discos", "DISCOS object class label"),
    },
    "Catalogue Metadata": {
        "Object type (SATCAT)": ("has_obj_type", "satcat", "PAY/R/B/DEB classification"),
        "RCS value": ("has_rcs", "satcat", "Radar cross-section from SATCAT"),
        "Operational status": ("has_ops_status", "satcat", "Operational status code"),
        "Owner/Country": ("has_owner", "satcat", "Country/owner information"),
        "Orbit class (UCS)": ("has_orbit_class", "ucs", "LEO/MEO/GEO classification"),
        "Purpose (UCS)": ("has_purpose", "ucs", "Mission purpose"),
        "Launch mass (UCS)": ("has_launch_mass", "ucs", "Launch mass from UCS"),
    },
}

PREDICTION_TARGETS = {
    "A/m Ratio Estimation": {
        "col": "discos_am_ratio_avg",
        "type": "regression",
        "desc": "Predict area-to-mass ratio (m2/kg). Ground truth from DISCOS mass + cross-section.",
    },
    "Object Type Classification": {
        "col": "satcat_object_type",
        "type": "classification",
        "desc": "Predict PAY/R/B/DEB. Ground truth from SATCAT.",
    },
    "DISCOS Object Class": {
        "col": "discos_object_class",
        "type": "classification",
        "desc": "Predict Payload/Rocket Body/Debris. Ground truth from DISCOS.",
    },
    "Shape Classification": {
        "col": "discos_shape",
        "type": "classification",
        "desc": "Predict object shape (Box, Cyl, Sphere, etc). Ground truth from DISCOS.",
    },
    "Operational Status": {
        "col": "satcat_ops_status",
        "type": "classification",
        "desc": "Predict operational status code. Ground truth from SATCAT.",
    },
    "Orbit Class": {
        "col": "ucs_orbit_class",
        "type": "classification",
        "desc": "Predict LEO/MEO/GEO/Elliptical. Ground truth from UCS.",
    },
    "RCS Size Category": {
        "col": "tle_rcs_size",
        "type": "classification",
        "desc": "Predict SMALL/MEDIUM/LARGE. Ground truth from GP data.",
    },
    "Attitude State (derived)": {
        "col": "_derived_attitude",
        "type": "classification",
        "desc": "Predict stable/tumbling. Derived from light curve variability.",
    },
    "Orbital Lifetime (has decayed)": {
        "col": "satcat_decay_date",
        "type": "regression",
        "desc": "Predict whether/when object decays. Ground truth from SATCAT decay dates.",
    },
}


class AnalysisTab:
    def __init__(self, parent, app):
        self.app = app
        t = T()
        self.frame = tk.Frame(parent, bg=t["bg"])

        # Loaded data
        self.lc_cat = None
        self.tle_idx = None
        self.discos = None
        self.merged = None

        # Feature/target selections
        self.feature_vars = {}
        self.target_vars = {}

        self._build_ui()

    def _build_ui(self):
        # -- Top: Data status bar --
        status_bar = tk.Frame(self.frame, bg=T()["bg_card"],
                              highlightbackground=T()["border"],
                              highlightthickness=1)
        status_bar.pack(fill="x", padx=10, pady=(6, 3))

        styled_label(status_bar, text="Data Sources",
                     font=FONT_SMALL_B, parent_bg=T()["bg_card"]
                     ).pack(side="left", padx=(10, 6), pady=6)
        styled_label(status_bar, text="(loaded from Merge Data tab)",
                     dim=True, parent_bg=T()["bg_card"]
                     ).pack(side="left", padx=(0, 10), pady=6)
        self.load_status = styled_label(status_bar, text="No data loaded",
                                        dim=True, parent_bg=T()["bg_card"])
        self.load_status.pack(side="right", padx=10, pady=6)

        # -- Middle: Feature and Target Selection --
        selection_frame = tk.Frame(self.frame, bg=T()["bg"])
        selection_frame.pack(fill="x", padx=10, pady=(3, 3))

        # Left: Input features
        feat_frame = styled_lf(selection_frame, text="Input Features (tick what your model needs)", padx=6, pady=4, font=(FONT, 9, "bold"))
        feat_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        feat_canvas = tk.Canvas(feat_frame, bg=T()["bg"], highlightthickness=0, height=200)
        feat_scrollbar = tk.Scrollbar(feat_frame, orient="vertical", command=feat_canvas.yview)
        feat_inner = tk.Frame(feat_canvas, bg=T()["bg"])

        feat_inner.bind("<Configure>", lambda e: feat_canvas.configure(scrollregion=feat_canvas.bbox("all")))
        feat_canvas.create_window((0, 0), window=feat_inner, anchor="nw")
        feat_canvas.configure(yscrollcommand=feat_scrollbar.set)

        feat_canvas.pack(side="left", fill="both", expand=True)
        feat_scrollbar.pack(side="right", fill="y")

        for group_name, features in INPUT_FEATURES.items():
            styled_label(feat_inner, text=group_name, font=FONT_SMALL_B, fg=T()["fg_bright"]).pack(anchor="w", pady=(4, 0), padx=3)
            for display_name, (key, source, desc) in features.items():
                var = tk.BooleanVar(value=False)
                self.feature_vars[key] = (var, source, display_name)
                cb = styled_cb(feat_inner, text=display_name, variable=var, command=self._on_selection_change)
                cb.pack(anchor="w", padx=15)

        # Right: Prediction targets
        target_frame = styled_lf(selection_frame, text="Prediction Targets (tick what to predict)", padx=6, pady=4, font=(FONT, 9, "bold"))
        target_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

        for target_name, info in PREDICTION_TARGETS.items():
            var = tk.BooleanVar(value=False)
            self.target_vars[target_name] = (var, info)
            frame = tk.Frame(target_frame, bg=T()["bg"])
            frame.pack(anchor="w", fill="x", padx=3, pady=1)
            styled_cb(frame, text=target_name, variable=var, command=self._on_selection_change, font=FONT_SMALL_B).pack(side="left")
            tag = "REG" if info["type"] == "regression" else "CLS"
            color = T()["orange"] if info["type"] == "regression" else T()["accent"]
            styled_label(frame, text=f"[{tag}]", font=FONT_TINY, fg=color).pack(side="left", padx=3)
            styled_label(frame, text=info["desc"], font=FONT_TINY, fg=T()["fg_dim"], wraplength=350, justify="left").pack(side="left", padx=3)

        # Preset buttons
        preset_frame = tk.Frame(selection_frame, bg=T()["bg"])
        preset_frame.pack(side="right", fill="y", padx=5)

        styled_label(preset_frame, text="Presets", font=FONT_SMALL_B).pack(pady=(0, 3))

        presets = [
            ("A/m from LC+TLE", self._preset_am_lc_tle),
            ("A/m from all data", self._preset_am_all),
            ("Object type from LC+TLE", self._preset_objtype),
            ("Shape from LC+TLE", self._preset_shape),
            ("Multi-task (all 3)", self._preset_multitask),
            ("Clear all", self._preset_clear),
        ]
        for name, cmd in presets:
            styled_button(preset_frame, text=name, command=cmd, style="flat", width=20, font=FONT_TINY).pack(pady=1)

        # -- Bottom: Results --
        paned = tk.PanedWindow(self.frame, orient="horizontal", bg=T()["border"], sashwidth=4)
        paned.pack(fill="both", expand=True, padx=10, pady=(3, 8))

        # Left: text report
        report_frame = tk.Frame(paned, bg=T()["bg"])
        paned.add(report_frame, width=550)

        self.report_text = styled_text(report_frame, terminal=True, state="disabled")
        report_sb = tk.Scrollbar(report_frame, command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=report_sb.set)
        self.report_text.pack(side="left", fill="both", expand=True)
        report_sb.pack(side="right", fill="y")

        # Right: plots
        plot_frame = tk.Frame(paned, bg=T()["bg_plot"])
        paned.add(plot_frame, width=600)

        self.fig = Figure(figsize=(7, 6), dpi=100, facecolor=T()["bg_plot"])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _log(self, msg):
        self.report_text.configure(state="normal")
        self.report_text.insert("end", msg + "\n")
        self.report_text.see("end")
        self.report_text.configure(state="disabled")
        self.frame.update_idletasks()

    def _clear_log(self):
        self.report_text.configure(state="normal")
        self.report_text.delete("1.0", "end")
        self.report_text.configure(state="disabled")

    # -- Data Loading --

    def load_data(self, df, *args):
        """Called when merged data is available from other tabs (merger or explorer)."""
        if df is not None:
            self.merged = df
            self._update_load_status()
            self._on_selection_change()

    def set_source_data(self, lc_cat=None, sdlcd_cat=None, tle_idx=None, discos=None, merged=None):
        """Accept pre-loaded DataFrames from other tabs."""
        if lc_cat is not None:
            self.lc_cat = lc_cat
        if sdlcd_cat is not None:
            self.sdlcd_cat = sdlcd_cat
        if tle_idx is not None:
            self.tle_idx = tle_idx
        if discos is not None:
            self.discos = discos
        if merged is not None:
            self.merged = merged
        self._update_load_status()
        self._on_selection_change()

    def _update_load_status(self):
        parts = []
        if self.lc_cat is not None:
            parts.append(f"MMT-9: {len(self.lc_cat):,}")
        if self.sdlcd_cat is not None:
            parts.append(f"SDLCD: {len(self.sdlcd_cat):,}")
        if self.tle_idx is not None:
            parts.append(f"TLE: {len(self.tle_idx):,}")
        if self.discos is not None:
            parts.append(f"DISCOS: {len(self.discos):,}")
        if self.merged is not None:
            parts.append(f"Merged: {len(self.merged):,}")
        if parts:
            self.load_status.configure(text=" | ".join(parts), fg=T()["success"])
        else:
            self.load_status.configure(text="No data loaded", fg=T()["fg_dim"])

    # -- Feature Availability Checking --

    def _check_feature(self, key, norad_ids):
        """Return set of NORAD IDs that satisfy a given feature requirement."""
        if key == "has_lc" and self.lc_cat is not None:
            return set(self.lc_cat["norad_id"].dropna().astype(int))

        elif key == "lc_obs_100" and self.lc_cat is not None:
            if "total_observations" in self.lc_cat.columns:
                return set(self.lc_cat[self.lc_cat["total_observations"] >= 100]["norad_id"].dropna().astype(int))
            return set()

        elif key == "lc_obs_1000" and self.lc_cat is not None:
            if "total_observations" in self.lc_cat.columns:
                return set(self.lc_cat[self.lc_cat["total_observations"] >= 1000]["norad_id"].dropna().astype(int))
            return set()

        elif key == "lc_tracks_3" and self.lc_cat is not None:
            if "num_tracks" in self.lc_cat.columns:
                return set(self.lc_cat[self.lc_cat["num_tracks"] >= 3]["norad_id"].dropna().astype(int))
            return set()

        elif key == "lc_mag_stats" and self.lc_cat is not None:
            if "mean_stdmag" in self.lc_cat.columns:
                return set(self.lc_cat[self.lc_cat["mean_stdmag"].notna()]["norad_id"].dropna().astype(int))
            return set()

        # -- SDLCD Light Curve features --
        elif key == "has_sdlcd_lc" and self.sdlcd_cat is not None:
            return set(self.sdlcd_cat["norad_id"].dropna().astype(int))

        elif key == "sdlcd_obs_100" and self.sdlcd_cat is not None:
            if "total_observations" in self.sdlcd_cat.columns:
                return set(self.sdlcd_cat[self.sdlcd_cat["total_observations"] >= 100]["norad_id"].dropna().astype(int))
            return set()

        elif key == "sdlcd_obs_1000" and self.sdlcd_cat is not None:
            if "total_observations" in self.sdlcd_cat.columns:
                return set(self.sdlcd_cat[self.sdlcd_cat["total_observations"] >= 1000]["norad_id"].dropna().astype(int))
            return set()

        elif key == "sdlcd_tracks_3" and self.sdlcd_cat is not None:
            if "num_tracks" in self.sdlcd_cat.columns:
                return set(self.sdlcd_cat[self.sdlcd_cat["num_tracks"] >= 3]["norad_id"].dropna().astype(int))
            return set()

        elif key == "sdlcd_mag_stats" and self.sdlcd_cat is not None:
            if "mean_stdmag" in self.sdlcd_cat.columns:
                return set(self.sdlcd_cat[self.sdlcd_cat["mean_stdmag"].notna()]["norad_id"].dropna().astype(int))
            return set()

        # -- Any light curve source --
        elif key == "has_any_lc":
            ids = set()
            if self.lc_cat is not None:
                ids |= set(self.lc_cat["norad_id"].dropna().astype(int))
            if self.sdlcd_cat is not None:
                ids |= set(self.sdlcd_cat["norad_id"].dropna().astype(int))
            return ids

        elif key == "has_tle" and self.tle_idx is not None:
            return set(self.tle_idx["norad_id"].dropna().astype(int))

        elif key == "tle_50" and self.tle_idx is not None:
            return set(self.tle_idx[self.tle_idx["total_gp_records"] >= 50]["norad_id"].dropna().astype(int))

        elif key == "tle_500" and self.tle_idx is not None:
            return set(self.tle_idx[self.tle_idx["total_gp_records"] >= 500]["norad_id"].dropna().astype(int))

        elif key in ("has_bstar", "has_mean_motion", "has_orbital") and self.tle_idx is not None:
            # These three filters are aliases of has_tle. BSTAR, MEAN_MOTION,
            # INCLINATION and ECCENTRICITY are standard fields in every TLE
            # record, so any satellite with TLE history has them all. The
            # previous implementation looked for aggregated mean_* columns
            # in tle_index.csv that the main pipeline does not write, which
            # caused the filters to report 0 satellites. Physical-significance
            # filtering (e.g. BSTAR != 0 for drag-affected orbits) belongs in
            # the experiment scripts where the semantics can be made explicit.
            return set(self.tle_idx["norad_id"].dropna().astype(int))

        elif key == "has_mass" and self.discos is not None:
            return set(self.discos[self.discos["mass_kg"].notna()]["norad_id"].dropna().astype(int))

        elif key == "has_xsect" and self.discos is not None:
            return set(self.discos[self.discos["xsect_avg_m2"].notna()]["norad_id"].dropna().astype(int))

        elif key == "has_shape" and self.discos is not None:
            if "shape" in self.discos.columns:
                return set(self.discos[self.discos["shape"].notna()]["norad_id"].dropna().astype(int))
            return set()

        elif key == "has_dims" and self.discos is not None:
            if all(c in self.discos.columns for c in ["length_m", "height_m", "depth_m"]):
                mask = self.discos[["length_m", "height_m", "depth_m"]].notna().all(axis=1)
                return set(self.discos[mask]["norad_id"].dropna().astype(int))
            return set()

        elif key == "has_am" and self.discos is not None:
            return set(self.discos[self.discos["am_ratio_avg"].notna()]["norad_id"].dropna().astype(int))

        elif key == "has_discos_class" and self.discos is not None:
            if "object_class" in self.discos.columns:
                return set(self.discos[self.discos["object_class"].notna()]["norad_id"].dropna().astype(int))
            return set()

        elif key == "has_obj_type" and self.merged is not None:
            if "satcat_object_type" in self.merged.columns:
                return set(self.merged[self.merged["satcat_object_type"].notna()]["norad_id"].dropna().astype(int))
            return set()

        elif key == "has_rcs" and self.merged is not None:
            if "satcat_rcs_m2" in self.merged.columns:
                return set(self.merged[self.merged["satcat_rcs_m2"].notna()]["norad_id"].dropna().astype(int))
            return set()

        elif key == "has_ops_status" and self.merged is not None:
            if "satcat_ops_status" in self.merged.columns:
                return set(self.merged[self.merged["satcat_ops_status"].notna()]["norad_id"].dropna().astype(int))
            return set()

        elif key == "has_owner" and self.merged is not None:
            if "satcat_owner" in self.merged.columns:
                return set(self.merged[self.merged["satcat_owner"].notna()]["norad_id"].dropna().astype(int))
            return set()

        elif key == "has_orbit_class" and self.merged is not None:
            if "ucs_orbit_class" in self.merged.columns:
                return set(self.merged[self.merged["ucs_orbit_class"].notna()]["norad_id"].dropna().astype(int))
            return set()

        elif key == "has_purpose" and self.merged is not None:
            if "ucs_purpose" in self.merged.columns:
                return set(self.merged[self.merged["ucs_purpose"].notna()]["norad_id"].dropna().astype(int))
            return set()

        elif key == "has_launch_mass" and self.merged is not None:
            if "ucs_launch_mass_kg" in self.merged.columns:
                return set(self.merged[self.merged["ucs_launch_mass_kg"].notna()]["norad_id"].dropna().astype(int))
            return set()

        return set()

    def _check_target(self, target_name, norad_ids):
        """Return set of NORAD IDs that have ground truth for a target."""
        info = PREDICTION_TARGETS[target_name]
        col = info["col"]

        if col == "_derived_attitude":
            # Derived from light curve variability - need LC data from either source
            ids = set()
            if self.lc_cat is not None and "std_stdmag" in self.lc_cat.columns:
                ids |= set(self.lc_cat[self.lc_cat["std_stdmag"].notna()]["norad_id"].dropna().astype(int))
            if self.sdlcd_cat is not None and "std_stdmag" in self.sdlcd_cat.columns:
                ids |= set(self.sdlcd_cat[self.sdlcd_cat["std_stdmag"].notna()]["norad_id"].dropna().astype(int))
            return ids

        if col == "satcat_decay_date":
            # For decay prediction, objects that HAVE decayed provide ground truth
            if self.merged is not None and col in self.merged.columns:
                return set(self.merged[self.merged[col].notna()]["norad_id"].dropna().astype(int))
            return set()

        # Check in merged first, then individual sources
        for df in [self.merged, self.discos, self.tle_idx]:
            if df is not None and col in df.columns:
                return set(df[df[col].notna()]["norad_id"].dropna().astype(int))

        return set()

    # -- Analysis --

    def _on_selection_change(self):
        """Recalculate when any checkbox changes."""
        if not any([self.lc_cat is not None, self.tle_idx is not None,
                    self.discos is not None, self.merged is not None]):
            return

        self._clear_log()

        # Get selected features
        selected_features = []
        for key, (var, source, name) in self.feature_vars.items():
            if var.get():
                selected_features.append((key, source, name))

        # Get selected targets
        selected_targets = []
        for tname, (var, info) in self.target_vars.items():
            if var.get():
                selected_targets.append((tname, info))

        if not selected_features and not selected_targets:
            self._log("Tick input features and prediction targets to see")
            self._log("how many satellites have complete data.")
            return

        self._log("=" * 50)
        self._log("  ML DATASET FEASIBILITY ANALYSIS")
        self._log("=" * 50)

        # Check each feature individually
        feature_sets = {}
        self._log(f"\n  SELECTED INPUT FEATURES:")
        self._log(f"  {'-' * 45}")

        for key, source, name in selected_features:
            ids = self._check_feature(key, None)
            feature_sets[key] = ids
            self._log(f"    {name:<35s} {len(ids):>8,}")

        # Intersection of all selected features
        if feature_sets:
            all_feature_ids = set.intersection(*feature_sets.values())
        else:
            all_feature_ids = set()

        self._log(f"\n  Combined (all features):            {len(all_feature_ids):>8,}")

        # Check each target
        target_sets = {}
        if selected_targets:
            self._log(f"\n  SELECTED PREDICTION TARGETS:")
            self._log(f"  {'-' * 45}")

            for tname, info in selected_targets:
                ids = self._check_target(tname, None)
                target_sets[tname] = ids
                tag = "REG" if info["type"] == "regression" else "CLS"
                self._log(f"    [{tag}] {tname:<30s} {len(ids):>8,}")

        # Combined: features AND each target
        if feature_sets and target_sets:
            self._log(f"\n  USABLE TRAINING SETS (features + target):")
            self._log(f"  {'-' * 45}")

            training_sizes = {}
            for tname, info in selected_targets:
                usable = all_feature_ids & target_sets[tname]
                training_sizes[tname] = usable
                self._log(f"    {tname:<35s} {len(usable):>8,}")

            # Multi-task: intersection of ALL targets
            if len(target_sets) > 1:
                all_targets_ids = set.intersection(*target_sets.values())
                multi_task = all_feature_ids & all_targets_ids
                self._log(f"\n    Multi-task (all targets):          {len(multi_task):>8,}")

            # Detailed breakdown for each training set
            for tname, usable_ids in training_sizes.items():
                if not usable_ids:
                    continue

                info = [i for t, i in selected_targets if t == tname][0]
                self._log(f"\n  {'=' * 50}")
                self._log(f"  DETAIL: {tname}")
                self._log(f"  {'=' * 50}")
                self._log(f"  Usable samples: {len(usable_ids):,}")

                # Object class breakdown if available
                if self.merged is not None and "satcat_object_type" in self.merged.columns:
                    subset = self.merged[self.merged["norad_id"].isin(usable_ids)]
                    if not subset.empty:
                        self._log(f"\n  By object type:")
                        for t, c in subset["satcat_object_type"].value_counts().items():
                            pct = 100 * c / len(subset)
                            self._log(f"    {str(t):<20s} {c:>6,}  ({pct:.1f}%)")

                # Target distribution
                if info["type"] == "regression":
                    col = info["col"]
                    for df in [self.merged, self.discos]:
                        if df is not None and col in df.columns:
                            vals = df[df["norad_id"].isin(usable_ids)][col].dropna().astype(float)
                            if not vals.empty:
                                self._log(f"\n  Target distribution:")
                                self._log(f"    Mean:     {vals.mean():>12.6f}")
                                self._log(f"    Median:   {vals.median():>12.6f}")
                                self._log(f"    Std:      {vals.std():>12.6f}")
                                self._log(f"    Range:    [{vals.min():.6f}, {vals.max():.6f}]")
                            break

                elif info["type"] == "classification":
                    col = info["col"]
                    if col != "_derived_attitude":
                        for df in [self.merged, self.discos]:
                            if df is not None and col in df.columns:
                                vals = df[df["norad_id"].isin(usable_ids)][col].dropna()
                                if not vals.empty:
                                    self._log(f"\n  Class distribution:")
                                    for cls, cnt in vals.value_counts().items():
                                        pct = 100 * cnt / len(vals)
                                        self._log(f"    {str(cls):<25s} {cnt:>6,}  ({pct:.1f}%)")
                                break

                # Suggested split
                n = len(usable_ids)
                if n > 0:
                    self._log(f"\n  Suggested split:")
                    self._log(f"    Train (70%): {int(n * 0.7):>8,}")
                    self._log(f"    Val   (15%): {int(n * 0.15):>8,}")
                    self._log(f"    Test  (15%): {n - int(n * 0.7) - int(n * 0.15):>8,}")

        # Update plots
        self._update_plots(feature_sets, target_sets, all_feature_ids,
                           training_sizes if (feature_sets and target_sets) else {})

    def _update_plots(self, feature_sets, target_sets, all_feature_ids, training_sizes):
        self.fig.clear()

        n_plots = 0
        if feature_sets:
            n_plots += 1
        if target_sets:
            n_plots += 1
        if training_sizes:
            n_plots += 1
        if any(training_sizes.values()):
            n_plots += 1

        if n_plots == 0:
            self.canvas.draw()
            return

        plot_idx = 1

        # Plot 1: Feature availability
        if feature_sets:
            ax = self.fig.add_subplot(2, 2, plot_idx)
            ax.set_facecolor(T()["bg_plot_ax"])
            names = [self.feature_vars[k][2][:25] for k in feature_sets]
            counts = [len(v) for v in feature_sets.values()]
            colors = ["#3498db"] * len(counts)
            ax.barh(names, counts, color=colors, edgecolor=T()["bg_plot_ax"], linewidth=0.5)
            ax.axvline(x=len(all_feature_ids), color=T()["danger"], linestyle="--",
                       linewidth=1, label=f"Combined: {len(all_feature_ids):,}")
            ax.legend(fontsize=7)
            ax.set_xlabel("Satellites", fontsize=8)
            ax.set_title("Feature Availability", fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=7)
            plot_idx += 1

        # Plot 2: Training set sizes
        if training_sizes:
            ax = self.fig.add_subplot(2, 2, plot_idx)
            ax.set_facecolor(T()["bg_plot_ax"])
            names = [n[:30] for n in training_sizes.keys()]
            sizes = [len(v) for v in training_sizes.values()]
            colors = ["#27ae60" if s >= 1000 else "#f39c12" if s >= 100 else "#e74c3c"
                      for s in sizes]
            bars = ax.barh(names, sizes, color=colors, edgecolor=T()["bg_plot_ax"], linewidth=0.5)
            for bar, val in zip(bars, sizes):
                ax.text(bar.get_width() + max(sizes) * 0.02 if sizes else 0,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:,}", va="center", fontsize=7)
            ax.set_xlabel("Usable Training Samples", fontsize=8)
            ax.set_title("Training Set Sizes", fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=7)
            plot_idx += 1

        # Plot 3: Object type breakdown for largest training set
        if training_sizes and self.merged is not None:
            largest_name = max(training_sizes, key=lambda k: len(training_sizes[k]))
            largest_ids = training_sizes[largest_name]
            if largest_ids and "satcat_object_type" in self.merged.columns:
                subset = self.merged[self.merged["norad_id"].isin(largest_ids)]
                type_counts = subset["satcat_object_type"].value_counts()
                if not type_counts.empty:
                    ax = self.fig.add_subplot(2, 2, plot_idx)
                    ax.set_facecolor(T()["bg_plot_ax"])
                    pie_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
                    ax.pie(type_counts.values, labels=type_counts.index,
                           autopct="%1.1f%%",
                           colors=pie_colors[:len(type_counts)],
                           textprops={"fontsize": 7})
                    ax.set_title(f"Object Types: {largest_name[:25]}", fontsize=8, fontweight="bold")
                    plot_idx += 1

        # Plot 4: Target distribution for regression targets
        if training_sizes:
            for tname, usable_ids in training_sizes.items():
                tinfo = PREDICTION_TARGETS[tname]
                if tinfo["type"] == "regression" and usable_ids and plot_idx <= 4:
                    col = tinfo["col"]
                    for df in [self.merged, self.discos]:
                        if df is not None and col in df.columns:
                            vals = df[df["norad_id"].isin(usable_ids)][col].dropna().astype(float)
                            if not vals.empty and len(vals) > 10:
                                ax = self.fig.add_subplot(2, 2, plot_idx)
                                ax.set_facecolor(T()["bg_plot_ax"])
                                log_vals = np.log10(vals[vals > 0])
                                ax.hist(log_vals, bins=50, color="#8e44ad",
                                        alpha=0.8, edgecolor=T()["bg_plot_ax"], linewidth=0.5)
                                ax.set_xlabel(f"log10({col.split('_')[-1]})", fontsize=8)
                                ax.set_ylabel("Count", fontsize=8)
                                ax.set_title(f"Target: {tname[:25]}", fontsize=8, fontweight="bold")
                                ax.tick_params(labelsize=7)
                                plot_idx += 1
                            break

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()

    # -- Presets --

    def _preset_clear(self):
        for key, (var, _, _) in self.feature_vars.items():
            var.set(False)
        for tname, (var, _) in self.target_vars.items():
            var.set(False)
        self._on_selection_change()

    def _preset_am_lc_tle(self):
        self._preset_clear()
        for key in ["has_any_lc", "lc_obs_100", "sdlcd_obs_100", "has_tle", "tle_50", "has_bstar"]:
            if key in self.feature_vars:
                self.feature_vars[key][0].set(True)
        for tname in ["A/m Ratio Estimation"]:
            if tname in self.target_vars:
                self.target_vars[tname][0].set(True)
        self._on_selection_change()

    def _preset_am_all(self):
        self._preset_clear()
        for key in ["has_any_lc", "lc_obs_100", "sdlcd_obs_100", "has_tle", "tle_50", "has_bstar",
                     "has_orbital", "has_rcs", "has_obj_type"]:
            if key in self.feature_vars:
                self.feature_vars[key][0].set(True)
        for tname in ["A/m Ratio Estimation"]:
            if tname in self.target_vars:
                self.target_vars[tname][0].set(True)
        self._on_selection_change()

    def _preset_objtype(self):
        self._preset_clear()
        for key in ["has_any_lc", "has_tle", "tle_50", "has_bstar"]:
            if key in self.feature_vars:
                self.feature_vars[key][0].set(True)
        for tname in ["Object Type Classification"]:
            if tname in self.target_vars:
                self.target_vars[tname][0].set(True)
        self._on_selection_change()

    def _preset_multitask(self):
        self._preset_clear()
        for key in ["has_any_lc", "lc_obs_100", "sdlcd_obs_100", "has_tle", "tle_50", "has_bstar",
                     "has_orbital"]:
            if key in self.feature_vars:
                self.feature_vars[key][0].set(True)
        for tname in ["A/m Ratio Estimation", "Object Type Classification",
                       "Shape Classification"]:
            if tname in self.target_vars:
                self.target_vars[tname][0].set(True)
        self._on_selection_change()

    def _preset_shape(self):
        self._preset_clear()
        for key in ["has_any_lc", "lc_obs_100", "sdlcd_obs_100", "has_tle", "tle_50", "has_bstar"]:
            if key in self.feature_vars:
                self.feature_vars[key][0].set(True)
        for tname in ["Shape Classification"]:
            if tname in self.target_vars:
                self.target_vars[tname][0].set(True)
        self._on_selection_change()