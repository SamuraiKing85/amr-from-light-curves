"""
gui/tab_tle.py
===============
Tab 4: TLE / Orbital history visualisations.
"""

import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from gui.theme import (T, FONT, FONT_BODY, FONT_SMALL, FONT_SMALL_B,
                       FONT_MONO_SM, styled_frame, card_frame, styled_label,
                       styled_entry, styled_button, styled_listbox,
                       style_figure, style_axes, style_axes_empty,
                       style_legend, plot_color)
from data.loaders import load_tle_history


PLOT_TYPES = [
    ("Altitude (Apogee/Perigee)", "altitude"),
    ("Semi-Major Axis", "sma"),
    ("B* Drag Term", "bstar"),
    ("Mean Motion", "mean_motion"),
    ("Eccentricity", "eccentricity"),
    ("Inclination", "inclination"),
    ("Period", "period"),
    ("Mean Motion Derivative (Decay Rate)", "ndot"),
    ("All Orbital Elements", "all_elements"),
]


class TLETab:
    def __init__(self, parent, app):
        self.app = app
        t = T()
        self.frame = tk.Frame(parent, bg=t["bg"])
        self.df = None
        self.tle_dir = None
        self.current_tle = None
        self.current_norad = None
        # Pre-built {norad_id: name} mapping, rebuilt on every load_data().
        # See LightCurveTab._build_name_lookup for the rationale.
        self._name_lookup = {}
        self._build_ui()

    def _build_ui(self):
        t = T()

        # -- Top bar --
        top = tk.Frame(self.frame, bg=t["bg_card"], pady=6)
        top.pack(fill="x", padx=8, pady=(8, 0))

        styled_label(top, text="Search:", parent_bg=t["bg_card"]
                     ).pack(side="left", padx=(8, 4))

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._filter_list())
        styled_entry(top, textvariable=self.search_var, width=25
                     ).pack(side="left", padx=(0, 10), ipady=2)

        self.status_label = styled_label(top, text="No data loaded", dim=True,
                                         parent_bg=t["bg_card"])
        self.status_label.pack(side="left", padx=10)

        # Plot type selector
        opt_frame = tk.Frame(top, bg=t["bg_card"])
        opt_frame.pack(side="right", padx=8)
        styled_label(opt_frame, text="Plot:", parent_bg=t["bg_card"]
                     ).pack(side="left", padx=(0, 4))

        self.plot_type = tk.StringVar(value="altitude")
        self.plot_combo = ttk.Combobox(
            opt_frame, textvariable=self.plot_type, width=30,
            values=[name for name, _ in PLOT_TYPES],
            state="readonly", font=FONT_SMALL)
        self.plot_combo.pack(side="left", padx=3)
        self.plot_combo.current(0)
        self.plot_combo.bind("<<ComboboxSelected>>", lambda _: self._replot())

        # -- Main content --
        body = tk.Frame(self.frame, bg=t["bg"])
        body.pack(fill="both", expand=True, padx=8, pady=8)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Left: satellite list
        left = card_frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        hdr = tk.Frame(left, bg=t["bg_card"])
        hdr.pack(fill="x", padx=10, pady=(10, 4))
        styled_label(hdr, text="Satellites with TLE Data", heading=True,
                     parent_bg=t["bg_card"]).pack(anchor="w")

        list_container = tk.Frame(left, bg=t["bg_card"])
        list_container.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        self.sat_listbox = styled_listbox(list_container)
        lb_scroll = tk.Scrollbar(list_container, command=self.sat_listbox.yview)
        self.sat_listbox.configure(yscrollcommand=lb_scroll.set)
        self.sat_listbox.pack(side="left", fill="both", expand=True)
        lb_scroll.pack(side="right", fill="y")
        self.sat_listbox.bind("<<ListboxSelect>>", self._on_select)

        info_card = tk.Frame(left, bg=t["list_bg"],
                             highlightbackground=t["border"],
                             highlightthickness=1)
        info_card.pack(fill="x", padx=6, pady=(2, 8))
        self.info_label = tk.Label(info_card, text="", font=FONT_SMALL,
                                   fg=t["fg_dim"], bg=t["list_bg"],
                                   wraplength=260, justify="left", padx=8, pady=6)
        self.info_label.pack(fill="x")

        # Navigation buttons
        nav_frame = tk.Frame(left, bg=t["bg_card"])
        nav_frame.pack(fill="x", padx=6, pady=(0, 6))
        self.nav_explorer_btn = styled_button(nav_frame, text="Explorer \u2192",
                                              command=self._nav_to_explorer, style="primary")
        self.nav_explorer_btn.configure(pady=2, width=12)
        self.nav_lc_btn = styled_button(nav_frame, text="Light Curve \u2192",
                                        command=self._nav_to_lc, style="primary")
        self.nav_lc_btn.configure(pady=2, width=12)
        # Start hidden — shown when a satellite is selected

        # Right: plot
        right = card_frame(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        self.fig = Figure(figsize=(10, 6), dpi=100)
        style_figure(self.fig)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(right, bg=t["bg_card"])
        toolbar_frame.pack(side="bottom", fill="x")
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        # Placeholder
        ax = self.fig.add_subplot(111)
        style_axes_empty(ax, "Select a satellite to view its orbital history")
        self.canvas.draw()

        self._sat_list_data = []

    # -- Data ----------------------------------------------------------

    def load_data(self, df, tle_dir=None):
        self.df = df
        if tle_dir:
            self.tle_dir = tle_dir
        self._build_name_lookup()
        self._populate_list()

    def _build_name_lookup(self):
        """Build the {norad_id -> name} mapping in one pass over the merged
        DataFrame. Replaces the previous per-row boolean-mask filter, which
        was O(N) per call and O(N^2) overall when populating long lists."""
        self._name_lookup = {}
        df = self.df
        if df is None or 'norad_id' not in df.columns:
            return
        name_cols = [c for c in ('satcat_object_name', 'discos_name',
                                 'ucs_name', 'mmt9_name')
                     if c in df.columns]
        if not name_cols:
            return
        sub = df[['norad_id'] + name_cols].dropna(subset=['norad_id'])
        nids = sub['norad_id'].tolist()
        col_lists = [sub[c].tolist() for c in name_cols]
        for i, nid in enumerate(nids):
            try:
                nid_int = int(nid)
            except (ValueError, TypeError):
                continue
            if nid_int in self._name_lookup:
                continue  # first match wins (matches previous behaviour)
            for lst in col_lists:
                v = lst[i]
                if v is not None and pd.notna(v) and str(v) != 'nan':
                    self._name_lookup[nid_int] = str(v)
                    break

    def _populate_list(self):
        self.sat_listbox.delete(0, "end")
        self._sat_list_data = []

        if self.df is None or not self.tle_dir:
            self.status_label.configure(text="No TLE directory set")
            return

        tle_path = Path(self.tle_dir)
        if not tle_path.is_dir():
            self.status_label.configure(text=f"Directory not found: {self.tle_dir}")
            return

        tle_files = {int(f.stem): f for f in tle_path.glob("*.parquet") if f.stem.isdigit()}
        if not tle_files:
            self.status_label.configure(text="No TLE files found")
            return

        entries = []
        for norad_id in sorted(tle_files.keys()):
            name = self._get_sat_name(norad_id)
            display = f"{norad_id:>6d}  {name}" if name else f"{norad_id:>6d}"
            entries.append((norad_id, display))

        self._sat_list_data = entries
        for _, display in entries:
            self.sat_listbox.insert("end", display)

        self.status_label.configure(text=f"{len(entries)} satellites with TLE data")

    def _get_sat_name(self, norad_id):
        """O(1) name lookup against the pre-built dict."""
        try:
            return self._name_lookup.get(int(norad_id), "")
        except (ValueError, TypeError):
            return ""

    def _filter_list(self):
        search = self.search_var.get().strip().lower()
        self.sat_listbox.delete(0, "end")
        for norad_id, display in self._sat_list_data:
            if search in display.lower() or search in str(norad_id):
                self.sat_listbox.insert("end", display)

    def _on_select(self, event):
        selection = self.sat_listbox.curselection()
        if not selection:
            return
        display_text = self.sat_listbox.get(selection[0])
        try:
            norad_id = int(display_text.strip().split()[0])
        except (ValueError, IndexError):
            return
        self._load_and_plot(norad_id)

    def _load_and_plot(self, norad_id):
        if not self.tle_dir:
            return

        tle = load_tle_history(self.tle_dir, norad_id)
        if tle.empty:
            self.info_label.configure(text="No TLE data found")
            return

        self.current_tle = tle
        self.current_norad = norad_id

        t = T()
        n_records = len(tle)
        epoch_min = tle["EPOCH"].min().strftime("%Y-%m-%d") if pd.notna(tle["EPOCH"].min()) else "?"
        epoch_max = tle["EPOCH"].max().strftime("%Y-%m-%d") if pd.notna(tle["EPOCH"].max()) else "?"

        info_parts = [f"Records: {n_records:,}", f"Epochs: {epoch_min} to {epoch_max}"]
        if "BSTAR" in tle.columns:
            bstar = tle["BSTAR"].dropna()
            if not bstar.empty:
                info_parts.append(f"B* mean: {bstar.mean():.6f}")
        if "APOAPSIS" in tle.columns and "PERIAPSIS" in tle.columns:
            apo = tle["APOAPSIS"].dropna()
            peri = tle["PERIAPSIS"].dropna()
            if not apo.empty:
                info_parts.append(f"Apogee: {apo.iloc[-1]:.0f} km")
            if not peri.empty:
                info_parts.append(f"Perigee: {peri.iloc[-1]:.0f} km")

        self.info_label.configure(text="\n".join(info_parts), fg=t["fg"])
        self._update_nav_buttons(norad_id)
        self._replot()

    # -- Plotting ------------------------------------------------------

    def _replot(self):
        if self.current_tle is None:
            return

        t = T()
        tle = self.current_tle
        norad_id = self.current_norad
        name = self._get_sat_name(norad_id)
        title = f"NORAD {norad_id}" + (f" - {name}" if name else "")

        selected_name = self.plot_type.get()
        plot_key = "altitude"
        for pname, pkey in PLOT_TYPES:
            if pname == selected_name:
                plot_key = pkey
                break

        self.fig.clear()
        style_figure(self.fig)

        if plot_key == "all_elements":
            self._plot_all_elements(tle, title)
        elif plot_key == "altitude":
            self._plot_altitude(tle, title)
        elif plot_key == "sma":
            self._plot_single(tle, "SEMIMAJOR_AXIS", "Semi-Major Axis (km)", title, 0)
        elif plot_key == "bstar":
            self._plot_single(tle, "BSTAR", "B* Drag Term", title, 4)
        elif plot_key == "mean_motion":
            self._plot_single(tle, "MEAN_MOTION", "Mean Motion (rev/day)", title, 0)
        elif plot_key == "eccentricity":
            self._plot_single(tle, "ECCENTRICITY", "Eccentricity", title, 3)
        elif plot_key == "inclination":
            self._plot_single(tle, "INCLINATION", "Inclination (deg)", title, 5)
        elif plot_key == "period":
            self._plot_single(tle, "PERIOD", "Period (minutes)", title, 6)
        elif plot_key == "ndot":
            self._plot_single(tle, "MEAN_MOTION_DOT", "Mean Motion Derivative (rev/day\u00b2)", title, 1)

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_altitude(self, tle, title):
        t = T()
        ax = self.fig.add_subplot(111)
        if "APOAPSIS" in tle.columns:
            mask = tle["APOAPSIS"].notna()
            ax.plot(tle.loc[mask, "EPOCH"], tle.loc[mask, "APOAPSIS"],
                    linewidth=0.8, color=plot_color(1), alpha=0.8, label="Apogee")
        if "PERIAPSIS" in tle.columns:
            mask = tle["PERIAPSIS"].notna()
            ax.plot(tle.loc[mask, "EPOCH"], tle.loc[mask, "PERIAPSIS"],
                    linewidth=0.8, color=plot_color(0), alpha=0.8, label="Perigee")
        if "APOAPSIS" in tle.columns and "PERIAPSIS" in tle.columns:
            both = tle[tle["APOAPSIS"].notna() & tle["PERIAPSIS"].notna()]
            if not both.empty:
                ax.fill_between(both["EPOCH"], both["PERIAPSIS"], both["APOAPSIS"],
                                alpha=0.12, color=plot_color(4))
        style_axes(ax, title=title, xlabel="Epoch", ylabel="Altitude (km)")
        style_legend(ax)

    def _plot_single(self, tle, column, ylabel, title, color_idx=0):
        t = T()
        ax = self.fig.add_subplot(111)
        if column not in tle.columns or tle[column].notna().sum() == 0:
            style_axes_empty(ax, f"No {column} data available")
            return
        mask = tle[column].notna()
        ax.plot(tle.loc[mask, "EPOCH"], tle.loc[mask, column],
                linewidth=0.8, color=plot_color(color_idx), alpha=0.8)
        style_axes(ax, title=title, xlabel="Epoch", ylabel=ylabel)

    def _plot_all_elements(self, tle, title):
        t = T()
        plots = [
            ("APOAPSIS", "Apogee (km)", 1),
            ("PERIAPSIS", "Perigee (km)", 0),
            ("INCLINATION", "Inclination (\u00b0)", 5),
            ("ECCENTRICITY", "Eccentricity", 3),
            ("BSTAR", "B* Drag", 4),
            ("MEAN_MOTION", "Mean Motion", 6),
        ]
        available = [(c, l, ci) for c, l, ci in plots
                     if c in tle.columns and tle[c].notna().any()]
        n = len(available)
        if n == 0:
            ax = self.fig.add_subplot(111)
            style_axes_empty(ax, "No orbital element data available")
            return

        for i, (col, ylabel, ci) in enumerate(available, 1):
            ax = self.fig.add_subplot(n, 1, i)
            mask = tle[col].notna()
            ax.plot(tle.loc[mask, "EPOCH"], tle.loc[mask, col],
                    linewidth=0.6, color=plot_color(ci), alpha=0.8)
            style_axes(ax, ylabel=ylabel)
            ax.tick_params(labelsize=7)
            if i == 1:
                ax.set_title(title, fontsize=10, fontweight="bold",
                             color=t["plot_text"])
            if i < n:
                ax.set_xticklabels([])
        if available:
            ax.set_xlabel("Epoch", fontsize=8, color=t["plot_text_dim"])

    # -- Cross-tab navigation ------------------------------------------

    def select_satellite(self, norad_id):
        """Select and plot a satellite by NORAD ID (called from other tabs)."""
        self.search_var.set(str(norad_id))
        for i in range(self.sat_listbox.size()):
            text = self.sat_listbox.get(i)
            try:
                if int(text.strip().split()[0]) == norad_id:
                    self.sat_listbox.selection_clear(0, "end")
                    self.sat_listbox.selection_set(i)
                    self.sat_listbox.see(i)
                    self._load_and_plot(norad_id)
                    return
            except (ValueError, IndexError):
                continue

    def _nav_to_lc(self):
        if self.current_norad is not None:
            self.app.navigate_to_lightcurve(self.current_norad)

    def _nav_to_explorer(self):
        if self.current_norad is not None:
            self.app.navigate_to_explorer(self.current_norad)

    def _update_nav_buttons(self, norad_id):
        """Show/hide nav buttons based on data availability."""
        from pathlib import Path
        has_lc = False
        if self.app.mmt9_lc_dir:
            has_lc = Path(self.app.mmt9_lc_dir, f"{norad_id}.parquet").exists()
        if not has_lc and self.app.sdlcd_lc_dir:
            has_lc = Path(self.app.sdlcd_lc_dir, f"{norad_id}.parquet").exists()
        has_explorer = self.app.merged_df is not None

        # Hide all first, then show available ones
        self.nav_explorer_btn.pack_forget()
        self.nav_lc_btn.pack_forget()
        if has_explorer:
            self.nav_explorer_btn.pack(side="left", padx=(0, 4), fill="x", expand=True)
        if has_lc:
            self.nav_lc_btn.pack(side="left", fill="x", expand=True)