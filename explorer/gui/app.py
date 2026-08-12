"""
gui/app.py – Main application window
"""
import tkinter as tk
from tkinter import ttk, messagebox
from gui.theme import (T, init_theme, is_dark, set_theme, FONT,
                       pref, set_pref, save_prefs, styled_label)
from gui.tab_merger import MergerTab
from gui.tab_explorer import ExplorerTab
from gui.tab_lightcurves import LightCurveTab
from gui.tab_tle import TLETab
from gui.tab_analysis import AnalysisTab


class SatelliteExplorerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.merged_df = None
        self.mmt9_lc_dir = pref("mmt9_lc_dir")
        self.sdlcd_lc_dir = pref("sdlcd_lc_dir")
        self.tle_dir = pref("tle_dir")

        # Auto-discover canonical paths from the wider space-debris-ml project.
        # Used only as a fallback — a saved preference (or a stale one that
        # still points at an existing path) takes priority. Returns {} when
        # the explorer is run standalone, so manual workflow still works.
        from data.autodiscover import discover_project_paths
        self.detected_paths = discover_project_paths()

        def _fallback(saved, key):
            """Use saved preference when it still resolves to an existing
            path; otherwise fall back to the auto-detected value."""
            from pathlib import Path as _P
            if saved and _P(saved).exists():
                return saved
            return self.detected_paths.get(key) or saved or None

        self.mmt9_lc_dir = _fallback(self.mmt9_lc_dir, "mmt9_lc_dir")
        self.sdlcd_lc_dir = _fallback(self.sdlcd_lc_dir, "sdlcd_lc_dir")
        self.tle_dir = _fallback(self.tle_dir, "tle_dir")

        # Apply saved geometry
        geom = pref("geometry", "1400x850")
        root.geometry(geom)
        root.minsize(1100, 700)

        init_theme(root)
        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        t = T()
        self.root.configure(bg=t["bg"])

        # Header
        hdr = tk.Frame(self.root, bg=t["header_bg"], pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Satellite Explorer", font=(FONT, 18, "bold"),
                 fg=t["header_fg"], bg=t["header_bg"]).pack(side="left", padx=15)
        tk.Label(hdr, text="SATCAT \u00b7 UCS \u00b7 DISCOS \u00b7 MMT-9 \u00b7 SDLCD \u00b7 TLE",
                 font=(FONT, 10), fg=t["header_sub"], bg=t["header_bg"]
                 ).pack(side="left", padx=10)

        # Theme indicator + switch button
        icon = "\u2600 Light mode" if is_dark() else "\u263e Dark mode"
        tk.Button(hdr, text=icon, font=(FONT, 8), relief="flat", bd=0,
                  bg=t["header_bg"], fg=t["header_sub"], cursor="hand2",
                  activebackground=t["header_bg"], activeforeground=t["header_fg"],
                  command=self._switch_theme).pack(side="right", padx=15)

        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self.merger_tab = MergerTab(self.notebook, self)
        self.notebook.add(self.merger_tab.frame, text="  1. Merge Data  ")

        self.explorer_tab = ExplorerTab(self.notebook, self)
        self.notebook.add(self.explorer_tab.frame, text="  2. Explore Satellites  ")
        self.explorer_tab.on_data_loaded = self._on_explorer_direct_load

        self.lc_tab = LightCurveTab(self.notebook, self)
        self.notebook.add(self.lc_tab.frame, text="  3. Light Curves  ")

        self.tle_tab = TLETab(self.notebook, self)
        self.notebook.add(self.tle_tab.frame, text="  4. TLE / Orbital  ")

        self.analysis_tab = AnalysisTab(self.notebook, self)
        self.notebook.add(self.analysis_tab.frame, text="  5. ML Readiness  ")

    def _switch_theme(self):
        new = "light" if is_dark() else "dark"
        set_theme(new)
        messagebox.showinfo("Theme Changed",
            f"Theme set to {new}. Restart the app to apply.")

    def _on_close(self):
        set_pref("geometry", self.root.geometry())
        # Save directory paths (prefer merger tab StringVars as most current)
        mt = self.merger_tab
        mmt9_lc = mt.mmt9_lc_var.get().strip() or (self.mmt9_lc_dir or "")
        sdlcd_lc = mt.sdlcd_lc_var.get().strip() or (self.sdlcd_lc_dir or "")
        tle = mt.tle_dir_var.get().strip() or (self.tle_dir or "")
        if mmt9_lc:  set_pref("mmt9_lc_dir", mmt9_lc)
        if sdlcd_lc: set_pref("sdlcd_lc_dir", sdlcd_lc)
        if tle:      set_pref("tle_dir", tle)
        # Save source file paths from merger tab
        for key, src in mt.sources.items():
            p = src['path'].get().strip()
            if p: set_pref(f"path_{key}", p)
        save_prefs()
        self.root.destroy()

    # -- Data flow ---------------------------------------------------------
    def set_merged_data(self, df, mmt9_lc_dir=None, sdlcd_lc_dir=None, tle_dir=None):
        self.merged_df = df
        if mmt9_lc_dir:  self.mmt9_lc_dir = mmt9_lc_dir
        if sdlcd_lc_dir: self.sdlcd_lc_dir = sdlcd_lc_dir
        if tle_dir:      self.tle_dir = tle_dir
        # Also check merger tab StringVars directly (in case dirs were set
        # after last merge, or restored from prefs but not passed through)
        mt = self.merger_tab
        if not self.mmt9_lc_dir:
            v = mt.mmt9_lc_var.get().strip()
            if v: self.mmt9_lc_dir = v
        if not self.sdlcd_lc_dir:
            v = mt.sdlcd_lc_var.get().strip()
            if v: self.sdlcd_lc_dir = v
        if not self.tle_dir:
            v = mt.tle_dir_var.get().strip()
            if v: self.tle_dir = v
        # Show wait cursor while tabs populate
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            self.explorer_tab.load_data(df)
            self.lc_tab.load_data(df, self.mmt9_lc_dir, self.sdlcd_lc_dir)
            self.tle_tab.load_data(df, self.tle_dir)
        finally:
            self.root.configure(cursor="")

    def _on_explorer_direct_load(self, df):
        self.merged_df = df
        # Pull dirs from merger tab if app doesn't have them
        mt = self.merger_tab
        if not self.mmt9_lc_dir:
            v = mt.mmt9_lc_var.get().strip()
            if v: self.mmt9_lc_dir = v
        if not self.sdlcd_lc_dir:
            v = mt.sdlcd_lc_var.get().strip()
            if v: self.sdlcd_lc_dir = v
        if not self.tle_dir:
            v = mt.tle_dir_var.get().strip()
            if v: self.tle_dir = v
        self.lc_tab.load_data(df, self.mmt9_lc_dir, self.sdlcd_lc_dir)
        self.tle_tab.load_data(df, self.tle_dir)

    # -- Cross-tab navigation ------------------------------------------
    def navigate_to_lightcurve(self, norad_id):
        """Switch to Light Curves tab and select a specific satellite."""
        self.notebook.select(2)  # Tab index 2 = Light Curves
        self.lc_tab.select_satellite(norad_id)

    def navigate_to_tle(self, norad_id):
        """Switch to TLE tab and select a specific satellite."""
        self.notebook.select(3)  # Tab index 3 = TLE / Orbital
        self.tle_tab.select_satellite(norad_id)

    def navigate_to_explorer(self, norad_id):
        """Switch to Explorer tab and search for a specific satellite."""
        self.notebook.select(1)  # Tab index 1 = Explorer
        self.explorer_tab.search_var.set(str(norad_id))