"""
gui/tab_lightcurves.py
=======================
Tab 3: View MMT-9 and SDLCD light curves for individual satellites.
Features source filtering and dual-source display.
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
                       styled_rb, styled_cb,
                       style_figure, style_axes, style_axes_empty,
                       style_legend)
from data.loaders import load_mmt9_lightcurve, load_sdlcd_lightcurve


class LightCurveTab:
    def __init__(self, parent, app):
        self.app = app
        t = T()
        self.frame = tk.Frame(parent, bg=t["bg"])
        self.df = None
        self.lc_dir = None
        self.sdlcd_dir = None
        self.current_lc = None
        self.current_norad = None
        # Pre-built {norad_id: name} mapping for O(1) name lookups during
        # list population. Rebuilt every time load_data() is called.
        self._name_lookup = {}
        self._build_ui()

    def _build_ui(self):
        t = T()

        # -- Top toolbar --------------------------------------------------
        toolbar = tk.Frame(self.frame, bg=t["bg_card"], pady=6)
        toolbar.pack(fill="x", padx=8, pady=(8, 0))

        # Search
        search_frame = tk.Frame(toolbar, bg=t["bg_card"])
        search_frame.pack(side="left", padx=(8, 16))

        styled_label(search_frame, text="Search:", parent_bg=t["bg_card"]
                     ).pack(side="left", padx=(0, 4))

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._filter_list())
        styled_entry(search_frame, textvariable=self.search_var, width=22
                     ).pack(side="left", ipady=2)

        # -- Source filter buttons -------------------------------------
        src_frame = tk.Frame(toolbar, bg=t["bg_card"])
        src_frame.pack(side="left", padx=(0, 16))

        styled_label(src_frame, text="Show:", dim=True,
                     parent_bg=t["bg_card"]).pack(side="left", padx=(0, 6))

        self.source_filter = tk.StringVar(value="all")
        self._filter_buttons = {}
        filter_defs = [
            ("all",   "All Sources"),
            ("mmt9",  "MMT-9 Only"),
            ("sdlcd", "SDLCD Only"),
            ("both",  "Both Sources"),
        ]
        for val, text in filter_defs:
            btn = styled_button(src_frame, text=text, style="flat",
                                command=lambda v=val: self._set_source_filter(v))
            btn.configure(padx=10, pady=2)
            btn.pack(side="left", padx=2)
            self._filter_buttons[val] = btn
        self._update_filter_button_styles()

        # Status
        self.status_label = styled_label(toolbar, text="No data loaded",
                                         dim=True, parent_bg=t["bg_card"])
        self.status_label.pack(side="left", padx=16)

        # -- Plot options (right) --------------------------------------
        opt_frame = tk.Frame(toolbar, bg=t["bg_card"])
        opt_frame.pack(side="right", padx=8)

        self.plot_mode = tk.StringVar(value="combined")
        for txt, val in [("Combined", "combined"), ("Per Track", "per_track")]:
            styled_rb(opt_frame, text=txt, variable=self.plot_mode,
                               value=val, command=self._replot,
                               parent_bg=t["bg_card"]
                               ).pack(side="left", padx=4)

        tk.Frame(opt_frame, width=1, bg=t["border"]).pack(
            side="left", fill="y", padx=6, pady=2)

        self.mag_type = tk.StringVar(value="StdMag")
        for txt, val in [("StdMag", "StdMag"), ("Raw Mag", "Mag")]:
            styled_rb(opt_frame, text=txt, variable=self.mag_type,
                               value=val, command=self._replot,
                               parent_bg=t["bg_card"]
                               ).pack(side="left", padx=4)

        tk.Frame(opt_frame, width=1, bg=t["border"]).pack(
            side="left", fill="y", padx=6, pady=2)

        self.invert_var = tk.BooleanVar(value=True)
        styled_cb(opt_frame, text="Invert Y", variable=self.invert_var,
                           command=self._replot, parent_bg=t["bg_card"]
                           ).pack(side="left", padx=4)

        # -- Main content ---------------------------------------------
        body = tk.Frame(self.frame, bg=t["bg"])
        body.pack(fill="both", expand=True, padx=8, pady=8)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Left panel: satellite list
        left = card_frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        hdr = tk.Frame(left, bg=t["bg_card"])
        hdr.pack(fill="x", padx=10, pady=(10, 4))
        styled_label(hdr, text="Satellites", heading=True,
                     parent_bg=t["bg_card"]).pack(side="left")
        self.list_count_label = styled_label(hdr, text="", dim=True,
                                             parent_bg=t["bg_card"])
        self.list_count_label.pack(side="right")

        list_container = tk.Frame(left, bg=t["bg_card"])
        list_container.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        self.sat_listbox = styled_listbox(list_container)
        lb_scroll = tk.Scrollbar(list_container, command=self.sat_listbox.yview)
        self.sat_listbox.configure(yscrollcommand=lb_scroll.set)
        self.sat_listbox.pack(side="left", fill="both", expand=True)
        lb_scroll.pack(side="right", fill="y")
        self.sat_listbox.bind("<<ListboxSelect>>", self._on_select)

        # Info card
        info_card = tk.Frame(left, bg=t["list_bg"],
                             highlightbackground=t["border"],
                             highlightthickness=1)
        info_card.pack(fill="x", padx=6, pady=(2, 8))

        self.track_info = tk.Label(
            info_card, text="Select a satellite from the list above",
            font=FONT_SMALL, fg=t["fg_dim"], bg=t["list_bg"],
            wraplength=260, justify="left", padx=8, pady=6)
        self.track_info.pack(fill="x")

        # Navigation buttons
        nav_frame = tk.Frame(left, bg=t["bg_card"])
        nav_frame.pack(fill="x", padx=6, pady=(0, 6))
        self.nav_explorer_btn = styled_button(nav_frame, text="Explorer \u2192",
                                              command=self._nav_to_explorer, style="primary")
        self.nav_explorer_btn.configure(pady=2, width=12)
        self.nav_tle_btn = styled_button(nav_frame, text="TLE \u2192",
                                         command=self._nav_to_tle, style="primary")
        self.nav_tle_btn.configure(pady=2, width=12)
        # Start hidden — shown when a satellite is selected
        self._nav_frame = nav_frame

        # Right panel: plot
        right = card_frame(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        self.fig = Figure(figsize=(10, 5), dpi=100)
        style_figure(self.fig)
        self.ax = self.fig.add_subplot(111)
        style_axes_empty(self.ax, "Select a satellite to view its light curve")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(right, bg=t["bg_card"])
        toolbar_frame.pack(side="bottom", fill="x")
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        self._sat_list_data = []

    # -- Source filter -------------------------------------------------

    def _set_source_filter(self, value):
        self.source_filter.set(value)
        self._update_filter_button_styles()
        self._filter_list()

    def _update_filter_button_styles(self):
        t = T()
        active = self.source_filter.get()
        color_map = {
            "all": t["fg_bright"], "mmt9": t["src_mmt9"],
            "sdlcd": t["src_sdlcd"], "both": t["src_both"],
        }
        for val, btn in self._filter_buttons.items():
            if val == active:
                btn.configure(bg=t["border"], fg=color_map[val],
                              font=FONT_SMALL_B)
            else:
                btn.configure(bg=t["bg_card"], fg=t["fg_dim"],
                              font=FONT_SMALL)

    def _matches_source_filter(self, sources_set):
        filt = self.source_filter.get()
        if filt == "all":
            return True
        elif filt == "mmt9":
            return "mmt9" in sources_set
        elif filt == "sdlcd":
            return "sdlcd" in sources_set
        elif filt == "both":
            return "mmt9" in sources_set and "sdlcd" in sources_set
        return True

    # -- Data loading --------------------------------------------------

    def load_data(self, df, lc_dir=None, sdlcd_dir=None):
        self.df = df
        self.lc_dir = lc_dir
        self.sdlcd_dir = sdlcd_dir
        self._build_name_lookup()
        self._populate_list()

    def _build_name_lookup(self):
        """Build the {norad_id -> name} mapping in one pass over the merged
        DataFrame. Replaces the previous per-row boolean-mask filter, which
        was O(N) per call and O(N^2) overall when populating a 16K-row list.
        Called whenever a fresh DataFrame is handed in via load_data().
        """
        self._name_lookup = {}
        df = self.df
        if df is None or 'norad_id' not in df.columns:
            return
        name_cols = [c for c in ('satcat_object_name', 'discos_name',
                                 'ucs_name', 'mmt9_name',
                                 'sdlcd_object_name')
                     if c in df.columns]
        if not name_cols:
            return
        # tolist() converts each whole column once (cheap for PyArrow/object
        # dtypes alike); per-row iteration then runs in pure Python.
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

        if self.df is None or (not self.lc_dir and not self.sdlcd_dir):
            self.status_label.configure(text="No light curve directory set")
            self.list_count_label.configure(text="")
            return

        lc_sources = {}
        if self.lc_dir:
            lc_path = Path(self.lc_dir)
            if lc_path.is_dir():
                for f in lc_path.glob("*.parquet"):
                    if f.stem.isdigit():
                        lc_sources.setdefault(int(f.stem), set()).add("mmt9")
        if self.sdlcd_dir:
            sdlcd_path = Path(self.sdlcd_dir)
            if sdlcd_path.is_dir():
                for f in sdlcd_path.glob("*.parquet"):
                    if f.stem.isdigit():
                        lc_sources.setdefault(int(f.stem), set()).add("sdlcd")

        if not lc_sources:
            self.status_label.configure(text="No light curve files found")
            self.list_count_label.configure(text="")
            return

        entries = []
        for norad_id in sorted(lc_sources.keys()):
            name = self._get_sat_name(norad_id)
            srcs = lc_sources[norad_id]
            display = self._format_entry(norad_id, name, srcs)
            entries.append((norad_id, display, srcs))

        self._sat_list_data = entries
        self._filter_list()

        n_mmt9  = sum(1 for _, _, s in entries if 'mmt9' in s)
        n_sdlcd = sum(1 for _, _, s in entries if 'sdlcd' in s)
        n_both  = sum(1 for _, _, s in entries if len(s) == 2)
        parts = []
        if n_mmt9:  parts.append(f"MMT-9: {n_mmt9:,}")
        if n_sdlcd: parts.append(f"SDLCD: {n_sdlcd:,}")
        if n_both:  parts.append(f"Both: {n_both:,}")
        bullet = " \u2022 "
        self.status_label.configure(
            text=f"{len(entries):,} satellites  \u2022  {bullet.join(parts)}")

    def _get_sat_name(self, norad_id):
        """O(1) name lookup against the pre-built dict. Empty string if the
        merged DataFrame is absent or this norad_id has no name."""
        try:
            return self._name_lookup.get(int(norad_id), "")
        except (ValueError, TypeError):
            return ""

    @staticmethod
    def _format_entry(norad_id, name, srcs):
        if "mmt9" in srcs and "sdlcd" in srcs:
            badge = "M+S"
        elif "mmt9" in srcs:
            badge = "MMT"
        else:
            badge = "SLC"
        return f"{norad_id:>6d}  {badge}  {name}" if name else f"{norad_id:>6d}  {badge}"

    # -- Filtering / selection -----------------------------------------

    def _filter_list(self):
        t = T()
        search = self.search_var.get().strip().lower()
        self.sat_listbox.delete(0, "end")

        visible = 0
        for norad_id, display, sources in self._sat_list_data:
            if not self._matches_source_filter(sources):
                continue
            if search and search not in display.lower() and search not in str(norad_id):
                continue
            self.sat_listbox.insert("end", display)
            idx = visible
            if "mmt9" in sources and "sdlcd" in sources:
                self.sat_listbox.itemconfig(idx, fg=t["src_both"])
            elif "sdlcd" in sources:
                self.sat_listbox.itemconfig(idx, fg=t["src_sdlcd"])
            else:
                self.sat_listbox.itemconfig(idx, fg=t["src_mmt9"])
            visible += 1

        self.list_count_label.configure(text=f"{visible:,} shown")

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

    # -- Load & plot ---------------------------------------------------

    def _load_and_plot(self, norad_id):
        if not self.lc_dir and not self.sdlcd_dir:
            return

        frames, sources_found = [], []
        if self.lc_dir:
            mmt9_lc = load_mmt9_lightcurve(self.lc_dir, norad_id)
            if not mmt9_lc.empty:
                if 'Source' not in mmt9_lc.columns:
                    mmt9_lc['Source'] = 'MMT-9'
                frames.append(mmt9_lc)
                sources_found.append('MMT-9')
        if self.sdlcd_dir:
            sdlcd_lc = load_sdlcd_lightcurve(self.sdlcd_dir, norad_id)
            if not sdlcd_lc.empty:
                if 'Source' not in sdlcd_lc.columns:
                    sdlcd_lc['Source'] = 'SDLCD'
                frames.append(sdlcd_lc)
                sources_found.append('SDLCD')

        if not frames:
            t = T()
            self.track_info.configure(text="No light curve data found",
                                      fg=t["fg_dim"])
            return

        lc = pd.concat(frames, ignore_index=True)
        lc.sort_values('Datetime', inplace=True)
        lc.reset_index(drop=True, inplace=True)

        self.current_lc = lc
        self.current_norad = norad_id

        t = T()
        n_tracks = lc['Track'].nunique()
        n_obs = len(lc)
        date_min = lc['Datetime'].min().strftime("%Y-%m-%d")
        date_max = lc['Datetime'].max().strftime("%Y-%m-%d")
        mag_min, mag_max = lc['StdMag'].min(), lc['StdMag'].max()
        source_str = " + ".join(sources_found)

        parts = []
        for src in sources_found:
            src_df = lc[lc['Source'] == src]
            parts.append(f"  {src}: {len(src_df):,} obs, "
                         f"{src_df['Track'].nunique()} tracks")

        info = (f"Source:  {source_str}\n"
                f"Obs:  {n_obs:,}  |  Tracks:  {n_tracks}\n"
                f"Date range:  {date_min}  \u2192  {date_max}\n"
                f"StdMag:  {mag_min:.2f}  \u2013  {mag_max:.2f}")
        if len(sources_found) > 1:
            info += "\n" + "\n".join(parts)

        self.track_info.configure(text=info, fg=t["fg"])
        self._update_nav_buttons(norad_id)
        self._replot()

    def _replot(self):
        if self.current_lc is None:
            return

        t = T()
        lc = self.current_lc
        norad_id = self.current_norad
        mag_col = self.mag_type.get()
        mode = self.plot_mode.get()

        self.fig.clear()
        style_figure(self.fig)
        ax = self.fig.add_subplot(111)

        name = self._get_sat_name(norad_id) if self.df is not None else ""
        title = f"NORAD {norad_id}"
        if name:
            title += f"  \u2014  {name}"

        SRC_COLORS = {'MMT-9': t["src_mmt9"], 'SDLCD': t["src_sdlcd"]}

        if mode == "combined":
            if 'Source' in lc.columns and lc['Source'].nunique() > 1:
                for src_name, grp in lc.groupby('Source'):
                    c = SRC_COLORS.get(src_name, t["accent"])
                    ax.scatter(grp['Datetime'], grp[mag_col], s=1.5, alpha=0.55,
                               c=c, edgecolors='none', label=src_name, rasterized=True)
                style_axes(ax, title=title, xlabel="Time (UTC)", ylabel=mag_col)
                style_legend(ax, markerscale=6)
            else:
                src_name = lc['Source'].iloc[0] if 'Source' in lc.columns else 'MMT-9'
                c = SRC_COLORS.get(src_name, t["accent"])
                ax.scatter(lc['Datetime'], lc[mag_col], s=1.5, alpha=0.55,
                           c=c, edgecolors='none', rasterized=True)
                style_axes(ax, title=title, xlabel="Time (UTC)", ylabel=mag_col)
        else:
            tracks = lc['Track'].unique()
            cmap = matplotlib.colormaps.get_cmap('tab10')
            for i, track in enumerate(tracks):
                t_data = lc[lc['Track'] == track]
                color = cmap(i % 10)
                t0 = t_data['Datetime'].min()
                dt_sec = (t_data['Datetime'] - t0).dt.total_seconds()
                src = t_data['Source'].iloc[0] if 'Source' in t_data.columns else ''
                label = f"Trk {track}" + (f" ({src})" if src else "")
                ax.scatter(dt_sec, t_data[mag_col], s=2, alpha=0.65,
                           color=color, label=label, edgecolors='none',
                           rasterized=True)
            style_axes(ax, title=title, xlabel="Time within track (s)",
                       ylabel=mag_col)
            if len(tracks) <= 12:
                style_legend(ax, fontsize=7, loc='upper right', markerscale=4)

        if self.invert_var.get():
            ax.invert_yaxis()

        self.fig.tight_layout(pad=1.2)
        self.canvas.draw()

    # -- Cross-tab navigation ------------------------------------------

    def select_satellite(self, norad_id):
        """Select and plot a satellite by NORAD ID (called from other tabs)."""
        # Reset filter to show all sources so the satellite is visible
        self._set_source_filter("all")
        self.search_var.set(str(norad_id))
        # Find and select in listbox
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

    def _nav_to_tle(self):
        if self.current_norad is not None:
            self.app.navigate_to_tle(self.current_norad)

    def _nav_to_explorer(self):
        if self.current_norad is not None:
            self.app.navigate_to_explorer(self.current_norad)

    def _update_nav_buttons(self, norad_id):
        """Show/hide nav buttons based on data availability."""
        from pathlib import Path
        # Check TLE
        has_tle = False
        if self.app.tle_dir:
            has_tle = Path(self.app.tle_dir, f"{norad_id}.parquet").exists()
        # Explorer always available if merged data exists
        has_explorer = self.app.merged_df is not None

        # Hide all first, then show available ones
        self.nav_explorer_btn.pack_forget()
        self.nav_tle_btn.pack_forget()
        if has_explorer:
            self.nav_explorer_btn.pack(side="left", padx=(0, 4), fill="x", expand=True)
        if has_tle:
            self.nav_tle_btn.pack(side="left", fill="x", expand=True)