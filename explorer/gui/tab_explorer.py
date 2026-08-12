"""
gui/tab_explorer.py
====================
Tab 2: Browse and filter the merged satellite catalogue with advanced
multi-parameter filtering.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from pathlib import Path
from gui.theme import (T, FONT, FONT_BODY, FONT_SMALL, FONT_SMALL_B,
                       FONT_MONO_SM, styled_frame, card_frame, styled_label,
                       styled_entry, styled_button, styled_listbox, styled_lf,
                       styled_cb, styled_text)


# Columns to show in the main table (in order)
DISPLAY_COLS = [
    ('norad_id', 'NORAD ID', 70),
    ('satcat_object_name', 'Name', 180),
    ('satcat_cospar_id', 'COSPAR', 100),
    ('satcat_object_type', 'Type', 60),
    ('satcat_ops_status', 'Status', 50),
    ('ucs_orbit_class', 'Orbit', 50),
    ('satcat_perigee_km', 'Perigee', 70),
    ('satcat_apogee_km', 'Apogee', 70),
    ('satcat_inclination_deg', 'Inc°', 55),
    ('discos_mass_kg', 'Mass(kg)', 75),
    ('satcat_rcs_m2', 'RCS(m²)', 70),
    ('discos_xsect_avg_m2', 'XSect(m²)', 75),
    ('discos_am_ratio_avg', 'A/m', 65),
    ('discos_object_class', 'DISCOS Class', 90),
    ('ucs_purpose', 'Purpose', 120),
    ('satcat_owner', 'Owner', 60),
]


class ExplorerTab:
    def __init__(self, parent, app):
        self.app = app
        t = T()
        self.frame = tk.Frame(parent, bg=t["bg"])
        self.df = None
        self.filtered_df = None
        self.on_data_loaded = None  # Callback for when data is loaded directly
        self._build_ui()

    def _build_ui(self):
        # -- Top: Text Search + Load CSV --
        search_frame = tk.Frame(self.frame, bg=T()["bg"])
        search_frame.pack(fill="x", padx=10, pady=(8, 2))

        styled_label(search_frame, text="Search (name / NORAD / COSPAR):",
                     parent_bg=T()["bg"]).pack(side="left", padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._apply_filters())
        styled_entry(search_frame, textvariable=self.search_var, width=30).pack(side="left", padx=(0, 15))

        self.count_label = styled_label(search_frame, text="No data loaded", font=(FONT, 10, "bold"), fg=T()["accent"])
        self.count_label.pack(side="right", padx=10)

        styled_button(search_frame, text="Reset Filters",
                      command=self._reset_filters, style="danger"
                      ).pack(side="right", padx=5)

        # -- Filter Panel (collapsible-style LabelFrame) --
        filter_outer = styled_lf(self.frame, text="Advanced Filters", padx=8, pady=6, font=(FONT, 9, "bold"))
        filter_outer.pack(fill="x", padx=10, pady=(2, 4))

        # Row 1: Dropdown filters
        row1 = tk.Frame(filter_outer, bg=T()["bg"])
        row1.pack(fill="x", pady=(0, 4))

        # Object Type
        self._add_dropdown(row1, "Type:", "type", ["All"], col=0)
        # DISCOS Class
        self._add_dropdown(row1, "DISCOS Class:", "discos_class", ["All"], col=2)
        # Orbit Class
        self._add_dropdown(row1, "Orbit:", "orbit", ["All"], col=4)
        # Ops Status
        self._add_dropdown(row1, "Status:", "ops_status", ["All"], col=6)
        # Shape
        self._add_dropdown(row1, "Shape:", "shape", ["All"], col=8)
        # Owner
        self._add_dropdown(row1, "Owner:", "owner", ["All"], col=10)

        # Row 1b: More dropdowns
        row1b = tk.Frame(filter_outer, bg=T()["bg"])
        row1b.pack(fill="x", pady=(0, 4))
        # Purpose
        self._add_dropdown(row1b, "Purpose:", "purpose", ["All"], col=0)

        # Row 2: Numeric range filters
        row2 = tk.Frame(filter_outer, bg=T()["bg"])
        row2.pack(fill="x", pady=(0, 4))

        self.range_filters = {}
        self._add_range(row2, "Perigee (km):", "perigee", col=0)
        self._add_range(row2, "Apogee (km):", "apogee", col=3)
        self._add_range(row2, "Inclination (°):", "inclination", col=6)
        self._add_range(row2, "Mass (kg):", "mass", col=9)

        # Row 3: More numeric ranges + checkboxes
        row3 = tk.Frame(filter_outer, bg=T()["bg"])
        row3.pack(fill="x", pady=(0, 2))

        self._add_range(row3, "RCS (m²):", "rcs", col=0)
        self._add_range(row3, "XSect (m²):", "xsect", col=3)
        self._add_range(row3, "A/m (m²/kg):", "am_ratio", col=6)
        self._add_range(row3, "Period (min):", "period", col=9)

        # Row 4: Data availability checkboxes
        row4 = tk.Frame(filter_outer, bg=T()["bg"])
        row4.pack(fill="x", pady=(2, 0))

        styled_label(row4, text="Data availability:", font=(FONT, 8, "italic"), fg=T()["fg_dim"]).pack(side="left", padx=(0, 8))

        self.has_mass_var = tk.BooleanVar(value=False)
        styled_cb(row4, text="Has mass", variable=self.has_mass_var, command=self._apply_filters).pack(side="left", padx=4)

        self.has_rcs_var = tk.BooleanVar(value=False)
        styled_cb(row4, text="Has RCS", variable=self.has_rcs_var, command=self._apply_filters).pack(side="left", padx=4)

        self.has_xsect_var = tk.BooleanVar(value=False)
        styled_cb(row4, text="Has cross-section", variable=self.has_xsect_var, command=self._apply_filters).pack(side="left", padx=4)

        self.has_am_var = tk.BooleanVar(value=False)
        styled_cb(row4, text="Has A/m", variable=self.has_am_var, command=self._apply_filters).pack(side="left", padx=4)

        self.has_lc_var = tk.BooleanVar(value=False)
        styled_cb(row4, text="Has light curve", variable=self.has_lc_var, command=self._apply_filters).pack(side="left", padx=4)

        self.has_ucs_var = tk.BooleanVar(value=False)
        styled_cb(row4, text="In UCS", variable=self.has_ucs_var, command=self._apply_filters).pack(side="left", padx=4)

        self.in_orbit_var = tk.BooleanVar(value=False)
        styled_cb(row4, text="Still in orbit", variable=self.in_orbit_var, command=self._apply_filters).pack(side="left", padx=4)

        self.has_tle_var = tk.BooleanVar(value=False)
        styled_cb(row4, text="Has TLE data", variable=self.has_tle_var, command=self._apply_filters).pack(side="left", padx=4)

        # -- Main table + detail split --
        paned = tk.PanedWindow(self.frame, orient="horizontal", bg="#cccccc", sashwidth=4)
        paned.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        # Left: table
        table_frame = tk.Frame(paned)
        paned.add(table_frame, width=850)

        cols = [c[0] for c in DISPLAY_COLS]
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")

        for col_id, heading, width in DISPLAY_COLS:
            self.tree.heading(col_id, text=heading, command=lambda c=col_id: self._sort_column(c))
            self.tree.column(col_id, width=width, minwidth=40)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Right: detail panel
        detail_frame = tk.Frame(paned, bg=T()["bg_card"])
        paned.add(detail_frame, width=450)

        # Header with nav buttons inline
        detail_hdr = tk.Frame(detail_frame, bg=T()["bg_card"])
        detail_hdr.pack(fill="x", padx=10, pady=(8, 4))
        styled_label(detail_hdr, text="Satellite Details", heading=True,
                     parent_bg=T()["bg_card"]).pack(side="left")
        self.nav_tle_btn = styled_button(detail_hdr, text="TLE \u2192",
                                         command=self._nav_to_tle, style="primary")
        self.nav_tle_btn.configure(padx=8, pady=1)
        # Start hidden — shown when a satellite with data is selected
        self.nav_lc_btn = styled_button(detail_hdr, text="Light Curve \u2192",
                                        command=self._nav_to_lc, style="primary")
        self.nav_lc_btn.configure(padx=8, pady=1)
        self._selected_norad = None

        self.detail_text = styled_text(detail_frame, state="disabled", padx=10, pady=8)
        detail_sb = tk.Scrollbar(detail_frame, command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_sb.set)
        detail_sb.pack(side="right", fill="y", padx=(0, 2), pady=5)
        self.detail_text.pack(fill="both", expand=True, padx=(5, 0), pady=5)

        # Sort state
        self._sort_reverse = {}

    # -- Helper: dropdown filter --

    def _add_dropdown(self, parent, label, key, values, col):
        styled_label(parent, text=label, font=FONT_SMALL
                 ).grid(row=0, column=col, padx=(8, 2), sticky="e")
        var = tk.StringVar(value="All")
        combo = ttk.Combobox(parent, textvariable=var, width=14,
                             values=values, state="readonly", font=("Segoe UI", 8))
        combo.grid(row=0, column=col + 1, padx=(0, 4))
        combo.bind("<<ComboboxSelected>>", lambda _: self._apply_filters())

        if not hasattr(self, '_dropdown_vars'):
            self._dropdown_vars = {}
            self._dropdown_combos = {}
        self._dropdown_vars[key] = var
        self._dropdown_combos[key] = combo

    # -- Helper: numeric range filter --

    def _add_range(self, parent, label, key, col):
        styled_label(parent, text=label, font=FONT_SMALL
                 ).grid(row=0, column=col, padx=(8, 2), sticky="e")

        min_var = tk.StringVar()
        max_var = tk.StringVar()

        min_entry = styled_entry(parent, textvariable=min_var, width=8)
        min_entry.grid(row=0, column=col + 1, padx=1)
        min_entry.bind("<Return>", lambda _: self._apply_filters())
        min_entry.bind("<FocusOut>", lambda _: self._apply_filters())

        styled_label(parent, text="–", font=FONT_SMALL
                 ).grid(row=0, column=col + 1, padx=(0, 0), sticky="e")

        max_entry = styled_entry(parent, textvariable=max_var, width=8)
        max_entry.grid(row=0, column=col + 2, padx=1)
        max_entry.bind("<Return>", lambda _: self._apply_filters())
        max_entry.bind("<FocusOut>", lambda _: self._apply_filters())

        self.range_filters[key] = (min_var, max_var)

    # -- Mapping from filter keys to DataFrame columns --

    _DROPDOWN_COL_MAP = {
        'type': 'satcat_object_type',
        'discos_class': 'discos_object_class',
        'orbit': 'ucs_orbit_class',
        'ops_status': 'satcat_ops_status',
        'owner': 'satcat_owner',
        'purpose': 'ucs_purpose',
        'shape': 'discos_shape',
    }

    _RANGE_COL_MAP = {
        'perigee': 'satcat_perigee_km',
        'apogee': 'satcat_apogee_km',
        'inclination': 'satcat_inclination_deg',
        'mass': 'discos_mass_kg',
        'rcs': 'satcat_rcs_m2',
        'xsect': 'discos_xsect_avg_m2',
        'am_ratio': 'discos_am_ratio_avg',
        'period': 'satcat_period_min',
        'bstar': 'tle_mean_bstar',
        'mean_motion': 'tle_mean_mean_motion',
    }

    # -- Filter logic --

    def _reset_filters(self):
        """Reset all filters to defaults."""
        self.search_var.set("")
        for var in self._dropdown_vars.values():
            var.set("All")
        for min_var, max_var in self.range_filters.values():
            min_var.set("")
            max_var.set("")
        self.has_mass_var.set(False)
        self.has_rcs_var.set(False)
        self.has_xsect_var.set(False)
        self.has_am_var.set(False)
        self.has_lc_var.set(False)
        self.has_ucs_var.set(False)
        self.in_orbit_var.set(False)
        self.has_tle_var.set(False)
        self._apply_filters()


    def load_data(self, df: pd.DataFrame):
        """Called when merged data is ready."""
        self.df = df
        self._populate_filter_options()
        self._apply_filters()

    def _populate_filter_options(self):
        if self.df is None:
            return

        for key, col in self._DROPDOWN_COL_MAP.items():
            if col in self.df.columns and key in self._dropdown_combos:
                vals = sorted(self.df[col].dropna().unique().astype(str).tolist())
                self._dropdown_combos[key]['values'] = ["All"] + vals

    def _apply_filters(self):
        if self.df is None:
            return

        mask = pd.Series(True, index=self.df.index)

        # Text search
        search = self.search_var.get().strip().lower()
        if search:
            text_mask = pd.Series(False, index=self.df.index)
            for col in ['satcat_object_name', 'satcat_cospar_id', 'ucs_name',
                        'discos_name', 'discos_cospar_id']:
                if col in self.df.columns:
                    text_mask |= self.df[col].astype(str).str.lower().str.contains(search, na=False)
            text_mask |= self.df['norad_id'].astype(str).str.contains(search, na=False)
            mask &= text_mask

        # Dropdown filters
        for key, col in self._DROPDOWN_COL_MAP.items():
            if key not in self._dropdown_vars:
                continue
            val = self._dropdown_vars[key].get()
            if val != "All" and col in self.df.columns:
                mask &= self.df[col].astype(str) == val

        # Numeric range filters
        for key, col in self._RANGE_COL_MAP.items():
            if key not in self.range_filters or col not in self.df.columns:
                continue
            min_var, max_var = self.range_filters[key]
            min_str = min_var.get().strip()
            max_str = max_var.get().strip()
            if min_str:
                try:
                    mask &= self.df[col] >= float(min_str)
                except ValueError:
                    pass
            if max_str:
                try:
                    mask &= self.df[col] <= float(max_str)
                except ValueError:
                    pass

        # Data availability checkboxes
        if self.has_mass_var.get() and 'discos_mass_kg' in self.df.columns:
            mask &= self.df['discos_mass_kg'].notna()
        if self.has_rcs_var.get() and 'satcat_rcs_m2' in self.df.columns:
            mask &= self.df['satcat_rcs_m2'].notna()
        if self.has_xsect_var.get() and 'discos_xsect_avg_m2' in self.df.columns:
            mask &= self.df['discos_xsect_avg_m2'].notna()
        if self.has_am_var.get() and 'discos_am_ratio_avg' in self.df.columns:
            mask &= self.df['discos_am_ratio_avg'].notna()
        if self.has_lc_var.get() and 'has_mmt9' in self.df.columns:
            mask &= self.df['has_mmt9'] == True
        if self.has_ucs_var.get() and 'has_ucs' in self.df.columns:
            mask &= self.df['has_ucs'] == True
        if self.in_orbit_var.get() and 'satcat_decay_date' in self.df.columns:
            mask &= self.df['satcat_decay_date'].isna()
        if self.has_tle_var.get() and 'has_tle' in self.df.columns:
            mask &= self.df['has_tle'] == True

        self.filtered_df = self.df[mask]
        self._populate_table()

    def _populate_table(self):
        """Fill the treeview with filtered data."""
        self.tree.delete(*self.tree.get_children())

        if self.filtered_df is None or self.filtered_df.empty:
            self.count_label.configure(text="0 results")
            return

        display_limit = 5000
        df_show = self.filtered_df.head(display_limit)
        cols = [c[0] for c in DISPLAY_COLS]

        for _, row in df_show.iterrows():
            values = []
            for col in cols:
                val = row.get(col, "")
                if pd.isna(val):
                    val = ""
                elif isinstance(val, float):
                    if col in ('discos_am_ratio_avg',):
                        val = f"{val:.6f}" if val != 0 else "0"
                    elif val == int(val):
                        val = str(int(val))
                    else:
                        val = f"{val:.2f}"
                values.append(val)
            self.tree.insert("", "end", values=values)

        total = len(self.filtered_df)
        shown = min(total, display_limit)
        if total > display_limit:
            self.count_label.configure(text=f"{shown:,} / {total:,} shown")
        else:
            self.count_label.configure(text=f"{total:,} results")

    def _sort_column(self, col):
        """Sort table by column."""
        if self.filtered_df is None:
            return
        reverse = self._sort_reverse.get(col, False)
        self._sort_reverse[col] = not reverse
        try:
            self.filtered_df = self.filtered_df.sort_values(col, ascending=not reverse, na_position='last')
        except Exception:
            pass
        self._populate_table()

    def _on_select(self, event):
        """Show details for selected satellite."""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        values = item['values']
        col_names = [c[0] for c in DISPLAY_COLS]
        norad_id = values[col_names.index('norad_id')] if 'norad_id' in col_names else None

        if norad_id is None or self.df is None:
            return
        try:
            norad_id = int(norad_id)
        except (ValueError, TypeError):
            return

        row = self.df[self.df['norad_id'] == norad_id]
        if row.empty:
            return
        row = row.iloc[0]

        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")

        sources_order = ['satcat', 'ucs', 'discos', 'mmt9', 'sdlcd', 'tle']
        source_labels = {
            'satcat': 'SATCAT (CelesTrak)',
            'ucs': 'UCS Satellite Database',
            'discos': 'ESA DISCOS',
            'mmt9': 'MMT-9 Observations',
            'sdlcd': 'SDLCD Observations',
            'tle': 'TLE / GP History',
        }

        self.detail_text.insert("end", f"NORAD ID: {norad_id}\n")
        self.detail_text.insert("end", "=" * 45 + "\n\n")

        for src in sources_order:
            src_cols = [c for c in row.index if c.startswith(f'{src}_')]
            if not src_cols:
                continue

            has_data = any(pd.notna(row[c]) for c in src_cols)
            if not has_data:
                continue

            self.detail_text.insert("end", f"-- {source_labels.get(src, src.upper())} --\n")
            for col in src_cols:
                val = row[col]
                if pd.isna(val):
                    continue
                display_name = col.replace(f'{src}_', '').replace('_', ' ').title()
                if isinstance(val, float):
                    if val == int(val) and abs(val) < 1e10:
                        val = int(val)
                    else:
                        val = f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"
                self.detail_text.insert("end", f"  {display_name:<28s} {val}\n")
            self.detail_text.insert("end", "\n")

        # Source presence flags
        flags = [c for c in row.index if c.startswith('has_')]
        if flags:
            self.detail_text.insert("end", "-- Data Sources Present --\n")
            for f in flags:
                src_name = f.replace('has_', '').upper()
                val = "Yes" if row[f] else "No"
                self.detail_text.insert("end", f"  {src_name:<28s} {val}\n")

        self.detail_text.configure(state="disabled")

        # Update navigation buttons
        self._selected_norad = norad_id
        self._update_nav_buttons(norad_id, row)

    # -- Cross-tab navigation ------------------------------------------

    def _nav_to_lc(self):
        if self._selected_norad is not None:
            self.app.navigate_to_lightcurve(self._selected_norad)

    def _nav_to_tle(self):
        if self._selected_norad is not None:
            self.app.navigate_to_tle(self._selected_norad)

    def _update_nav_buttons(self, norad_id, row=None):
        """Show/hide nav buttons based on data availability."""
        from pathlib import Path
        # Check light curve availability
        has_lc = False
        if row is not None:
            for flag in ['has_mmt9', 'has_sdlcd']:
                if flag in row.index and row[flag]:
                    has_lc = True
                    break
        if not has_lc:
            if self.app.mmt9_lc_dir:
                has_lc = Path(self.app.mmt9_lc_dir, f"{norad_id}.parquet").exists()
            if not has_lc and self.app.sdlcd_lc_dir:
                has_lc = Path(self.app.sdlcd_lc_dir, f"{norad_id}.parquet").exists()

        # Check TLE availability
        has_tle = False
        if row is not None and 'has_tle' in row.index:
            has_tle = bool(row['has_tle'])
        if not has_tle and self.app.tle_dir:
            has_tle = Path(self.app.tle_dir, f"{norad_id}.parquet").exists()

        # Show or hide — pack TLE first (rightmost), LC second, so visual
        # order is [LC →] [TLE →] matching tab order
        self.nav_lc_btn.pack_forget()
        self.nav_tle_btn.pack_forget()
        if has_tle:
            self.nav_tle_btn.pack(side="right", padx=(4, 0))
        if has_lc:
            self.nav_lc_btn.pack(side="right", padx=(4, 0))