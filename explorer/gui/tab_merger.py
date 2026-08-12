"""
gui/tab_merger.py – Tab 1: Configure data sources, merge catalogues, export.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pandas as pd
from pathlib import Path

from gui.theme import (T, FONT, FONT_BODY, FONT_SMALL, FONT_SMALL_B,
                       FONT_TINY, FONT_MONO_SM, pref, styled_frame,
                       styled_label, styled_entry, styled_button,
                       styled_text, styled_lf, styled_cb, styled_rb, card_frame)
from data.loaders import (load_satcat, load_ucs, load_discos,
                          load_mmt9_catalogue, load_mmt9_official_catalogue,
                          load_sdlcd_catalogue, load_tle_index)
from data.merger import merge_catalogues


class MergerTab:
    def __init__(self, parent, app):
        self.app = app
        t = T()
        self.frame = tk.Frame(parent, bg=t["bg"])

        # Pull auto-detected paths from the app (empty dict if the explorer
        # is running standalone). Used as a fallback only — saved prefs win
        # unless they point at a file/dir that no longer exists.
        det = getattr(app, "detected_paths", {})

        def _resolve(prefs_key):
            """Saved pref wins when it still resolves to an existing path;
            otherwise fall back to the auto-detected value; otherwise empty."""
            saved = pref(prefs_key, "")
            if saved and Path(saved).exists():
                return saved
            return det.get(prefs_key, "") or saved or ""

        # Track which fields were pre-filled by auto-detection (rather than
        # by a saved preference). Used for the status banner and to switch
        # the MMT-9 catalogue type to "custom" when auto-detected.
        self._autofilled = set()
        for k in ("path_discos", "path_mmt9", "path_sdlcd", "path_tle",
                  "mmt9_lc_dir", "sdlcd_lc_dir", "tle_dir"):
            saved = pref(k, "")
            if (not saved or not Path(saved).exists()) and det.get(k):
                self._autofilled.add(k)

        self.sources = {
            'satcat': {'path': tk.StringVar(value=pref("path_satcat","")), 'df': None,
                       'label': 'SATCAT', 'desc': 'CelesTrak satellite catalogue',
                       'loader': load_satcat, 'ext': [('CSV','*.csv')]},
            'ucs':    {'path': tk.StringVar(value=pref("path_ucs","")), 'df': None,
                       'label': 'UCS', 'desc': 'Union of Concerned Scientists database',
                       'loader': load_ucs, 'ext': [('Excel','*.xlsx *.xls')]},
            'discos': {'path': tk.StringVar(value=_resolve("path_discos")), 'df': None,
                       'label': 'DISCOS', 'desc': 'ESA DISCOS physical properties',
                       'loader': load_discos, 'ext': [('CSV','*.csv')]},
            'mmt9':   {'path': tk.StringVar(value=_resolve("path_mmt9")), 'df': None,
                       'label': 'MMT-9', 'desc': 'MMT-9 observation catalogue',
                       'loader': None, 'ext': [('All','*.csv *.txt')]},
            'sdlcd':  {'path': tk.StringVar(value=_resolve("path_sdlcd")), 'df': None,
                       'label': 'SDLCD', 'desc': 'SDLCD observation catalogue',
                       'loader': load_sdlcd_catalogue, 'ext': [('CSV','*.csv')]},
            'tle':    {'path': tk.StringVar(value=_resolve("path_tle")), 'df': None,
                       'label': 'TLE Index', 'desc': 'GP/TLE history index',
                       'loader': load_tle_index, 'ext': [('CSV','*.csv')]},
        }
        self.include_vars = {}
        self.mmt9_lc_var = tk.StringVar(value=_resolve("mmt9_lc_dir"))
        self.sdlcd_lc_var = tk.StringVar(value=_resolve("sdlcd_lc_dir"))
        # The auto-detected MMT-9 catalogue at data/processed/mmt9/catalogue.csv
        # is the project's custom processed format, not the official MMT-9
        # website .txt. When that path is auto-detected, default to "custom"
        # so the loader picks the right parser.
        default_mmt9_type = "custom" if "path_mmt9" in self._autofilled else "official"
        self.mmt9_cat_type = tk.StringVar(value=default_mmt9_type)
        self.tle_dir_var = tk.StringVar(value=_resolve("tle_dir"))
        self._build_ui()

    def _build_ui(self):
        t = T()

        # Scrollable content area
        outer = tk.Frame(self.frame, bg=t["bg"])
        outer.pack(fill="both", expand=True)

        # ━━ Auto-detect status (shown only when something was detected) ━━
        det = getattr(self.app, "detected_paths", {})
        root = det.get("_project_root")
        if root and self._autofilled:
            n = len(self._autofilled)
            banner = tk.Frame(outer, bg=t["bg"])
            banner.pack(fill="x", padx=12, pady=(6, 0))
            styled_label(banner,
                text=(f"\u2713 Detected space-debris-ml project at  {root}  "
                      f"\u2014  {n} path(s) pre-filled. Browse any field to override."),
                dim=True, wraplength=900, justify="left"
            ).pack(anchor="w")

        # ━━ SECTION 1: Data Directories ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        dir_frame = styled_lf(outer,
            text="Data Directories  \u2014  used by Light Curves, TLE, and other tabs")
        dir_frame.configure(padx=12, pady=8, font=(FONT, 10, "bold"))
        dir_frame.pack(fill="x", padx=12, pady=(4, 4))

        styled_label(dir_frame, text="These folders contain per-satellite parquet files "
                     "(e.g. 25544.parquet). They are used across the entire application "
                     "for plotting light curves and orbital histories, not just for merging.",
                     dim=True, wraplength=900, justify="left"
                     ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        dirs = [
            ("MMT-9 Light Curves", self.mmt9_lc_var, self._browse_lc,
             "Folder of MMT-9 per-satellite light curve parquet files"),
            ("SDLCD Light Curves", self.sdlcd_lc_var, self._browse_sdlcd_lc,
             "Folder of SDLCD per-satellite light curve parquet files"),
            ("TLE / GP Histories", self.tle_dir_var, self._browse_tle_dir,
             "Folder of per-satellite TLE/GP history parquet files"),
        ]
        for i, (label, var, cmd, hint) in enumerate(dirs):
            row = i + 1
            name_frame = tk.Frame(dir_frame, bg=t["bg"])
            name_frame.grid(row=row, column=0, sticky="w", padx=(4, 8))
            styled_label(name_frame, text=label, font=FONT_SMALL_B
                         ).pack(anchor="w")
            styled_label(name_frame, text=hint, dim=True,
                         font=FONT_TINY).pack(anchor="w")

            styled_entry(dir_frame, textvariable=var, width=62
                         ).grid(row=row, column=1, padx=4, pady=3)
            styled_button(dir_frame, text="Browse", style="flat",
                          command=cmd).grid(row=row, column=2, padx=2)

            # Show file count if path is set
            dir_status = styled_label(dir_frame, text="", dim=True,
                                      width=18, anchor="w")
            dir_status.grid(row=row, column=3, padx=4)
            self._update_dir_status(var, dir_status)
            var.trace_add("write", lambda *_, v=var, l=dir_status:
                          self._update_dir_status(v, l))

        # -- Quick import shortcut --
        import_frame = tk.Frame(outer, bg=t["bg"])
        import_frame.pack(fill="x", padx=12, pady=(6, 4))
        import_btn = styled_button(import_frame,
            text="\U0001f4c2  Load Previously Exported CSV",
            command=self._on_import)
        import_btn.configure(font=(FONT, 10, "bold"), padx=20, pady=5)
        import_btn.pack(side="left")
        styled_label(import_frame,
                     text="Skip merging \u2014 load a dataset you exported earlier",
                     dim=True).pack(side="left", padx=10)

        # ━━ SECTION 2: Catalogue Files + Merge Log (side by side) ━━━━━━━
        mid_frame = tk.Frame(outer, bg=t["bg"])
        mid_frame.pack(fill="both", expand=True, padx=12, pady=(4, 2))
        mid_frame.columnconfigure(0, weight=3)
        mid_frame.columnconfigure(1, weight=2)
        mid_frame.rowconfigure(0, weight=1)

        # Left: Catalogue Files
        cat_frame = styled_lf(mid_frame, text="Catalogue Files  \u2014  merged on NORAD ID")
        cat_frame.configure(padx=10, pady=6, font=(FONT, 10, "bold"))
        cat_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        # Explanation
        styled_label(cat_frame, text="Tick sources to include. "
                     "All loaded data is available to every tab.",
                     dim=True, wraplength=550, justify="left"
                     ).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 4))

        # Column headers
        for ci, (txt, w) in enumerate([("", 3), ("Source", 14),
                                        ("File Path", 45), ("", 6), ("Status", 14)]):
            styled_label(cat_frame, text=txt, font=FONT_SMALL_B,
                         fg=t["fg_dim"], anchor="w"
                         ).grid(row=1, column=ci, sticky="w", padx=2)

        for i, (key, s) in enumerate(self.sources.items()):
            row = i + 2
            var = tk.BooleanVar(value=True)
            self.include_vars[key] = var

            styled_cb(cat_frame, variable=var, parent_bg=t["bg"]
                      ).grid(row=row, column=0, padx=(2, 1))

            name_frame = tk.Frame(cat_frame, bg=t["bg"])
            name_frame.grid(row=row, column=1, sticky="w", padx=(0, 2))
            styled_label(name_frame, text=s['label'], font=FONT_SMALL_B
                         ).pack(anchor="w")
            styled_label(name_frame, text=s['desc'], dim=True,
                         font=FONT_TINY).pack(anchor="w")

            styled_entry(cat_frame, textvariable=s['path'], width=42
                         ).grid(row=row, column=2, padx=3, pady=2)

            styled_button(cat_frame, text="Browse", style="flat",
                          command=lambda k=key: self._browse(k)
                          ).grid(row=row, column=3, padx=1)

            lbl = styled_label(cat_frame, text="", dim=True, width=14, anchor="w")
            lbl.grid(row=row, column=4, padx=2)
            s['status_label'] = lbl

        # MMT-9 format radio
        fmt_row = len(self.sources) + 2
        fmt_frame = tk.Frame(cat_frame, bg=t["bg"])
        fmt_frame.grid(row=fmt_row, column=1, columnspan=3, sticky="w", pady=(2, 0))
        styled_label(fmt_frame, text="MMT-9 format:", dim=True
                     ).pack(side="left", padx=(0, 6))
        styled_rb(fmt_frame, text="Official (website TXT)",
                  variable=self.mmt9_cat_type, value="official"
                  ).pack(side="left", padx=(0, 8))
        styled_rb(fmt_frame, text="Processed CSV",
                  variable=self.mmt9_cat_type, value="custom"
                  ).pack(side="left")

        # Right: Merge Log
        log_frame = styled_lf(mid_frame, text="Merge Log")
        log_frame.configure(padx=8, pady=6, font=(FONT, 10, "bold"))
        log_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        self.summary_text = styled_text(log_frame, terminal=True, state="disabled")
        sb = tk.Scrollbar(log_frame, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=sb.set)
        self.summary_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # ━━ SECTION 3: Actions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        act_frame = tk.Frame(outer, bg=t["bg"])
        act_frame.pack(fill="x", padx=12, pady=(4, 2))

        self.merge_btn = styled_button(act_frame,
            text="\u25b6  Merge Selected Catalogues", command=self._on_merge)
        self.merge_btn.configure(font=(FONT, 11, "bold"), padx=20, pady=5)
        self.merge_btn.pack(side="left", padx=(0, 10))

        self.export_btn = styled_button(act_frame,
            text="\U0001f4be  Export Merged CSV", command=self._on_export,
            style="success")
        self.export_btn.configure(font=(FONT, 10), padx=15, pady=4,
                                  state="disabled")
        self.export_btn.pack(side="left", padx=(0, 10))

        self.progress = ttk.Progressbar(act_frame, mode="indeterminate",
                                        length=200)
        self.progress.pack(side="left", padx=10)

    # -- Helpers -----------------------------------------------------------

    def _update_dir_status(self, var, label):
        """Show parquet count for a directory path."""
        t = T()
        path = var.get().strip()
        if not path:
            label.configure(text="", fg=t["fg_dim"])
            return
        p = Path(path)
        if p.is_dir():
            n = sum(1 for f in p.glob("*.parquet") if f.stem.isdigit())
            label.configure(text=f"{n:,} satellite files", fg=t["success"])
        else:
            label.configure(text="Not found", fg=t["danger"])

    def _browse(self, key):
        s = self.sources[key]
        ft = s['ext'] + [('All files', '*.*')]
        p = filedialog.askopenfilename(title=f"Select {s['label']}",
                                       filetypes=ft)
        if p:
            s['path'].set(p)
            # Save immediately so paths survive crashes
            from gui.theme import set_pref, save_prefs
            set_pref(f"path_{key}", p)
            save_prefs()

    def _browse_lc(self):
        p = filedialog.askdirectory(title="Select MMT-9 lightcurves directory")
        if p:
            self.mmt9_lc_var.set(p)
            self._save_dir_prefs()
            self._push_dirs_to_app()

    def _browse_sdlcd_lc(self):
        p = filedialog.askdirectory(title="Select SDLCD lightcurves directory")
        if p:
            self.sdlcd_lc_var.set(p)
            self._save_dir_prefs()
            self._push_dirs_to_app()

    def _browse_tle_dir(self):
        p = filedialog.askdirectory(title="Select TLE histories directory")
        if p:
            self.tle_dir_var.set(p)
            self._save_dir_prefs()
            self._push_dirs_to_app()

    def _save_dir_prefs(self):
        """Persist directory paths immediately."""
        from gui.theme import set_pref, save_prefs
        for key, var in [("mmt9_lc_dir", self.mmt9_lc_var),
                         ("sdlcd_lc_dir", self.sdlcd_lc_var),
                         ("tle_dir", self.tle_dir_var)]:
            v = var.get().strip()
            if v:
                set_pref(key, v)
        save_prefs()

    def _push_dirs_to_app(self):
        """Immediately update the app's shared directory state and refresh
        downstream tabs if data is already loaded."""
        lc = self.mmt9_lc_var.get().strip() or None
        sd = self.sdlcd_lc_var.get().strip() or None
        td = self.tle_dir_var.get().strip() or None
        if lc: self.app.mmt9_lc_dir = lc
        if sd: self.app.sdlcd_lc_dir = sd
        if td: self.app.tle_dir = td
        # Refresh downstream tabs if merged data already exists
        if self.app.merged_df is not None:
            self.app.lc_tab.load_data(self.app.merged_df,
                                      self.app.mmt9_lc_dir,
                                      self.app.sdlcd_lc_dir)
            self.app.tle_tab.load_data(self.app.merged_df,
                                       self.app.tle_dir)

    def _log(self, msg):
        self.summary_text.configure(state="normal")
        self.summary_text.insert("end", msg + "\n")
        self.summary_text.see("end")
        self.summary_text.configure(state="disabled")
        self.frame.update_idletasks()

    # -- Merge logic -------------------------------------------------------

    def _on_merge(self):
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.configure(state="disabled")
        self.merge_btn.configure(state="disabled")
        self.progress.start(15)
        threading.Thread(target=self._merge_worker, daemon=True).start()

    def _merge_worker(self):
        t = T()
        try:
            loaded = {}
            for key, s in self.sources.items():
                if not self.include_vars[key].get():
                    continue
                path = s['path'].get().strip()
                if not path:
                    continue
                self._log(f"Loading {s['label']}: {Path(path).name}...")
                try:
                    if key == 'mmt9':
                        loader = (load_mmt9_official_catalogue
                                  if self.mmt9_cat_type.get() == "official"
                                  else load_mmt9_catalogue)
                    else:
                        loader = s['loader']
                    df = loader(path)
                    s['df'] = df
                    loaded[key] = df
                    self._log(f"  \u2192 {len(df):,} rows, {len(df.columns)} cols")
                    self.frame.after(0, lambda s=s, n=len(df):
                        s['status_label'].configure(
                            text=f"\u2713 {n:,} rows", fg=t["success"]))
                except Exception as e:
                    self._log(f"  ERROR: {e}")
                    self.frame.after(0, lambda s=s:
                        s['status_label'].configure(
                            text="\u2717 Failed", fg=t["danger"]))

            if not loaded:
                self._log("\nNo sources loaded. Select files and try again.")
                return

            self._log(f"\nMerging {len(loaded)} source(s) on NORAD ID...")
            base = ('satcat' if 'satcat' in loaded
                    else max(loaded, key=lambda k: len(loaded[k])))
            merged = merge_catalogues(loaded, base=base)

            sep = "=" * 55
            self._log(f"\n{sep}")
            self._log("MERGE COMPLETE")
            self._log(sep)
            self._log(f"  Total rows:    {len(merged):,}")
            self._log(f"  Total columns: {len(merged.columns)}\n")

            for key in loaded:
                flag = f'has_{key}'
                if flag in merged.columns:
                    n = merged[flag].sum()
                    self._log(f"  {key.upper():8s}  {n:>7,} objects")

            self._log("\nKey Field Completeness:")
            hdr = f"  {'Field':<35s} {'Non-null':>10s} {'%':>8s}"
            self._log(hdr)
            self._log("  " + "-" * 55)
            kf = [c for c in merged.columns if any(
                k in c for k in ['mass', 'rcs', 'xsect', 'apogee', 'perigee',
                                  'inclination', 'period', 'object_type',
                                  'object_class', 'purpose', 'orbit_class'])]
            for col in sorted(kf):
                nn = merged[col].notna().sum()
                pct = 100 * nn / len(merged) if len(merged) else 0
                self._log(f"  {col:<35s} {nn:>10,} {pct:>7.1f}%")

            lc = self.mmt9_lc_var.get().strip() or None
            sd = self.sdlcd_lc_var.get().strip() or None
            td = self.tle_dir_var.get().strip() or None
            self.frame.after(0, lambda: self.app.set_merged_data(
                merged, lc, sd, td))
            self.frame.after(0, lambda: self.export_btn.configure(
                state="normal"))

        except Exception as e:
            self._log(f"\nFATAL ERROR: {e}")
            import traceback
            self._log(traceback.format_exc())
        finally:
            self.frame.after(0, lambda: self.merge_btn.configure(
                state="normal"))
            self.frame.after(0, self.progress.stop)

    def _on_export(self):
        if self.app.merged_df is None:
            return
        p = filedialog.asksaveasfilename(
            title="Export Merged Dataset", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if p:
            self.app.merged_df.to_csv(p, index=False)
            messagebox.showinfo("Exported",
                f"Saved {len(self.app.merged_df):,} rows to:\n{p}")

    def _on_import(self):
        """Load a previously exported merged CSV, bypassing the merge step."""
        path = filedialog.askopenfilename(
            title="Load Merged Dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        # Disable UI and show explicit loading state
        self.merge_btn.configure(state="disabled")
        self.progress.start(15)
        self._log(f"Loading CSV: {Path(path).name}")
        self._log("  Reading file (this may take a moment for large datasets)...")
        threading.Thread(target=self._import_worker, args=(path,),
                         daemon=True).start()

    def _import_worker(self, path):
        """Background worker for CSV import."""
        try:
            self._log("  Parsing CSV data...")
            df = pd.read_csv(path, low_memory=False)
            if 'norad_id' not in df.columns:
                self.frame.after(0, lambda: messagebox.showerror(
                    "Invalid File",
                    "CSV must contain a 'norad_id' column.\n"
                    "Use a merged dataset exported from this tab."))
                return

            df['norad_id'] = pd.to_numeric(
                df['norad_id'], errors='coerce').astype('Int64')

            n_rows = len(df)
            n_cols = len(df.columns)
            self._log(f"  Loaded {n_rows:,} rows, {n_cols} columns")
            self._log("  Populating Explorer tab...")

            lc = self.mmt9_lc_var.get().strip() or None
            sd = self.sdlcd_lc_var.get().strip() or None
            td = self.tle_dir_var.get().strip() or None

            def _finish():
                self._log("  Populating Light Curve and TLE tabs...")
                self.app.set_merged_data(df, lc, sd, td)
                self.export_btn.configure(state="normal")
                if hasattr(self.app, 'analysis_tab'):
                    self._log("  Populating ML Readiness tab...")
                    self._extract_analysis_data(df)
                self._log("Import complete.\n")
                messagebox.showinfo("Import Successful",
                    f"Loaded {n_rows:,} satellites ({n_cols} columns)"
                    f"\nfrom: {Path(path).name}")

            self.frame.after(0, _finish)

        except Exception as e:
            self._log(f"Import error: {e}")
            self.frame.after(0, lambda: messagebox.showerror(
                "Import Error", f"Failed to load CSV:\n{e}"))
        finally:
            self.frame.after(0, lambda: self.merge_btn.configure(
                state="normal"))
            self.frame.after(0, self.progress.stop)

    def _extract_analysis_data(self, df):
        """Extract per-source pseudo-DataFrames for the ML analysis tab."""
        def _extract(prefix):
            cols = [c for c in df.columns if c.startswith(prefix + '_')]
            if not cols:
                return None
            sub = df[['norad_id'] + cols].copy()
            sub.columns = [c.replace(prefix + '_', '') if c != 'norad_id'
                           else c for c in sub.columns]
            sub = sub.dropna(subset=[c for c in sub.columns if c != 'norad_id'],
                             how='all')
            return sub if len(sub) > 0 else None

        self.app.analysis_tab.set_source_data(
            lc_cat=_extract('mmt9'),
            sdlcd_cat=_extract('sdlcd'),
            tle_idx=_extract('tle'),
            discos=_extract('discos'),
            merged=df,
        )