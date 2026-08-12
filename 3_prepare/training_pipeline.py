"""
ML Data Normalisation Pipeline
================================
Processes raw light curve data (MMT-9, SDLCD, or any future source), TLE orbital data,
and DISCOS labels into a single training-ready dataset.

10-step pipeline with checkpoint/resume, interactive menu, and comprehensive reporting.

Usage:
    python training_pipeline.py                    # Interactive menu
    python training_pipeline.py --mmt9_dir ./output --sdlcd_dir ./sdlcd_output \
        --tle_dir ./output/tle_histories --discos ./discos_catalogue.csv --output ./ml_ready
"""

__version__ = "2.0.0"

import os
import sys
import json
import hashlib
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import PROC_MMT9, PROC_SDLCD, PROC_TLE, PROC_DISCOS, TRAINING_DIR, RAW_DISCOS, ensure_dirs
from common.menu import (
    print_header as _print_header, prompt_choice, prompt_confirm,
    print_section, print_summary_table, print_warning, print_error,
)
from common.logging_setup import setup_logging

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


#  DEFAULT CONFIGURATION


DEFAULT_CONFIG = {
    'resample_length': 1024,
    'gap_threshold_multiplier': 10,
    'min_points_per_track': 20,
    'min_duration_seconds': 5.0,
    'max_nan_fraction': 0.10,
    'tle_window_days': 30,
    'tle_trends_enabled': True,
    'split_ratios': [0.70, 0.15, 0.15],
    'random_seed': 42,
    'checkpoint_batch_size': 1000,
}

TYPE_MAP = {
    "Payload": "Payload",
    "Rocket Body": "Rocket Body",
    "Payload Fragmentation Debris": "Debris",
    "Rocket Fragmentation Debris": "Debris",
    "Payload Mission Related Object": "Debris",
    "Rocket Mission Related Object": "Debris",
    "Payload Debris": "Debris",
    "Rocket Debris": "Debris",
    "Other Debris": "Debris",
    "Other Mission Related Object": "Debris",
    "Unknown": None,
}

SHAPE_MAP = {
    "Cyl": "Cylinder", "Cyl + 1 Nozzle": "Cylinder", "Cyl + Cone": "Cylinder",
    "Cyl + 2 Pan": "Cylinder", "Sphere + Cyl": "Cylinder",
    "Box": "Box", "Box + 1 Pan": "Box-wing", "Box + 2 Pan": "Box-wing",
    "Trap Box + 2 Pan": "Box-wing", "Hex Poly + 2 Pan": "Box-wing",
    "Box + 2 Ant": "Box-wing",
    "Sphere": "Sphere", "Sphere + Cone": "Sphere",
    "Cone": "Cone",
    "Irr": "Irregular",
}



#  CHECKPOINT SYSTEM


class CheckpointManager:
    """Manages pipeline checkpoints for resume capability."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.intermediate_dir = self.output_dir / '.intermediate'
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.intermediate_dir / 'checkpoint.json'
        self.state = self._load()

    def _load(self):
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            'completed_steps': [],
            'config_hash': None,
            'started_at': datetime.now().isoformat(),
            'last_updated': None,
            'step_progress': {},
            'report_data': {},
        }

    def save(self):
        self.state['last_updated'] = datetime.now().isoformat()
        tmp = str(self.checkpoint_path) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
        if self.checkpoint_path.exists():
            os.remove(self.checkpoint_path)
        os.rename(tmp, self.checkpoint_path)

    def config_hash(self, config):
        return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]

    def is_step_done(self, step_name):
        return step_name in self.state['completed_steps']

    def mark_step_done(self, step_name):
        if step_name not in self.state['completed_steps']:
            self.state['completed_steps'].append(step_name)
        self.save()

    def get_step_progress(self, step_name):
        return self.state.get('step_progress', {}).get(step_name, 0)

    def save_step_progress(self, step_name, progress):
        if 'step_progress' not in self.state:
            self.state['step_progress'] = {}
        self.state['step_progress'][step_name] = progress
        self.save()

    def save_intermediate(self, step_name, df):
        path = self.intermediate_dir / f'{step_name}.parquet'
        tmp = str(path) + '.tmp'
        df.to_parquet(tmp, index=False)
        if path.exists():
            os.remove(path)
        os.rename(tmp, path)

    def load_intermediate(self, step_name):
        path = self.intermediate_dir / f'{step_name}.parquet'
        if path.exists():
            return pd.read_parquet(path)
        return None

    def save_report_data(self, key, data):
        self.state['report_data'][key] = data
        self.save()

    def get_report_data(self, key):
        return self.state.get('report_data', {}).get(key, {})

    def check_config(self, config):
        new_hash = self.config_hash(config)
        old_hash = self.state.get('config_hash')
        if old_hash and old_hash != new_hash and self.state['completed_steps']:
            return False  # Config changed
        self.state['config_hash'] = new_hash
        self.save()
        return True

    def reset(self):
        """Clear all checkpoints and intermediate files."""
        import shutil
        if self.intermediate_dir.exists():
            shutil.rmtree(self.intermediate_dir)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.state = {
            'completed_steps': [],
            'config_hash': None,
            'started_at': datetime.now().isoformat(),
            'last_updated': None,
            'step_progress': {},
            'report_data': {},
        }
        self.save()



#  MODULAR DATA LOADERS


def load_mmt9_tracks(lc_dir, gap_threshold=10):
    """Load MMT-9 light curve files and return standardised tracks."""
    lc_path = Path(lc_dir) / 'lightcurves'
    if not lc_path.exists():
        print(f"  [WARN] MMT-9 lightcurves dir not found: {lc_path}")
        return [], {'satellites': 0, 'raw_tracks': 0, 'raw_points': 0, 'gap_splits': 0}

    sat_files = sorted(lc_path.glob('*.parquet'))
    if not sat_files:
        sat_files = sorted(lc_path.glob('*.csv'))  # Fallback for pre-conversion data
    reader = pd.read_parquet if sat_files and sat_files[0].suffix == '.parquet' else pd.read_csv
    tracks = []
    stats = {'satellites': 0, 'raw_tracks': 0, 'raw_points': 0, 'gap_splits': 0}

    for sat_file in tqdm(sat_files, desc="  Loading MMT-9", unit="sat"):
        try:
            norad_id = int(sat_file.stem)
        except ValueError:
            continue

        try:
            df = reader(sat_file)
            if df.empty or 'StdMag' not in df.columns:
                continue
        except Exception:
            continue

        stats['satellites'] += 1
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed', errors='coerce')
        df = df.dropna(subset=['Datetime', 'StdMag'])

        # Group by Track column
        for track_id, group in df.groupby('Track'):
            group = group.sort_values('Datetime').reset_index(drop=True)
            if len(group) < 2:
                continue

            stats['raw_tracks'] += 1
            stats['raw_points'] += len(group)

            datetimes = group['Datetime'].values
            magnitudes = group['StdMag'].values.astype(np.float64)
            distances = group['Distance_km'].values.astype(np.float64) if 'Distance_km' in group.columns else None
            phases = group['Phase_deg'].values.astype(np.float64) if 'Phase_deg' in group.columns else None
            filt = group['Filter'].iloc[0] if 'Filter' in group.columns else 'Clear'

            # Gap splitting
            sub_tracks = _split_on_gaps(datetimes, magnitudes, gap_threshold,
                                        distances=distances, phases=phases)

            for i, st in enumerate(sub_tracks):
                if i > 0:
                    stats['gap_splits'] += 1
                tid = int(track_id) * 100 + i if len(sub_tracks) > 1 else int(track_id)
                tracks.append({
                    'norad_id': norad_id,
                    'track_id': tid,
                    'source': 'MMT9',
                    'filter': str(filt),
                    'datetimes': st['datetimes'],
                    'magnitudes': st['magnitudes'],
                    'mag_errors': None,
                    'distances_km': st.get('distances'),
                    'phase_angles_deg': st.get('phases'),
                })

    return tracks, stats


def load_sdlcd_tracks(lc_dir, gap_threshold=10):
    """Load SDLCD light curve files and return standardised tracks."""
    lc_path = Path(lc_dir) / 'lightcurves'
    if not lc_path.exists():
        print(f"  [WARN] SDLCD lightcurves dir not found: {lc_path}")
        return [], {'satellites': 0, 'raw_tracks': 0, 'raw_points': 0, 'gap_splits': 0}

    sat_files = sorted(lc_path.glob('*.parquet'))
    if not sat_files:
        sat_files = sorted(lc_path.glob('*.csv'))
    reader = pd.read_parquet if sat_files and sat_files[0].suffix == '.parquet' else pd.read_csv
    tracks = []
    stats = {'satellites': 0, 'raw_tracks': 0, 'raw_points': 0, 'gap_splits': 0}

    for sat_file in tqdm(sat_files, desc="  Loading SDLCD", unit="sat"):
        try:
            norad_id = int(sat_file.stem)
        except ValueError:
            continue

        try:
            df = reader(sat_file)
            if df.empty or 'StdMag' not in df.columns:
                continue
        except Exception:
            continue

        stats['satellites'] += 1
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed', errors='coerce')
        df = df.dropna(subset=['Datetime', 'StdMag'])

        for track_id, group in df.groupby('Track'):
            group = group.sort_values('Datetime').reset_index(drop=True)
            if len(group) < 2:
                continue

            stats['raw_tracks'] += 1
            stats['raw_points'] += len(group)

            datetimes = group['Datetime'].values
            magnitudes = group['StdMag'].values.astype(np.float64)
            mag_errors = group['Mag_err'].values.astype(np.float64) if 'Mag_err' in group.columns and group['Mag_err'].notna().any() else None
            filt = group['Filter'].iloc[0] if 'Filter' in group.columns else 'R'

            sub_tracks = _split_on_gaps(datetimes, magnitudes, gap_threshold,
                                        mag_errors=mag_errors)

            for i, st in enumerate(sub_tracks):
                if i > 0:
                    stats['gap_splits'] += 1
                tid = int(track_id) * 100 + i if len(sub_tracks) > 1 else int(track_id)
                tracks.append({
                    'norad_id': norad_id,
                    'track_id': tid,
                    'source': 'SDLCD',
                    'filter': str(filt),
                    'datetimes': st['datetimes'],
                    'magnitudes': st['magnitudes'],
                    'mag_errors': st.get('mag_errors'),
                    'distances_km': None,
                    'phase_angles_deg': None,
                })

    return tracks, stats


# Loader registry - add new sources here
LOADERS = OrderedDict([
    ("MMT9", load_mmt9_tracks),
    ("SDLCD", load_sdlcd_tracks),
])


def _split_on_gaps(datetimes, magnitudes, gap_threshold, **extra_arrays):
    """Split a track into sub-tracks at long temporal gaps."""
    if len(datetimes) < 2:
        result = {'datetimes': datetimes, 'magnitudes': magnitudes}
        for k, v in extra_arrays.items():
            if v is not None:
                result[k] = v
        return [result]

    # Compute time deltas in seconds
    dt_ns = np.diff(datetimes.astype('datetime64[ns]').astype(np.int64))
    deltas_s = dt_ns / 1e9
    deltas_s = deltas_s[deltas_s > 0]

    if len(deltas_s) == 0:
        result = {'datetimes': datetimes, 'magnitudes': magnitudes}
        for k, v in extra_arrays.items():
            if v is not None:
                result[k] = v
        return [result]

    median_interval = np.median(deltas_s)
    if median_interval <= 0:
        median_interval = 1.0

    threshold = gap_threshold * median_interval

    # Find split points
    all_deltas = np.diff(datetimes.astype('datetime64[ns]').astype(np.int64)) / 1e9
    split_indices = np.where(all_deltas > threshold)[0] + 1

    if len(split_indices) == 0:
        result = {'datetimes': datetimes, 'magnitudes': magnitudes}
        for k, v in extra_arrays.items():
            if v is not None:
                result[k] = v
        return [result]

    # Split at gap points
    all_indices = np.concatenate([[0], split_indices, [len(datetimes)]])
    sub_tracks = []
    for j in range(len(all_indices) - 1):
        s, e = all_indices[j], all_indices[j + 1]
        if e - s < 2:
            continue
        st = {
            'datetimes': datetimes[s:e],
            'magnitudes': magnitudes[s:e],
        }
        for k, v in extra_arrays.items():
            if v is not None:
                st[k] = v[s:e]
        sub_tracks.append(st)

    return sub_tracks if sub_tracks else [{'datetimes': datetimes, 'magnitudes': magnitudes}]



#  STEP 2: FILTER & VALIDATE


def filter_tracks(tracks, config):
    """Filter tracks based on quality criteria. Returns (valid_tracks, filter_report)."""
    min_pts = config['min_points_per_track']
    min_dur = config['min_duration_seconds']
    max_nan = config['max_nan_fraction']

    report = {
        'too_few_points': 0, 'too_short': 0,
        'zero_variance': 0, 'excessive_nan': 0,
        'total_input': len(tracks), 'total_output': 0,
    }

    valid = []
    for t in tqdm(tracks, desc="  Filtering", unit="trk"):
        mags = t['magnitudes']
        dts = t['datetimes']

        # Check point count
        if len(mags) < min_pts:
            report['too_few_points'] += 1
            continue

        # Check duration
        duration_s = (dts[-1] - dts[0]) / np.timedelta64(1, 's') if len(dts) > 1 else 0
        if duration_s < min_dur:
            report['too_short'] += 1
            continue

        # Check NaN/inf
        bad = np.isnan(mags) | np.isinf(mags)
        if bad.sum() / len(mags) > max_nan:
            report['excessive_nan'] += 1
            continue

        # Remove NaNs
        good_mask = ~bad
        if good_mask.sum() < min_pts:
            report['too_few_points'] += 1
            continue

        # Check variance
        clean_mags = mags[good_mask]
        if np.std(clean_mags) < 1e-10:
            report['zero_variance'] += 1
            continue

        # Keep only clean points
        t_clean = dict(t)
        t_clean['magnitudes'] = mags[good_mask]
        t_clean['datetimes'] = dts[good_mask]
        if t.get('mag_errors') is not None:
            t_clean['mag_errors'] = t['mag_errors'][good_mask]
        if t.get('distances_km') is not None:
            t_clean['distances_km'] = t['distances_km'][good_mask]
        if t.get('phase_angles_deg') is not None:
            t_clean['phase_angles_deg'] = t['phase_angles_deg'][good_mask]

        valid.append(t_clean)

    report['total_output'] = len(valid)
    return valid, report



#  STEP 3: EXTRACT RAW TRACK FEATURES


def extract_raw_features(tracks):
    """Compute pre-normalisation statistics for each track."""
    features = []
    for t in tqdm(tracks, desc="  Raw features", unit="trk"):
        mags = t['magnitudes']
        dts = t['datetimes']
        duration_s = (dts[-1] - dts[0]) / np.timedelta64(1, 's')

        feat = {
            'norad_id': t['norad_id'],
            'track_id': t['track_id'],
            'source': t['source'],
            'filter': t['filter'],
            'observation_datetime': str(dts[0]),
            'raw_mean_mag': float(np.mean(mags)),
            'raw_std_mag': float(np.std(mags)),
            'raw_amplitude': float(np.max(mags) - np.min(mags)),
            'observation_duration_s': float(duration_s),
            'original_num_points': len(mags),
            'sampling_rate_hz': float(len(mags) / duration_s) if duration_s > 0 else 0.0,
            'mean_phase_angle_deg': float(np.nanmean(t['phase_angles_deg'])) if t.get('phase_angles_deg') is not None else np.nan,
            'mean_distance_km': float(np.nanmean(t['distances_km'])) if t.get('distances_km') is not None else np.nan,
        }
        features.append(feat)

    return pd.DataFrame(features)



#  STEP 4: NORMALISE & RESAMPLE


def normalise_and_resample(tracks, resample_length, checkpoint_mgr=None, step_name='step4'):
    """Per-track normalise and resample to fixed length with binary mask.
    
    Uses vectorised operations for speed:
    - np.interp for magnitude interpolation (already vectorised)
    - np.searchsorted + distance check for mask (O(n log n) instead of O(n²))
    
    The mask is TRUTHFUL: a grid point is marked 1 only if a real data point
    exists within one median sampling interval of that grid position.
    Points in gaps or beyond the data edges are marked 0.
    """

    start_idx = checkpoint_mgr.get_step_progress(step_name) if checkpoint_mgr else 0
    results = []

    # Load partial results if resuming
    if start_idx > 0 and checkpoint_mgr:
        partial = checkpoint_mgr.load_intermediate(step_name + '_partial')
        if partial is not None:
            results = partial.to_dict('records')
            print(f"  Resuming from track {start_idx:,}/{len(tracks):,}")

    for i in tqdm(range(start_idx, len(tracks)), desc="  Normalising",
                  unit="trk", initial=start_idx, total=len(tracks)):
        t = tracks[i]
        mags = t['magnitudes']
        dts = t['datetimes']

        # Per-track normalisation
        mean_mag = np.mean(mags)
        std_mag = np.std(mags)
        if std_mag < 1e-10:
            std_mag = 1.0
        norm_mags = (mags - mean_mag) / std_mag

        # Convert times to float64 nanoseconds for arithmetic
        orig_ns = dts.astype('datetime64[ns]').astype(np.float64)
        t_start = orig_ns[0]
        t_end = orig_ns[-1]

        # Uniform time grid
        grid_ns = np.linspace(t_start, t_end, resample_length)

        # Interpolate normalised magnitudes onto grid (already vectorised)
        resampled = np.interp(grid_ns, orig_ns, norm_mags)

        # Build mask using searchsorted (vectorised nearest-neighbour)
        # For each grid point, find the nearest original data point
        # and check if it's within one median sampling interval
        if len(orig_ns) > 1:
            median_interval_ns = float(np.median(np.diff(orig_ns)))
            if median_interval_ns <= 0:
                median_interval_ns = 1e9  # 1 second fallback
        else:
            median_interval_ns = 1e9

        # searchsorted gives the insertion index - nearest is either idx or idx-1
        insert_idx = np.searchsorted(orig_ns, grid_ns, side='left')
        insert_idx = np.clip(insert_idx, 0, len(orig_ns) - 1)

        # Distance to the point at insert_idx
        dist_right = np.abs(orig_ns[insert_idx] - grid_ns)

        # Distance to the point at insert_idx - 1 (if it exists)
        left_idx = np.clip(insert_idx - 1, 0, len(orig_ns) - 1)
        dist_left = np.abs(orig_ns[left_idx] - grid_ns)

        # Nearest distance is the minimum of left and right
        nearest_dist = np.minimum(dist_right, dist_left)

        # Mask: 1 if nearest real point is within one median interval
        mask = (nearest_dist <= median_interval_ns).astype(np.int8)

        # Where mask is 0 (in a gap), set magnitude to 0 (neutral after normalisation)
        resampled[mask == 0] = 0.0

        # Resample mag errors if available
        mag_err_resampled = None
        if t.get('mag_errors') is not None:
            mag_err_resampled = np.interp(grid_ns, orig_ns, t['mag_errors']).tolist()

        results.append({
            'norad_id': t['norad_id'],
            'track_id': t['track_id'],
            'lightcurve': resampled.tolist(),
            'mask': mask.tolist(),
            'mag_errors_resampled': mag_err_resampled,
            'mask_coverage': float(mask.sum() / resample_length),
        })

        # Checkpoint every N tracks
        if checkpoint_mgr and (i + 1) % 1000 == 0:
            checkpoint_mgr.save_intermediate(step_name + '_partial', pd.DataFrame(results))
            checkpoint_mgr.save_step_progress(step_name, i + 1)

    return pd.DataFrame(results)



#  STEP 5: TLE FEATURE ENGINEERING


def extract_tle_features(features_df, tle_dir, config, checkpoint_mgr=None, step_name='step5'):
    """Match TLE data to each track and extract orbital features."""
    tle_path = Path(tle_dir)
    window_days = config['tle_window_days']
    compute_trends = config['tle_trends_enabled']

    start_idx = checkpoint_mgr.get_step_progress(step_name) if checkpoint_mgr else 0
    tle_rows = []

    if start_idx > 0 and checkpoint_mgr:
        partial = checkpoint_mgr.load_intermediate(step_name + '_partial')
        if partial is not None:
            tle_rows = partial.to_dict('records')
            print(f"  Resuming TLE matching from {start_idx}/{len(features_df)}")

    # Cache for loaded TLE files
    tle_cache = {}

    for i in tqdm(range(start_idx, len(features_df)), desc="  TLE matching",
                  unit="trk", initial=start_idx, total=len(features_df)):
        row = features_df.iloc[i]
        norad_id = int(row['norad_id'])
        obs_dt_str = row['observation_datetime']

        try:
            obs_dt = pd.to_datetime(obs_dt_str)
        except Exception:
            tle_rows.append(_empty_tle_row(norad_id, row['track_id']))
            continue

        # Load TLE data (with caching)
        if norad_id not in tle_cache:
            sat_tle_path = tle_path / f"{norad_id}.parquet"
            if not sat_tle_path.exists():
                sat_tle_path = tle_path / f"{norad_id}.csv"  # Fallback
            if sat_tle_path.exists():
                try:
                    tle_df = pd.read_parquet(sat_tle_path) if sat_tle_path.suffix == '.parquet' else pd.read_csv(sat_tle_path)
                    tle_df['EPOCH'] = pd.to_datetime(tle_df['EPOCH'], format='mixed', errors='coerce')
                    tle_df = tle_df.dropna(subset=['EPOCH'])
                    tle_cache[norad_id] = tle_df
                except Exception:
                    tle_cache[norad_id] = None
            else:
                tle_cache[norad_id] = None

            # Keep cache reasonable size
            if len(tle_cache) > 5000:
                oldest = list(tle_cache.keys())[0]
                del tle_cache[oldest]

        tle_df = tle_cache.get(norad_id)

        if tle_df is None or tle_df.empty:
            tle_rows.append(_empty_tle_row(norad_id, row['track_id']))
            continue

        # Find nearest TLE
        time_diffs = (tle_df['EPOCH'] - obs_dt).abs()
        nearest_idx = time_diffs.idxmin()
        nearest = tle_df.loc[nearest_idx]
        tle_age_days = time_diffs.loc[nearest_idx].total_seconds() / 86400.0

        # Extract features
        ecc = _safe_float(nearest.get('ECCENTRICITY', np.nan))
        bstar = _safe_float(nearest.get('BSTAR', np.nan))
        mm = _safe_float(nearest.get('MEAN_MOTION', np.nan))

        tle_row = {
            'norad_id': norad_id,
            'track_id': row['track_id'],
            'has_tle': 1,
            'tle_age_days': float(tle_age_days),
            'inclination_deg': _safe_float(nearest.get('INCLINATION', np.nan)),
            'eccentricity': ecc,
            'eccentricity_log': float(np.log10(ecc + 1e-8)) if not np.isnan(ecc) else np.nan,
            'mean_motion': mm,
            'bstar': bstar,
            'bstar_log': float(np.log10(abs(bstar) + 1e-10)) * (1 if bstar >= 0 else -1) if not np.isnan(bstar) else np.nan,
            'semimajor_axis_km': _safe_float(nearest.get('SEMIMAJOR_AXIS', np.nan)),
            'period_min': _safe_float(nearest.get('PERIOD', np.nan)),
            'perigee_km': _safe_float(nearest.get('PERIAPSIS', np.nan)),
            'apogee_km': _safe_float(nearest.get('APOAPSIS', np.nan)),
            'rcs_size': nearest.get('RCS_SIZE', None) if 'RCS_SIZE' in tle_df.columns else None,
        }

        # Windowed trends
        if compute_trends:
            window_start = obs_dt - timedelta(days=window_days)
            window_end = obs_dt + timedelta(days=window_days)
            window_df = tle_df[(tle_df['EPOCH'] >= window_start) & (tle_df['EPOCH'] <= window_end)]

            if len(window_df) >= 3:
                epoch_days = (window_df['EPOCH'] - obs_dt).dt.total_seconds() / 86400.0
                tle_row['bstar_trend'] = _linear_slope(epoch_days.values, pd.to_numeric(window_df['BSTAR'], errors='coerce').values)
                tle_row['mean_motion_trend'] = _linear_slope(epoch_days.values, pd.to_numeric(window_df['MEAN_MOTION'], errors='coerce').values)
                tle_row['eccentricity_trend'] = _linear_slope(epoch_days.values, pd.to_numeric(window_df['ECCENTRICITY'], errors='coerce').values)
            else:
                tle_row['bstar_trend'] = 0.0
                tle_row['mean_motion_trend'] = 0.0
                tle_row['eccentricity_trend'] = 0.0
        else:
            tle_row['bstar_trend'] = np.nan
            tle_row['mean_motion_trend'] = np.nan
            tle_row['eccentricity_trend'] = np.nan

        tle_rows.append(tle_row)

        if checkpoint_mgr and (i + 1) % 1000 == 0:
            checkpoint_mgr.save_intermediate(step_name + '_partial', pd.DataFrame(tle_rows))
            checkpoint_mgr.save_step_progress(step_name, i + 1)

    return pd.DataFrame(tle_rows)


def _empty_tle_row(norad_id, track_id):
    return {
        'norad_id': norad_id, 'track_id': track_id, 'has_tle': 0,
        'tle_age_days': np.nan, 'inclination_deg': np.nan,
        'eccentricity': np.nan, 'eccentricity_log': np.nan,
        'mean_motion': np.nan, 'bstar': np.nan, 'bstar_log': np.nan,
        'semimajor_axis_km': np.nan, 'period_min': np.nan,
        'perigee_km': np.nan, 'apogee_km': np.nan, 'rcs_size': None,
        'bstar_trend': np.nan, 'mean_motion_trend': np.nan,
        'eccentricity_trend': np.nan,
    }


def _safe_float(val):
    try:
        v = float(val)
        return v if not np.isnan(v) and not np.isinf(v) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _linear_slope(x, y):
    """Compute linear regression slope, handling NaNs."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return 0.0
    x_clean, y_clean = x[mask], y[mask]
    try:
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        return float(slope) if np.isfinite(slope) else 0.0
    except (np.linalg.LinAlgError, ValueError):
        return 0.0



#  STEP 6: ORBIT REGIME


def classify_orbit_regime(row):
    """Classify orbit regime from TLE orbital elements."""
    perigee = row.get('perigee_km', np.nan)
    apogee = row.get('apogee_km', np.nan)
    inc = row.get('inclination_deg', np.nan)
    ecc = row.get('eccentricity', np.nan)

    if pd.isna(perigee) or pd.isna(apogee):
        return 'UNKNOWN'

    if perigee < 2000:
        if not pd.isna(inc) and 96 <= inc <= 100:
            return 'SSO'
        return 'LEO'
    elif 2000 <= perigee < 20000:
        return 'MEO'
    elif 35000 <= perigee <= 36500 and (pd.isna(ecc) or ecc < 0.01):
        return 'GEO'
    elif not pd.isna(ecc) and ecc > 0.1 and apogee > 30000:
        if not pd.isna(inc) and 62 <= inc <= 65:
            return 'MOLNIYA'
        return 'HEO'
    else:
        return 'OTHER'



#  STEP 7: JOIN LABELS


def join_labels(df, discos_path):
    """Join DISCOS ground truth labels by NORAD ID."""
    discos = pd.read_csv(discos_path)
    discos['norad_id'] = pd.to_numeric(discos['norad_id'], errors='coerce').astype('Int64')
    discos = discos.dropna(subset=['norad_id'])

    # A/M ratio
    am = discos[['norad_id', 'am_ratio_avg']].dropna(subset=['am_ratio_avg']).copy()
    am['am_ratio_log'] = np.log10(am['am_ratio_avg'])
    df = df.merge(am, on='norad_id', how='left')

    # Object type
    oc = discos[['norad_id', 'object_class']].dropna(subset=['object_class']).copy()
    oc['object_type'] = oc['object_class'].map(TYPE_MAP)
    df = df.merge(oc[['norad_id', 'object_type']], on='norad_id', how='left')

    # Shape
    sh = discos[['norad_id', 'shape']].dropna(subset=['shape']).copy()
    sh['shape_broad'] = sh['shape'].map(SHAPE_MAP).fillna('Other')
    df = df.merge(sh[['norad_id', 'shape_broad']], on='norad_id', how='left')

    # Mass and cross-section (additional labels)
    extra = discos[['norad_id', 'mass_kg', 'xsect_avg_m2']].copy()
    df = df.merge(extra, on='norad_id', how='left')

    return df



#  STEP 8: SCALE FEATURES


def scale_features(df, split_col='split'):
    """Standard scale numeric features, fitted on train split only."""
    feature_cols = [
        'raw_mean_mag', 'raw_std_mag', 'raw_amplitude',
        'observation_duration_s', 'original_num_points', 'sampling_rate_hz',
        'inclination_deg', 'eccentricity_log', 'mean_motion', 'bstar_log',
        'semimajor_axis_km', 'period_min', 'perigee_km', 'apogee_km',
        'tle_age_days',
    ]

    # Add trend columns if they exist and aren't all NaN
    for col in ['bstar_trend', 'mean_motion_trend', 'eccentricity_trend']:
        if col in df.columns and df[col].notna().any():
            feature_cols.append(col)

    # Filter to columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Fit on train only
    train_mask = df[split_col] == 'train'
    scaler_params = {}

    for col in feature_cols:
        train_vals = pd.to_numeric(df.loc[train_mask, col], errors='coerce')
        mean_val = float(train_vals.mean()) if train_vals.notna().any() else 0.0
        std_val = float(train_vals.std()) if train_vals.notna().any() and train_vals.std() > 1e-10 else 1.0

        scaler_params[col] = {'mean': mean_val, 'std': std_val}
        df[col + '_scaled'] = (pd.to_numeric(df[col], errors='coerce') - mean_val) / std_val

    return df, scaler_params, feature_cols



#  STEP 9: TRAIN/VAL/TEST SPLIT


def split_dataset(df, ratios, seed):
    """Per-satellite stratified split."""
    np.random.seed(seed)

    # Get unique satellites with their object type
    sat_info = df.groupby('norad_id').agg(
        object_type=('object_type', 'first'),
        num_tracks=('track_id', 'count'),
    ).reset_index()

    # Fill missing type with 'Unknown' for stratification
    sat_info['strat_type'] = sat_info['object_type'].fillna('Unknown')

    assignments = {}
    train_r, val_r, test_r = ratios

    for stype, group in sat_info.groupby('strat_type'):
        ids = group['norad_id'].values.copy()
        np.random.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_r)
        n_val = int(n * val_r)

        for nid in ids[:n_train]:
            assignments[nid] = 'train'
        for nid in ids[n_train:n_train + n_val]:
            assignments[nid] = 'val'
        for nid in ids[n_train + n_val:]:
            assignments[nid] = 'test'

    df['split'] = df['norad_id'].map(assignments)
    return df, assignments



#  STEP 10: ANALYSIS REPORT


def generate_report(df, source_stats, filter_report, config, output_path):
    """Generate comprehensive dataset analysis report."""
    lines = []
    lines.append("=" * 62)
    lines.append("  ML DATASET ANALYSIS REPORT")
    lines.append("=" * 62)

    # Section 1: Source summary
    lines.append("\n  1. SOURCE DATA SUMMARY")
    lines.append("  " + "-" * 58)
    total_sats = 0
    total_tracks = 0
    total_pts = 0
    for src, stats in source_stats.items():
        lines.append(f"  {src:10s}  {stats['satellites']:>8,} sats  {stats['raw_tracks']:>10,} tracks  {stats['raw_points']:>14,} pts")
        total_sats += stats['satellites']
        total_tracks += stats['raw_tracks']
        total_pts += stats['raw_points']
    lines.append(f"  {'TOTAL':10s}  {total_sats:>8,} sats  {total_tracks:>10,} tracks  {total_pts:>14,} pts")

    # Section 2: Filtering
    lines.append(f"\n  2. FILTERING REPORT")
    lines.append("  " + "-" * 58)
    fr = filter_report
    lines.append(f"  Too few points (<{config['min_points_per_track']}):  {fr['too_few_points']:>8,}")
    lines.append(f"  Too short (<{config['min_duration_seconds']}s):       {fr['too_short']:>8,}")
    lines.append(f"  Zero variance:                {fr['zero_variance']:>8,}")
    lines.append(f"  Excessive NaN (>{config['max_nan_fraction']*100:.0f}%):       {fr['excessive_nan']:>8,}")
    gap_total = sum(s.get('gap_splits', 0) for s in source_stats.values())
    lines.append(f"  Gap splits created:           +{gap_total:>7,}")
    removed = fr['total_input'] - fr['total_output']
    pct = removed / fr['total_input'] * 100 if fr['total_input'] > 0 else 0
    lines.append(f"  Tracks after filtering:       {fr['total_output']:>8,}  ({100-pct:.1f}% retained)")

    # Section 3: TLE coverage
    if 'has_tle' in df.columns:
        lines.append(f"\n  3. TLE COVERAGE")
        lines.append("  " + "-" * 58)
        has_tle = df['has_tle'].sum()
        no_tle = (df['has_tle'] == 0).sum()
        lines.append(f"  Tracks with TLE:    {has_tle:>8,}  ({has_tle/len(df)*100:.1f}%)")
        lines.append(f"  Tracks without TLE: {no_tle:>8,}  ({no_tle/len(df)*100:.1f}%)")
        if 'tle_age_days' in df.columns:
            age = df['tle_age_days'].dropna()
            lines.append(f"  Mean TLE age:       {age.mean():>8.1f} days")
            lines.append(f"  Median TLE age:     {age.median():>8.1f} days")

    # Section 4: Label availability
    lines.append(f"\n  4. LABEL AVAILABILITY")
    lines.append("  " + "-" * 58)
    has_am = df['am_ratio_avg'].notna().sum() if 'am_ratio_avg' in df.columns else 0
    has_type = df['object_type'].notna().sum() if 'object_type' in df.columns else 0
    has_shape = df['shape_broad'].notna().sum() if 'shape_broad' in df.columns else 0
    n = len(df)
    lines.append(f"  Has A/M ratio:    {has_am:>8,} tracks  ({has_am/n*100:.1f}%)")
    lines.append(f"  Has object type:  {has_type:>8,} tracks  ({has_type/n*100:.1f}%)")
    lines.append(f"  Has shape:        {has_shape:>8,} tracks  ({has_shape/n*100:.1f}%)")

    # Section 5: Class distribution
    if 'split' in df.columns and 'object_type' in df.columns:
        lines.append(f"\n  5. CLASS DISTRIBUTION BY SPLIT")
        lines.append("  " + "-" * 58)
        ct = df.groupby(['split', 'object_type']).size().unstack(fill_value=0)
        for split in ['train', 'val', 'test']:
            if split in ct.index:
                parts = [f"{col}: {ct.loc[split, col]:,}" for col in ct.columns if col in ct.loc[split].index]
                lines.append(f"  {split:6s}  {' | '.join(parts)}")

    # Section 6: A/M distribution
    if 'am_ratio_avg' in df.columns:
        lines.append(f"\n  6. A/M RATIO DISTRIBUTION")
        lines.append("  " + "-" * 58)
        am = df['am_ratio_avg'].dropna()
        if len(am) > 0:
            lines.append(f"  Count:   {len(am):>10,}")
            lines.append(f"  Min:     {am.min():>10.6f} m²/kg")
            lines.append(f"  Median:  {am.median():>10.6f} m²/kg")
            lines.append(f"  Max:     {am.max():>10.6f} m²/kg")

    # Section 7: Resampling quality
    if 'mask_coverage' in df.columns:
        lines.append(f"\n  7. RESAMPLING QUALITY")
        lines.append("  " + "-" * 58)
        mc = df['mask_coverage']
        lines.append(f"  Resampled to: {config['resample_length']} points")
        lines.append(f"  Mean mask coverage:  {mc.mean()*100:.1f}% real data")
        lines.append(f"  Median:              {mc.median()*100:.1f}% real data")
        lines.append(f"  <25% real:           {(mc < 0.25).sum():,} tracks")
        lines.append(f"  >90% real:           {(mc > 0.90).sum():,} tracks")
        for src in df['source'].unique():
            src_mc = df.loc[df['source'] == src, 'mask_coverage']
            lines.append(f"  {src}: mean {src_mc.mean()*100:.1f}% real")

    # Section 8: Per-source contribution
    lines.append(f"\n  8. PER-SOURCE CONTRIBUTION")
    lines.append("  " + "-" * 58)
    for src in df['source'].unique():
        s = df[df['source'] == src]
        n_sats = s['norad_id'].nunique()
        n_trk = len(s)
        n_am = s['am_ratio_avg'].notna().sum() if 'am_ratio_avg' in s.columns else 0
        lines.append(f"  {src:10s}  {n_sats:>6,} sats  {n_trk:>8,} tracks  {n_am:>8,} with A/M")

    lines.append("\n" + "=" * 62)

    report_text = '\n'.join(lines)
    print(report_text)

    # Save report
    report_path = output_path / 'dataset_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)

    return report_text



#  MAIN PIPELINE


def run_pipeline(config, paths):
    """Execute the full 10-step pipeline."""
    output_path = Path(paths['output'])
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(output_path)

    # Check config consistency
    if not ckpt.check_config(config):
        print("\n  [WARN] Configuration has changed since last run.")
        try:
            ans = input("  Reset and start fresh? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = 'n'
        if ans in ('y', 'yes'):
            ckpt.reset()
        else:
            print("  Continuing with existing checkpoints (results may be inconsistent).")

    source_stats = {}
    all_tracks = []

    # ── Step 1: Ingest ──
    if not ckpt.is_step_done('step1'):
        print(f"\n{'─'*60}")
        print("  STEP 1: Ingesting light curve data")
        print(f"{'─'*60}")

        for source_name, loader_fn in LOADERS.items():
            src_dir = paths.get(source_name.lower() + '_dir')
            if not src_dir:
                continue
            tracks, stats = loader_fn(src_dir, config['gap_threshold_multiplier'])
            all_tracks.extend(tracks)
            source_stats[source_name] = stats
            print(f"  {source_name}: {stats['satellites']:,} sats, {stats['raw_tracks']:,} tracks, {stats['raw_points']:,} points")

        if not all_tracks:
            print("  [ERROR] No tracks loaded from any source.")
            return

        # Sort deterministically
        all_tracks.sort(key=lambda t: (t['norad_id'], t['track_id']))

        ckpt.save_report_data('source_stats', source_stats)
        ckpt.mark_step_done('step1')
        print(f"  Total: {len(all_tracks):,} tracks from {len(source_stats)} source(s)")
    else:
        print("  Step 1: Already done, loading...")
        source_stats = ckpt.get_report_data('source_stats')
        # Reload tracks (needed for subsequent steps)
        for source_name, loader_fn in LOADERS.items():
            src_dir = paths.get(source_name.lower() + '_dir')
            if not src_dir:
                continue
            tracks, _ = loader_fn(src_dir, config['gap_threshold_multiplier'])
            all_tracks.extend(tracks)
        all_tracks.sort(key=lambda t: (t['norad_id'], t['track_id']))

    # ── Step 2: Filter ──
    if not ckpt.is_step_done('step2'):
        print(f"\n{'─'*60}")
        print("  STEP 2: Filtering & validating tracks")
        print(f"{'─'*60}")
        all_tracks, filter_report = filter_tracks(all_tracks, config)
        ckpt.save_report_data('filter_report', filter_report)
        ckpt.mark_step_done('step2')
        print(f"  Kept {filter_report['total_output']:,} / {filter_report['total_input']:,} tracks")
    else:
        print("  Step 2: Already done, re-filtering...")
        all_tracks, filter_report = filter_tracks(all_tracks, config)

    # ── Step 3: Raw features ──
    if not ckpt.is_step_done('step3'):
        print(f"\n{'─'*60}")
        print("  STEP 3: Extracting raw track features")
        print(f"{'─'*60}")
        features_df = extract_raw_features(all_tracks)
        ckpt.save_intermediate('step3', features_df)
        ckpt.mark_step_done('step3')
    else:
        print("  Step 3: Loading from checkpoint...")
        features_df = ckpt.load_intermediate('step3')

    # ── Step 4: Normalise & resample ──
    if not ckpt.is_step_done('step4'):
        print(f"\n{'─'*60}")
        print(f"  STEP 4: Normalising & resampling to {config['resample_length']} points")
        print(f"{'─'*60}")
        norm_df = normalise_and_resample(all_tracks, config['resample_length'], ckpt)
        ckpt.save_intermediate('step4', norm_df)
        ckpt.mark_step_done('step4')
    else:
        print("  Step 4: Loading from checkpoint...")
        norm_df = ckpt.load_intermediate('step4')

    # Merge features + normalised data
    df = features_df.merge(norm_df, on=['norad_id', 'track_id'], how='inner')

    # ── Step 5: TLE features ──
    if not ckpt.is_step_done('step5'):
        print(f"\n{'─'*60}")
        print("  STEP 5: Extracting TLE features")
        print(f"{'─'*60}")
        tle_df = extract_tle_features(features_df, paths['tle_dir'], config, ckpt)
        ckpt.save_intermediate('step5', tle_df)
        ckpt.mark_step_done('step5')
    else:
        print("  Step 5: Loading from checkpoint...")
        tle_df = ckpt.load_intermediate('step5')

    df = df.merge(tle_df, on=['norad_id', 'track_id'], how='left')

    # ── Step 6: Orbit regime ──
    if not ckpt.is_step_done('step6'):
        print(f"\n{'─'*60}")
        print("  STEP 6: Classifying orbit regimes")
        print(f"{'─'*60}")
        df['orbit_regime'] = df.apply(classify_orbit_regime, axis=1)
        # One-hot encode
        for regime in ['LEO', 'MEO', 'GEO', 'HEO', 'SSO', 'MOLNIYA', 'OTHER', 'UNKNOWN']:
            df[f'orbit_{regime}'] = (df['orbit_regime'] == regime).astype(int)
        regime_counts = df['orbit_regime'].value_counts()
        for regime, count in regime_counts.items():
            print(f"  {regime:10s}: {count:>8,}")
        ckpt.mark_step_done('step6')
    else:
        print("  Step 6: Already done.")

    # ── Step 7: Labels ──
    if not ckpt.is_step_done('step7'):
        print(f"\n{'─'*60}")
        print("  STEP 7: Joining DISCOS labels")
        print(f"{'─'*60}")
        df = join_labels(df, paths['discos'])
        has_am = df['am_ratio_avg'].notna().sum()
        has_type = df['object_type'].notna().sum()
        print(f"  With A/M ratio:   {has_am:,} tracks")
        print(f"  With object type: {has_type:,} tracks")
        ckpt.mark_step_done('step7')
    else:
        print("  Step 7: Already done.")
        if 'am_ratio_avg' not in df.columns:
            df = join_labels(df, paths['discos'])

    # ── Step 9: Split (before scaling so we fit scaler on train only) ──
    if not ckpt.is_step_done('step9'):
        print(f"\n{'─'*60}")
        print(f"  STEP 9: Train/val/test split ({'/'.join(str(int(r*100)) for r in config['split_ratios'])})")
        print(f"{'─'*60}")
        df, assignments = split_dataset(df, config['split_ratios'], config['random_seed'])
        for split in ['train', 'val', 'test']:
            n = (df['split'] == split).sum()
            n_sats = df.loc[df['split'] == split, 'norad_id'].nunique()
            print(f"  {split:6s}: {n:>8,} tracks ({n_sats:,} satellites)")
        ckpt.mark_step_done('step9')
    else:
        print("  Step 9: Already done.")
        if 'split' not in df.columns:
            df, assignments = split_dataset(df, config['split_ratios'], config['random_seed'])

    # ── Step 8: Scale features (after split so we fit on train only) ──
    if not ckpt.is_step_done('step8'):
        print(f"\n{'─'*60}")
        print("  STEP 8: Scaling features (fitted on train split)")
        print(f"{'─'*60}")
        df, scaler_params, feature_cols = scale_features(df)
        print(f"  Scaled {len(feature_cols)} feature columns")
        ckpt.mark_step_done('step8')
    else:
        print("  Step 8: Already done.")
        if not any(c.endswith('_scaled') for c in df.columns):
            df, scaler_params, feature_cols = scale_features(df)
        else:
            scaler_params = {}
            feature_cols = []

    # ── Step 10: Save & Report ──
    print(f"\n{'─'*60}")
    print("  STEP 10: Saving output & generating report")
    print(f"{'─'*60}")

    # Save per-split parquet files
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        split_path = output_path / f'{split}.parquet'
        split_df.to_parquet(split_path, index=False)
        print(f"  Saved {split}.parquet: {len(split_df):,} tracks")

    # Save full dataset
    df.to_parquet(output_path / 'full_dataset.parquet', index=False)

    # Save normalisation params
    params = {
        'config': config,
        'feature_scaler': scaler_params,
        'scaled_feature_columns': feature_cols,
        'type_encoding': {'Payload': 0, 'Rocket Body': 1, 'Debris': 2},
        'orbit_regime_categories': ['LEO', 'MEO', 'GEO', 'HEO', 'SSO', 'MOLNIYA', 'OTHER', 'UNKNOWN'],
        'sources_used': list(source_stats.keys()),
        'total_tracks': len(df),
        'created_at': datetime.now().isoformat(),
    }
    params_path = output_path / 'normalisation_params.json'
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2, default=str)

    # Save split assignments
    split_assign = df[['norad_id', 'split']].drop_duplicates()
    split_assign.to_csv(output_path / 'split_assignments.csv', index=False)

    # Save dataset summary
    summary = df.groupby('norad_id').agg(
        source=('source', 'first'),
        num_tracks=('track_id', 'count'),
        split=('split', 'first'),
        has_am=('am_ratio_avg', lambda x: x.notna().any()),
        has_type=('object_type', lambda x: x.notna().any()),
        orbit_regime=('orbit_regime', 'first'),
    ).reset_index()
    summary.to_csv(output_path / 'dataset_summary.csv', index=False)

    # Generate report
    filter_report = ckpt.get_report_data('filter_report') or {}
    generate_report(df, source_stats, filter_report, config, output_path)

    # Offer to clean up intermediate files
    print(f"\n  Intermediate files in {ckpt.intermediate_dir}")
    try:
        ans = input("  Delete intermediate files to free space? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = 'n'
    if ans in ('y', 'yes'):
        import shutil
        shutil.rmtree(ckpt.intermediate_dir)
        print("  Intermediate files deleted.")

    print(f"\n  Output saved to: {output_path}")
    print("  Done!")



#  INTERACTIVE MENU


def print_header():
    _print_header("ML Normalisation Pipeline", __version__,
                  "10-step data preparation for training")


def _auto_detect_paths() -> dict:
    """Auto-detect input paths from standard project directories."""
    paths = {
        'mmt9_dir': None, 'sdlcd_dir': None,
        'tle_dir': None, 'discos': None, 'output': str(TRAINING_DIR),
    }

    # MMT-9
    if (PROC_MMT9 / "lightcurves").exists() and list((PROC_MMT9 / "lightcurves").glob("*.parquet")):
        paths['mmt9_dir'] = str(PROC_MMT9)

    # SDLCD
    if (PROC_SDLCD / "lightcurves").exists() and list((PROC_SDLCD / "lightcurves").glob("*.parquet")):
        paths['sdlcd_dir'] = str(PROC_SDLCD)

    # TLE
    tle_hist = PROC_TLE / "tle_histories"
    if tle_hist.exists() and list(tle_hist.glob("*.parquet")):
        paths['tle_dir'] = str(tle_hist)

    # DISCOS
    for discos_candidate in [
        PROC_DISCOS / "discos_catalogue.csv",
        RAW_DISCOS / "discos_catalogue.csv",
    ]:
        if discos_candidate.exists():
            paths['discos'] = str(discos_candidate)
            break

    return paths


def interactive_menu():
    """Run the interactive menu interface."""
    print_header()

    config = dict(DEFAULT_CONFIG)
    paths = _auto_detect_paths()

    # Show auto-detected paths
    print("  Auto-detected data directories:")
    for key, label in [('mmt9_dir', 'MMT-9'), ('sdlcd_dir', 'SDLCD'),
                        ('tle_dir', 'TLE'), ('discos', 'DISCOS')]:
        val = paths.get(key)
        if val:
            print(f"    {label:<10s}  {val}")
        else:
            print(f"    {label:<10s}  (not found)")
    print(f"    {'Output':<10s}  {paths['output']}")
    print()

    while True:
        print(f"{'─'*60}")

        options = []
        if not paths['mmt9_dir'] and not paths['sdlcd_dir']:
            options.append(("setup", "Setup - Configure input paths"))
        else:
            options.append(("setup", "Setup - Change input paths"))
            options.append(("config", "Configure - Adjust processing parameters"))
            options.append(("preview", "Preview - Show data stats"))
            options.append(("run", "Run - Execute full pipeline"))

        options.append(("quit", "Quit"))

        print("\n  What would you like to do?")
        for i, (_, label) in enumerate(options, 1):
            print(f"    [{i}] {label}")
        print()

        try:
            raw = input("  Enter choice: ").strip()
            idx = int(raw) if raw.isdigit() else -1
            choice = options[idx - 1][0] if 1 <= idx <= len(options) else None
        except (ValueError, IndexError, EOFError, KeyboardInterrupt):
            choice = None

        if choice is None or choice == 'quit':
            print("\n  Goodbye!\n")
            break

        elif choice == 'setup':
            print()
            for key, label, default in [
                ('mmt9_dir', 'MMT-9 output directory', paths.get('mmt9_dir', '')),
                ('sdlcd_dir', 'SDLCD output directory', paths.get('sdlcd_dir', '')),
                ('tle_dir', 'TLE histories directory', paths.get('tle_dir', '')),
                ('discos', 'DISCOS catalogue CSV', paths.get('discos', '')),
                ('output', 'Output directory', paths.get('output', './ml_ready')),
            ]:
                try:
                    current = f" [{default}]" if default else ""
                    val = input(f"  {label}{current}: ").strip().strip('"').strip("'")
                    if val:
                        paths[key] = val
                    elif default:
                        paths[key] = default
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
            print()

        elif choice == 'config':
            print(f"\n  Current configuration:")
            print(f"    Resample length:     {config['resample_length']}")
            print(f"    Gap threshold:       {config['gap_threshold_multiplier']}x median interval")
            print(f"    Min points/track:    {config['min_points_per_track']}")
            print(f"    Min duration:        {config['min_duration_seconds']}s")
            print(f"    TLE window:          ±{config['tle_window_days']} days")
            print(f"    TLE trends:          {'enabled' if config['tle_trends_enabled'] else 'disabled'}")
            print(f"    Split ratios:        {'/'.join(str(int(r*100)) for r in config['split_ratios'])}")
            print(f"    Random seed:         {config['random_seed']}")
            print()

            for key, prompt, conv in [
                ('resample_length', 'Resample length', int),
                ('gap_threshold_multiplier', 'Gap threshold multiplier', int),
                ('min_points_per_track', 'Min points per track', int),
                ('min_duration_seconds', 'Min duration (seconds)', float),
                ('tle_window_days', 'TLE window (days)', int),
                ('random_seed', 'Random seed', int),
            ]:
                try:
                    val = input(f"  {prompt} [{config[key]}]: ").strip()
                    if val:
                        config[key] = conv(val)
                except (ValueError, EOFError, KeyboardInterrupt):
                    pass

            try:
                val = input(f"  TLE trends enabled [{config['tle_trends_enabled']}] (y/n): ").strip().lower()
                if val in ('y', 'yes', 'true'):
                    config['tle_trends_enabled'] = True
                elif val in ('n', 'no', 'false'):
                    config['tle_trends_enabled'] = False
            except (EOFError, KeyboardInterrupt):
                pass

            try:
                val = input(f"  Split ratios [{'/'.join(str(int(r*100)) for r in config['split_ratios'])}] (e.g. 70/15/15): ").strip()
                if val:
                    parts = [int(x) for x in val.split('/')]
                    if len(parts) == 3 and sum(parts) == 100:
                        config['split_ratios'] = [p / 100 for p in parts]
            except (ValueError, EOFError, KeyboardInterrupt):
                pass
            print()

        elif choice == 'preview':
            print("\n  Scanning data directories...")
            for name, key in [('MMT-9', 'mmt9_dir'), ('SDLCD', 'sdlcd_dir')]:
                d = paths.get(key)
                if d:
                    lc_path = Path(d) / 'lightcurves'
                    if lc_path.exists():
                        n = len(list(lc_path.glob('*.parquet'))) or len(list(lc_path.glob('*.csv')))
                        print(f"  {name}: {n:,} satellite files")
                    else:
                        print(f"  {name}: directory not found")

            tle_d = paths.get('tle_dir')
            if tle_d and Path(tle_d).exists():
                n = len(list(Path(tle_d).glob('*.parquet'))) or len(list(Path(tle_d).glob('*.csv')))
                print(f"  TLE: {n:,} files")

            discos_p = paths.get('discos')
            if discos_p and Path(discos_p).exists():
                d = pd.read_csv(discos_p, usecols=['norad_id', 'am_ratio_avg'])
                n_am = d['am_ratio_avg'].notna().sum()
                print(f"  DISCOS: {len(d):,} objects ({n_am:,} with A/M ratio)")
            print()

        elif choice == 'run':
            # Validate paths
            missing = []
            if not paths.get('mmt9_dir') and not paths.get('sdlcd_dir'):
                missing.append("At least one light curve source")
            if not paths.get('tle_dir'):
                missing.append("TLE histories directory")
            if not paths.get('discos'):
                missing.append("DISCOS catalogue")
            if missing:
                print(f"\n  Missing required paths:")
                for m in missing:
                    print(f"    - {m}")
                print("  Use Setup to configure paths.\n")
                continue

            run_pipeline(config, paths)



#  CLI ENTRY POINT


def main():
    parser = argparse.ArgumentParser(
        description='ML Data Normalisation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training_pipeline.py                    # Interactive menu (auto-detects paths)
  python training_pipeline.py --mmt9_dir ./output --sdlcd_dir ./sdlcd_output \\
      --tle_dir ./output/tle_histories --discos ./discos_catalogue.csv --output ./ml_ready
        """)

    parser.add_argument('--mmt9_dir', help='MMT-9 processed directory')
    parser.add_argument('--sdlcd_dir', help='SDLCD processed directory')
    parser.add_argument('--tle_dir', help='TLE histories directory')
    parser.add_argument('--discos', help='DISCOS catalogue CSV')
    parser.add_argument('--output', default=str(TRAINING_DIR), help='Output directory')
    parser.add_argument('--resample_length', type=int, default=1024)
    parser.add_argument('--split', nargs=3, type=int, default=[70, 15, 15],
                        help='Train/val/test split percentages')
    parser.add_argument('--tle_window', type=int, default=30)
    parser.add_argument('--tle_trends', action='store_true', default=True)
    parser.add_argument('--no_tle_trends', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reset', action='store_true', help='Reset checkpoints')

    args = parser.parse_args()

    ensure_dirs()

    try:
        # If no source dirs given, launch interactive menu
        if not args.mmt9_dir and not args.sdlcd_dir:
            interactive_menu()
            return

        config = dict(DEFAULT_CONFIG)
        config['resample_length'] = args.resample_length
        config['split_ratios'] = [r / 100 for r in args.split]
        config['tle_window_days'] = args.tle_window
        config['tle_trends_enabled'] = not args.no_tle_trends
        config['random_seed'] = args.seed

        paths = {
            'mmt9_dir': args.mmt9_dir,
            'sdlcd_dir': args.sdlcd_dir,
            'tle_dir': args.tle_dir,
            'discos': args.discos,
            'output': args.output,
        }

        if args.reset:
            ckpt = CheckpointManager(args.output)
            ckpt.reset()
            print("Checkpoints reset.")

        print_header()
        run_pipeline(config, paths)

    except KeyboardInterrupt:
        print("\n\n  Interrupted. Progress saved via checkpoints.")
        sys.exit(2)


if __name__ == '__main__':
    main()
