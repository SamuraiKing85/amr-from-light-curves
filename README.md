# Machine learning characterisation of space objects from light curves and orbital parameters

Code for the paper *Fusing Light Curves and Orbital Parameters for Machine
Learning-Based Characterisation of Space Objects: A Reproducible Study on
Public Data*.

The pipeline estimates area-to-mass ratio (A/m) and classifies object type
(payload, rocket body, debris) by combining public optical light curves with
orbital parameters derived from two-line element histories, validated against
ESA DISCOS ground truth.

**Headline results.** Orbital geometry dominates A/m estimation
(R² 0.6220); light curves alone contribute little (R² 0.2189); fusing the two
gives no statistically significant gain (R² 0.6235). Gradient boosting on the
same orbital features outperforms every neural variant (R² 0.6619). The B*
drag term, treated operationally as the primary A/m proxy, ranks ninth of
eighteen features and reaches only R² 0.0261 alone.

---

## Pipeline

Four stages, run in order:

| Stage | What it does |
|---|---|
| `1_acquire/` | Downloads light curves (MMT-9, SDLCD), TLE histories (Space-Track) and ground truth (ESA DISCOS) |
| `2_process/` | Parses and normalises each source into a common format |
| `3_prepare/` | Gap-splits, filters, resamples to 1,024 points, merges modalities, builds the satellite-level split |
| `4_train/` | Trains the fusion network and ablations, runs tree baselines, evaluates, and produces explainability outputs |

Shared utilities are in `common/`, paths and credentials handling in `config/`.

`explorer/` contains Satellite Explorer, a desktop application for merging and
visually inspecting the multi-source satellite catalogue. It is a companion
tool rather than part of the training pipeline, and is not required to
reproduce the results.

## Data

None of the data is redistributed here. All of it is public, and you will need
accounts for two of the four sources:

| Source | What it provides | Access |
|---|---|---|
| Mini-MegaTORTORA (MMT-9) | Optical light curves, predominantly LEO | http://mmt9.ru/satellites |
| Space Debris Light Curve Database (SDLCD) | Light curves for GEO, GTO, Molniya and GNSS orbits | Astronomical and Geophysical Observatory, Modra |
| Space-Track.org | Two-line element histories | Free account required |
| ESA DISCOS | Area-to-mass ratio and object type ground truth | Free API token required |

Copy `config/credentials.example.json` to `config/credentials.json` and fill in
your Space-Track username and password and your DISCOS token. That file is
gitignored and must never be committed.

**On exact reproduction.** MMT-9 and SDLCD grow over time. The published
figures use a snapshot taken during the study, giving 636,036 sub-tracks from
16,351 satellites. A later download will produce more tracks, so the numbers
will differ slightly. `dataset_analysis_report.txt` records the exact counts
from the snapshot used, including the retention arithmetic: 346,835 raw tracks
plus 674,158 gap splits gives 1,020,993 sub-tracks, of which 636,036 survive
quality filtering (62.3%).

## Setup

Python 3.10 or later.

```bash
git clone https://github.com/<your-username>/amr-from-light-curves.git
cd amr-from-light-curves
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python 1_acquire/mmt9_downloader.py
python 1_acquire/sdlcd_downloader.py
python 1_acquire/tle_downloader.py
python 1_acquire/discos_downloader.py

python 2_process/mmt9_processor.py
python 2_process/sdlcd_processor.py
python 2_process/tle_manager.py

python 3_prepare/training_pipeline.py

python 4_train/train.py
python 4_train/evaluate.py
python 4_train/baselines.py
python 4_train/analysis.py --all
python 4_train/explain.py --model fusion
```

Most scripts accept `--help`. `analysis.py` and `explain.py` support `--all`
to run across every trained model.

Acquisition and preparation are the long stages: the full download is several
hundred gigabytes of raw photometry before filtering.

## Figures and explainability

`4_train/analysis.py` produces the scatter, residual, confusion-matrix,
ablation and per-class figures from saved evaluation results, without
retraining:

```bash
python 4_train/analysis.py --all
```

`4_train/explain.py` produces the permutation-importance and Grad-CAM outputs:

```bash
python 4_train/explain.py --model fusion            # both
python 4_train/explain.py --model fusion --gradcam  # Grad-CAM only
```

Note that `--permutation` is the expensive step: eighteen features by five
repeats is ninety full passes over the test set, which takes hours on CPU. The
results are written to `permutation_importance.json` before the chart is
drawn, so the figure can be redrawn without recomputing.

**On the figures in the paper.** These scripts produce the analysis figures at
their default sizes. The versions printed in the paper were re-rendered at the
publisher's text width so that axis lettering remained legible at print size;
the underlying values are identical and come from the same saved results.

## Model

A late-fusion network. The convolutional branch takes the normalised magnitude
sequence stacked with a binary mask channel marking real versus interpolated
positions, through three 1D convolutional layers (32/64/128) to a 64-dimensional
embedding. The orbital branch takes eighteen TLE-derived features through a
two-layer MLP to a matching embedding. The two are concatenated before two
shared layers and two heads, one regressing log A/m and one classifying object
type.

Trained with AdamW, learning rate 3×10⁻⁴, weight decay 5×10⁻⁴, cosine
annealing, batch size 64, seed 42. The split is stratified at satellite level,
not track level, so all tracks from a given object fall in one partition.

## Citation

```bibtex
@inproceedings{pinkney2026fusing,
  author    = {Pinkney, Jay and Moorton, Zoe},
  title     = {Fusing Light Curves and Orbital Parameters for Machine
               Learning-Based Characterisation of Space Objects: A
               Reproducible Study on Public Data},
  booktitle = {Proceedings of ICGS3-26},
  year      = {2026},
  publisher = {Springer},
}
```

## Licence

MIT. See `LICENSE`.

## Acknowledgements

This work relies on public archives. Thanks to the teams behind the
Mini-MegaTORTORA system at the Special Astrophysical Observatory of the
Russian Academy of Sciences, the Space Debris Light Curve Database at the
Astronomical and Geophysical Observatory in Modra, Comenius University in
Bratislava, the ESA Space Debris Office for DISCOS, and Space-Track.org.
