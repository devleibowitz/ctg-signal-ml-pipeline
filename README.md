# Machine Learning for Time-Series Signal Processing & Prediction

A preprocessing and feature-engineering pipeline for **cardiotocography (CTG)** fetal monitor data. Raw continuous HR and uterine activity signals are cleaned, aligned, and formatted into uniform tensors for sequence models, alongside a tabular set of clinically interpretable features for risk analysis and prediction.

## Overview

The goal is to transform heterogeneous CSV monitor exports into two model-ready artifacts:

1. **3D tensors** — `(num_samples × time_steps × 2_features)` with FHR (`HR1`) and TOCO channels at a fixed sequence length.
2. **Feature table** — one row per patient sequence with validity metrics and engineered clinical descriptors.

The pipeline is modular: `loader.py` → `preprocess.py` → `pipeline.py` (tensor formatting) → `features.py` → `label_tabular_ctg_features.py`, configured via `config.yaml`.

## Tech Stack

| Component | Libraries |
|-----------|-----------|
| Language | Python 3.12+ |
| Data & signals | Pandas, NumPy, SciPy |
| Tensors | PyTorch |
| Visualization | Matplotlib, Seaborn, Plotly |
| Configuration | YAML (`config.yaml`) |

Install dependencies with Poetry (`poetry install`) or pip (`pip install -r requirements.txt`).

## Quick Start

1. Point `input_dir` in `config.yaml` at your directory of encrypted monitor CSV files.
2. Run the pipeline:

```bash
python pipeline.py
```

Outputs are written to `output_dir` (default: `output/`):

| File | Description |
|------|-------------|
| `tensor.pt` | PyTorch tensor + patient ID list |
| `extracted_clinical_features.csv` | Per-sequence clinical features |
| `labeled_clinical_features.csv` | Quality-filtered features with binary CTG labels |
| `summary.csv` | Pipeline-level statistics |
| `per_patient_stats.csv` | Per-patient preprocessing metrics |
| `plots/` | Before/after signal plots and duration distributions |

## Pipeline Architecture

```
Raw CSV files
    │
    ▼
┌─────────────────┐
│  Load & Merge   │  Extract patient IDs; merge files within 2 h gaps
└────────┬────────┘
         ▼
┌─────────────────┐
│ Signal Process  │  Trim, outlier removal, resample, smooth
└────────┬────────┘
         ├──────────────────────┐
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Tensor Format   │    │ Feature Extract │  Validity masking → clinical metrics
└────────┬────────┘    └────────┬────────┘
         │                      ▼
         │             ┌─────────────────┐
         │             │ Clinical Label  │  CTG inclusion / exclusion rules
         │             └────────┬────────┘
         ▼                      ▼
    tensor.pt          labeled_clinical_features.csv
```

---

## Signal Processing

Signal processing turns raw, irregular monitor exports into clean, uniformly sampled time series suitable for deep learning and downstream statistics.

### 1. Data loading & patient merging

- **Input:** CSV files containing `Monitor_Date`, `HR1` (fetal heart rate), and `TOCO` (uterine contractions). `HR2` and `HRM` are dropped.
- **Patient identification:** A `pat_[ID]` is extracted from each filename.
- **Merging:** Multiple files for the same patient whose start/end timestamps fall within a **2-hour window** are concatenated vertically into a single continuous record.
- **Validation:** Time gaps and overlaps between merged segments are logged so downstream analysis is not silently corrupted.

### 2. Edge trimming

Leading and trailing rows where **both** `HR1` and `TOCO` are zero are stripped. These zeros typically represent idle monitor time before recording starts or after it ends, not physiological signal.

### 3. Outlier handling

Values outside physiologically plausible ranges are flagged and replaced with `NaN`:

| Signal | Valid range (default) |
|--------|----------------------|
| FHR (`HR1`) | 50–240 bpm |
| TOCO | 0–100 units |

Missing values are then **interpolated** so the time series remains continuous for resampling and modeling.

### 4. Uniform resampling

Monitor files may have irregular timestamps. All signals are resampled to a **fixed sampling interval** (default: 1 second) so every patient shares the same temporal grid.

### 5. Light smoothing

A Savitzky–Golay filter is applied as a lightweight denoising step. Smoothing is conservative — the priority is preserving clinically meaningful FHR variability and contraction morphology.

### 6. Sequence formatting (tensor generation)

Processed signals are shaped to a fixed length defined by `min_duration_minutes` (default: **90 minutes** = 5,400 samples at 1 Hz):

| Condition | Action |
|-----------|--------|
| Signal **longer** than `min_duration` | Keep only the **last** `min_duration` (most recent window before delivery) |
| Signal **shorter** than `min_duration` | **Zero-pad at the beginning** |

The result is a float32 tensor of shape `(N, T, 2)` where channel 0 = FHR and channel 1 = TOCO.

---

## Feature Extraction

Feature extraction produces a tabular representation complementary to the raw tensor — useful for classical ML models, interpretability, and longitudinal risk analysis.

### Validity masking (critical first step)

Before any metric is computed, the pipeline identifies **artifact segments** where the monitor interpolated a straight line across missing data (sensor dropout). The method:

1. Compute the discrete **second derivative** of the signal.
2. Mark segments where `|d²/dt²| < 1e-6` for **≥ 15 consecutive seconds** as invalid flatlines.
3. Build a boolean **validity mask** for both FHR and TOCO.

All rolling statistics, counts, and averages are computed **only on valid samples**. Each output row includes `valid_fhr_ratio` and `valid_toco_ratio` so downstream models can weight or filter low-quality recordings.

### FHR features

| Feature | Description |
|---------|-------------|
| Baseline FHR | Median of the central 90% of valid FHR values |
| Baseline variability | Std dev of deviations from baseline |
| STV | Short-term variability — mean absolute beat-to-beat difference |
| LTV | Long-term variability — std dev of 1-minute medians |
| Acceleration / deceleration counts | Episodes ≥ 15 s, ≥ 15 bpm from baseline |
| Prolonged decelerations | Decelerations lasting ≥ 120 s |
| Time in bradycardia / tachycardia | Fraction of valid time below 110 or above 160 bpm |
| ApEn / SampEn | Approximate and sample entropy — signal complexity |
| PSD | Power spectral density via Welch's method |

### TOCO features

| Feature | Description |
|---------|-------------|
| Baseline resting tone | Median of valid TOCO between contractions |
| Contraction frequency | Peaks per hour above resting tone |
| Contraction duration | Mean width of detected contraction peaks |
| Peak intensity | Mean maximum amplitude per contraction |
| Uterine work (AUC) | Area under the curve above resting tone |
| Relaxation time | Mean time to return to baseline after a contraction |

### Combined FHR–TOCO features

| Feature | Description |
|---------|-------------|
| Lag time (phase shift) | Cross-correlation lag between TOCO and FHR |
| Deceleration classification | Early / late / variable deceleration typing |
| Recovery time | Mean time for FHR to return to baseline after a decel |
| FHR response area | Integrated FHR deviation during decelerations |
| Cross-correlation coefficient | Peak normalized cross-correlation between channels |

### Output schema

Each row in `extracted_clinical_features.csv` represents one processed patient sequence:

1. `file_name`
2. `patient_id`
3. `birth_time` — maximum original timestamp before truncation/padding
4. `valid_fhr_ratio`
5. `valid_toco_ratio`
6. All engineered clinical feature columns

---

## Clinical Labeling (Non-Reassuring FHR)

After feature extraction, `label_tabular_ctg_features.py` applies standard CTG clinical guidelines to assign a strict binary target for supervised learning. This step runs automatically as part of `pipeline.py` and can also be run standalone:

```bash
python label_tabular_ctg_features.py
```

### Label semantics

| `criteria_non_reassuring` | Meaning |
|---------------------------|---------|
| `1` | **Non-reassuring (pathological)** — at least one pathological rule fired |
| `0` | **Reassuring (normal)** — all reassuring criteria met |

Rows that fail quality checks or fall into the clinical **indeterminate** zone are **excluded** from the labeled dataset.

### Exclusion criteria

Records are dropped if they meet **any** of the following:

| Step | Criterion | Threshold |
|------|-----------|-----------|
| **7A — Low FHR validity** | `valid_fhr_ratio` | < 60% |
| **7A — Low TOCO validity** | `valid_toco_ratio` | < 60% |
| **7D — Indeterminate** | Neither pathological nor reassuring rules apply | — |

### Non-reassuring inclusion (`criteria_non_reassuring = 1`)

A record is labeled **non-reassuring** if **any** of the following pathological rules fire (rules are evaluated with OR logic; multiple rules may overlap on the same record):

| Rule | Criteria |
|------|----------|
| **Hypoxia signature** | Baseline variability < 5 bpm **and** late deceleration present |
| **Severe bradycardia** | Baseline FHR < 110 bpm **and** baseline variability < 5 bpm |
| **Severe tachycardia** | Baseline FHR > 160 bpm **and** baseline variability < 5 bpm |
| **Sustained distress** | Prolonged deceleration count > 0 |

Pathological labels take precedence over reassuring criteria when both could apply.

### Reassuring inclusion (`criteria_non_reassuring = 0`)

A record is labeled **reassuring** only if **all** of the following are true:

- Baseline FHR between 110 and 160 bpm (inclusive)
- Baseline variability between 5 and 25 bpm (inclusive)
- No late decelerations
- No prolonged decelerations

### Output

`labeled_clinical_features.csv` contains all columns from `extracted_clinical_features.csv` plus `criteria_non_reassuring`, restricted to records that passed quality filtering and received a definitive label.

---

## Configuration

Key parameters in `config.yaml`:

```yaml
min_duration_minutes: 90      # fixed sequence length
resample_freq_seconds: 1      # target sampling rate
merge_gap_hours: 2.0          # patient file merge window
hr1_min / hr1_max             # FHR physiological bounds
toco_min / toco_max           # TOCO physiological bounds
```

## Project Structure

```
├── config.py          # Config dataclass and YAML loader
├── config.yaml        # Pipeline parameters
├── loader.py          # CSV loading and patient merging
├── preprocess.py      # Signal cleaning and resampling
├── features.py        # Validity masking and clinical features
├── label_tabular_ctg_features.py  # CTG clinical labeling (Step 7)
├── pipeline.py        # End-to-end orchestration
├── visualize.py       # Before/after plots and reporting
└── explore_patient.ipynb  # Interactive exploration notebook
```

## Visualization & Reporting

- **Per-patient plots:** Side-by-side comparison of raw vs. preprocessed FHR and TOCO.
- **Duration distribution:** Histogram of signal lengths before and after truncation.
- **Summary statistics:** Total files processed, merge count, duration min/max/median, and missingness/interpolation rates.
