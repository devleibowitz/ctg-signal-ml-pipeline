"""Step 7: Clinical Labeling Engine.

Reads ``extracted_clinical_features.csv``, applies standard CTG clinical
guidelines to assign a strict binary label, and writes
``labeled_clinical_features.csv``.

Label semantics
---------------
    1  =  Non-Reassuring (Pathological)
    0  =  Reassuring (Normal)

Pipeline
--------
    7A  Data-quality exclusions  (low signal validity  → drop)
    7B  Pathological rules       (any rule fires       → label = 1)
    7C  Reassuring criteria      (all criteria met     → label = 0)
    7D  Indeterminate exclusion  (neither 0 nor 1      → drop)

Usage
-----
    python label_tabular_ctg_features.py
    python label_tabular_ctg_features.py --input path/to/features.csv --output path/to/labeled.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Column names ──────────────────────────────────────────────────────────
_VALID_FHR   = "valid_fhr_ratio"
_VALID_TOCO  = "valid_toco_ratio"
_BASELINE    = "fhr_baseline_bpm"
_VARIABILITY = "fhr_baseline_variability_bpm"
_PROLONGED   = "fhr_prolonged_deceleration_count"
_DECEL_TYPE  = "combined_deceleration_type"

# ── Thresholds (from CLAUDE.md Step 7) ───────────────────────────────────
_QUALITY_FHR_MIN  = 0.60
_QUALITY_TOCO_MIN = 0.60
_BASELINE_LOW     = 110.0
_BASELINE_HIGH    = 160.0
_VAR_LOW          =   5.0
_VAR_HIGH         =  25.0
_PROLONGED_THRESHOLD = 1

_REQUIRED_COLS = {_VALID_FHR, _VALID_TOCO, _BASELINE, _VARIABILITY, _PROLONGED, _DECEL_TYPE}


# ─────────────────────────────────────────────────────────────────────────
# Core labeling logic
# ─────────────────────────────────────────────────────────────────────────

def _check_required_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is absent."""
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")


def apply_quality_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Step 7A — Drop rows with insufficient signal validity.

    Args:
        df: Full features DataFrame.

    Returns:
        (retained DataFrame, exclusion_counts dict with per-reason tallies)
    """
    low_fhr  = df[_VALID_FHR]  < _QUALITY_FHR_MIN
    low_toco = df[_VALID_TOCO] < _QUALITY_TOCO_MIN

    counts = {
        "low_fhr_validity":  int(low_fhr.sum()),
        "low_toco_validity": int(low_toco.sum()),
        "total_excluded":    int((low_fhr | low_toco).sum()),
    }

    retained = df[~(low_fhr | low_toco)].copy()
    return retained, counts


def compute_label_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[str, pd.Series]]:
    """Steps 7B & 7C — Compute boolean masks for label = 1, label = 0.

    Args:
        df: Quality-filtered features DataFrame.

    Returns:
        (label_1_mask, label_0_mask, rule_masks)
        where rule_masks maps each named criterion to its boolean Series.
    """
    has_late_decel  = df[_DECEL_TYPE] == "Late"
    low_var         = df[_VARIABILITY] < _VAR_LOW
    has_prolonged   = df[_PROLONGED]  > _PROLONGED_THRESHOLD

    # ── Step 7B: Pathological rules (any → label = 1) ────────────────────
    rule_masks: dict[str, pd.Series] = {
        "hypoxia_signature":  low_var & has_late_decel,
        "severe_bradycardia": (df[_BASELINE] < _BASELINE_LOW)  & low_var,
        "severe_tachycardia": (df[_BASELINE] > _BASELINE_HIGH) & low_var,
        "sustained_distress": has_prolonged,
    }
    label_1 = rule_masks["hypoxia_signature"]  \
            | rule_masks["severe_bradycardia"] \
            | rule_masks["severe_tachycardia"] \
            | rule_masks["sustained_distress"]

    # ── Step 7C: Reassuring criteria (all → label = 0) ───────────────────
    label_0 = (
        (df[_BASELINE]    >= _BASELINE_LOW)  &
        (df[_BASELINE]    <= _BASELINE_HIGH) &
        (df[_VARIABILITY] >= _VAR_LOW)       &
        (df[_VARIABILITY] <= _VAR_HIGH)      &
        ~has_late_decel                      &
        ~has_prolonged
    )

    # Label 1 takes precedence if a row somehow satisfies both masks
    label_0 = label_0 & ~label_1

    return label_1, label_0, rule_masks


def assign_labels(
    df: pd.DataFrame,
    label_1: pd.Series,
    label_0: pd.Series,
) -> tuple[pd.DataFrame, int]:
    """Step 7D — Assign labels and drop indeterminate rows.

    Args:
        df: Quality-filtered DataFrame.
        label_1: Boolean Series for pathological cases.
        label_0: Boolean Series for reassuring cases.

    Returns:
        (labeled DataFrame with 'label' column, n_indeterminate_dropped)
    """
    n_indeterminate = int((~label_1 & ~label_0).sum())

    labeled = df[label_1 | label_0].copy()
    labeled["criteria_non_reassuring"] = 0
    labeled.loc[label_1[label_1 | label_0], "criteria_non_reassuring"] = 1

    return labeled, n_indeterminate


def print_summary(
    n_total:          int,
    exclusion_counts: dict[str, int],
    rule_masks:       dict[str, pd.Series],
    label_1:          pd.Series,
    label_0:          pd.Series,
    n_indeterminate:  int,
    output_path:      Path,
) -> None:
    """Print a formatted labeling summary to stdout and logger.

    Args:
        n_total:          Total rows in the input CSV.
        exclusion_counts: Counts from apply_quality_exclusions().
        rule_masks:       Per-rule boolean Series from compute_label_masks().
        label_1:          Final label-1 mask (quality-filtered rows only).
        label_0:          Final label-0 mask (quality-filtered rows only).
        n_indeterminate:  Rows dropped as indeterminate.
        output_path:      Path where labeled CSV was written.
    """
    n_path1 = int(label_1.sum())
    n_path0 = int(label_0.sum())
    n_labeled = n_path1 + n_path0

    sep = "─" * 58

    lines = [
        "",
        "╔══════════════════════════════════════════════════════╗",
        "║          Clinical Labeling Engine — Summary          ║",
        "╚══════════════════════════════════════════════════════╝",
        "",
        f"  Input records            : {n_total:>6}",
        sep,
        "  Step 7A — Quality Exclusions",
        f"    Low FHR validity        : {exclusion_counts['low_fhr_validity']:>6}  (valid_fhr_ratio < {_QUALITY_FHR_MIN:.0%})",
        f"    Low TOCO validity       : {exclusion_counts['low_toco_validity']:>6}  (valid_toco_ratio < {_QUALITY_TOCO_MIN:.0%})",
        f"    Total excluded (7A)     : {exclusion_counts['total_excluded']:>6}",
        sep,
        f"  Step 7D — Indeterminate   : {n_indeterminate:>6}  (dropped)",
        sep,
        f"  Final labeled records     : {n_labeled:>6}",
        "",
        "  Step 7B — Label = 1  (Non-Reassuring / Pathological)",
        f"    Total                   : {n_path1:>6}",
        f"    · Hypoxia signature     : {int(rule_masks['hypoxia_signature'].sum()):>6}  (variability<5 & late decel)",
        f"    · Severe bradycardia    : {int(rule_masks['severe_bradycardia'].sum()):>6}  (baseline<110 & variability<5)",
        f"    · Severe tachycardia    : {int(rule_masks['severe_tachycardia'].sum()):>6}  (baseline>160 & variability<5)",
        f"    · Sustained distress    : {int(rule_masks['sustained_distress'].sum()):>6}  (prolonged_decel > 0)",
        "    (rules are non-exclusive; counts may overlap)",
        "",
        "  Step 7C — Label = 0  (Reassuring / Normal)",
        f"    Total                   : {n_path0:>6}",
        sep,
        f"  Output → {output_path}",
        "",
    ]

    report = "\n".join(lines)
    print(report)
    logger.info("Labeling complete: %d labeled (%d pathological, %d reassuring)",
                n_labeled, n_path1, n_path0)


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

def run_labeling(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Execute the full labeling pipeline.

    Args:
        input_path:  Path to extracted_clinical_features.csv.
        output_path: Destination path for labeled_clinical_features.csv.

    Returns:
        Labeled DataFrame.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If required columns are absent from the input.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading features from %s", input_path)
    df = pd.read_csv(input_path)
    n_total = len(df)
    logger.info("Loaded %d records with %d columns", n_total, len(df.columns))

    _check_required_columns(df)

    # 7A — quality exclusions
    df_filtered, excl_counts = apply_quality_exclusions(df)
    logger.info(
        "7A: dropped %d records (low_fhr=%d  low_toco=%d)  →  %d retained",
        excl_counts["total_excluded"],
        excl_counts["low_fhr_validity"],
        excl_counts["low_toco_validity"],
        len(df_filtered),
    )

    if df_filtered.empty:
        logger.error("No records survived quality exclusions — aborting.")
        sys.exit(1)

    # 7B & 7C — label masks
    label_1, label_0, rule_masks = compute_label_masks(df_filtered)

    # 7D — assign + drop indeterminate
    labeled_df, n_indeterminate = assign_labels(df_filtered, label_1, label_0)
    logger.info(
        "7D: dropped %d indeterminate records  →  %d labeled (%d path / %d normal)",
        n_indeterminate, len(labeled_df),
        int(label_1.sum()), int(label_0.sum()),
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(output_path, index=False)
    logger.info("Labeled dataset saved → %s", output_path)

    print_summary(
        n_total, excl_counts, rule_masks,
        label_1, label_0, n_indeterminate,
        output_path,
    )

    return labeled_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CTG Clinical Labeling Engine (Step 7)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("output/extracted_clinical_features.csv"),
        help="Path to extracted_clinical_features.csv",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output/labeled_clinical_features.csv"),
        help="Destination path for labeled_clinical_features.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_labeling(args.input, args.output)
