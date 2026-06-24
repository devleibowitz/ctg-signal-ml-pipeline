"""End-to-end CTG signal processing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import PipelineConfig, load_config
from features import extract_all_features
from loader import load_all_patients
from preprocess import PreprocessStats, preprocess_all
from visualize import plot_duration_distribution, plot_patient_signals

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tensor formatting
# ---------------------------------------------------------------------------

def format_tensor(
    processed: dict[str, pd.DataFrame],
    config: PipelineConfig,
) -> tuple[torch.Tensor, list[str]]:
    """Pack processed patient DataFrames into a (N, T, 2) float32 tensor.

    Each signal is shaped to exactly T = config.min_duration_steps samples:
    - Longer signals are **truncated from the front** (last T steps kept).
    - Shorter signals are **zero-padded at the beginning**.

    Args:
        processed: Mapping of patient_id → processed DataFrame.
        config: Pipeline configuration.

    Returns:
        (tensor of shape [num_samples, T, 2], ordered list of patient IDs)
    """
    T = config.min_duration_steps
    samples: list[np.ndarray] = []
    patient_ids: list[str] = []

    for pid, df in processed.items():
        arr = df[["HR1", "TOCO"]].to_numpy(dtype=np.float32)  # (t, 2)

        if len(arr) >= T:
            arr = arr[-T:]                                         # keep last T
        else:
            pad = np.zeros((T - len(arr), 2), dtype=np.float32)
            arr = np.vstack([pad, arr])                            # pad at beginning

        samples.append(arr)
        patient_ids.append(pid)

    tensor = torch.tensor(np.stack(samples), dtype=torch.float32)  # (N, T, 2)
    logger.info("Output tensor shape: %s  (samples × timesteps × features)", tuple(tensor.shape))
    return tensor, patient_ids


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def build_summary(
    all_stats: list[PreprocessStats],
    processed: dict[str, pd.DataFrame],
    config: PipelineConfig,
    n_files_total: int,
    n_files_merged: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a pipeline-level summary and a per-patient detail DataFrame.

    Args:
        all_stats: Per-patient PreprocessStats from preprocess_all().
        processed: Processed patient DataFrames (used for post-truncation durations).
        config: Pipeline configuration.
        n_files_total: Total CSV files discovered in input_dir.
        n_files_merged: Files that were merged into a sibling record.

    Returns:
        (summary_df with one aggregate row, per_patient_df with one row per patient)
    """
    T_minutes = config.min_duration_steps * config.resample_freq_seconds / 60

    per_rows = []
    for s in all_stats:
        in_processed = s.patient_id in processed
        post_trunc = min(s.duration_post_resample_minutes, T_minutes) if in_processed else None
        per_rows.append({
            "patient_id":                  s.patient_id,
            "retained":                    in_processed,
            "duration_pre_trim_min":       round(s.duration_pre_trim_minutes, 2),
            "duration_post_trim_min":      round(s.duration_post_trim_minutes, 2),
            "duration_post_resample_min":  round(s.duration_post_resample_minutes, 2),
            "duration_post_truncation_min": round(post_trunc, 2) if post_trunc is not None else None,
            "hr1_outliers":                s.n_hr1_outliers,
            "toco_outliers":               s.n_toco_outliers,
            "n_interpolated":              s.n_interpolated,
            "missingness_pct":             round(s.missingness_pct, 3),
        })

    per_patient_df = pd.DataFrame(per_rows)

    pre_durs  = per_patient_df.loc[per_patient_df["retained"], "duration_post_trim_min"].dropna()
    post_durs = per_patient_df.loc[per_patient_df["retained"], "duration_post_truncation_min"].dropna()

    def _stats(series: pd.Series, label: str) -> dict:
        return {
            f"{label}_min":    round(float(series.min()), 2)    if not series.empty else None,
            f"{label}_max":    round(float(series.max()), 2)    if not series.empty else None,
            f"{label}_median": round(float(series.median()), 2) if not series.empty else None,
        }

    summary_df = pd.DataFrame([{
        "files_discovered":        n_files_total,
        "files_merged":            n_files_merged,
        "patients_loaded":         len(all_stats),
        "patients_retained":       int(per_patient_df["retained"].sum()),
        **_stats(pre_durs,  "pre_truncation_duration_min"),
        **_stats(post_durs, "post_truncation_duration_min"),
        "total_hr1_outliers":      int(per_patient_df["hr1_outliers"].sum()),
        "total_toco_outliers":     int(per_patient_df["toco_outliers"].sum()),
        "total_interpolated":      int(per_patient_df["n_interpolated"].sum()),
        "mean_missingness_pct":    round(float(per_patient_df["missingness_pct"].mean()), 3),
    }])

    return summary_df, per_patient_df


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: str | Path = "config.yaml",
    visualize_ids: list[str] | None = None,
    plot_durations: bool = True,
    save_tensor: bool = True,
) -> tuple[torch.Tensor, list[str], pd.DataFrame, pd.DataFrame]:
    """Execute the complete CTG signal processing pipeline.

    Steps:
        1. Load and merge patient CSV files (loader)
        2. Preprocess each patient signal (preprocess)
        3. Format into a 3D tensor (format_tensor)
        4. Generate visualizations (visualize)
        5. Produce and save summary statistics (build_summary)

    Args:
        config_path: Path to the YAML configuration file.
        visualize_ids: Patient IDs to generate before/after signal plots for.
            Pass ``"all"`` as the first element to plot every patient.
        plot_durations: If True, save a duration-distribution histogram.
        save_tensor: If True, save tensor.pt to config.output_dir.

    Returns:
        (tensor, patient_ids, summary_df, per_patient_df)

    Raises:
        RuntimeError: If no data survives loading or preprocessing.
    """
    config = load_config(config_path)
    plots_dir = config.output_dir / "plots"

    # 1. Load
    raw_patients, n_files_merged = load_all_patients(config)
    if not raw_patients:
        raise RuntimeError("No patient data loaded — check input_dir in config.yaml")

    n_files_total = sum(1 for _ in config.input_dir.glob("*.csv")) if config.input_dir.exists() else 0

    # 2. Preprocess
    processed, all_stats = preprocess_all(raw_patients, config)
    if not processed:
        raise RuntimeError("No patients remained after preprocessing")

    # 3. Tensor
    tensor, patient_ids = format_tensor(processed, config)

    # 4. Feature extraction
    features_df = extract_all_features(processed, config)
    features_path = config.output_dir / "extracted_clinical_features.csv"
    features_df.to_csv(features_path, index=False)
    logger.info("Clinical features saved → %s  (%d rows × %d cols)",
                features_path, len(features_df), len(features_df.columns))

    # 5. Visualize
    ids_to_plot: list[str] = []
    if visualize_ids:
        ids_to_plot = list(processed.keys()) if visualize_ids[0] == "all" else visualize_ids

    for pid in ids_to_plot:
        if pid not in raw_patients or pid not in processed:
            logger.warning("Cannot plot %s — missing from raw or processed data", pid)
            continue
        plot_patient_signals(pid, raw_patients[pid], processed[pid], output_dir=plots_dir)

    # 6. Summary
    summary_df, per_patient_df = build_summary(
        all_stats, processed, config, n_files_total, n_files_merged,
    )

    if plot_durations:
        plot_duration_distribution(per_patient_df, output_dir=plots_dir)

    if save_tensor:
        tensor_path = config.output_dir / "tensor.pt"
        torch.save({"tensor": tensor, "patient_ids": patient_ids}, tensor_path)
        logger.info("Tensor saved → %s", tensor_path)

    summary_df.to_csv(config.output_dir / "summary.csv", index=False)
    per_patient_df.to_csv(config.output_dir / "per_patient_stats.csv", index=False)

    logger.info("\n=== Pipeline Summary ===\n%s", summary_df.T.to_string())

    return tensor, patient_ids, summary_df, per_patient_df


if __name__ == "__main__":
    run_pipeline()
