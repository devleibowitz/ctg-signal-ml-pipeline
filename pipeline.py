"""End-to-end CTG signal processing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from config import PipelineConfig, load_config
from features import extract_all_features
from label_tabular_ctg_features import run_labeling
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
# Tensor index mapping & lookup
# ---------------------------------------------------------------------------

TENSOR_INDEX_MAP_FILENAME = "tensor_index_map.csv"
TENSOR_ARTIFACT_FILENAME = "tensor.pt"


def build_tensor_index_map(
    patient_ids: list[str],
    processed: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a per-row lookup table aligned with tensor sample indices.

    Row ``i`` corresponds to ``tensor[i]`` and ``patient_ids[i]``.  Metadata is
    taken from the processed DataFrames (same source as feature extraction).

    Args:
        patient_ids: Ordered patient/session IDs from :func:`format_tensor`.
        processed: Mapping of patient_id → processed DataFrame.

    Returns:
        DataFrame with columns ``tensor_index``, ``patient_id``, ``file_name``,
        and ``birth_time``.

    Raises:
        KeyError: If a ``patient_ids`` entry is missing from ``processed``.
    """
    rows: list[dict[str, Any]] = []

    for i, pid in enumerate(patient_ids):
        if pid not in processed:
            raise KeyError(f"patient_id {pid!r} missing from processed data")
        df = processed[pid]
        source_files = list(df.attrs.get("source_files", []))
        file_name = ";".join(source_files) if source_files else pid
        rows.append({
            "tensor_index": i,
            "patient_id":     pid,
            "file_name":      file_name,
            "birth_time":     df["Monitor_Date"].max(),
        })

    return pd.DataFrame(rows)


def tensor_index_to_meta(
    index: int,
    patient_ids: list[str],
    index_map: pd.DataFrame,
) -> dict[str, Any]:
    """Return metadata for a single tensor row.

    Args:
        index: Tensor sample index (0-based).
        patient_ids: Ordered ID list from :func:`format_tensor`.
        index_map: Lookup table from :func:`build_tensor_index_map`.

    Returns:
        Dict with ``tensor_index``, ``patient_id``, ``file_name``, and
        ``birth_time``.

    Raises:
        IndexError: If ``index`` is out of range.
    """
    if index < 0 or index >= len(patient_ids):
        raise IndexError(
            f"tensor index {index} out of range for {len(patient_ids)} samples"
        )
    row = index_map.iloc[index]
    if int(row["tensor_index"]) != index:
        row = index_map.loc[index_map["tensor_index"] == index].iloc[0]
    return row.to_dict()


def patient_id_to_index(patient_id: str, patient_ids: list[str]) -> int:
    """Return the first tensor index for an exact ``patient_id`` match.

    Args:
        patient_id: Patient/session identifier.
        patient_ids: Ordered ID list from :func:`format_tensor`.

    Returns:
        Matching tensor index.

    Raises:
        KeyError: If ``patient_id`` is not present.
    """
    try:
        return patient_ids.index(patient_id)
    except ValueError as exc:
        raise KeyError(f"No tensor index for patient_id={patient_id!r}") from exc


def patient_id_to_indices(patient_id: str, patient_ids: list[str]) -> list[int]:
    """Return all tensor indices matching ``patient_id`` exactly.

    For a base patient ID shared across split sessions (``pid_0``, ``pid_1``),
    call this once per session ID or filter :func:`build_tensor_index_map` by
    ``patient_id`` prefix.

    Args:
        patient_id: Patient/session identifier.
        patient_ids: Ordered ID list from :func:`format_tensor`.

    Returns:
        List of matching indices (empty when no match).
    """
    return [i for i, pid in enumerate(patient_ids) if pid == patient_id]


def save_tensor_artifacts(
    tensor: torch.Tensor,
    patient_ids: list[str],
    index_map: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist the tensor bundle and a CSV index map for offline lookup.

    Args:
        tensor: Output tensor of shape ``(N, T, 2)``.
        patient_ids: Ordered ID list aligned with tensor rows.
        index_map: Lookup table from :func:`build_tensor_index_map`.
        output_dir: Directory for ``tensor.pt`` and ``tensor_index_map.csv``.

    Returns:
        ``(tensor_path, index_map_path)``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_path = output_dir / TENSOR_ARTIFACT_FILENAME
    map_path = output_dir / TENSOR_INDEX_MAP_FILENAME

    birth_times = pd.to_datetime(index_map["birth_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    torch.save(
        {
            "tensor":      tensor,
            "patient_ids": patient_ids,
            "file_names":  index_map["file_name"].tolist(),
            "birth_times": birth_times.tolist(),
        },
        tensor_path,
    )
    index_map.to_csv(map_path, index=False)

    logger.info("Tensor saved → %s", tensor_path)
    logger.info("Tensor index map saved → %s", map_path)
    return tensor_path, map_path


def load_tensor_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Load the tensor bundle and index map written by :func:`save_tensor_artifacts`.

    Args:
        output_dir: Directory containing ``tensor.pt`` and ``tensor_index_map.csv``.

    Returns:
        Dict with keys ``tensor``, ``patient_ids``, ``file_names``, ``birth_times``,
        and ``index_map``.
    """
    output_dir = Path(output_dir)
    data = torch.load(output_dir / TENSOR_ARTIFACT_FILENAME, weights_only=False)
    index_map = pd.read_csv(output_dir / TENSOR_INDEX_MAP_FILENAME)
    index_map["birth_time"] = pd.to_datetime(index_map["birth_time"])
    return {
        "tensor":      data["tensor"],
        "patient_ids": data["patient_ids"],
        "file_names":  data.get("file_names"),
        "birth_times": data.get("birth_times"),
        "index_map":   index_map,
    }


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
    label_features: bool = True,
) -> tuple[torch.Tensor, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Execute the complete CTG signal processing pipeline.

    Steps:
        1. Load and merge patient CSV files (loader)
        2. Preprocess each patient signal (preprocess)
        3. Format into a 3D tensor (format_tensor)
        4. Extract clinical features (features)
        5. Apply clinical labeling rules (label_tabular_ctg_features)
        6. Generate visualizations (visualize)
        7. Produce and save summary statistics (build_summary)

    Args:
        config_path: Path to the YAML configuration file.
        visualize_ids: Patient IDs to generate before/after signal plots for.
            Pass ``"all"`` as the first element to plot every patient.
        plot_durations: If True, save a duration-distribution histogram.
        save_tensor: If True, save ``tensor.pt`` and ``tensor_index_map.csv`` to
            config.output_dir.
        label_features: If True, run the clinical labeling engine on extracted features.

    Returns:
        (tensor, patient_ids, summary_df, per_patient_df, labeled_df)
        labeled_df is None when label_features is False.

    Raises:
        RuntimeError: If no data survives loading or preprocessing.
    """
    config = load_config(config_path)
    plots_dir = config.output_dir / "plots"

    # 1. Load
    raw_patients, n_files_merged, _load_summary = load_all_patients(config)
    if not raw_patients:
        raise RuntimeError("No patient data loaded — check input_source and input paths in config.yaml")

    input_dir = config.input_dir
    if config.input_source.lower() == "csv" and input_dir is not None:
        n_files_total = sum(1 for _ in input_dir.glob("*.csv"))
    elif config.input_source.lower() == "parquet" and input_dir is not None:
        n_files_total = sum(1 for _ in input_dir.rglob("*.parquet"))
    else:
        n_files_total = 0

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

    # 5. Clinical labeling
    labeled_df: pd.DataFrame | None = None
    if label_features:
        labeled_path = config.output_dir / "labeled_clinical_features.csv"
        labeled_df = run_labeling(features_path, labeled_path)

    # 6. Visualize
    ids_to_plot: list[str] = []
    if visualize_ids:
        ids_to_plot = list(processed.keys()) if visualize_ids[0] == "all" else visualize_ids

    for pid in ids_to_plot:
        if pid not in raw_patients or pid not in processed:
            logger.warning("Cannot plot %s — missing from raw or processed data", pid)
            continue
        plot_patient_signals(pid, raw_patients[pid], processed[pid], output_dir=plots_dir)

    # 7. Summary
    summary_df, per_patient_df = build_summary(
        all_stats, processed, config, n_files_total, n_files_merged,
    )

    if plot_durations:
        plot_duration_distribution(per_patient_df, output_dir=plots_dir)

    index_map = build_tensor_index_map(patient_ids, processed)

    if save_tensor:
        save_tensor_artifacts(tensor, patient_ids, index_map, config.output_dir)

    summary_df.to_csv(config.output_dir / "summary.csv", index=False)
    per_patient_df.to_csv(config.output_dir / "per_patient_stats.csv", index=False)

    logger.info("\n=== Pipeline Summary ===\n%s", summary_df.T.to_string())
    print(f"\n=== Pipeline Summary ===\n{summary_df.T.to_string()}")

    return tensor, patient_ids, summary_df, per_patient_df, labeled_df


if __name__ == "__main__":
    run_pipeline()
