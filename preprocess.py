"""Signal preprocessing: trim, outlier removal, resampling, interpolation, and smoothing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class PreprocessStats:
    """Per-patient statistics collected during preprocessing."""

    patient_id: str
    n_hr1_outliers: int = 0
    n_toco_outliers: int = 0
    n_interpolated: int = 0
    duration_pre_trim_minutes: float = 0.0
    duration_post_trim_minutes: float = 0.0
    duration_post_resample_minutes: float = 0.0
    missingness_pct: float = 0.0


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _trim_leading_trailing_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where both HR1 and TOCO are zero at the signal edges.

    Finds the first and last row where at least one signal is non-zero,
    then slices to that range.

    Args:
        df: DataFrame with HR1 and TOCO columns.

    Returns:
        Trimmed DataFrame (same index labels preserved).
    """
    valid = (df["HR1"] != 0) | (df["TOCO"] != 0)
    indices = valid[valid].index

    if indices.empty:
        logger.warning("All rows are zero — returning empty DataFrame")
        return df.iloc[0:0]

    trimmed = df.loc[indices[0] : indices[-1]].copy()
    n_removed = len(df) - len(trimmed)
    if n_removed:
        logger.debug("Trimmed %d leading/trailing zero rows", n_removed)
    return trimmed


def _replace_outliers(
    df: pd.DataFrame,
    col: str,
    lower: float,
    upper: float,
) -> tuple[pd.DataFrame, int]:
    """Replace physiologically impossible values with NaN.

    Args:
        df: Input DataFrame.
        col: Column to inspect.
        lower: Inclusive minimum valid value.
        upper: Inclusive maximum valid value.

    Returns:
        (modified copy of df, number of outliers replaced)
    """
    mask = (df[col] < lower) | (df[col] > upper)
    n = int(mask.sum())
    if n:
        df = df.copy()
        df.loc[mask, col] = np.nan
        logger.debug("%s: flagged %d outliers as NaN", col, n)
    return df, n


def _resample(df: pd.DataFrame, freq_seconds: int) -> pd.DataFrame:
    """Resample to a uniform time grid by averaging within each bin.

    Gaps in the original data become NaN rows — they will be handled by
    the interpolation step that follows.

    Args:
        df: DataFrame with Monitor_Date column.
        freq_seconds: Target sampling interval in seconds.

    Returns:
        Resampled DataFrame with Monitor_Date as a regular column.
    """
    df = df.set_index("Monitor_Date")
    df = df.resample(f"{freq_seconds}s").mean()
    df = df.reset_index()
    return df


def _interpolate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Time-aware linear interpolation of NaN values in HR1 and TOCO.

    Uses the DatetimeIndex for accurate gap-proportional fills.  ``limit_direction``
    is set to ``"both"`` so edge NaNs (leading/trailing) are also filled.

    Args:
        df: DataFrame with Monitor_Date column and possible NaNs.

    Returns:
        (filled DataFrame, number of points interpolated)
    """
    n_before = int(df[["HR1", "TOCO"]].isna().sum().sum())
    df = df.set_index("Monitor_Date")
    for col in ("HR1", "TOCO"):
        df[col] = df[col].interpolate(method="time", limit_direction="both")
    df = df.reset_index()
    n_after = int(df[["HR1", "TOCO"]].isna().sum().sum())
    n_filled = n_before - n_after
    if n_filled:
        logger.debug("Interpolated %d NaN values", n_filled)
    return df, n_filled


def _smooth(df: pd.DataFrame, window: int = 11, polyorder: int = 2) -> pd.DataFrame:
    """Apply a light Savitzky-Golay filter to HR1 and TOCO.

    Preserves signal shape while reducing high-frequency noise.  Skips a
    column if it is too short for the chosen window.

    Args:
        df: DataFrame with HR1 and TOCO columns (no NaNs expected).
        window: Filter window length in samples (must be odd, > polyorder).
        polyorder: Polynomial order of the fitting function.

    Returns:
        DataFrame with smoothed signals.
    """
    if window % 2 == 0:
        window += 1  # Savitzky-Golay requires odd window

    df = df.copy()
    for col in ("HR1", "TOCO"):
        if df[col].notna().sum() >= window:
            df[col] = savgol_filter(df[col].to_numpy(), window_length=window, polyorder=polyorder)
        else:
            logger.debug("%s: too short for smoothing window=%d — skipped", col, window)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_patient(
    df: pd.DataFrame,
    config: PipelineConfig,
    patient_id: str = "",
) -> tuple[pd.DataFrame, PreprocessStats]:
    """Run the full preprocessing pipeline on one patient's raw DataFrame.

    Pipeline order:
        1. Trim leading/trailing zeros
        2. Replace physiological outliers with NaN
        3. Resample to uniform grid (config.resample_freq_seconds)
        4. Interpolate all NaNs (outliers + resampling gaps)
        5. Light Savitzky-Golay smoothing

    Args:
        df: Raw patient DataFrame containing Monitor_Date, HR1, TOCO.
        config: Pipeline configuration with bounds and resampling parameters.
        patient_id: Identifier used in log messages and returned stats.

    Returns:
        (processed DataFrame, PreprocessStats)
    """
    stats = PreprocessStats(patient_id=patient_id)

    stats.duration_pre_trim_minutes = (
        (df["Monitor_Date"].max() - df["Monitor_Date"].min()).total_seconds() / 60
    )
    total_pre = len(df) * 2  # HR1 + TOCO
    stats.missingness_pct = df[["HR1", "TOCO"]].isna().sum().sum() / max(total_pre, 1) * 100

    # 1. Trim
    df = _trim_leading_trailing_zeros(df)
    if df.empty:
        logger.error("Patient %s: empty after trimming", patient_id)
        return df, stats

    stats.duration_post_trim_minutes = (
        (df["Monitor_Date"].max() - df["Monitor_Date"].min()).total_seconds() / 60
    )

    # 2. Outlier removal
    df, stats.n_hr1_outliers = _replace_outliers(df, "HR1", config.hr1_min, config.hr1_max)
    df, stats.n_toco_outliers = _replace_outliers(df, "TOCO", config.toco_min, config.toco_max)

    # 3. Resample
    df = _resample(df, config.resample_freq_seconds)
    stats.duration_post_resample_minutes = (
        (df["Monitor_Date"].max() - df["Monitor_Date"].min()).total_seconds() / 60
    )

    # 4. Interpolate
    df, stats.n_interpolated = _interpolate(df)

    # 5. Smooth
    df = _smooth(df)

    logger.info(
        "Patient %-10s | raw %.1f min → trimmed %.1f min → resampled %.1f min "
        "| outliers HR1=%d TOCO=%d | interpolated=%d",
        patient_id,
        stats.duration_pre_trim_minutes,
        stats.duration_post_trim_minutes,
        stats.duration_post_resample_minutes,
        stats.n_hr1_outliers,
        stats.n_toco_outliers,
        stats.n_interpolated,
    )
    return df, stats


def preprocess_all(
    patients: dict[str, pd.DataFrame],
    config: PipelineConfig,
) -> tuple[dict[str, pd.DataFrame], list[PreprocessStats]]:
    """Preprocess every patient DataFrame from loader.load_all_patients().

    Args:
        patients: Mapping of patient_id → raw DataFrame.
        config: Pipeline configuration.

    Returns:
        (mapping of patient_id → processed DataFrame, list of per-patient stats)
    """
    processed: dict[str, pd.DataFrame] = {}
    all_stats: list[PreprocessStats] = []

    for pid, df in patients.items():
        result, stats = preprocess_patient(df, config, patient_id=pid)
        all_stats.append(stats)
        if not result.empty:
            processed[pid] = result

    logger.info(
        "Preprocessing complete: %d / %d patients retained",
        len(processed), len(patients),
    )
    return processed, all_stats
