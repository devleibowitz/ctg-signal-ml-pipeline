"""Clinical feature extraction from preprocessed CTG signals.

All metrics are computed exclusively on valid signal segments — artificial
flatlines (sensor dropouts) are identified and masked out before any
calculation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import correlate, find_peaks, welch

from config import PipelineConfig

logger = logging.getLogger(__name__)

# ── Physiological thresholds ──────────────────────────────────────────────
_BRADYCARDIA_THRESH      = 110.0   # bpm
_TACHYCARDIA_THRESH      = 160.0   # bpm
_ACCEL_AMP_THRESH        = 15.0    # bpm above baseline
_DECEL_AMP_THRESH        = 15.0    # bpm below baseline
_EVENT_MIN_DUR_SEC       = 15.0    # minimum duration for accel/decel
_PROLONGED_DUR_SEC       = 120.0   # seconds — prolonged deceleration
_LATE_DECEL_LAG_SEC      = 30.0    # lag threshold for "Late" classification
_TOCO_CONTRACTION_MIN_RISE  = 25.0 # tocometry units above resting tone
_TOCO_CONTRACTION_PROMINENCE = 20.0


# ═══════════════════════════════════════════════════════════════════════════
# Validity masking
# ═══════════════════════════════════════════════════════════════════════════

def _validity_mask(signal: np.ndarray, fs: float, min_flat_sec: float = 15.0) -> np.ndarray:
    """Boolean mask where True = valid, False = artificial flatline / dropout.

    Flatline detection: segments where |second discrete derivative| < 1e-6
    for >= min_flat_sec consecutive seconds — the device interpolated a
    straight line across a missing-data gap.

    Args:
        signal: 1-D float array.
        fs: Sampling frequency in Hz.
        min_flat_sec: Minimum consecutive flat seconds to mark invalid.

    Returns:
        Boolean array of length len(signal).
    """
    if len(signal) < 3:
        return np.ones(len(signal), dtype=bool)

    d2 = np.abs(np.diff(signal, n=2))
    d2 = np.pad(d2, (1, 1), mode="edge")   # restore original length

    near_zero   = d2 < 1e-6
    min_flat    = max(2, int(min_flat_sec * fs))
    invalid     = np.zeros(len(signal), dtype=bool)

    i = 0
    while i < len(near_zero):
        if near_zero[i]:
            j = i
            while j < len(near_zero) and near_zero[j]:
                j += 1
            if j - i >= min_flat:
                invalid[i:j] = True
            i = j
        else:
            i += 1

    return ~invalid


# ═══════════════════════════════════════════════════════════════════════════
# FHR features
# ═══════════════════════════════════════════════════════════════════════════

def _fhr_baseline(fhr: np.ndarray) -> float:
    """Median of the central 90 % of valid FHR values (trims extreme excursions)."""
    if len(fhr) == 0:
        return np.nan
    lo, hi = np.percentile(fhr, [5, 95])
    central = fhr[(fhr >= lo) & (fhr <= hi)]
    return float(np.median(central)) if len(central) > 0 else float(np.median(fhr))


def _baseline_variability(fhr: np.ndarray) -> float:
    """Std dev of absolute deviations from baseline — bandwidth of oscillation."""
    if len(fhr) < 2:
        return np.nan
    return float(np.std(np.abs(fhr - _fhr_baseline(fhr))))


def _stv(fhr: np.ndarray) -> float:
    """Short-term variability: mean absolute beat-to-beat difference (bpm)."""
    if len(fhr) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(fhr))))


def _ltv(fhr: np.ndarray, fs: float, epoch_sec: float = 60.0) -> float:
    """Long-term variability: mean peak-to-trough range within 1-min epochs (bpm)."""
    epoch_n = max(1, int(epoch_sec * fs))
    if len(fhr) < epoch_n:
        return float(np.ptp(fhr)) if len(fhr) > 1 else np.nan
    ranges = [float(np.ptp(fhr[s : s + epoch_n]))
              for s in range(0, len(fhr) - epoch_n + 1, epoch_n)]
    return float(np.mean(ranges)) if ranges else np.nan


def _event_counts(fhr: np.ndarray, baseline: float, fs: float) -> tuple[int, int, int]:
    """Count accelerations, decelerations, and prolonged decelerations.

    Uses run-length analysis on the deviation signal.

    Args:
        fhr: Valid FHR samples.
        baseline: Estimated baseline bpm.
        fs: Sampling frequency.

    Returns:
        (accel_count, decel_count, prolonged_decel_count)
    """
    if len(fhr) == 0 or np.isnan(baseline):
        return 0, 0, 0

    dev         = fhr - baseline
    min_samp    = max(1, int(_EVENT_MIN_DUR_SEC * fs))
    prolong_samp = max(1, int(_PROLONGED_DUR_SEC * fs))

    def _runs(arr: np.ndarray, thresh: float) -> list[tuple[int, int]]:
        above = arr > thresh
        runs, in_run, start = [], False, 0
        for i, a in enumerate(above):
            if a and not in_run:
                in_run, start = True, i
            elif not a and in_run:
                runs.append((start, i)); in_run = False
        if in_run:
            runs.append((start, len(above)))
        return [(s, e) for s, e in runs if e - s >= min_samp]

    accel_runs = _runs(dev,  _ACCEL_AMP_THRESH)
    decel_runs = _runs(-dev, _DECEL_AMP_THRESH)
    prolonged  = sum(1 for s, e in decel_runs if e - s >= prolong_samp)
    return len(accel_runs), len(decel_runs), prolonged


def _time_in_range_min(fhr: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Minutes with FHR in [lo, hi]."""
    if len(fhr) == 0:
        return 0.0
    return float(np.sum((fhr >= lo) & (fhr <= hi))) / fs / 60.0


def _sample_entropy(signal: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Sample Entropy (SampEn) — vectorised, subsampled for performance.

    Args:
        signal: 1-D array.
        m: Embedding dimension.
        r_factor: Tolerance as fraction of signal std.

    Returns:
        SampEn scalar, or NaN if computation is infeasible.
    """
    max_pts = 300
    if len(signal) > max_pts:
        idx = np.linspace(0, len(signal) - 1, max_pts, dtype=int)
        signal = signal[idx]

    N = len(signal)
    if N < m + 2:
        return np.nan
    r = r_factor * float(np.std(signal))
    if r < 1e-10:
        return np.nan

    def _count(m_val: int) -> int:
        tmpl = np.array([signal[i : i + m_val] for i in range(N - m_val)])
        # Chebyshev distance between all pairs; exclude self-matches
        dist = np.abs(tmpl[:, None, :] - tmpl[None, :, :]).max(axis=2)
        np.fill_diagonal(dist, r + 1.0)
        return int(np.sum(dist < r))

    B = _count(m)
    A = _count(m + 1)
    if B == 0:
        return np.nan
    return float(-np.log(A / B)) if A > 0 else np.inf


def _psd_peak_freq(signal: np.ndarray, fs: float) -> float:
    """Dominant frequency (Hz) from Welch PSD in the FHR variability band 0.03–0.5 Hz."""
    if len(signal) < 64:
        return np.nan
    freqs, power = welch(signal, fs=fs, nperseg=min(512, len(signal)))
    band = (freqs >= 0.03) & (freqs <= 0.5)
    if not np.any(band):
        return float(freqs[np.argmax(power)])
    return float(freqs[band][np.argmax(power[band])])


# ═══════════════════════════════════════════════════════════════════════════
# TOCO features
# ═══════════════════════════════════════════════════════════════════════════

def _toco_baseline(toco: np.ndarray) -> float:
    """Resting uterine tone estimated as the 10th percentile of valid TOCO."""
    return float(np.percentile(toco, 10)) if len(toco) > 0 else np.nan


def _detect_contractions(
    toco: np.ndarray, fs: float, resting_tone: float
) -> tuple[np.ndarray, dict]:
    """Detect contraction peaks in the TOCO signal via find_peaks.

    A contraction is a peak ≥ 25 units above resting tone, separated from
    the previous peak by at least 1 minute, with a width ≥ 20 seconds.
    """
    if len(toco) == 0 or np.isnan(resting_tone):
        return np.array([], dtype=int), {}

    peaks, props = find_peaks(
        toco,
        height=resting_tone + _TOCO_CONTRACTION_MIN_RISE,
        distance=max(1, int(60.0 * fs)),
        prominence=_TOCO_CONTRACTION_PROMINENCE,
        width=max(1, int(20.0 * fs)),
        rel_height=0.5,
    )
    return peaks, props


def _contraction_frequency(n: int, duration_min: float) -> float:
    """Contractions per 10 minutes."""
    return (n / duration_min * 10.0) if duration_min > 0 else np.nan


def _uterine_work(toco: np.ndarray, resting_tone: float, fs: float) -> float:
    """Area under TOCO curve above resting tone (bpm·s), trapezoidal integration."""
    if len(toco) == 0 or np.isnan(resting_tone):
        return np.nan
    return float(np.trapezoid(np.maximum(toco - resting_tone, 0.0), dx=1.0 / fs))


# ═══════════════════════════════════════════════════════════════════════════
# Combined FHR + TOCO features
# ═══════════════════════════════════════════════════════════════════════════

def _fhr_toco_lag(fhr: np.ndarray, toco: np.ndarray, fs: float) -> float:
    """Phase lag (seconds) between TOCO peaks and FHR dips via cross-correlation.

    Positive → FHR responds after TOCO peak (late deceleration pattern).
    Negative / zero → early or coincident.
    """
    if len(fhr) < 32 or len(toco) < 32:
        return np.nan

    def _z(x: np.ndarray) -> np.ndarray:
        s = float(np.std(x))
        return (x - float(np.mean(x))) / (s if s > 1e-9 else 1.0)

    fhr_inv = np.max(fhr) - fhr          # invert so decels → peaks
    corr    = correlate(_z(fhr_inv), _z(toco), mode="full")
    lags    = np.arange(-(len(fhr) - 1), len(toco))
    return float(lags[np.argmax(corr)]) / fs


def _classify_deceleration(lag_sec: float, decel_count: int) -> str:
    if decel_count == 0:
        return "None"
    if np.isnan(lag_sec):
        return "Unknown"
    if lag_sec <= 0:
        return "Early"
    if lag_sec > _LATE_DECEL_LAG_SEC:
        return "Late"
    return "Variable"


def _recovery_times(
    fhr: np.ndarray,
    toco_peaks: np.ndarray,
    toco_props: dict,
    baseline: float,
    fs: float,
) -> np.ndarray:
    """Seconds for FHR to recover to within 5 bpm of baseline after each deceleration."""
    if len(toco_peaks) == 0 or len(fhr) == 0 or np.isnan(baseline):
        return np.array([])

    threshold = baseline - 5.0
    times = []
    for peak in toco_peaks:
        search_start = max(0, int(peak - 60 * fs))
        search_end   = min(len(fhr), int(peak + 120 * fs))
        window = fhr[search_start:search_end]
        if len(window) == 0:
            continue
        dip_global = search_start + int(np.argmin(window))
        post = fhr[dip_global : min(len(fhr), dip_global + int(300 * fs))]
        recovered = np.where(post >= threshold)[0]
        if len(recovered) > 0:
            times.append(float(recovered[0]) / fs)

    return np.array(times)


def _fhr_response_area(
    fhr: np.ndarray,
    toco_peaks: np.ndarray,
    baseline: float,
    fs: float,
    window_sec: float = 60.0,
) -> float:
    """Total area (bpm·s) of FHR below baseline inside contraction windows."""
    if len(toco_peaks) == 0 or len(fhr) == 0 or np.isnan(baseline):
        return 0.0
    half = int(window_sec * fs / 2)
    total = 0.0
    for peak in toco_peaks:
        seg = fhr[max(0, int(peak) - half) : min(len(fhr), int(peak) + half)]
        total += float(np.trapezoid(np.maximum(baseline - seg, 0.0), dx=1.0 / fs))
    return total


def _cross_correlation(fhr: np.ndarray, toco: np.ndarray) -> float:
    """Pearson correlation between inverted FHR and TOCO (aligned valid segments)."""
    if len(fhr) < 2 or len(toco) < 2:
        return np.nan
    fhr_inv = np.max(fhr) - fhr
    if np.std(fhr_inv) < 1e-9 or np.std(toco) < 1e-9:
        return np.nan
    return float(np.corrcoef(fhr_inv, toco)[0, 1])


# ═══════════════════════════════════════════════════════════════════════════
# Per-patient entry point
# ═══════════════════════════════════════════════════════════════════════════

def extract_patient_features(
    patient_id: str,
    processed_df: pd.DataFrame,
    config: PipelineConfig,
    source_files: list[str] | None = None,
) -> dict[str, Any]:
    """Extract all clinical features for one patient.

    Args:
        patient_id: Patient identifier string.
        processed_df: Post-preprocessing DataFrame (Monitor_Date, HR1, TOCO).
        config: Pipeline config (provides resample_freq_seconds).
        source_files: Original CSV filenames; joined with ";" for file_name column.

    Returns:
        Flat dict matching the required CSV column order.
    """
    fs = 1.0 / config.resample_freq_seconds

    fhr_arr  = processed_df["HR1"].to_numpy(dtype=float)
    toco_arr = processed_df["TOCO"].to_numpy(dtype=float)

    birth_time = processed_df["Monitor_Date"].max()
    file_name  = ";".join(source_files) if source_files else patient_id
    duration_min = len(processed_df) / fs / 60.0

    # ── 1. Validity masks ──────────────────────────────────────────────────
    fhr_mask  = _validity_mask(fhr_arr,  fs)
    toco_mask = _validity_mask(toco_arr, fs)

    fhr_valid  = fhr_arr[fhr_mask]
    toco_valid = toco_arr[toco_mask]

    valid_fhr_ratio  = float(np.mean(fhr_mask))
    valid_toco_ratio = float(np.mean(toco_mask))

    # ── 2. FHR features ───────────────────────────────────────────────────
    baseline = _fhr_baseline(fhr_valid)
    accel, decel, prolonged = _event_counts(fhr_valid, baseline, fs)

    fhr_feats: dict[str, Any] = {
        "fhr_baseline_bpm":                 baseline,
        "fhr_baseline_variability_bpm":     _baseline_variability(fhr_valid),
        "fhr_stv_bpm":                      _stv(fhr_valid),
        "fhr_ltv_bpm":                      _ltv(fhr_valid, fs),
        "fhr_acceleration_count":           accel,
        "fhr_deceleration_count":           decel,
        "fhr_prolonged_deceleration_count": prolonged,
        "fhr_time_bradycardia_min":         _time_in_range_min(fhr_valid, fs, 0.0, _BRADYCARDIA_THRESH),
        "fhr_time_tachycardia_min":         _time_in_range_min(fhr_valid, fs, _TACHYCARDIA_THRESH, 9999.0),
        "fhr_sample_entropy":               _sample_entropy(fhr_valid),
        "fhr_psd_peak_freq_hz":             _psd_peak_freq(fhr_valid, fs),
    }

    # ── 3. TOCO features ──────────────────────────────────────────────────
    resting_tone = _toco_baseline(toco_valid)
    toco_peaks, toco_props = _detect_contractions(toco_valid, fs, resting_tone)

    durations   = toco_props.get("widths", np.array([])) / fs
    relaxations = _relaxation_times_from_props(toco_peaks, toco_props, fs)

    toco_feats: dict[str, Any] = {
        "toco_baseline_resting_tone":           resting_tone,
        "toco_contraction_frequency_per_10min": _contraction_frequency(len(toco_peaks), duration_min),
        "toco_contraction_duration_mean_sec":   float(np.mean(durations))              if len(durations)   > 0 else np.nan,
        "toco_peak_intensity_mean":             float(np.mean(toco_valid[toco_peaks])) if len(toco_peaks)  > 0 else np.nan,
        "toco_uterine_work_auc":               _uterine_work(toco_valid, resting_tone, fs),
        "toco_relaxation_time_mean_sec":        float(np.mean(relaxations))            if len(relaxations) > 0 else np.nan,
    }

    # ── 4. Combined features — use co-valid mask so indices align ─────────
    min_len      = min(len(fhr_arr), len(toco_arr))
    both_valid   = fhr_mask[:min_len] & toco_mask[:min_len]
    fhr_aligned  = fhr_arr[:min_len][both_valid]
    toco_aligned = toco_arr[:min_len][both_valid]

    # Detect contractions on aligned TOCO for FHR-response metrics
    toco_peaks_a, toco_props_a = _detect_contractions(toco_aligned, fs, resting_tone)
    recovery = _recovery_times(fhr_aligned, toco_peaks_a, toco_props_a, baseline, fs)
    lag_sec  = _fhr_toco_lag(fhr_aligned, toco_aligned, fs)

    combined_feats: dict[str, Any] = {
        "combined_lag_time_sec":            lag_sec,
        "combined_deceleration_type":       _classify_deceleration(lag_sec, decel),
        "combined_recovery_time_mean_sec":  float(np.mean(recovery)) if len(recovery) > 0 else np.nan,
        "combined_fhr_response_area":       _fhr_response_area(fhr_aligned, toco_peaks_a, baseline, fs),
        "combined_cross_correlation":       _cross_correlation(fhr_aligned, toco_aligned),
    }

    logger.info(
        "Patient %-22s | valid FHR %.0f%% TOCO %.0f%% | "
        "baseline %.1f stv %.2f | accels %d decels %d contractions %d",
        patient_id[:22],
        valid_fhr_ratio * 100, valid_toco_ratio * 100,
        baseline if not np.isnan(baseline) else -1,
        fhr_feats["fhr_stv_bpm"] if not np.isnan(fhr_feats["fhr_stv_bpm"]) else -1,
        accel, decel, len(toco_peaks),
    )

    return {
        "file_name":        file_name,
        "patient_id":       patient_id,
        "birth_time":       birth_time,
        "valid_fhr_ratio":  round(valid_fhr_ratio,  4),
        "valid_toco_ratio": round(valid_toco_ratio, 4),
        **fhr_feats,
        **toco_feats,
        **combined_feats,
    }


def _relaxation_times_from_props(
    peaks: np.ndarray, props: dict, fs: float
) -> np.ndarray:
    """Inter-contraction relaxation gaps (seconds): from one contraction end to the next start."""
    if len(peaks) < 2:
        return np.array([])
    rights = props.get("right_ips", peaks.astype(float))
    lefts  = props.get("left_ips",  peaks.astype(float))
    gaps   = (lefts[1:] - rights[:-1]) / fs
    return gaps[gaps > 0]


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline-level entry point
# ═══════════════════════════════════════════════════════════════════════════

def extract_all_features(
    processed: dict[str, pd.DataFrame],
    config: PipelineConfig,
) -> pd.DataFrame:
    """Extract clinical features for every patient and return a consolidated DataFrame.

    Args:
        processed: Mapping of patient_id → processed DataFrame (from preprocess_all).
        config: Pipeline configuration.

    Returns:
        DataFrame with one row per patient in the required column order.
    """
    rows: list[dict[str, Any]] = []

    for pid, df in processed.items():
        source_files = list(df.attrs.get("source_files", []))
        try:
            rows.append(extract_patient_features(pid, df, config, source_files))
        except Exception as exc:
            logger.error("Feature extraction failed for %s: %s", pid, exc, exc_info=True)

    if not rows:
        logger.warning("No features extracted")
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    n_feat = len(result.columns) - 5   # subtract the 5 metadata cols
    logger.info(
        "Feature extraction complete: %d patients × %d clinical features",
        len(result), n_feat,
    )
    return result
