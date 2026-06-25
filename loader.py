"""Data loading and patient-file merging for CTG monitor parquet files."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from config import PipelineConfig

logger = logging.getLogger(__name__)

TARGET_COLS = ["Monitor_Date", "HR1", "TOCO"]

# Matches "pat_12345", "pat-12345", "PAT_12345", etc.
_PATIENT_ID_RE = re.compile(r"pat[_\-]?(\w+)", re.IGNORECASE)
_FILE_DATETIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2}\.\d{3})")


def extract_patient_id(filename: str) -> str | None:
    """Parse a patient ID from a CTG filename using the pat_[ID] convention.

    Args:
        filename: Bare filename or full path string.

    Returns:
        The extracted ID string, or None if no match is found.
    """
    stem = Path(filename).stem
    match = _PATIENT_ID_RE.search(stem)
    if match:
        return match.group(1)
    logger.warning("No patient ID found in filename: %s", filename)
    return None


def _read_parquet(path: Path) -> pd.DataFrame | None:
    """Read one CTG parquet, keeping only Monitor_Date / HR1 / TOCO.

    Args:
        path: Path to a parquet file.

    Returns:
        Sorted DataFrame with TARGET_COLS, or None on failure / empty file.
    """
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.error("Cannot read %s: %s", path, exc)
        return None

    # Strip any accidental whitespace in column names before anything else
    df.columns = df.columns.str.strip()

    # Handle parquet files produced from CSV-as-string conversion:
    # - Two columns where first row is "HRM", "TOCO"
    # - No explicit Monitor_Date column, so reconstruct from filename timestamp.
    if "Monitor_Date" not in df.columns and df.shape[1] == 2:
        reconstructed = _reconstruct_from_string_parquet(df, path)
        if reconstructed is None:
            logger.warning("Skipping %s — could not reconstruct target columns", path.name)
            return None
        df = reconstructed

    if "Monitor_Date" in df.columns:
        df["Monitor_Date"] = pd.to_datetime(df["Monitor_Date"], errors="coerce")

    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        logger.warning("Skipping %s — missing columns: %s", path.name, missing)
        return None

    if df.empty:
        logger.debug("Skipping empty file %s", path.name)
        return None

    df = df[TARGET_COLS].copy()
    rows_before_coercion = len(df)
    df["HR1"] = pd.to_numeric(df["HR1"], errors="coerce")
    df["TOCO"] = pd.to_numeric(df["TOCO"], errors="coerce")
    df.dropna(subset=TARGET_COLS, inplace=True)
    dropped_rows = rows_before_coercion - len(df)
    logger.info(
        "Numeric coercion audit [%s]: dropped %d/%d rows",
        path.name,
        dropped_rows,
        rows_before_coercion,
    )
    if df.empty:
        logger.debug("Skipping empty file %s after type conversion", path.name)
        return None
    df["HR1"] = df["HR1"].astype(int)
    df["TOCO"] = df["TOCO"].astype(int)
    df.sort_values("Monitor_Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.attrs["numeric_coercion_dropped_rows"] = dropped_rows
    df.attrs["rows_before_numeric_coercion"] = rows_before_coercion
    return df


def _parse_start_datetime_from_filename(path: Path) -> pd.Timestamp | None:
    """Parse start timestamp from parquet filename suffix."""
    match = _FILE_DATETIME_RE.search(path.stem)
    if not match:
        return None
    dt_str = match.group(1).replace("_", ":")
    return pd.to_datetime(dt_str, errors="coerce")


def _reconstruct_from_string_parquet(df_raw: pd.DataFrame, path: Path) -> pd.DataFrame | None:
    """Rebuild Monitor_Date/HR1/TOCO from 2-column string parquet exports."""
    start_dt = _parse_start_datetime_from_filename(path)
    if start_dt is None or pd.isna(start_dt):
        logger.warning("Could not parse start datetime from filename: %s", path.name)
        return None

    df = df_raw.copy()
    col0, col1 = df.columns[:2]
    # Converted files include a first signal-header row ("HRM", "TOCO").
    if not df.empty:
        first_left = str(df.iloc[0, 0]).strip().upper()
        first_right = str(df.iloc[0, 1]).strip().upper()
        if first_left in {"HRM", "HR1"} and first_right == "TOCO":
            df = df.iloc[1:].copy()

    hr1 = pd.to_numeric(df[col0], errors="coerce")
    toco = pd.to_numeric(df[col1], errors="coerce")
    valid_mask = hr1.notna() & toco.notna()
    hr1 = hr1[valid_mask]
    toco = toco[valid_mask]
    if hr1.empty:
        return None

    monitor_date = pd.date_range(start=start_dt, periods=len(hr1), freq="1s")
    return pd.DataFrame(
        {
            "Monitor_Date": monitor_date,
            "HR1": hr1.astype(int).to_numpy(),
            "TOCO": toco.astype(int).to_numpy(),
        }
    )


def _folder_sort_key(path: Path) -> tuple[int, str]:
    """Sort folders by numeric prefix (e.g., 2_... before 10_...)."""
    prefix = path.name.split("_", 1)[0]
    return (int(prefix), path.name) if prefix.isdigit() else (10**9, path.name)


def _get_selected_parquet_files(config: PipelineConfig) -> list[Path]:
    """Return parquet files from the first N subfolders of config.input_dir."""
    folder_limit = config.parquet_folders_to_load
    if folder_limit <= 0:
        logger.warning("parquet_folders_to_load=%d; no folders will be loaded", folder_limit)
        return []

    folders = sorted((p for p in config.input_dir.iterdir() if p.is_dir()), key=_folder_sort_key)
    selected_folders = folders[:folder_limit]
    if not selected_folders:
        return []

    parquet_files: list[Path] = []
    for folder in selected_folders:
        parquet_files.extend(sorted(folder.glob("*.parquet")))

    logger.info(
        "Selected %d/%d parquet folders (parquet_folders_to_load=%d)",
        len(selected_folders),
        len(folders),
        folder_limit,
    )
    return parquet_files


def _within_merge_window(df_a: pd.DataFrame, df_b: pd.DataFrame, gap_hours: float) -> bool:
    """Return True if the gap between the end of df_a and the start of df_b is ≤ gap_hours.

    Args:
        df_a: Earlier DataFrame (must be pre-sorted by Monitor_Date).
        df_b: Later DataFrame.
        gap_hours: Maximum allowed gap in hours to qualify for merging.
    """
    end_a = df_a["Monitor_Date"].max()
    start_b = df_b["Monitor_Date"].min()
    gap_h = (start_b - end_a).total_seconds() / 3600
    return gap_h <= gap_hours


def _concat_and_audit(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate frames sorted by time and log any inter-segment gaps or overlaps.

    Args:
        frames: List of DataFrames in chronological order.

    Returns:
        Single merged and sorted DataFrame.
    """
    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values("Monitor_Date", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    for i in range(1, len(frames)):
        prev_end = frames[i - 1]["Monitor_Date"].max()
        curr_start = frames[i]["Monitor_Date"].min()
        delta_min = (curr_start - prev_end).total_seconds() / 60
        if delta_min > 0:
            logger.info("  Segment %d→%d: gap of %.1f min", i, i + 1, delta_min)
        elif delta_min < 0:
            logger.warning("  Segment %d→%d: overlap of %.1f min", i, i + 1, abs(delta_min))

    return merged


def load_all_patients(config: PipelineConfig) -> tuple[dict[str, pd.DataFrame], int]:
    """Load parquet files from config.input_dir, merge per-patient files, and return a dict.

    Files that share the same pat_[ID] and whose timestamps fall within
    config.merge_gap_hours of each other are merged vertically.  If a patient
    has non-contiguous recording sessions that exceed the gap threshold they
    are kept as separate entries (keyed as ``{pid}_0``, ``{pid}_1``, …).

    Args:
        config: Populated PipelineConfig instance.

    Returns:
        Tuple of (patient_id → merged DataFrame, number of files merged).
    """
    parquet_files = _get_selected_parquet_files(config)
    if not parquet_files:
        logger.error("No parquet files found in selected folders under %s", config.input_dir)
        return {}, 0
    logger.info("Discovered %d parquet files", len(parquet_files))

    # Group paths by patient ID
    grouped: dict[str, list[Path]] = {}
    for path in parquet_files:
        pid = extract_patient_id(path.name)
        if pid is not None:
            grouped.setdefault(pid, []).append(path)

    result: dict[str, pd.DataFrame] = {}
    files_merged = 0
    total_rows_before_coercion = 0
    total_rows_dropped_coercion = 0
    files_with_coercion_drops = 0

    for pid, paths in grouped.items():
        # Keep path alongside frame so we can tag df.attrs["source_files"] later
        loaded = [(p, _read_parquet(p)) for p in sorted(paths)]
        loaded = [(p, f) for p, f in loaded if f is not None]

        if not loaded:
            logger.warning("Patient %s: all files failed to load", pid)
            continue

        frame_paths = [p for p, _ in loaded]
        frames      = [f for _, f in loaded]

        for frame in frames:
            rows_before = int(frame.attrs.get("rows_before_numeric_coercion", len(frame)))
            rows_dropped = int(frame.attrs.get("numeric_coercion_dropped_rows", 0))
            total_rows_before_coercion += rows_before
            total_rows_dropped_coercion += rows_dropped
            if rows_dropped > 0:
                files_with_coercion_drops += 1

        if len(frames) == 1:
            result[pid] = frames[0]
            result[pid].attrs["source_files"] = [frame_paths[0].name]
            continue

        # Sort chronologically then greedily group consecutive mergeable frames
        paired = sorted(zip(frames, frame_paths), key=lambda t: t[0]["Monitor_Date"].min())
        frames      = [f for f, _ in paired]
        frame_paths = [p for _, p in paired]

        sessions:      list[list[pd.DataFrame]] = [[frames[0]]]
        session_paths: list[list[Path]]         = [[frame_paths[0]]]

        for frame, fpath in zip(frames[1:], frame_paths[1:]):
            tail = sessions[-1][-1]
            if _within_merge_window(tail, frame, config.merge_gap_hours):
                logger.info("Patient %s: merging segment starting %s", pid, frame["Monitor_Date"].min())
                sessions[-1].append(frame)
                session_paths[-1].append(fpath)
            else:
                logger.warning(
                    "Patient %s: segment starting %s exceeds %.1fh gap — new session",
                    pid, frame["Monitor_Date"].min(), config.merge_gap_hours,
                )
                sessions.append([frame])
                session_paths.append([fpath])

        files_merged += len(frames) - len(sessions)

        if len(sessions) == 1:
            result[pid] = _concat_and_audit(sessions[0])
            result[pid].attrs["source_files"] = [p.name for p in session_paths[0]]
        else:
            for i, (session, spaths) in enumerate(zip(sessions, session_paths)):
                key = f"{pid}_{i}"
                result[key] = _concat_and_audit(session)
                result[key].attrs["source_files"] = [p.name for p in spaths]

    logger.info(
        "Loaded %d patient records (%d file merges performed)",
        len(result), files_merged,
    )
    logger.info(
        "Numeric coercion audit [batch]: dropped %d/%d rows across %d files (%d files with drops)",
        total_rows_dropped_coercion,
        total_rows_before_coercion,
        len(parquet_files),
        files_with_coercion_drops,
    )
    return result, files_merged


def load_patient_by_id(patient_id: str, config: PipelineConfig) -> pd.DataFrame | None:
    """Load and merge all files for a single patient without scanning the full directory.

    Useful for interactive / notebook workflows where loading all patients is unnecessary.

    Args:
        patient_id: The patient ID string (as extracted by extract_patient_id).
        config: Pipeline configuration (provides input_dir and merge_gap_hours).

    Returns:
        Merged DataFrame for the patient, or None if no matching files are found.
    """
    parquet_files = _get_selected_parquet_files(config)
    paths = [p for p in parquet_files if extract_patient_id(p.name) == patient_id]

    if not paths:
        logger.warning("No files found for patient ID: %s", patient_id)
        return None

    frames = [f for p in paths if (f := _read_parquet(p)) is not None]

    if not frames:
        return None

    if len(frames) == 1:
        return frames[0]

    frames.sort(key=lambda d: d["Monitor_Date"].min())
    return _concat_and_audit(frames)
