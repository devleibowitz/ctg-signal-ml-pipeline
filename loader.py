"""Data loading and patient-file merging for CTG monitor CSV files."""

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


def _read_csv(path: Path) -> pd.DataFrame | None:
    """Read one CTG CSV, keeping only Monitor_Date / HR1 / TOCO.

    Files use a 2-row header: row 0 is patient-ID metadata, row 1 contains
    the actual column names (Monitor_Date, HR1, HR2, HRM, TOCO).  Files
    that contain only the metadata row (no signal data) are silently skipped.

    Args:
        path: Path to a CSV file.

    Returns:
        Sorted DataFrame with TARGET_COLS, or None on failure / empty file.
    """
    try:
        # Read without parse_dates — strip column-name whitespace first,
        # then convert explicitly so the column is always a Timestamp dtype.
        df = pd.read_csv(path, header=1)
    except ValueError as exc:
        # "only N lines in file" — metadata-only file with no signal rows
        logger.debug("Skipping empty file %s (%s)", path.name, exc)
        return None
    except Exception as exc:
        logger.error("Cannot read %s: %s", path, exc)
        return None

    # Strip any accidental whitespace in column names before anything else
    df.columns = df.columns.str.strip()

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
    df.sort_values("Monitor_Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


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
    """Load every CSV from config.input_dir, merge per-patient files, and return a dict.

    Files that share the same pat_[ID] and whose timestamps fall within
    config.merge_gap_hours of each other are merged vertically.  If a patient
    has non-contiguous recording sessions that exceed the gap threshold they
    are kept as separate entries (keyed as ``{pid}_0``, ``{pid}_1``, …).

    Args:
        config: Populated PipelineConfig instance.

    Returns:
        Tuple of (patient_id → merged DataFrame, number of files merged).
    """
    csv_files = sorted(config.input_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found in %s", config.input_dir)
        return {}
    logger.info("Discovered %d CSV files", len(csv_files))

    # Group paths by patient ID
    grouped: dict[str, list[Path]] = {}
    for path in csv_files:
        pid = extract_patient_id(path.name)
        if pid is not None:
            grouped.setdefault(pid, []).append(path)

    result: dict[str, pd.DataFrame] = {}
    files_merged = 0

    for pid, paths in grouped.items():
        # Keep path alongside frame so we can tag df.attrs["source_files"] later
        loaded = [(p, _read_csv(p)) for p in sorted(paths)]
        loaded = [(p, f) for p, f in loaded if f is not None]

        if not loaded:
            logger.warning("Patient %s: all files failed to load", pid)
            continue

        frame_paths = [p for p, _ in loaded]
        frames      = [f for _, f in loaded]

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
    pattern = f"*pat*{patient_id}*.csv"
    paths = sorted(config.input_dir.glob(pattern))

    if not paths:
        logger.warning("No files found for patient ID: %s", patient_id)
        return None

    frames = [f for p in paths if (f := _read_csv(p)) is not None]

    if not frames:
        return None

    if len(frames) == 1:
        return frames[0]

    frames.sort(key=lambda d: d["Monitor_Date"].min())
    return _concat_and_audit(frames)
