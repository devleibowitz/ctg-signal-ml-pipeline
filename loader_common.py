"""Shared utilities for CTG file loading and patient-session merging."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config import PipelineConfig

logger = logging.getLogger(__name__)

TARGET_COLS = ["Monitor_Date", "HR1", "TOCO"]

_PATIENT_ID_RE = re.compile(r"pat[_\-]?(0x[0-9A-Fa-f]+)", re.IGNORECASE)


@dataclass
class LoadSummary:
    """Batch-level load/merge audit counters."""

    input_format: str
    files_discovered: int = 0
    files_loaded: int = 0
    patient_records: int = 0
    files_merged: int = 0
    discards: dict[str, int] = field(default_factory=dict)
    discard_details: list[dict[str, str]] = field(default_factory=list)

    def record_discard(self, filename: str, reason: str) -> None:
        """Increment a discard reason counter and store per-file detail."""
        self.discards[reason] = self.discards.get(reason, 0) + 1
        self.discard_details.append({"file_name": filename, "reason": reason})

    @property
    def files_discarded(self) -> int:
        return sum(self.discards.values())

    def to_summary_df(self) -> pd.DataFrame:
        """Return a one-row batch summary DataFrame."""
        return pd.DataFrame(
            [
                {
                    "input_format": self.input_format,
                    "files_discovered": self.files_discovered,
                    "files_loaded": self.files_loaded,
                    "files_discarded": self.files_discarded,
                    "patient_records_after_processing": self.patient_records,
                    "files_merged": self.files_merged,
                    **{f"discarded_{reason}": count for reason, count in sorted(self.discards.items())},
                }
            ]
        )

    def save(self, output_dir: Path) -> tuple[Path, Path | None]:
        """Write batch summary and per-file discard detail CSVs."""
        output_dir = Path(output_dir)
        summary_path = output_dir / "load_summary.csv"
        self.to_summary_df().to_csv(summary_path, index=False)

        details_path: Path | None = None
        if self.discard_details:
            details_path = output_dir / "load_discards.csv"
            pd.DataFrame(self.discard_details).to_csv(details_path, index=False)

        logger.info("Load summary saved → %s", summary_path)
        if details_path:
            logger.info("Load discard details saved → %s", details_path)
        return summary_path, details_path


def extract_patient_id(filename: str) -> str | None:
    """Parse a patient ID from a CTG filename using the pat_[ID] convention."""
    stem = Path(filename).stem
    match = _PATIENT_ID_RE.search(stem)
    if match:
        return match.group(1)
    return None


def finalize_dataframe(df: pd.DataFrame, path: Path, summary: LoadSummary) -> pd.DataFrame | None:
    """Coerce TARGET_COLS to datetime/int and record row-level coercion drops."""
    if "Monitor_Date" in df.columns:
        df["Monitor_Date"] = pd.to_datetime(df["Monitor_Date"], errors="coerce")

    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        summary.record_discard(path.name, "missing_columns")
        logger.debug("Skipping %s — missing columns: %s", path.name, missing)
        return None

    if df.empty:
        summary.record_discard(path.name, "empty_file")
        logger.debug("Skipping empty file %s", path.name)
        return None

    df = df[TARGET_COLS].copy()
    rows_before = len(df)
    df["HR1"] = pd.to_numeric(df["HR1"], errors="coerce")
    df["TOCO"] = pd.to_numeric(df["TOCO"], errors="coerce")
    df.dropna(subset=TARGET_COLS, inplace=True)

    if df.empty:
        summary.record_discard(path.name, "invalid_numeric_values")
        logger.debug("Skipping %s — no valid rows after type coercion", path.name)
        return None

    dropped_rows = rows_before - len(df)
    if dropped_rows:
        logger.debug(
            "Numeric coercion audit [%s]: dropped %d/%d rows",
            path.name,
            dropped_rows,
            rows_before,
        )

    df["HR1"] = df["HR1"].astype(int)
    df["TOCO"] = df["TOCO"].astype(int)
    df.sort_values("Monitor_Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.attrs["numeric_coercion_dropped_rows"] = dropped_rows
    df.attrs["rows_before_numeric_coercion"] = rows_before
    return df


def within_merge_window(df_a: pd.DataFrame, df_b: pd.DataFrame, gap_hours: float) -> bool:
    """Return True if the gap between end of df_a and start of df_b is ≤ gap_hours."""
    end_a = df_a["Monitor_Date"].max()
    start_b = df_b["Monitor_Date"].min()
    gap_h = (start_b - end_a).total_seconds() / 3600
    return gap_h <= gap_hours


def concat_and_audit(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate chronologically ordered frames and log inter-segment gaps."""
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


def merge_patient_files(
    pid: str,
    loaded: list[tuple[Path, pd.DataFrame]],
    config: PipelineConfig,
) -> tuple[dict[str, pd.DataFrame], int]:
    """Merge loadable frames for one patient into one or more session records."""
    result: dict[str, pd.DataFrame] = {}
    files_merged = 0

    frame_paths = [p for p, _ in loaded]
    frames = [f for _, f in loaded]

    if len(frames) == 1:
        result[pid] = frames[0]
        result[pid].attrs["source_files"] = [frame_paths[0].name]
        return result, files_merged

    paired = sorted(zip(frames, frame_paths), key=lambda t: t[0]["Monitor_Date"].min())
    frames = [f for f, _ in paired]
    frame_paths = [p for _, p in paired]

    sessions: list[list[pd.DataFrame]] = [[frames[0]]]
    session_paths: list[list[Path]] = [[frame_paths[0]]]

    for frame, fpath in zip(frames[1:], frame_paths[1:]):
        tail = sessions[-1][-1]
        if within_merge_window(tail, frame, config.merge_gap_hours):
            logger.info("Patient %s: merging segment starting %s", pid, frame["Monitor_Date"].min())
            sessions[-1].append(frame)
            session_paths[-1].append(fpath)
        else:
            logger.warning(
                "Patient %s: segment starting %s exceeds %.1fh gap — new session",
                pid,
                frame["Monitor_Date"].min(),
                config.merge_gap_hours,
            )
            sessions.append([frame])
            session_paths.append([fpath])

    files_merged = len(frames) - len(sessions)

    if len(sessions) == 1:
        result[pid] = concat_and_audit(sessions[0])
        result[pid].attrs["source_files"] = [p.name for p in session_paths[0]]
    else:
        for i, (session, spaths) in enumerate(zip(sessions, session_paths)):
            key = f"{pid}_{i}"
            result[key] = concat_and_audit(session)
            result[key].attrs["source_files"] = [p.name for p in spaths]

    return result, files_merged


def load_grouped_patients(
    source_files: list[Path],
    read_fn,
    config: PipelineConfig,
    summary: LoadSummary,
) -> dict[str, pd.DataFrame]:
    """Group files by patient ID, load, merge, and populate summary counters."""
    grouped: dict[str, list[Path]] = {}
    for path in source_files:
        pid = extract_patient_id(path.name)
        if pid is None:
            summary.record_discard(path.name, "no_patient_id")
            continue
        grouped.setdefault(pid, []).append(path)

    result: dict[str, pd.DataFrame] = {}
    total_rows_before = 0
    total_rows_dropped = 0

    for pid, paths in grouped.items():
        loaded: list[tuple[Path, pd.DataFrame]] = []
        for path in sorted(paths):
            frame = read_fn(path)
            if frame is not None:
                loaded.append((path, frame))
                summary.files_loaded += 1
                total_rows_before += int(frame.attrs.get("rows_before_numeric_coercion", len(frame)))
                total_rows_dropped += int(frame.attrs.get("numeric_coercion_dropped_rows", 0))

        if not loaded:
            if len(paths) > 1:
                logger.warning("Patient %s: all %d files failed to load", pid, len(paths))
            continue

        patient_records, merges = merge_patient_files(pid, loaded, config)
        result.update(patient_records)
        summary.files_merged += merges

    summary.patient_records = len(result)
    logger.info(
        "Loaded %d patient records (%d file merges performed)",
        summary.patient_records,
        summary.files_merged,
    )
    logger.info(
        "File load audit [batch]: loaded %d/%d files; discarded %d file(s)",
        summary.files_loaded,
        summary.files_discovered,
        summary.files_discarded,
    )
    if summary.discards:
        logger.info("Discard reasons: %s", summary.discards)
    if total_rows_dropped:
        logger.info(
            "Numeric coercion audit [batch]: dropped %d/%d rows across loaded files",
            total_rows_dropped,
            total_rows_before,
        )

    return result
