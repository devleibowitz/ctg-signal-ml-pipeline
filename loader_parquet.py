"""Parquet batch-folder loading for CTG monitor files."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from config import PipelineConfig
from loader_common import (
    LoadSummary,
    finalize_dataframe,
    load_grouped_patients,
    merge_patient_files,
)

logger = logging.getLogger(__name__)

_FILE_DATETIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2}\.\d{3})")


def _folder_sort_key(path: Path) -> tuple[int, str]:
    """Sort folders by numeric prefix (e.g., 2_... before 10_...)."""
    prefix = path.name.split("_", 1)[0]
    return (int(prefix), path.name) if prefix.isdigit() else (10**9, path.name)


def get_parquet_files(config: PipelineConfig) -> list[Path]:
    """Return parquet files from the first N subfolders of config.input_dir_parquet."""
    if config.input_dir_parquet is None:
        logger.error("input_source=parquet but input_dir_parquet is not set")
        return []

    folder_limit = config.parquet_folders_to_load
    file_limit = config.parquet_files_to_load
    if folder_limit <= 0:
        logger.warning("parquet_folders_to_load=%d; no folders will be loaded", folder_limit)
        return []

    folders = sorted(
        (p for p in config.input_dir_parquet.iterdir() if p.is_dir()),
        key=_folder_sort_key,
    )
    selected_folders = folders[:folder_limit]
    if not selected_folders:
        return []

    parquet_files: list[Path] = []
    for folder in selected_folders:
        parquet_files.extend(sorted(folder.glob("*.parquet")))

    total_files = len(parquet_files)
    limit_to_apply = total_files
    if file_limit is not None and file_limit > 0:
        limit_to_apply = min(total_files, file_limit)

    logger.info(
        "Selected %d/%d parquet folders (parquet_folders_to_load=%d); "
        "returning %d/%d file(s) (parquet_files_to_load=%s)",
        len(selected_folders),
        len(folders),
        folder_limit,
        limit_to_apply,
        total_files,
        file_limit if file_limit is not None else "all",
    )
    return parquet_files[:limit_to_apply]


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


def read_parquet_file(path: Path, summary: LoadSummary) -> pd.DataFrame | None:
    """Read one CTG parquet file, reconstructing columns when needed."""
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        summary.record_discard(path.name, "read_error")
        logger.error("Cannot read %s: %s", path, exc)
        return None

    df.columns = df.columns.str.strip()

    if "Monitor_Date" not in df.columns and df.shape[1] == 2:
        reconstructed = _reconstruct_from_string_parquet(df, path)
        if reconstructed is None:
            summary.record_discard(path.name, "reconstruction_failed")
            logger.warning("Skipping %s — could not reconstruct target columns", path.name)
            return None
        df = reconstructed

    return finalize_dataframe(df, path, summary)


def load_all_patients_parquet(config: PipelineConfig) -> tuple[dict[str, pd.DataFrame], int, LoadSummary]:
    """Load and merge parquet files from batched subfolders."""
    summary = LoadSummary(input_format="parquet")
    source_files = get_parquet_files(config)
    summary.files_discovered = len(source_files)

    if not source_files:
        logger.error("No parquet files found under %s", config.input_dir_parquet)
        summary.save(config.output_dir)
        return {}, 0, summary

    def _reader(path: Path) -> pd.DataFrame | None:
        return read_parquet_file(path, summary)

    patients = load_grouped_patients(source_files, _reader, config, summary)
    summary.save(config.output_dir)
    return patients, summary.files_merged, summary


def load_patient_by_id_parquet(patient_id: str, config: PipelineConfig) -> pd.DataFrame | None:
    """Load and merge parquet files for a single patient ID."""
    from loader_common import concat_and_audit, extract_patient_id

    summary = LoadSummary(input_format="parquet")
    paths = [
        p for p in get_parquet_files(config)
        if extract_patient_id(p.name) == patient_id
    ]
    if not paths:
        logger.warning("No parquet files found for patient ID: %s", patient_id)
        return None

    loaded = [(p, f) for p in paths if (f := read_parquet_file(p, summary)) is not None]
    if not loaded:
        return None

    if len(loaded) == 1:
        return loaded[0][1]

    records, _ = merge_patient_files(patient_id, loaded, config)
    if len(records) == 1:
        return next(iter(records.values()))

    frames = sorted(records.values(), key=lambda d: d["Monitor_Date"].min())
    return concat_and_audit(frames)
