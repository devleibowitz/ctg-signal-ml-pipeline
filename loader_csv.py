"""CSV directory loading for CTG monitor files."""

from __future__ import annotations

import logging
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


def get_csv_files(config: PipelineConfig) -> list[Path]:
    """Return all CSV files from config.input_dir_csv."""
    if config.input_dir_csv is None:
        logger.error("input_source=csv but input_dir_csv is not set")
        return []

    csv_files = sorted(config.input_dir_csv.glob("*.csv"))
    logger.info("Discovered %d CSV files in %s", len(csv_files), config.input_dir_csv)
    return csv_files


def read_csv_file(path: Path, summary: LoadSummary) -> pd.DataFrame | None:
    """Read one CTG CSV and return Monitor_Date / HR1 / TOCO with enforced dtypes.

    Files use a 2-row header: row 0 is patient-ID metadata, row 1 contains
    the actual column names (Monitor_Date, HR1, HR2, HRM, TOCO).
    """
    try:
        df = pd.read_csv(path, header=1)
    except (ValueError, pd.errors.ParserError) as exc:
        summary.record_discard(path.name, "metadata_only")
        logger.debug("Skipping metadata-only file %s (%s)", path.name, exc)
        return None
    except Exception as exc:
        summary.record_discard(path.name, "read_error")
        logger.error("Cannot read %s: %s", path, exc)
        return None

    df.columns = df.columns.str.strip()
    return finalize_dataframe(df, path, summary)


def load_all_patients_csv(config: PipelineConfig) -> tuple[dict[str, pd.DataFrame], int, LoadSummary]:
    """Load and merge all CSV files from config.input_dir_csv."""
    summary = LoadSummary(input_format="csv")
    source_files = get_csv_files(config)
    summary.files_discovered = len(source_files)

    if not source_files:
        logger.error("No CSV files found in %s", config.input_dir_csv)
        summary.save(config.output_dir)
        return {}, 0, summary

    def _reader(path: Path) -> pd.DataFrame | None:
        return read_csv_file(path, summary)

    patients = load_grouped_patients(source_files, _reader, config, summary)
    summary.save(config.output_dir)
    return patients, summary.files_merged, summary


def load_patient_by_id_csv(patient_id: str, config: PipelineConfig) -> pd.DataFrame | None:
    """Load and merge CSV files for a single patient ID."""
    from loader_common import concat_and_audit, extract_patient_id

    summary = LoadSummary(input_format="csv")
    paths = [
        p for p in get_csv_files(config)
        if extract_patient_id(p.name) == patient_id
    ]
    if not paths:
        logger.warning("No CSV files found for patient ID: %s", patient_id)
        return None

    loaded = [(p, f) for p in paths if (f := read_csv_file(p, summary)) is not None]
    if not loaded:
        return None

    if len(loaded) == 1:
        return loaded[0][1]

    records, _ = merge_patient_files(patient_id, loaded, config)
    if len(records) == 1:
        return next(iter(records.values()))

    frames = sorted(records.values(), key=lambda d: d["Monitor_Date"].min())
    return concat_and_audit(frames)
