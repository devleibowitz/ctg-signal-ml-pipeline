"""Facade for CTG monitor file loading (CSV or parquet)."""

from __future__ import annotations

import logging

import pandas as pd

from config import PipelineConfig
from loader_common import LoadSummary, extract_patient_id
from loader_csv import load_all_patients_csv, load_patient_by_id_csv
from loader_parquet import load_all_patients_parquet, load_patient_by_id_parquet

logger = logging.getLogger(__name__)

__all__ = [
    "LoadSummary",
    "extract_patient_id",
    "load_all_patients",
    "load_patient_by_id",
]


def load_all_patients(
    config: PipelineConfig,
) -> tuple[dict[str, pd.DataFrame], int, LoadSummary]:
    """Load patient files using the configured input source.

    Dispatches to loader_csv or loader_parquet based on config.input_source.
    Writes load_summary.csv (and load_discards.csv when applicable) to output_dir.

    Returns:
        Tuple of (patient_id → merged DataFrame, files merged, LoadSummary).
    """
    source = config.input_source.lower()
    if source == "csv":
        return load_all_patients_csv(config)
    if source == "parquet":
        return load_all_patients_parquet(config)
    raise ValueError(f"Unsupported input_source: {config.input_source!r}")


def load_patient_by_id(patient_id: str, config: PipelineConfig) -> pd.DataFrame | None:
    """Load and merge all files for a single patient ID."""
    source = config.input_source.lower()
    if source == "csv":
        return load_patient_by_id_csv(patient_id, config)
    if source == "parquet":
        return load_patient_by_id_parquet(patient_id, config)
    raise ValueError(f"Unsupported input_source: {config.input_source!r}")
