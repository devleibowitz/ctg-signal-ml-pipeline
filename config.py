"""Pipeline configuration: load and validate parameters from a YAML file."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    output_dir: Path
    input_source: str = "parquet"
    input_dir_csv: Path | None = None
    input_dir_parquet: Path | None = None
    parquet_folders_to_load: int = 18
    parquet_files_to_load: int | None = None
    min_duration_minutes: int = 90
    resample_freq_seconds: int = 1
    merge_gap_hours: float = 2.0
    hr1_min: float = 50.0
    hr1_max: float = 240.0
    toco_min: float = 0.0
    toco_max: float = 100.0
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.input_dir_csv is not None:
            self.input_dir_csv = Path(self.input_dir_csv)
        if self.input_dir_parquet is not None:
            self.input_dir_parquet = Path(self.input_dir_parquet)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        source = self.input_source.lower()
        if source not in {"csv", "parquet"}:
            raise ValueError(f"input_source must be 'csv' or 'parquet', got {self.input_source!r}")

        active = self.input_dir
        if active is None:
            logger.warning("No input directory configured for input_source=%s", self.input_source)
        elif not active.exists():
            logger.warning("Input directory does not exist: %s", active)

    @property
    def input_dir(self) -> Path | None:
        """Active input directory based on ``input_source``."""
        if self.input_source.lower() == "csv":
            return self.input_dir_csv
        return self.input_dir_parquet

    @property
    def min_duration_steps(self) -> int:
        """Fixed sequence length in samples (minutes → samples at resample_freq_seconds)."""
        return self.min_duration_minutes * 60 // self.resample_freq_seconds


def load_config(config_path: str | Path = "config.yaml") -> PipelineConfig:
    """Load and return a PipelineConfig from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Populated PipelineConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required field is missing from the YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as fh:
        raw: dict = yaml.safe_load(fh)

    # Backward-compat: legacy single input_dir key
    if "input_dir" in raw:
        legacy_dir = raw.pop("input_dir")
        if "input_dir_parquet" not in raw and "input_dir_csv" not in raw:
            raw["input_dir_parquet"] = legacy_dir
            logger.warning("Config key 'input_dir' is deprecated; mapped to 'input_dir_parquet'.")

    cfg = PipelineConfig(**raw)
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))
    logger.info("Config loaded from %s (input_source=%s)", config_path, cfg.input_source)
    return cfg
