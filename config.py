"""Pipeline configuration: load and validate parameters from a YAML file."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    min_duration_minutes: int = 90
    resample_freq_seconds: int = 1
    merge_gap_hours: float = 2.0
    hr1_min: float = 50.0
    hr1_max: float = 240.0
    toco_min: float = 0.0
    toco_max: float = 100.0
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.input_dir.exists():
            logger.warning("Input directory does not exist: %s", self.input_dir)

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

    cfg = PipelineConfig(**raw)
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))
    logger.info("Config loaded from %s", config_path)
    return cfg
