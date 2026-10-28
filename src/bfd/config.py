"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RigConfig:
    sample_rate: float = 12000.0
    n_samples: int = 4096
    samples_per_rev: int = 512
    order_revolutions: int = 8
    envelope_samples_per_rev: int = 64
    envelope_revolutions: int = 8


@dataclass
class DomainConfig:
    """Which operating points the model may learn from, and which it is tested on."""

    source: list[str] = field(default_factory=lambda: ["A"])
    targets: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])


@dataclass
class DataConfig:
    per_class: dict[str, int] = field(
        default_factory=lambda: {"train": 1500, "val": 300, "test": 500}
    )
    seeds: dict[str, int] = field(default_factory=lambda: {"train": 101, "val": 202, "test": 303})
    out_dir: str = "data"


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 6
    seed: int = 7
    width: int = 32


@dataclass
class Config:
    rig: RigConfig = field(default_factory=RigConfig)
    domains: DomainConfig = field(default_factory=DomainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")

        sections: dict[str, Any] = {}
        for f in fields(cls):
            section = f.default_factory()  # type: ignore[misc]
            for key, value in (raw.get(f.name) or {}).items():
                if not hasattr(section, key):
                    raise ValueError(f"unknown config key {f.name}.{key}")
                setattr(section, key, value)
            sections[f.name] = section
        return cls(**sections)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
