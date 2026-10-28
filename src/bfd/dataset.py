"""Dataset generation, organised by operating condition.

A dataset here is always tied to one operating point. That is the unit of
domain shift: the model trains on the source condition(s) and is scored on
every condition separately, so the cost of a speed change is visible rather
than averaged away.

Each record carries its shaft speed. The feature-based model is allowed to use
it -- a real installation knows its own shaft speed -- and the raw-waveform CNN
is not, which is exactly the asymmetry the experiment measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bfd.config import Config
from bfd.envelope import envelope_order_track, order_track
from bfd.geometry import SKF_6205, Bearing
from bfd.simulate import CLASSES, CONDITIONS, OperatingCondition, RigParams, simulate


@dataclass
class VibrationDataset:
    """Vibration records with labels, shaft speeds, and their condition name."""

    x: np.ndarray  # (N, n_samples) float32
    y: np.ndarray  # (N,) int64
    shaft_hz: np.ndarray  # (N,) float64
    condition: str

    def __len__(self) -> int:
        return len(self.y)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, x=self.x, y=self.y, shaft_hz=self.shaft_hz, condition=self.condition
        )

    @classmethod
    def load(cls, path: str | Path) -> VibrationDataset:
        with np.load(path) as f:
            return cls(
                x=f["x"], y=f["y"], shaft_hz=f["shaft_hz"], condition=str(f["condition"])
            )


def rig_params(cfg: Config) -> RigParams:
    """Measurement-chain parameters implied by the config."""
    return RigParams(sample_rate=cfg.rig.sample_rate, n_samples=cfg.rig.n_samples)


def generate(
    cfg: Config,
    split: str,
    condition_name: str,
    bearing: Bearing = SKF_6205,
) -> VibrationDataset:
    """Generate one split at one operating condition, balanced over fault classes."""
    if split not in cfg.data.per_class:
        raise KeyError(f"no example count configured for split {split!r}")
    if condition_name not in CONDITIONS:
        raise KeyError(f"unknown condition {condition_name!r}")

    condition: OperatingCondition = CONDITIONS[condition_name]
    rig = rig_params(cfg)
    per_class = cfg.data.per_class[split]

    # Seed per (split, condition) so regenerating one does not disturb another.
    seed = cfg.data.seeds[split] + 1000 * (ord(condition_name) - ord("A"))
    rng = np.random.default_rng(seed)

    total = len(CLASSES) * per_class
    x = np.empty((total, rig.n_samples), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)
    shaft = np.empty(total, dtype=np.float64)

    index = 0
    for label, fault in enumerate(CLASSES):
        for _ in range(per_class):
            x[index] = simulate(fault, condition, bearing, rig, rng)
            y[index] = label
            # A small speed wobble, as any real drive has. The exact value is
            # what a tachometer would report, so downstream code may use it.
            shaft[index] = condition.shaft_hz * rng.normal(1.0, 0.005)
            index += 1

    permutation = rng.permutation(total)
    return VibrationDataset(
        x=x[permutation], y=y[permutation], shaft_hz=shaft[permutation], condition=condition_name
    )


def to_order_domain(data: VibrationDataset, cfg: Config) -> np.ndarray:
    """Angular-resampled records, cropped to a fixed number of revolutions.

    Every output row covers the same number of shaft revolutions regardless of
    speed, so a fault at 3.58x shaft rate lands in the same place for all of
    them.
    """
    length = cfg.rig.samples_per_rev * cfg.rig.order_revolutions
    out = np.zeros((len(data), length), dtype=np.float32)
    for i, (record, hz) in enumerate(zip(data.x, data.shaft_hz, strict=True)):
        tracked = order_track(
            record.astype(np.float64), cfg.rig.sample_rate, float(hz), cfg.rig.samples_per_rev
        )
        usable = min(length, len(tracked))
        out[i, :usable] = tracked[:usable]
    return out


def build_all(cfg: Config, out_dir: str | Path | None = None) -> dict[tuple[str, str], Path]:
    """Generate every (split, condition) pair the config asks for."""
    root = Path(out_dir or cfg.data.out_dir)
    needed = sorted(set(cfg.domains.source) | set(cfg.domains.targets))
    paths: dict[tuple[str, str], Path] = {}

    for condition in needed:
        for split in cfg.data.per_class:
            # Only the source conditions need training and validation data.
            if split in ("train", "val") and condition not in cfg.domains.source:
                continue
            data = generate(cfg, split, condition)
            path = root / f"{split}_{condition}.npz"
            data.save(path)
            paths[(split, condition)] = path
    return paths


def to_envelope_order_domain(data: VibrationDataset, cfg: Config) -> np.ndarray:
    """Demodulated, angular-resampled records, cropped to a fixed revolution count.

    See :func:`bfd.envelope.envelope_order_track` for why this differs from
    :func:`to_order_domain` and why the difference matters.
    """
    length = cfg.rig.envelope_samples_per_rev * cfg.rig.envelope_revolutions
    out = np.zeros((len(data), length), dtype=np.float32)
    for i, (record, hz) in enumerate(zip(data.x, data.shaft_hz, strict=True)):
        tracked = envelope_order_track(
            record.astype(np.float64),
            cfg.rig.sample_rate,
            float(hz),
            cfg.rig.envelope_samples_per_rev,
        )
        usable = min(length, len(tracked))
        out[i, :usable] = tracked[:usable]
        # Scale-free: sensor gain must not be a feature.
        std = out[i].std()
        if std > 0:
            out[i] /= std
    return out
