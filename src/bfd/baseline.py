"""Envelope-feature baseline: gradient boosting on physics-derived features."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bfd.dataset import VibrationDataset
from bfd.features import FEATURE_NAMES, extract
from bfd.geometry import SKF_6205, Bearing


def build_model(seed: int = 7) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.1,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=seed,
                ),
            ),
        ]
    )


def featurise(
    data: VibrationDataset, sample_rate: float, bearing: Bearing = SKF_6205
) -> np.ndarray:
    return extract(data.x.astype(np.float64), sample_rate, data.shaft_hz, bearing)


def fit(
    train: VibrationDataset, sample_rate: float, seed: int = 7, bearing: Bearing = SKF_6205
) -> tuple[Pipeline, dict[str, Any]]:
    t0 = time.perf_counter()
    features = featurise(train, sample_rate, bearing)
    feature_seconds = time.perf_counter() - t0

    model = build_model(seed)
    t0 = time.perf_counter()
    model.fit(features, train.y)

    return model, {
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_extraction_seconds": round(feature_seconds, 2),
        "fit_seconds": round(time.perf_counter() - t0, 2),
        "n_train": int(len(train)),
    }


def predict(
    model: Pipeline, data: VibrationDataset, sample_rate: float, bearing: Bearing = SKF_6205
) -> np.ndarray:
    return model.predict(featurise(data, sample_rate, bearing))


def save(model: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load(path: str | Path) -> Pipeline:
    return joblib.load(path)
