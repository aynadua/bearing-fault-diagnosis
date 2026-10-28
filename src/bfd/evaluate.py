"""Cross-domain evaluation.

The headline artefact is a matrix: rows are models, columns are the operating
conditions they are tested on, and the source condition is marked. The number
that matters is not the best cell -- it is the drop from the source column to
the furthest target column.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bfd.simulate import CLASSES


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(cm, (y_true, y_pred), 1.0)
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    return {CLASSES[i]: float(matrix[i, i]) for i in range(n_classes)}


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Accuracy, macro recall and per-class recall for one (model, condition) cell."""
    recalls = per_class_recall(y_true, y_pred, len(CLASSES))
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_recall": float(np.mean(list(recalls.values()))),
        "per_class_recall": recalls,
        "confusion_matrix": confusion_matrix(y_true, y_pred, len(CLASSES)).tolist(),
        "n": int(len(y_true)),
    }


def domain_gap(cells: dict[str, dict[str, Any]], source: list[str]) -> dict[str, float]:
    """How much accuracy is lost moving away from the training condition."""
    source_scores = [cells[c]["accuracy"] for c in source if c in cells]
    target_scores = [cells[c]["accuracy"] for c in cells if c not in source]
    if not source_scores or not target_scores:
        return {}
    in_domain = float(np.mean(source_scores))
    return {
        "in_domain_accuracy": in_domain,
        "mean_out_of_domain_accuracy": float(np.mean(target_scores)),
        "worst_out_of_domain_accuracy": float(np.min(target_scores)),
        "mean_drop": in_domain - float(np.mean(target_scores)),
        "worst_drop": in_domain - float(np.min(target_scores)),
    }


def write_report(record: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return path


def format_matrix(records: list[dict[str, Any]], conditions: list[str], source: list[str]) -> str:
    """Markdown table: one row per model, one column per operating condition."""
    header_cells = [f"{c} (source)" if c in source else c for c in conditions]
    lines = [
        "| Model | " + " | ".join(header_cells) + " | worst drop |",
        "|---|" + "---:|" * (len(conditions) + 1),
    ]
    for record in records:
        cells = [f"{record['cells'][c]['accuracy']:.3f}" for c in conditions]
        drop = record.get("domain_gap", {}).get("worst_drop")
        drop_text = f"{drop:.3f}" if drop is not None else "-"
        lines.append(f"| {record['model']} | " + " | ".join(cells) + f" | {drop_text} |")
    return "\n".join(lines)
