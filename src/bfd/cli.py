"""Command line entry point: ``bfd <stage>``.

    bfd data                    # generate vibration records for every condition
    bfd baseline                # envelope features + gradient boosting
    bfd train --rep raw         # CNN on the raw waveform
    bfd train --rep order       # CNN on the order-tracked waveform
    bfd evaluate                # score every trained model on every condition
    bfd report                  # collect into reports/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bfd import baseline, evaluate, plots
from bfd.config import Config
from bfd.dataset import (
    VibrationDataset,
    build_all,
    to_envelope_order_domain,
    to_order_domain,
)
from bfd.simulate import CONDITIONS

REPORTS = Path("reports")
MODELS = Path("models")
REPRESENTATIONS = ("raw", "order", "envelope_order")


def _path(cfg: Config, split: str, condition: str) -> Path:
    return Path(cfg.data.out_dir) / f"{split}_{condition}.npz"


def _load(cfg: Config, split: str, condition: str) -> VibrationDataset:
    path = _path(cfg, split, condition)
    if not path.exists():
        raise SystemExit(f"missing {path}; run `bfd data` first")
    return VibrationDataset.load(path)


def _represent(data: VibrationDataset, cfg: Config, representation: str) -> np.ndarray:
    if representation == "raw":
        return data.x
    if representation == "order":
        return to_order_domain(data, cfg)
    if representation == "envelope_order":
        return to_envelope_order_domain(data, cfg)
    raise ValueError(f"unknown representation {representation!r}")


def cmd_data(cfg: Config, args: argparse.Namespace) -> None:
    paths = build_all(cfg)
    for (split, condition), path in sorted(paths.items()):
        data = VibrationDataset.load(path)
        rpm = CONDITIONS[condition].shaft_rpm
        print(f"{split:5s} {condition}  {rpm:5.0f} rpm  {len(data):5d} records  {path}")


def cmd_baseline(cfg: Config, args: argparse.Namespace) -> None:
    train_parts = [_load(cfg, "train", c) for c in cfg.domains.source]
    pooled = VibrationDataset(
        x=np.concatenate([p.x for p in train_parts]),
        y=np.concatenate([p.y for p in train_parts]),
        shaft_hz=np.concatenate([p.shaft_hz for p in train_parts]),
        condition="+".join(cfg.domains.source),
    )

    model, info = baseline.fit(pooled, cfg.rig.sample_rate, seed=cfg.train.seed)
    baseline.save(model, MODELS / "baseline.joblib")

    cells: dict[str, Any] = {}
    for condition in cfg.domains.targets:
        test = _load(cfg, "test", condition)
        predictions = baseline.predict(model, test, cfg.rig.sample_rate)
        cells[condition] = evaluate.score(test.y, predictions)
        print(f"  {condition}: accuracy {cells[condition]['accuracy']:.4f}")

    record = {
        "model": "envelope features + GBM",
        "cells": cells,
        "domain_gap": evaluate.domain_gap(cells, cfg.domains.source),
        **info,
    }
    evaluate.write_report(record, REPORTS / "baseline.json")


def cmd_train(cfg: Config, args: argparse.Namespace) -> None:
    from bfd import train as train_mod

    train_parts = [_load(cfg, "train", c) for c in cfg.domains.source]
    val_parts = [_load(cfg, "val", c) for c in cfg.domains.source]

    x_train = np.concatenate([_represent(p, cfg, args.rep) for p in train_parts])
    y_train = np.concatenate([p.y for p in train_parts])
    x_val = np.concatenate([_represent(p, cfg, args.rep) for p in val_parts])
    y_val = np.concatenate([p.y for p in val_parts])

    print(f"training CNN on the {args.rep} representation: {x_train.shape}")
    result = train_mod.train(x_train, y_train, x_val, y_val, cfg)
    path = train_mod.save_checkpoint(result, cfg, args.rep, MODELS / f"cnn_{args.rep}.pt")
    print(
        f"best val accuracy {result.best_val_accuracy:.4f} "
        f"({result.train_seconds:.0f}s) -> {path}"
    )


def cmd_evaluate(cfg: Config, args: argparse.Namespace) -> None:
    from bfd import train as train_mod

    for representation in REPRESENTATIONS:
        checkpoint = MODELS / f"cnn_{representation}.pt"
        if not checkpoint.exists():
            print(
                f"skipping {representation}: no checkpoint, "
                f"run `bfd train --rep {representation}`"
            )
            continue

        model, blob = train_mod.load_checkpoint(checkpoint)
        cells: dict[str, Any] = {}
        for condition in cfg.domains.targets:
            test = _load(cfg, "test", condition)
            predictions = train_mod.predict(model, _represent(test, cfg, representation))
            cells[condition] = evaluate.score(test.y, predictions)
            accuracy = cells[condition]["accuracy"]
            print(f"  CNN-{representation} {condition}: accuracy {accuracy:.4f}")

        record = {
            "model": f"CNN ({representation})",
            "cells": cells,
            "domain_gap": evaluate.domain_gap(cells, cfg.domains.source),
            "n_parameters": blob["n_parameters"],
            "best_val_accuracy": blob["best_val_accuracy"],
        }
        evaluate.write_report(record, REPORTS / f"cnn_{representation}.json")


def cmd_report(cfg: Config, args: argparse.Namespace) -> None:
    records: list[dict[str, Any]] = []
    for name in ("baseline.json", "cnn_raw.json", "cnn_order.json", "cnn_envelope_order.json"):
        path = REPORTS / name
        if path.exists():
            records.append(json.loads(path.read_text()))
    if not records:
        raise SystemExit("nothing to report; run `bfd baseline` and `bfd train` first")

    conditions = cfg.domains.targets
    plots.domain_matrix(records, conditions, cfg.domains.source, REPORTS / "domain_shift.png")
    plots.signal_gallery(cfg, REPORTS / "signals.png")
    plots.order_invariance(cfg, REPORTS / "order_invariance.png")
    for record in records:
        slug = record["model"].split()[0].lower()
        if "(" in record["model"]:
            slug = f"cnn_{record['model'].split('(')[1].rstrip(')')}"
        plots.confusion(record, conditions[-1], REPORTS / f"confusion_{slug}.png")

    speeds = ", ".join(f"{c}={CONDITIONS[c].shaft_rpm:.0f} rpm" for c in conditions)
    lines = [
        "# Results",
        "",
        f"Trained only on condition(s) **{', '.join(cfg.domains.source)}**, "
        f"tested on every condition. Operating points: {speeds}.",
        "",
        "## Accuracy by operating condition",
        "",
        evaluate.format_matrix(records, conditions, cfg.domains.source),
        "",
        "![domain shift](domain_shift.png)",
        "",
        "## Domain gap",
        "",
        "| Model | In-domain | Mean out-of-domain | Worst out-of-domain |",
        "|---|---:|---:|---:|",
    ]
    for record in records:
        gap = record.get("domain_gap", {})
        if gap:
            lines.append(
                f"| {record['model']} | {gap['in_domain_accuracy']:.3f} | "
                f"{gap['mean_out_of_domain_accuracy']:.3f} | "
                f"{gap['worst_out_of_domain_accuracy']:.3f} |"
            )

    path = REPORTS / "RESULTS.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bfd", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="stage", required=True)

    for name, handler in (
        ("data", cmd_data),
        ("baseline", cmd_baseline),
        ("evaluate", cmd_evaluate),
        ("report", cmd_report),
    ):
        sub.add_parser(name, help=name).set_defaults(handler=handler)

    train_parser = sub.add_parser("train", help="fit the CNN on one representation")
    train_parser.add_argument("--rep", choices=REPRESENTATIONS, default="raw")
    train_parser.set_defaults(handler=cmd_train)

    args = parser.parse_args(argv)
    args.handler(Config.load(args.config), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
