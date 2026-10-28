"""Figures for the README. Headless by construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bfd.config import Config  # noqa: E402
from bfd.envelope import envelope_spectrum, kurtogram_band  # noqa: E402
from bfd.geometry import SKF_6205, fault_frequencies  # noqa: E402
from bfd.simulate import CLASSES, CONDITIONS, RigParams, simulate  # noqa: E402


def domain_matrix(
    records: list[dict[str, Any]], conditions: list[str], source: list[str], path: str | Path
) -> Path:
    """Grouped bars: accuracy per operating condition, one group per model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    positions = np.arange(len(conditions))
    width = 0.8 / max(len(records), 1)

    for index, record in enumerate(records):
        values = [record["cells"][c]["accuracy"] for c in conditions]
        ax.bar(positions + index * width, values, width, label=record["model"])

    labels = [
        f"{c}\n{CONDITIONS[c].shaft_rpm:.0f} rpm" + ("\n(source)" if c in source else "")
        for c in conditions
    ]
    ax.set_xticks(positions + width * (len(records) - 1) / 2, labels)
    ax.axhline(1.0 / len(CLASSES), linestyle="--", linewidth=1, color="grey", label="chance")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fault classification under operating-condition shift")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def confusion(record: dict[str, Any], condition: str, path: str | Path) -> Path:
    """Confusion matrix for one model at one operating condition."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(record["cells"][condition]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=140)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{record['model']} at condition {condition}", fontsize=10)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value >= 0.01:
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.5 else "black", fontsize=8,
                )
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def signal_gallery(cfg: Config, path: str | Path, condition: str = "B") -> Path:
    """Waveform and envelope spectrum for each fault class, with theory marked."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    rig = RigParams(sample_rate=cfg.rig.sample_rate, n_samples=cfg.rig.n_samples)
    operating = CONDITIONS[condition]
    sites = fault_frequencies(SKF_6205, operating.shaft_hz)

    fig, axes = plt.subplots(len(CLASSES), 2, figsize=(11, 9), dpi=130)
    for row, fault in enumerate(CLASSES):
        signal = simulate(fault, operating, SKF_6205, rig, rng)
        time = np.arange(len(signal)) / rig.sample_rate

        axes[row, 0].plot(time * 1e3, signal, linewidth=0.5)
        axes[row, 0].set_ylabel(fault, fontsize=9)
        axes[row, 0].set_xlim(0, 120)
        if row == 0:
            axes[row, 0].set_title("Vibration waveform", fontsize=10)
        if row == len(CLASSES) - 1:
            axes[row, 0].set_xlabel("time (ms)")

        band = kurtogram_band(signal, rig.sample_rate)
        frequencies, spectrum = envelope_spectrum(signal, rig.sample_rate, band)
        keep = frequencies < 600
        axes[row, 1].plot(frequencies[keep], spectrum[keep], linewidth=0.7)
        for name, colour in (
            ("outer_race", "tab:red"),
            ("inner_race", "tab:green"),
            ("ball", "tab:purple"),
        ):
            axes[row, 1].axvline(
                sites[name], color=colour, linestyle="--", linewidth=1, alpha=0.8,
                label=name if row == 0 else None,
            )
        if row == 0:
            axes[row, 1].set_title(
                "Envelope spectrum (dashed = theoretical fault lines)", fontsize=10
            )
            axes[row, 1].legend(fontsize=7)
        if row == len(CLASSES) - 1:
            axes[row, 1].set_xlabel("frequency (Hz)")

    fig.suptitle(f"Condition {condition}: {operating.shaft_rpm:.0f} rpm", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def order_invariance(cfg: Config, path: str | Path) -> Path:
    """Why order tracking works: the same fault, at two speeds, in both domains."""
    from bfd.envelope import order_track

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    rig = RigParams(sample_rate=cfg.rig.sample_rate, n_samples=cfg.rig.n_samples)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)

    for condition_name, colour in (("A", "tab:blue"), ("D", "tab:orange")):
        operating = CONDITIONS[condition_name]
        signal = simulate("outer_race", operating, SKF_6205, rig, rng)
        label = f"{operating.shaft_rpm:.0f} rpm"

        frequencies, spectrum = envelope_spectrum(
            signal, rig.sample_rate, kurtogram_band(signal, rig.sample_rate)
        )
        keep = frequencies < 400
        axes[0].plot(
            frequencies[keep], spectrum[keep] / spectrum[keep].max(),
            linewidth=0.8, color=colour, label=label,
        )

        tracked = order_track(
            signal, rig.sample_rate, operating.shaft_hz, cfg.rig.samples_per_rev
        )
        orders, order_spectrum = envelope_spectrum(
            tracked, float(cfg.rig.samples_per_rev),
            kurtogram_band(tracked, float(cfg.rig.samples_per_rev)),
        )
        keep_orders = orders < 12
        axes[1].plot(
            orders[keep_orders], order_spectrum[keep_orders] / order_spectrum[keep_orders].max(),
            linewidth=0.8, color=colour, label=label,
        )

    axes[0].set_title("Time domain: the fault line moves with speed", fontsize=10)
    axes[0].set_xlabel("frequency (Hz)")
    axes[1].axvline(
        fault_frequencies(SKF_6205, 1.0)["outer_race"], color="black", linestyle="--",
        linewidth=1, label="BPFO = 3.58 orders",
    )
    axes[1].set_title("Order domain: it does not", fontsize=10)
    axes[1].set_xlabel("order (multiples of shaft rate)")
    for ax in axes:
        ax.set_ylabel("normalised envelope magnitude")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
