"""Vibration signal model for a faulted rolling-element bearing.

The model follows the standard description in Randall & Antoni, "Rolling
element bearing diagnostics -- a tutorial" (MSSP, 2011). A localised defect
produces a train of short impacts; each impact rings the structure at its
resonance; the resulting burst train is what a casing-mounted accelerometer
sees, buried in machine noise.

    x(t) = sum_i  A(t_i) * s(t - t_i)  +  shaft harmonics  +  noise
    t_i  = i / f_fault + jitter_i
    s(t) = exp(-zeta * 2*pi*fn*t) * sin(2*pi*fn*sqrt(1-zeta^2)*t)   for t >= 0

Three details carry all the diagnostic information, and getting them right is
the difference between a simulator and a random-number generator:

**Random slip.** The rolling elements slip slightly, so impacts are not exactly
periodic. Jitter of about 1 percent of the period is typical. It smears the raw
spectrum badly -- which is precisely why envelope analysis, not the raw
spectrum, is the industry method.

**Load-zone modulation.** This is what separates the three fault classes:

* an *outer-race* defect is stationary with respect to the load zone, so every
  impact has the same amplitude -- no modulation;
* an *inner-race* defect rotates with the shaft, entering and leaving the load
  zone once per revolution -- amplitude modulated at the shaft frequency;
* a *ball* defect rotates with the cage -- amplitude modulated at the cage
  frequency (FTF).

**Severity as a nuisance variable.** Defect size scales impact energy. It is
drawn at random and never given to the model, so the classifier cannot lean on
amplitude as a shortcut for fault type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bfd.geometry import Bearing, cage_frequency, fault_frequencies

# Fault classes, in label order. Do not reorder without regenerating datasets.
CLASSES: tuple[str, ...] = ("normal", "outer_race", "inner_race", "ball")


@dataclass(frozen=True)
class OperatingCondition:
    """A machine operating point. Domain shift in this project means this changing."""

    name: str
    shaft_rpm: float
    load_fraction: float

    @property
    def shaft_hz(self) -> float:
        return self.shaft_rpm / 60.0


# Four operating points spanning a variable-frequency-drive's range. The speed
# span is deliberately wider than CWRU's four motor loads (1730-1797 rpm, a
# 4 percent spread): a 4 percent shift is too small to expose the
# generalisation failure this project is about.
CONDITIONS: dict[str, OperatingCondition] = {
    "A": OperatingCondition("A", shaft_rpm=1500.0, load_fraction=0.25),
    "B": OperatingCondition("B", shaft_rpm=1800.0, load_fraction=0.50),
    "C": OperatingCondition("C", shaft_rpm=2400.0, load_fraction=0.75),
    "D": OperatingCondition("D", shaft_rpm=3000.0, load_fraction=1.00),
}


@dataclass(frozen=True)
class RigParams:
    """Properties of the measurement chain, shared by every operating point."""

    sample_rate: float = 12000.0
    n_samples: int = 2048
    resonance_hz_min: float = 2800.0
    resonance_hz_max: float = 4200.0
    damping_min: float = 0.02
    damping_max: float = 0.06
    slip_fraction: float = 0.01
    severity_min: float = 0.4
    severity_max: float = 1.6
    # Broadband background, as a fraction of the signal's own standard
    # deviation. This range puts the records at roughly -3 to +6 dB SNR, which
    # is where incipient bearing faults actually sit: the defect is a small
    # part of the total vibration long before it is a large one. It also makes
    # the kurtogram worth running -- at the 15 dB SNR a gentler setting would
    # give, demodulating the full band does just as well.
    noise_std_min: float = 0.5
    noise_std_max: float = 1.5


def impulse_response(
    n: int, sample_rate: float, resonance_hz: float, damping: float
) -> np.ndarray:
    """One decaying resonance burst, truncated once it has rung down."""
    t = np.arange(n) / sample_rate
    damped_hz = resonance_hz * np.sqrt(max(1.0 - damping**2, 1e-9))
    return np.exp(-damping * 2.0 * np.pi * resonance_hz * t) * np.sin(
        2.0 * np.pi * damped_hz * t
    )


def _impact_train(
    fault_hz: float,
    modulation_hz: float,
    modulation_depth: float,
    rig: RigParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit impulses at the fault rate, jittered and amplitude modulated."""
    duration = rig.n_samples / rig.sample_rate
    period = 1.0 / fault_hz
    n_impacts = int(np.ceil(duration / period)) + 2

    times = np.arange(n_impacts) * period
    times = times + rng.normal(0.0, rig.slip_fraction * period, n_impacts)
    # A random starting phase, so impacts are not aligned to sample 0.
    times = times - rng.uniform(0.0, period)

    train = np.zeros(rig.n_samples)
    for t in times:
        index = int(round(t * rig.sample_rate))
        if 0 <= index < rig.n_samples:
            envelope = 1.0 + modulation_depth * np.cos(2.0 * np.pi * modulation_hz * t)
            train[index] += max(envelope, 0.0)
    return train


def _shaft_harmonics(
    condition: OperatingCondition, rig: RigParams, rng: np.random.Generator
) -> np.ndarray:
    """Imbalance and misalignment components present whether or not there is a fault."""
    t = np.arange(rig.n_samples) / rig.sample_rate
    fr = condition.shaft_hz
    signal = np.zeros_like(t)
    for order, amplitude in ((1, 0.6), (2, 0.25), (3, 0.1)):
        signal += amplitude * np.sin(2.0 * np.pi * order * fr * t + rng.uniform(0, 2 * np.pi))
    return signal * (0.5 + condition.load_fraction)


def simulate(
    fault: str,
    condition: OperatingCondition,
    bearing: Bearing,
    rig: RigParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """One vibration record, normalised to unit standard deviation.

    Normalisation is deliberate: accelerometer sensitivity and amplifier gain
    vary between installations, so absolute amplitude is not a feature a
    deployable model may depend on.
    """
    if fault not in CLASSES:
        raise ValueError(f"unknown fault {fault!r}; expected one of {CLASSES}")

    signal = _shaft_harmonics(condition, rig, rng)

    if fault != "normal":
        frequencies = fault_frequencies(bearing, condition.shaft_hz)
        fault_hz = frequencies[fault]

        # Which rate the impact amplitude is modulated at -- see module docstring.
        if fault == "inner_race":
            modulation_hz = condition.shaft_hz
            depth = 0.7 * condition.load_fraction
        elif fault == "ball":
            modulation_hz = cage_frequency(bearing, condition.shaft_hz)
            depth = 0.6 * condition.load_fraction
        else:  # outer_race: the defect sits in the load zone and stays there
            modulation_hz = 0.0
            depth = 0.0

        train = _impact_train(fault_hz, modulation_hz, depth, rig, rng)
        response = impulse_response(
            rig.n_samples,
            rig.sample_rate,
            rng.uniform(rig.resonance_hz_min, rig.resonance_hz_max),
            rng.uniform(rig.damping_min, rig.damping_max),
        )
        severity = rng.uniform(rig.severity_min, rig.severity_max)
        bursts = np.convolve(train, response)[: rig.n_samples]
        signal = signal + severity * bursts / (np.std(bursts) + 1e-12)

    signal = signal + rng.normal(
        0.0, rng.uniform(rig.noise_std_min, rig.noise_std_max) * np.std(signal), rig.n_samples
    )
    return (signal / (np.std(signal) + 1e-12)).astype(np.float64)
