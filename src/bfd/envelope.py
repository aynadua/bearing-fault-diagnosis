"""Envelope analysis -- the method a vibration engineer actually uses.

A bearing defect does not show up cleanly in the raw spectrum. Random slip
smears the impact train, and the impacts are a small part of the total energy.
What *does* work is demodulation:

1. Band-pass the signal around the structural resonance the impacts excite.
   The resonance acts as a carrier; the fault repetition rate is the
   modulation.
2. Take the magnitude of the analytic signal (Hilbert transform) to recover
   that modulation -- the envelope.
3. Spectrum of the envelope. The fault frequency appears as a sharp line,
   usually with harmonics.

Step 1 needs a band. Picking it by eye does not scale, so the band is chosen
with a kurtogram (Antoni & Randall): split the spectrum into bands at several
resolutions, band-pass filter each, and keep the band whose filtered signal has
the highest kurtosis -- that is, the band where the impulsiveness lives.

:func:`order_track` is separate, and is the lever this project pulls. It
resamples from constant time steps to constant *shaft angle* steps. Under
angular sampling a fault frequency becomes a fault *order* -- a fixed multiple
of shaft rotation -- which does not move when the machine changes speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt


def bandpass(x: np.ndarray, sample_rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    nyquist = 0.5 * sample_rate
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 1.0 - 1e-4)
    if low >= high:
        return x
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def kurtosis(x: np.ndarray) -> float:
    """Fourth standardised moment. 3.0 for Gaussian noise, higher when impulsive."""
    centred = x - np.mean(x)
    variance = np.mean(centred**2)
    if variance <= 0:
        return 0.0
    return float(np.mean(centred**4) / variance**2)


@dataclass(frozen=True)
class Band:
    """A demodulation band and the kurtosis that justified choosing it."""

    low_hz: float
    high_hz: float
    kurtosis: float


def kurtogram_band(
    x: np.ndarray, sample_rate: float, levels: tuple[int, ...] = (2, 4, 8, 16)
) -> Band:
    """Choose the demodulation band by maximum kurtosis of the filtered signal."""
    nyquist = 0.5 * sample_rate
    best = Band(low_hz=0.0, high_hz=nyquist, kurtosis=kurtosis(x))

    for n_bands in levels:
        width = nyquist / n_bands
        for index in range(n_bands):
            low, high = index * width, (index + 1) * width
            if low <= 0:
                low = 1.0
            score = kurtosis(bandpass(x, sample_rate, low, high))
            if score > best.kurtosis:
                best = Band(low_hz=low, high_hz=high, kurtosis=score)
    return best


def envelope_spectrum(
    x: np.ndarray, sample_rate: float, band: Band | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frequencies, magnitude)`` of the squared-envelope spectrum."""
    filtered = bandpass(x, sample_rate, band.low_hz, band.high_hz) if band else x
    env = np.abs(hilbert(filtered))
    env = env - np.mean(env)  # strip DC, which otherwise dwarfs every line

    window = np.hanning(len(env))
    spectrum = np.abs(np.fft.rfft(env * window)) / (0.5 * np.sum(window))
    frequencies = np.fft.rfftfreq(len(env), d=1.0 / sample_rate)
    return frequencies, spectrum


def band_energy(
    frequencies: np.ndarray, spectrum: np.ndarray, centre_hz: float, tolerance_hz: float
) -> float:
    """Peak magnitude within +/- ``tolerance_hz`` of a target frequency."""
    mask = np.abs(frequencies - centre_hz) <= tolerance_hz
    return float(np.max(spectrum[mask])) if mask.any() else 0.0


def order_track(
    x: np.ndarray,
    sample_rate: float,
    shaft_hz: float,
    samples_per_rev: int = 512,
    strict: bool = True,
) -> np.ndarray:
    """Resample from constant time steps to constant shaft-angle steps.

    After this, one sample is a fixed fraction of a revolution regardless of
    speed, so a fault at 3.58x shaft rate sits at the same place in the
    transformed signal whether the machine runs at 1500 or 3000 rpm.

    **Choose ``samples_per_rev`` large enough.** The angular sample rate is
    ``samples_per_rev * shaft_hz``. If that falls below the original sample
    rate, this is a downsample, and the anti-alias filter inside
    ``resample_poly`` will quietly delete everything above the new Nyquist --
    including the structural resonance the bearing impacts ring at. The fault
    repetition rate survives, the impulsiveness that makes it detectable does
    not, and the result looks like order tracking while diagnosing nothing.

    At 12 kHz and a 1500 rpm shaft, 128 samples/rev gives a 1.6 kHz Nyquist
    and throws away a 3 kHz resonance entirely; 512 samples/rev gives 6.4 kHz
    and keeps it. ``strict`` raises instead of letting that happen silently.
    """
    angular_rate = samples_per_rev * shaft_hz
    if strict and angular_rate < sample_rate:
        raise ValueError(
            f"order tracking at {samples_per_rev} samples/rev and {shaft_hz:.1f} Hz shaft "
            f"speed gives an angular rate of {angular_rate:.0f} Hz, below the signal's "
            f"{sample_rate:.0f} Hz. That discards content above {angular_rate / 2:.0f} Hz. "
            f"Use at least {int(np.ceil(sample_rate / shaft_hz))} samples/rev, or pass "
            "strict=False if the loss is intended."
        )

    current_per_rev = sample_rate / shaft_hz
    ratio = Fraction(samples_per_rev / current_per_rev).limit_denominator(200)
    return resample_poly(x, ratio.numerator, ratio.denominator)


def envelope_order_track(
    x: np.ndarray,
    sample_rate: float,
    shaft_hz: float,
    samples_per_rev: int = 64,
    band: Band | None = None,
) -> np.ndarray:
    """Demodulate first, then resample by shaft angle.

    Plain :func:`order_track` fixes the fault repetition rate in order space
    but *not* the resonance that carries it. The structure rings at a constant
    frequency in Hz, so in orders it moves: a 3.5 kHz resonance sits at order
    140 on a 1500 rpm shaft and order 70 at 3000 rpm. A network reading the
    order-domain waveform still sees its carrier slide across the spectrum
    when the machine changes speed, which is why order tracking alone does not
    rescue it.

    Demodulating first removes the carrier entirely. What remains is the
    envelope -- the impact pattern -- whose rate is a fixed multiple of shaft
    speed. Order-tracking *that* is genuinely speed-invariant, and it is the
    representation condition monitoring has used for decades.

    The angular rate is far below the signal's own rate here, so the
    anti-alias filter inside the resampler discards content above the order
    Nyquist. For the raw waveform that would destroy the diagnosis; for the
    envelope it is the point, since everything above a few tens of orders is
    residual carrier leakage rather than fault information.
    """
    band = band or kurtogram_band(x, sample_rate)
    filtered = bandpass(x, sample_rate, band.low_hz, band.high_hz)
    envelope = np.abs(hilbert(filtered))
    envelope = envelope - np.mean(envelope)
    return order_track(envelope, sample_rate, shaft_hz, samples_per_rev, strict=False)
