"""Physics-informed features: the diagnostic a vibration engineer would compute.

Every frequency-domain feature is evaluated at a frequency *predicted from
bearing geometry and the measured shaft speed*, not at a learned absolute
frequency. That is what makes this feature set survive a change of operating
speed, and it is the whole reason it is in this repo: it is the control the
raw-waveform CNN is measured against.

Shaft speed is treated as known. On real machinery it is -- from a tachometer,
an encoder, or the drive itself -- and assuming otherwise would be modelling a
harder problem than the one the industry has.

The sideband features deserve a note. Load-zone modulation does not only
change an impact's amplitude; it puts sidebands around the fault line, spaced
at the modulating rate. An inner-race fault is modulated at shaft speed, so it
shows sidebands at ``BPFI +/- fr``. A ball fault is modulated at the cage rate,
so its sidebands sit at ``2*BSF +/- FTF``. An outer-race fault is unmodulated
and has none. Those three signatures are the textbook way to tell the faults
apart, so they are measured explicitly.
"""

from __future__ import annotations

import numpy as np

from bfd.envelope import band_energy, envelope_spectrum, kurtogram_band, kurtosis
from bfd.geometry import Bearing, cage_frequency, fault_frequencies

FAULT_SITES: tuple[str, ...] = ("outer_race", "inner_race", "ball")
HARMONICS: tuple[int, ...] = (1, 2, 3)

FEATURE_NAMES: tuple[str, ...] = (
    *(f"env_{site}_h{h}" for site in FAULT_SITES for h in HARMONICS),
    "sideband_inner_shaft",
    "sideband_ball_cage",
    "env_shaft_h1",
    "band_centre_norm",
    "band_width_norm",
    "band_kurtosis",
    "time_kurtosis",
    "crest_factor",
    "skewness",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    "rms_to_peak",
    "spectral_centroid_norm",
    "spectral_entropy",
)


def _time_domain(x: np.ndarray) -> dict[str, float]:
    """Classic scalar condition indicators."""
    absolute = np.abs(x)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(absolute))
    mean_abs = float(np.mean(absolute))
    mean_sqrt = float(np.mean(np.sqrt(absolute)) ** 2)
    centred = x - np.mean(x)
    std = float(np.std(x))

    return {
        "time_kurtosis": kurtosis(x),
        "crest_factor": peak / rms if rms > 0 else 0.0,
        "skewness": float(np.mean(centred**3) / std**3) if std > 0 else 0.0,
        "shape_factor": rms / mean_abs if mean_abs > 0 else 0.0,
        "impulse_factor": peak / mean_abs if mean_abs > 0 else 0.0,
        "clearance_factor": peak / mean_sqrt if mean_sqrt > 0 else 0.0,
        "rms_to_peak": rms / peak if peak > 0 else 0.0,
    }


def _spectral(x: np.ndarray, sample_rate: float) -> dict[str, float]:
    """Where the energy sits, and how spread out it is."""
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    frequencies = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    total = np.sum(spectrum)
    if total <= 0:
        return {"spectral_centroid_norm": 0.0, "spectral_entropy": 0.0}

    probability = spectrum / total
    nonzero = probability[probability > 0]
    return {
        "spectral_centroid_norm": float(np.sum(frequencies * probability) / (0.5 * sample_rate)),
        "spectral_entropy": float(-np.sum(nonzero * np.log(nonzero)) / np.log(len(probability))),
    }


def extract_one(
    x: np.ndarray, sample_rate: float, shaft_hz: float, bearing: Bearing
) -> np.ndarray:
    """Feature vector for one vibration record."""
    band = kurtogram_band(x, sample_rate)
    frequencies, spectrum = envelope_spectrum(x, sample_rate, band)

    # Normalise against the spectrum's own background, so a feature means "how
    # far this line stands above the surrounding noise" rather than "how hard
    # the accelerometer was driven".
    #
    # The background is the RMS of the envelope spectrum across the diagnostic
    # band, not its median. On a healthy bearing the envelope is almost flat,
    # its spectrum is near zero nearly everywhere, and the median collapses
    # towards zero -- which would send every ratio to infinity for exactly the
    # class that should score lowest.
    diagnostic = (frequencies > 0) & (frequencies < 15.0 * shaft_hz)
    floor = float(np.sqrt(np.mean(spectrum[diagnostic] ** 2))) + 1e-12
    resolution = float(frequencies[1] - frequencies[0])
    tolerance = max(2.0 * resolution, 0.02 * shaft_hz)

    sites = fault_frequencies(bearing, shaft_hz)
    features: dict[str, float] = {}
    for site in FAULT_SITES:
        for harmonic in HARMONICS:
            peak = band_energy(frequencies, spectrum, harmonic * sites[site], tolerance)
            features[f"env_{site}_h{harmonic}"] = peak / floor

    cage_hz = cage_frequency(bearing, shaft_hz)
    inner_sidebands = [
        band_energy(frequencies, spectrum, sites["inner_race"] + sign * shaft_hz, tolerance)
        for sign in (-1, 1)
    ]
    ball_sidebands = [
        band_energy(frequencies, spectrum, sites["ball"] + sign * cage_hz, tolerance)
        for sign in (-1, 1)
    ]
    features["sideband_inner_shaft"] = float(np.mean(inner_sidebands)) / floor
    features["sideband_ball_cage"] = float(np.mean(ball_sidebands)) / floor
    features["env_shaft_h1"] = band_energy(frequencies, spectrum, shaft_hz, tolerance) / floor

    nyquist = 0.5 * sample_rate
    features["band_centre_norm"] = 0.5 * (band.low_hz + band.high_hz) / nyquist
    features["band_width_norm"] = (band.high_hz - band.low_hz) / nyquist
    features["band_kurtosis"] = band.kurtosis

    features.update(_time_domain(x))
    features.update(_spectral(x, sample_rate))

    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)


def extract(
    x: np.ndarray, sample_rate: float, shaft_hz: np.ndarray, bearing: Bearing
) -> np.ndarray:
    """Feature matrix ``(N, F)`` for a batch of records with per-record speeds."""
    return np.stack(
        [
            extract_one(record, sample_rate, float(hz), bearing)
            for record, hz in zip(x, shaft_hz, strict=True)
        ]
    )
