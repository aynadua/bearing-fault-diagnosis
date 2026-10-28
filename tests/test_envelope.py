"""Envelope analysis and order tracking.

:func:`test_envelope_spectrum_peaks_at_the_fault_frequency` is the one that
ties the simulator and the diagnostic together: if either the physics model or
the demodulation is wrong, the peak lands somewhere else.
"""

import numpy as np
import pytest

from bfd.envelope import (
    band_energy,
    bandpass,
    envelope_spectrum,
    kurtogram_band,
    kurtosis,
    order_track,
)
from bfd.geometry import SKF_6205, fault_frequencies
from bfd.simulate import CONDITIONS, RigParams, simulate

RIG = RigParams(n_samples=4096)


def test_kurtosis_of_gaussian_noise_is_about_three():
    noise = np.random.default_rng(0).standard_normal(200_000)
    assert kurtosis(noise) == pytest.approx(3.0, abs=0.1)


def test_kurtosis_of_a_sine_is_one_and_a_half():
    sine = np.sin(np.linspace(0, 400 * np.pi, 100_000))
    assert kurtosis(sine) == pytest.approx(1.5, abs=0.01)


def test_bandpass_removes_out_of_band_energy():
    sample_rate = 12000.0
    t = np.arange(4096) / sample_rate
    signal = np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 4000 * t)
    filtered = bandpass(signal, sample_rate, 3000.0, 5000.0)

    spectrum = np.abs(np.fft.rfft(filtered))
    frequencies = np.fft.rfftfreq(len(filtered), d=1.0 / sample_rate)
    assert spectrum[np.argmin(np.abs(frequencies - 200))] < 0.05 * spectrum.max()


def test_kurtogram_band_is_more_impulsive_than_the_whole_signal():
    """The band it picks must beat the unfiltered signal on kurtosis.

    Note it does not always centre on the structural resonance. Impacts are
    broadband, and the selector maximises impulsiveness relative to whatever
    else is in the band -- so it tends to avoid the low-frequency region where
    the shaft harmonics live, even when that means sitting above the resonance.
    """
    rng = np.random.default_rng(3)
    for _ in range(5):
        signal = simulate("outer_race", CONDITIONS["B"], SKF_6205, RIG, rng)
        band = kurtogram_band(signal, RIG.sample_rate)
        assert band.kurtosis >= kurtosis(signal)
        assert band.low_hz < band.high_hz <= 0.5 * RIG.sample_rate


def test_kurtogram_band_sharpens_the_fault_line():
    """The practical justification: demodulating the chosen band beats not choosing."""
    rng = np.random.default_rng(7)
    operating = CONDITIONS["B"]
    expected = fault_frequencies(SKF_6205, operating.shaft_hz)["outer_race"]

    with_band, without_band = [], []
    for _ in range(8):
        signal = simulate("outer_race", operating, SKF_6205, RIG, rng)
        for band, bucket in (
            (kurtogram_band(signal, RIG.sample_rate), with_band),
            (None, without_band),
        ):
            frequencies, spectrum = envelope_spectrum(signal, RIG.sample_rate, band)
            search = (frequencies > 0.5 * operating.shaft_hz) & (frequencies < 400)
            peak = np.max(spectrum[np.abs(frequencies - expected) <= 8.0])
            bucket.append(peak / (np.sqrt(np.mean(spectrum[search] ** 2)) + 1e-12))

    assert np.mean(with_band) > np.mean(without_band)


@pytest.mark.parametrize("fault", ["outer_race", "inner_race", "ball"])
@pytest.mark.parametrize("condition", ["A", "B", "C", "D"])
def test_envelope_identifies_the_correct_fault_site(fault, condition):
    """The diagnostic criterion the feature extractor actually uses.

    Not "the tallest peak in the spectrum is the fault" -- at high load the
    tallest peak is often the shaft line. The question a diagnosis asks is
    narrower: of the three frequencies this bearing *could* fail at, which one
    carries the most envelope energy?
    """
    operating = CONDITIONS[condition]
    sites = fault_frequencies(SKF_6205, operating.shaft_hz)
    rng = np.random.default_rng(4)

    correct = 0
    trials = 12
    for _ in range(trials):
        signal = simulate(fault, operating, SKF_6205, RIG, rng)
        frequencies, spectrum = envelope_spectrum(
            signal, RIG.sample_rate, kurtogram_band(signal, RIG.sample_rate)
        )
        tolerance = max(2.0 * (frequencies[1] - frequencies[0]), 0.02 * operating.shaft_hz)
        amplitudes = {
            site: band_energy(frequencies, spectrum, sites[site], tolerance)
            for site in ("outer_race", "inner_race", "ball")
        }
        correct += max(amplitudes, key=amplitudes.get) == fault

    assert correct >= trials - 2, f"{correct}/{trials} at {condition}"


def test_order_tracking_puts_the_fault_at_the_same_order_at_any_speed():
    """The premise of the whole project, as an assertion.

    The statistic is the peak of the *averaged* envelope order spectrum, not
    the median of per-record peaks. At these SNRs a single record's peak is
    noisy, and averaging spectra before picking the peak is what a condition-
    monitoring system does anyway. It is also stable: this test gives the same
    four numbers for any random seed.
    """
    samples_per_rev = 512
    expected_order = fault_frequencies(SKF_6205, 1.0)["outer_race"]
    rng = np.random.default_rng(5)

    located = []
    for condition in ("A", "B", "C", "D"):
        operating = CONDITIONS[condition]
        accumulated, orders = None, None
        for _ in range(15):
            signal = simulate("outer_race", operating, SKF_6205, RIG, rng)
            tracked = order_track(signal, RIG.sample_rate, operating.shaft_hz, samples_per_rev)
            orders, spectrum = envelope_spectrum(
                tracked, float(samples_per_rev), kurtogram_band(tracked, float(samples_per_rev))
            )
            normalised = spectrum / spectrum.max()
            accumulated = normalised if accumulated is None else accumulated + normalised

        # Search around BPFO: the shaft line and its harmonics are present here too.
        search = (orders > 2.5) & (orders < 5.0)
        located.append(float(orders[search][np.argmax(accumulated[search])]))

    for order in located:
        assert order == pytest.approx(expected_order, rel=0.03)
    # And they must agree with each other, not merely with theory.
    assert max(located) - min(located) < 0.1


def test_order_tracking_changes_length_with_speed():
    """Faster shaft, more revolutions in the same record, so more angular samples."""
    rng = np.random.default_rng(6)
    record = simulate("normal", CONDITIONS["A"], SKF_6205, RIG, rng)
    tracked_slow = order_track(record, RIG.sample_rate, CONDITIONS["A"].shaft_hz, 512)
    tracked_fast = order_track(record, RIG.sample_rate, CONDITIONS["D"].shaft_hz, 512)
    assert len(tracked_fast) > len(tracked_slow)


def test_order_tracking_refuses_to_alias_away_the_resonance():
    """128 samples/rev at 1500 rpm would put Nyquist below the resonance."""
    rng = np.random.default_rng(7)
    record = simulate("outer_race", CONDITIONS["A"], SKF_6205, RIG, rng)
    with pytest.raises(ValueError, match="below the signal"):
        order_track(record, RIG.sample_rate, CONDITIONS["A"].shaft_hz, 128)

    # Explicitly opting in is allowed, because sometimes the loss is wanted.
    assert len(order_track(record, RIG.sample_rate, 25.0, 128, strict=False)) > 0


def test_order_tracking_covers_the_configured_revolutions_at_the_slowest_speed():
    """The crop in :func:`bfd.dataset.to_order_domain` must not run short."""
    rng = np.random.default_rng(8)
    slowest = min(CONDITIONS.values(), key=lambda c: c.shaft_rpm)
    record = simulate("normal", slowest, SKF_6205, RIG, rng)
    tracked = order_track(record, RIG.sample_rate, slowest.shaft_hz, 512)
    assert len(tracked) >= 8 * 512
