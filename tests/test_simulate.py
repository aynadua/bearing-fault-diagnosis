"""Simulator invariants, checked against what bearing physics requires."""

import numpy as np
import pytest

from bfd.envelope import kurtosis
from bfd.geometry import SKF_6205
from bfd.simulate import CLASSES, CONDITIONS, RigParams, impulse_response, simulate


@pytest.fixture
def rig():
    return RigParams(n_samples=4096)


@pytest.mark.parametrize("fault", CLASSES)
def test_output_shape_and_normalisation(fault, rig):
    signal = simulate(fault, CONDITIONS["B"], SKF_6205, rig, np.random.default_rng(0))
    assert signal.shape == (rig.n_samples,)
    assert np.isclose(np.std(signal), 1.0, atol=1e-6)
    assert np.all(np.isfinite(signal))


def test_unknown_fault_is_rejected(rig):
    with pytest.raises(ValueError, match="unknown fault"):
        simulate("cracked_cage", CONDITIONS["B"], SKF_6205, rig, np.random.default_rng(0))


@pytest.mark.parametrize("fault", ["outer_race", "inner_race", "ball"])
def test_faults_are_more_impulsive_than_a_healthy_bearing(fault, rig):
    """Kurtosis is the classic health indicator: faults ring, healthy bearings hum."""
    rng = np.random.default_rng(1)
    healthy = np.mean(
        [kurtosis(simulate("normal", CONDITIONS["B"], SKF_6205, rig, rng)) for _ in range(8)]
    )
    faulty = np.mean(
        [kurtosis(simulate(fault, CONDITIONS["B"], SKF_6205, rig, rng)) for _ in range(8)]
    )
    assert faulty > healthy + 1.0


def test_healthy_signal_is_not_impulsive(rig):
    """A sum of sinusoids plus noise has kurtosis below the Gaussian value of 3."""
    rng = np.random.default_rng(2)
    scores = [kurtosis(simulate("normal", CONDITIONS["B"], SKF_6205, rig, rng)) for _ in range(8)]
    assert np.mean(scores) < 3.0


def test_impulse_response_decays(rig):
    response = impulse_response(2048, 12000.0, 3500.0, 0.05)
    first, last = np.abs(response[:128]).max(), np.abs(response[-128:]).max()
    assert last < 0.01 * first


def test_impulse_response_rings_at_the_requested_resonance():
    n, sample_rate, resonance = 4096, 12000.0, 3500.0
    response = impulse_response(n, sample_rate, resonance, 0.01)
    spectrum = np.abs(np.fft.rfft(response))
    peak_hz = np.fft.rfftfreq(n, d=1.0 / sample_rate)[np.argmax(spectrum)]
    assert peak_hz == pytest.approx(resonance, rel=0.05)


def test_simulation_is_reproducible(rig):
    a = simulate("ball", CONDITIONS["C"], SKF_6205, rig, np.random.default_rng(9))
    b = simulate("ball", CONDITIONS["C"], SKF_6205, rig, np.random.default_rng(9))
    assert np.array_equal(a, b)


def test_conditions_cover_a_wide_speed_range():
    speeds = [c.shaft_rpm for c in CONDITIONS.values()]
    assert max(speeds) / min(speeds) >= 2.0, "the domain shift must be large enough to matter"
