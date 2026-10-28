"""Feature extraction.

The headline test is :func:`test_each_fault_lights_up_its_own_frequency`: the
feature matrix should be diagonal, with every fault class scoring highest on
its own characteristic frequency. That is the diagnostic a vibration engineer
reads, expressed as an assertion.
"""

import numpy as np
import pytest

from bfd.features import FAULT_SITES, FEATURE_NAMES, extract, extract_one
from bfd.geometry import SKF_6205
from bfd.simulate import CLASSES, CONDITIONS, RigParams, simulate

RIG = RigParams(n_samples=4096)
INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _mean_features(fault: str, condition: str, trials: int = 12) -> np.ndarray:
    rng = np.random.default_rng(0)
    operating = CONDITIONS[condition]
    return np.mean(
        [
            extract_one(
                simulate(fault, operating, SKF_6205, RIG, rng),
                RIG.sample_rate,
                operating.shaft_hz,
                SKF_6205,
            )
            for _ in range(trials)
        ],
        axis=0,
    )


@pytest.mark.parametrize("fault", FAULT_SITES)
@pytest.mark.parametrize("condition", ["A", "B", "C", "D"])
def test_each_fault_lights_up_its_own_frequency(fault, condition):
    features = _mean_features(fault, condition)
    scores = {site: features[INDEX[f"env_{site}_h1"]] for site in FAULT_SITES}
    assert max(scores, key=scores.get) == fault, scores


def test_a_healthy_bearing_lights_up_nothing():
    """No characteristic frequency should stand far above the background."""
    features = _mean_features("normal", "B")
    for site in FAULT_SITES:
        assert features[INDEX[f"env_{site}_h1"]] < 3.0


def test_faults_raise_kurtosis_above_healthy():
    healthy = _mean_features("normal", "B")[INDEX["time_kurtosis"]]
    for fault in FAULT_SITES:
        assert _mean_features(fault, "B")[INDEX["time_kurtosis"]] > healthy


def test_feature_vector_shape_and_finiteness():
    rng = np.random.default_rng(1)
    operating = CONDITIONS["B"]
    signal = simulate("ball", operating, SKF_6205, RIG, rng)
    vector = extract_one(signal, RIG.sample_rate, operating.shaft_hz, SKF_6205)
    assert vector.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(vector))


def test_batch_extraction_matches_single_extraction():
    rng = np.random.default_rng(2)
    operating = CONDITIONS["C"]
    signals = np.stack(
        [simulate(f, operating, SKF_6205, RIG, rng) for f in ("normal", "outer_race")]
    )
    speeds = np.array([operating.shaft_hz, operating.shaft_hz])

    batch = extract(signals, RIG.sample_rate, speeds, SKF_6205)
    assert batch.shape == (2, len(FEATURE_NAMES))
    assert np.allclose(
        batch[0], extract_one(signals[0], RIG.sample_rate, operating.shaft_hz, SKF_6205)
    )


def test_features_are_invariant_to_sensor_gain():
    """Accelerometer sensitivity varies between installations; features must not."""
    rng = np.random.default_rng(3)
    operating = CONDITIONS["B"]
    signal = simulate("inner_race", operating, SKF_6205, RIG, rng)

    plain = extract_one(signal, RIG.sample_rate, operating.shaft_hz, SKF_6205)
    amplified = extract_one(signal * 12.5, RIG.sample_rate, operating.shaft_hz, SKF_6205)

    scale_free = [n for n in FEATURE_NAMES if n not in ("band_centre_norm", "band_width_norm")]
    for name in scale_free:
        assert plain[INDEX[name]] == pytest.approx(amplified[INDEX[name]], rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("fault", CLASSES)
def test_extraction_is_deterministic(fault):
    rng_a, rng_b = np.random.default_rng(4), np.random.default_rng(4)
    operating = CONDITIONS["A"]
    a = extract_one(
        simulate(fault, operating, SKF_6205, RIG, rng_a), RIG.sample_rate,
        operating.shaft_hz, SKF_6205,
    )
    b = extract_one(
        simulate(fault, operating, SKF_6205, RIG, rng_b), RIG.sample_rate,
        operating.shaft_hz, SKF_6205,
    )
    assert np.array_equal(a, b)
