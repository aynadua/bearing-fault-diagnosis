"""Dataset assembly and the order-domain transform."""

import numpy as np
import pytest

from bfd.config import Config
from bfd.dataset import VibrationDataset, generate, to_order_domain
from bfd.simulate import CLASSES, CONDITIONS


@pytest.fixture
def tiny_config():
    cfg = Config()
    cfg.data.per_class = {"train": 6, "test": 4}
    cfg.data.seeds = {"train": 1, "test": 2}
    return cfg


def test_split_is_balanced_over_fault_classes(tiny_config):
    data = generate(tiny_config, "train", "B")
    assert len(data) == len(CLASSES) * 6
    assert np.array_equal(np.bincount(data.y), np.full(len(CLASSES), 6))


def test_records_carry_the_condition_speed(tiny_config):
    data = generate(tiny_config, "train", "C")
    expected = CONDITIONS["C"].shaft_hz
    assert data.condition == "C"
    # Within the simulated tachometer wobble.
    assert np.allclose(data.shaft_hz, expected, rtol=0.03)


def test_shapes_and_dtypes(tiny_config):
    data = generate(tiny_config, "train", "A")
    assert data.x.shape == (len(data), tiny_config.rig.n_samples)
    assert data.x.dtype == np.float32
    assert data.y.dtype == np.int64


def test_conditions_produce_different_data(tiny_config):
    a = generate(tiny_config, "train", "A")
    d = generate(tiny_config, "train", "D")
    assert not np.array_equal(a.x, d.x)


def test_train_and_test_splits_do_not_overlap(tiny_config):
    train = generate(tiny_config, "train", "A")
    test = generate(tiny_config, "test", "A")
    train_rows = {row.tobytes() for row in train.x}
    assert not any(row.tobytes() in train_rows for row in test.x)


def test_generation_is_reproducible(tiny_config):
    assert np.array_equal(
        generate(tiny_config, "train", "B").x, generate(tiny_config, "train", "B").x
    )


def test_unknown_split_and_condition_are_rejected(tiny_config):
    with pytest.raises(KeyError):
        generate(tiny_config, "holdout", "A")
    with pytest.raises(KeyError):
        generate(tiny_config, "train", "Z")


@pytest.mark.parametrize("condition", ["A", "B", "C", "D"])
def test_order_domain_has_a_fixed_length_at_every_speed(tiny_config, condition):
    """The point of the transform: one row covers the same angle at any speed."""
    data = generate(tiny_config, "train", condition)
    tracked = to_order_domain(data, tiny_config)
    expected = tiny_config.rig.samples_per_rev * tiny_config.rig.order_revolutions
    assert tracked.shape == (len(data), expected)
    # Nothing was left as zero padding -- the records really are long enough.
    assert np.all(tracked[:, -1] != 0)


def test_round_trip_through_disk(tiny_config, tmp_path):
    data = generate(tiny_config, "train", "A")
    data.save(tmp_path / "x.npz")
    loaded = VibrationDataset.load(tmp_path / "x.npz")
    assert np.array_equal(loaded.x, data.x)
    assert np.array_equal(loaded.y, data.y)
    assert loaded.condition == data.condition
