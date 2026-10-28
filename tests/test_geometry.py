"""Bearing geometry.

The multipliers are checked against the values Case Western Reserve University
publishes for the SKF 6205-2RS JEM drive-end bearing. Reproducing them is
strong evidence that the geometry code is right, and it is the link between
this simulator and the real dataset the loader can be pointed at.
"""

import numpy as np
import pytest

from bfd.geometry import (
    SKF_6205,
    Bearing,
    ball_defect_frequency,
    ball_spin_frequency,
    cage_frequency,
    fault_frequencies,
    inner_race_frequency,
    multipliers,
    outer_race_frequency,
)

# As published with the CWRU bearing dataset, in multiples of shaft speed.
CWRU_PUBLISHED = {
    "inner_race": 5.4152,
    "outer_race": 3.5848,
    "cage": 0.3983,
    "ball": 4.7135,
}


@pytest.mark.parametrize(("name", "expected"), CWRU_PUBLISHED.items())
def test_multipliers_match_the_published_cwru_values(name, expected):
    assert multipliers(SKF_6205)[name] == pytest.approx(expected, abs=5e-4)


def test_frequencies_scale_linearly_with_shaft_speed():
    at_25 = fault_frequencies(SKF_6205, 25.0)
    at_50 = fault_frequencies(SKF_6205, 50.0)
    for name, value in at_25.items():
        assert at_50[name] == pytest.approx(2.0 * value)


def test_inner_race_frequency_exceeds_outer_race():
    """BPFI > BPFO always: the inner race carries the extra (1 + d/D) term."""
    assert inner_race_frequency(SKF_6205, 30.0) > outer_race_frequency(SKF_6205, 30.0)


def test_cage_is_the_slowest_frequency():
    frequencies = fault_frequencies(SKF_6205, 30.0)
    assert frequencies["cage"] == min(frequencies.values())


def test_ball_defect_frequency_is_twice_ball_spin():
    """A ball defect strikes a race twice per ball revolution."""
    assert ball_defect_frequency(SKF_6205, 30.0) == pytest.approx(
        2.0 * ball_spin_frequency(SKF_6205, 30.0)
    )


def test_outer_and_inner_sum_to_the_ball_pass_total():
    """BPFO + BPFI = n * fr, because the (d/D) terms cancel."""
    shaft_hz = 29.0
    total = outer_race_frequency(SKF_6205, shaft_hz) + inner_race_frequency(SKF_6205, shaft_hz)
    assert total == pytest.approx(SKF_6205.n_balls * shaft_hz)


def test_contact_angle_reduces_the_geometry_ratio():
    angled = Bearing(n_balls=9, ball_diameter=0.3126, pitch_diameter=1.537, contact_angle_deg=30.0)
    assert angled.ratio == pytest.approx(SKF_6205.ratio * np.cos(np.radians(30.0)))


def test_cage_frequency_is_below_shaft_speed():
    """The cage always turns slower than the shaft."""
    assert cage_frequency(SKF_6205, 30.0) < 30.0
