"""Bearing geometry and the fault frequencies it implies.

The default geometry is the SKF 6205-2RS JEM deep-groove ball bearing used on
the drive end of the Case Western Reserve University test rig, so the
characteristic-frequency multipliers computed here reproduce the ones CWRU
publishes with that dataset. That matters for two reasons: it is a real
bearing rather than invented numbers, and it means a model trained on this
simulator is pointed at the same frequencies the real recordings contain.

For a bearing with ``n`` rolling elements, ball diameter ``d``, pitch diameter
``D`` and contact angle ``phi``, with the shaft turning at ``fr`` Hz:

    FTF  = (fr / 2) * (1 - (d/D) cos phi)              cage rotation
    BPFO = (n * fr / 2) * (1 - (d/D) cos phi)          ball passes a fixed outer-race point
    BPFI = (n * fr / 2) * (1 + (d/D) cos phi)          ball passes a point on the inner race
    BSF  = (D * fr / (2 * d)) * (1 - ((d/D) cos phi)^2) ball spin about its own axis

A defect *on a ball* strikes a race twice per ball revolution, so the
frequency seen in a spectrum is ``2 * BSF``. Tables that list a single "ball
fault frequency" -- CWRU's 4.7135 multiplier among them -- are quoting that
doubled value. :func:`ball_defect_frequency` returns it explicitly rather than
leaving the factor of two to be discovered later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bearing:
    """Rolling-element bearing geometry. Lengths may be in any consistent unit."""

    n_balls: int
    ball_diameter: float
    pitch_diameter: float
    contact_angle_deg: float = 0.0
    name: str = "unnamed"

    @property
    def ratio(self) -> float:
        """``(d / D) * cos(phi)``, the term every fault frequency is built from."""
        return (self.ball_diameter / self.pitch_diameter) * np.cos(
            np.radians(self.contact_angle_deg)
        )


# Drive-end bearing of the CWRU rig. Dimensions in inches, as published.
SKF_6205 = Bearing(
    n_balls=9,
    ball_diameter=0.3126,
    pitch_diameter=1.537,
    contact_angle_deg=0.0,
    name="SKF 6205-2RS JEM",
)


def cage_frequency(bearing: Bearing, shaft_hz: float) -> float:
    """Fundamental train (cage) frequency, Hz."""
    return 0.5 * shaft_hz * (1.0 - bearing.ratio)


def outer_race_frequency(bearing: Bearing, shaft_hz: float) -> float:
    """Ball pass frequency, outer race (BPFO), Hz."""
    return 0.5 * bearing.n_balls * shaft_hz * (1.0 - bearing.ratio)


def inner_race_frequency(bearing: Bearing, shaft_hz: float) -> float:
    """Ball pass frequency, inner race (BPFI), Hz."""
    return 0.5 * bearing.n_balls * shaft_hz * (1.0 + bearing.ratio)


def ball_spin_frequency(bearing: Bearing, shaft_hz: float) -> float:
    """Ball spin frequency (BSF), Hz -- one rotation of a ball about its axis."""
    return 0.5 * (bearing.pitch_diameter / bearing.ball_diameter) * shaft_hz * (
        1.0 - bearing.ratio**2
    )


def ball_defect_frequency(bearing: Bearing, shaft_hz: float) -> float:
    """Frequency at which a ball defect strikes a race, Hz (``2 * BSF``)."""
    return 2.0 * ball_spin_frequency(bearing, shaft_hz)


def fault_frequencies(bearing: Bearing, shaft_hz: float) -> dict[str, float]:
    """Every characteristic frequency, keyed by fault class name."""
    return {
        "cage": cage_frequency(bearing, shaft_hz),
        "outer_race": outer_race_frequency(bearing, shaft_hz),
        "inner_race": inner_race_frequency(bearing, shaft_hz),
        "ball": ball_defect_frequency(bearing, shaft_hz),
    }


def multipliers(bearing: Bearing) -> dict[str, float]:
    """Fault frequencies expressed as multiples of shaft speed."""
    return {name: value for name, value in fault_frequencies(bearing, 1.0).items()}
