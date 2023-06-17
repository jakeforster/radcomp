from dataclasses import dataclass
import numpy as np


@dataclass
class VoidingRule:
    """Voiding of compartments.

    Parameters
    ----------
    times : numpy.ndarray
        Void times (h).
    fractions : numpy.ndarray
        Fraction (0 to 1) in each layer-compartment voided at times in `times`. Shape (num_layers, num_compartments).
    """

    times: np.ndarray
    fractions: np.ndarray

    def __post_init__(self):
        self.times = np.array(self.times)
        self.fractions = np.array(self.fractions)

    def __eq__(self, other):
        if not isinstance(other, VoidingRule):
            return NotImplemented
        return np.array_equal(self.times, other.times) and np.array_equal(
            self.fractions, other.fractions
        )


@dataclass
class _VoidingEvent:
    """A voiding of compartments in a layer.

    Parameters
    ----------
    time : float
        Void time (h).
    fractions : numpy.ndarray
        Fraction (0 to 1) in each compartment of layer voided at `time`. Shape (num_compartments,).
    """

    time: float
    fractions: np.ndarray

    def __eq__(self, other):
        if not isinstance(other, _VoidingEvent):
            return NotImplemented
        return self.time == other.time and np.array_equal(
            self.fractions, other.fractions
        )


def _create_time_ordered_voids_for_layer(
    voiding_rules: list[VoidingRule], layer: int
) -> list[_VoidingEvent]:
    """Voids in a layer ordered by occurence."""
    voiding_events = [
        _VoidingEvent(time, rule.fractions[layer])
        for rule in voiding_rules
        for time in rule.times
        if any(rule.fractions[layer] != 0)
    ]
    return sorted(voiding_events, key=lambda event: event.time)
