from dataclasses import dataclass
import numpy as np


@dataclass
class VoidingRule:
    """A rule to specify how nuclei will be voided from compartments.

    Parameters
    ----------
    times : numpy.ndarray
        Void times (h).
    fractions : numpy.ndarray
        The fractions (0 to 1) of nuclei in each compartment and layer
        to be voided at times in ``times``. Shape (``num_layers``, ``num_compartments``).
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
    """Instantaneous voiding of compartments in a layer.

    Parameters
    ----------
    time : float
        Void time (h).
    fractions : numpy.ndarray
        Fraction (0 to 1) in each compartment of layer voided at `time`. Shape (num_compartments,).
    voiding_rules_index : int
        Index of associated voiding rule in list of voiding rules.
    voiding_rule_times_index : int
        Index of associated time in list of times in voiding rule.
    """

    time: float
    fractions: np.ndarray
    voiding_rules_index: int
    voiding_rule_times_index: int

    def __eq__(self, other):
        if not isinstance(other, _VoidingEvent):
            return NotImplemented
        return (
            self.time == other.time
            and np.array_equal(self.fractions, other.fractions)
            and self.voiding_rules_index == other.voiding_rules_index
            and self.voiding_rule_times_index == other.voiding_rule_times_index
        )


def _create_time_ordered_voids_for_layer(
    voiding_rules: list[VoidingRule], layer: int
) -> list[_VoidingEvent]:
    """Create time-ordered voids for layer out of voiding rules.

    Parameters
    ----------
    voiding_rules : list[VoidingRule]
    layer : int

    Returns
    -------
    list[_VoidingEvent]
    """
    voiding_events = [
        _VoidingEvent(time, rule.fractions[layer], i, j)
        for i, rule in enumerate(voiding_rules)
        for j, time in enumerate(rule.times)
        if any(rule.fractions[layer] != 0)
    ]
    return sorted(voiding_events, key=lambda event: event.time)
