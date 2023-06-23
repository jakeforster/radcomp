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
    voiding_rules_index: int
    voiding_rule_times_index: int

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
    # TODO add event number to remember order
    # voiding_rule index
    # time
    voiding_events = [
        _VoidingEvent(time, rule.fractions[layer], i, j)
        for i, rule in enumerate(voiding_rules)
        for j, time in enumerate(rule.times)
        if any(rule.fractions[layer] != 0)
    ]
    return sorted(voiding_events, key=lambda event: event.time)


def _put_it_back_together():
    # voided_nuclei has shape len(time_ordered_voids_for_layer), num_compartments
    # TODO inverse of above, for scoring voided_nuclei and voided_activity()
    # time_ordered_voids_for_layer
    # voided_nuclei has length n_compartments
    # x: list[np.ndarray],
    # where x has length voiding_rules
    # array has shape len(times) for voiding_rule, num_layers, num_compartments
    # to initialise x requires voiding_rules
    # x has to be initialised in _solve_dcm
    # partially fill x in _solve_dcm_layer:
    # when i get voided_nuclei it is x[voiding_event.voiding_rules_index][voiding_event.voiding_rule_times_index, layer, :]
    pass
