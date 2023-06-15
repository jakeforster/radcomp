from dataclasses import dataclass
import numpy as np


@dataclass
class FractionalVoiding:
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
        if not isinstance(other, FractionalVoiding):
            return NotImplemented
        return np.array_equal(self.times, other.times) and np.array_equal(
            self.fractions, other.fractions
        )


@dataclass
class _VoidLayer:
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
        if not isinstance(other, _VoidLayer):
            return NotImplemented
        return self.time == other.time and np.array_equal(
            self.fractions, other.fractions
        )


def _ordered_voids_in_layer(
    voiding: list[FractionalVoiding], layer: int
) -> list[_VoidLayer]:
    """Voids in a layer ordered by occurence."""
    void_layer_list = [
        _VoidLayer(time, fv.fractions[layer])
        for fv in voiding
        for time in fv.times
        if any(fv.fractions[layer] != 0)
    ]
    return sorted(void_layer_list, key=lambda x: x.time)
