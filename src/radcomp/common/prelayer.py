from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
from typing import TypeVar

from radcomp.common.utils import activity_to_nuclei


T = TypeVar("T", float, np.ndarray)


@dataclass
class Prelayer:
    """Input time-activity curves for a nuclide that is
    able to transition to one or more layers in a model.

    Parameters
    ----------
    trans_rate : float
        Transition rate (h-1) for nuclide in prelayer.
    branching_fracs : numpy.ndarray
        Branching fractions (0 to 1). Shape (num_layers,). Element i is for prelayer to layer i in model.
    activity_funcs : list[Callable[[T], T]]
        Prelayer activity (MBq) as a function of time (h) for each compartment in model. Length num_compartments. Element i is the function for compartment i.
    name : str
        Name for prelayer nuclide. Defaults to "prelayer".
    """

    trans_rate: float
    branching_fracs: np.ndarray
    activity_funcs: list[Callable[[T], T]]
    name: str = "prelayer"

    def nuclei_funcs(self) -> list[Callable[[T], T]]:
        """Time-nuclei curves.

        Returns
        -------
        list[Callable[[T], T]]
            Number of prelayer nuclei as a function of time (h) for each compartment in model.
            Length num_compartments. Element at index i is the function for compartment i.
        """
        return list(
            map(
                lambda a_cmpt: lambda t: activity_to_nuclei(a_cmpt(t), self.trans_rate),
                self.activity_funcs,
            )
        )
