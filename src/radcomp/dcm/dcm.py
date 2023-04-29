from dataclasses import dataclass
import numpy as np
from typing import Optional

from radcomp.common.utils import nuclei_to_activity, _save_arrays
from radcomp.common.prelayer import Prelayer
from radcomp.dcm.dcm_internal import (
    _include_prelayer_in_branching_frac,
    _solve_dcm,
    _plot_solved_tacs,
    _cumulated_activity,
    _info_xfer,
    _info_growth,
    _valid_dcm_input,
    _prelayer_as_tuple,
)
from radcomp.dcm.dcm_read_toml import _dcm_read_toml


@dataclass
class DetCompModelSol:
    """
    Holds the solution for a deterministic compartment model.

    Provides some convenient functions to inspect the model and its solution.

    NB. Users should not create an instance of this class directly; use the
    functions :func:`solve_dcm` or :func:`solve_dcm_from_toml` to get an instance.

    Parameters
    ----------
    trans_rates : numpy.ndarray
        Transition rates (h-1) of nuclides in layers. Shape (``num_layers``,).
    branching_fracs : numpy.ndarray
        Branching fractions (0 to 1). Shape (``num_layers``, ``num_layers``). Element [i, j] is for layer j to layer i.
    xfer_coeffs : numpy.ndarray
        Transfer coefficients (h-1) between compartments. Shape (``num_layers``, ``num_compartments``, ``num_compartments``). Element [i, j, k] is for compartment k to compartment j in layer i.
    t_eval : numpy.ndarray
        Times (h) at which to solve the model. Must be sorted (ascending).
    prelayer : Prelayer | None
        Input time-activity curves for a nuclide that is able to transition to one or more layers in the model.
    layer_names : list[str] | None
        Names of layers in model. Length ``num_layers``.
    compartment_names : list[str] | None
        Names of compartments in model. Length ``num_compartments``.
    num_layers : int
        Number of layers in model (excluding any prelayer).
    num_compartments: int
        Number of compartments in model.
    nuclei : numpy.ndarray
        Shape (``num_layers``, ``num_compartments``, len(``t_eval``)). The solution.
        Element [i, j, k] is the number of nuclei in layer i, compartment j
        at element k of ``t_eval``.
    """

    trans_rates: np.ndarray
    branching_fracs: np.ndarray
    xfer_coeffs: np.ndarray
    t_eval: np.ndarray
    prelayer: Prelayer | None
    layer_names: list[str] | None
    compartment_names: list[str] | None
    num_layers: int
    num_compartments: int
    nuclei: np.ndarray

    def activity(self) -> np.ndarray:
        """Activities (MBq) at times in ``t_eval``.

        Returns
        -------
        numpy.ndarray
            Shape (``num_layers``, ``num_compartments``, len(``t_eval``)). Element at
            index [i, j, k] is the activtiy (MBq) in layer i, compartment j at element
            k of ``t_eval``.
        """
        return np.array(
            [
                nuclei_to_activity(yl, tr)
                for yl, tr in zip(self.nuclei, self.trans_rates)
            ]
        )

    def cumulated_activity(self) -> np.ndarray:
        """Cumulated activity (MBq h) during ``t_eval``.

        Returns
        -------
        numpy.ndarray
            Shape (``num_layers``, ``num_compartments``). Element at index [i, j] is the
            cumulated activity (MBq h) in layer i, compartment j during ``t_eval``.
        """
        return _cumulated_activity(
            [self.t_eval] * self.num_layers, self.nuclei, self.trans_rates
        )

    def halflife(self) -> np.ndarray:
        """Half-lives (h) of nuclides in layers.

        Stable nuclides are assigned a half-life of numpy.inf.

        Returns
        -------
        numpy.ndarray
            Shape (``num_layers``,). Element at index i is the half-life (h)
            of nuclide in layer i.
        """
        return np.array([np.log(2) / r if r != 0 else np.inf for r in self.trans_rates])

    def plot(self) -> None:
        """Produce plots of time-activity curves or time-nuclei curves."""
        _plot_solved_tacs(
            [self.t_eval] * self.num_layers,
            self.nuclei,
            self.trans_rates,
            layer_names=self._get_layer_names(),
            compartment_names=self._get_compartment_names(),
        )

    def info_xfer(self) -> str:
        """Get information about transfer coefficients between compartments.

        Returns
        -------
        str
            Information.
        """
        return _info_xfer(
            self.xfer_coeffs,
            layer_names=self._get_layer_names(),
            compartment_names=self._get_compartment_names(),
        )

    def info_growth(self) -> str:
        """Get information about the growth of nuclides in layers.

        Returns
        -------
        str
            Information.
        """
        _, branching_fracs_prelayer, _ = _prelayer_as_tuple(
            self.prelayer, self.num_layers, self.num_compartments
        )
        branching_frac = _include_prelayer_in_branching_frac(
            self.branching_fracs, branching_fracs_prelayer
        )
        return _info_growth(
            branching_frac, layer_names=self._get_layer_names_incl_prelayer()
        )

    def save_arrays(self, filepath: str) -> None:
        """Save the solution to npz file.

        The saved arrays ``t_eval``, ``nuclei``, and the return of
        :meth:`activity`.
        The function :func:`read_arrays` can be used to read
        an npz file created using this method.

        Parameters
        ----------
        filepath : str
            Filepath of npz file to be created. Must end with ".npz".
        """
        _save_arrays(filepath, self.t_eval, self.nuclei, self.activity())

    def _get_layer_names(self) -> list[str]:
        """Names of layers in model, excluding any prelayer.

        Returns
        -------
        list[str]
            Length ``num_layers``.
        """
        if self.layer_names is None:
            return [f"Nuclide {layer+1}" for layer in range(self.num_layers)]
        else:
            return self.layer_names.copy()

    def _get_layer_names_incl_prelayer(self) -> list[str]:
        """Names of layers in model plus prelayer name prepended.

        If no prelayer, an empty string is prepended.

        Returns
        -------
        list[str]
            Length ``num_layers`` + 1.
        """
        layer_names = self._get_layer_names()
        layer_names[0:0] = [""] if self.prelayer is None else [self.prelayer.name]
        return layer_names

    def _get_compartment_names(self) -> list[str]:
        """
        Returns
        -------
        list[str]
            Length ``num_compartments``.
        """
        if self.compartment_names is None:
            return [f"Compartment {i+1}" for i in range(self.num_compartments)]
        else:
            return self.compartment_names.copy()


def solve_dcm(
    trans_rates: np.ndarray,
    branching_fracs: np.ndarray,
    xfer_coeffs: np.ndarray,
    initial_nuclei: np.ndarray,
    t_eval: np.ndarray,
    prelayer: Optional[Prelayer] = None,
    layer_names: Optional[list[str]] = None,
    compartment_names: Optional[list[str]] = None,
) -> DetCompModelSol:
    """Solve a deterministic compartment model.

    For more information about the model, see
    https://github.com/jakeforster/radcomp/blob/main/README.md.

    Parameters
    ----------
    trans_rates : numpy.ndarray
        Transition rates (h-1) of nuclides in layers. Shape (``num_layers``,).
    branching_fracs : numpy.ndarray
        Branching fractions (0 to 1). Shape (``num_layers``, ``num_layers``). Element [i, j] is for layer j to layer i.
    xfer_coeffs : numpy.ndarray
        Transfer coefficients (h-1) between compartments. Shape (``num_layers``, ``num_compartments``, ``num_compartments``). Element [i, j, k] is for compartment k to compartment j in layer i.
    initial_nuclei : numpy.ndarray
        Number of nuclei in each compartment in each layer at first element of ``t_eval``. Shape (``num_layers``, ``num_compartments``). Element [i, j] is for layer i, compartment j.
    t_eval : numpy.ndarray
        Times (h) at which to solve the model. Must be sorted (ascending).
    prelayer : Optional[Prelayer]
        Input time-activity curves for a nuclide that is able to transition to one or more layers in the model.
    layer_names : Optional[list[str]]
        Names of layers in model.
    compartment_names : Optional[list[str]]
        Names of compartments in model.

    Returns
    -------
    DetCompModelSol
        Solution for the model.
    """

    _valid_dcm_input(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
        prelayer,
        layer_names,
        compartment_names,
    )
    num_layers, num_compartments, _ = xfer_coeffs.shape
    t_span = (t_eval.min(), t_eval.max())
    _, nuclei = _solve_dcm(
        t_span,
        initial_nuclei,
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        t_eval=t_eval,
        prelayer_as_tuple=_prelayer_as_tuple(prelayer, num_layers, num_compartments),
    )
    return DetCompModelSol(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        t_eval,
        prelayer,
        layer_names,
        compartment_names,
        num_layers,
        num_compartments,
        np.array(nuclei),
    )


def solve_dcm_from_toml(
    filepath: str,
    t_eval: np.ndarray,
    prelayer: Optional[Prelayer] = None,
) -> DetCompModelSol:
    """Solve a deterministic compartment model from a
    TOML configuration file (except for a possible prelayer).

    For more information about the model or how to format the
    TOML file, see https://github.com/jakeforster/radcomp/blob/main/README.md.

    Parameters
    ----------
    filepath : str
        Filepath to TOML configuration file.
    t_eval : np.ndarray
        Times (h) at which to solve the model. Must be sorted (ascending).
    prelayer : Optional[Prelayer]
        Input time-activity curves for a nuclide that is able to transition to one or more layers in the model.

    Returns
    -------
    DetCompModelSol
        Solution for the model.
    """
    (
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        layer_names,
        compartment_names,
    ) = _dcm_read_toml(filepath)
    return solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
        prelayer=prelayer,
        layer_names=layer_names,
        compartment_names=compartment_names,
    )
