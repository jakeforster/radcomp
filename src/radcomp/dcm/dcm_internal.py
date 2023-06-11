import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from collections.abc import Callable
from typing import Optional

from radcomp.common.utils import nuclei_to_activity
from radcomp.common.prelayer import Prelayer
from radcomp.common.voiding import Voiding, _ordered_voids_in_layer


def _prelayer_as_tuple(
    prelayer: Prelayer | None,
    num_layers: int,
    num_compartments: int,
) -> tuple[float, np.ndarray, list[Callable[[float], float]]]:
    """Produce suitable input to function radcomp.dcm.dcm_internal._solve_dcm.

    Parameters
    ----------
    prelayer : None | Prelayer
    num_layers : int
    num_compartments : int

    Returns
    -------
    tuple[float, numpy.ndarray, list[Callable[[float], float]]]
        + Transition rate (h-1) for nuclide in prelayer.
        + Branching fractions (0 to 1). Shape (`num_layers`,).
          `branching_fracs`[i] is for prelayer to layer i in model.
        + Number of prelayer nuclei as a function of time (h) for
          each compartment in the model. Length `num_compartments`.
    """
    if prelayer is None:
        return (
            0,
            np.zeros(num_layers),
            [lambda _: 0] * num_compartments,
        )
    else:
        return (
            prelayer.trans_rate,
            prelayer.branching_fracs,
            prelayer.nuclei_funcs(),
        )


def _valid_dcm_input(
    trans_rates: np.ndarray,
    branching_fracs: np.ndarray,
    xfer_coeffs: np.ndarray,
    initial_nuclei: np.ndarray,
    t_eval: np.ndarray,
    prelayer: Optional[Prelayer] = None,
    layer_names: Optional[list[str]] = None,
    compartment_names: Optional[list[str]] = None,
) -> None:
    """Assert statements to check validity of input parameters
    for deterministic compartment model.

    Parameters
    ----------
    trans_rates : numpy.ndarray
        Transition rates (h-1) of nuclides in layers. Shape (num_layers,).
    branching_fracs : numpy.ndarray
        Branching fractions (0 to 1). Shape (num_layers, num_layers). `branching_fracs`[i,j] is for layer j to layer i.
    xfer_coeffs : numpy.ndarray
        Transfer coefficients (h-1) between compartments. Shape (num_layers, num_compartments, num_compartments). `xfer_coeffs`[i,j,k] is for compartment k to compartment j in layer i.
    initial_nuclei : numpy.ndarray
        Number of nuclei in each compartment in each layer at `t_eval[0]`. Shape (num_layers, num_compartments). `initial_nuclei`[i,j] is for layer i, compartment j.
    t_eval : numpy.ndarray
        Times (h) at which to solve the model. Must be sorted (ascending).
    prelayer : Optional[radcomp.Prelayer]
        Input time-activity curves for a nuclide that is able to transition to one or more layers in the model.
    layer_names : Optional[list[str]]
        Names of layers in model.
    compartment_names : Optional[list[str]]
        Names of compartments in model.
    """

    # size checks
    a, b, c = xfer_coeffs.shape
    assert b == c
    num_layers = a
    num_compartments = b
    assert trans_rates.shape == (num_layers,)
    assert branching_fracs.shape == (num_layers, num_layers)
    assert initial_nuclei.shape == (num_layers, num_compartments)
    if prelayer is not None:
        assert prelayer.branching_fracs.shape == (num_layers,)
        assert len(prelayer.activity_funcs) == num_compartments
    if layer_names is not None:
        assert len(layer_names) == num_layers
    if compartment_names is not None:
        assert len(compartment_names) == num_compartments

    # some type checks
    if prelayer is not None:
        assert all(callable(yp) for yp in prelayer.activity_funcs)

    # content checks
    assert np.all(xfer_coeffs >= 0)
    assert all(np.all(np.diag(xl) == 0) for xl in xfer_coeffs)

    assert np.all(trans_rates >= 0)

    assert np.all(np.triu(branching_fracs) == 0)
    assert np.all(branching_fracs >= 0)
    assert np.all(np.sum(branching_fracs, axis=0) <= 1)

    assert np.all(initial_nuclei >= 0)

    t_span = (t_eval.min(), t_eval.max())
    if prelayer is not None:
        assert prelayer.trans_rate > 0
        assert np.all(prelayer.branching_fracs >= 0)
        assert np.any(prelayer.branching_fracs != 0)
        assert all(
            np.all(
                act(
                    np.linspace(
                        t_span[0],
                        t_span[1],
                        num=int(np.round(60 * (t_span[1] - t_span[0]) + 1)),
                    )
                )
                >= 0
            )
            for act in prelayer.activity_funcs
        )


def _solve_dcm(
    t_span: tuple[float, float] | list[float],
    initial_nuclei: np.ndarray,
    trans_rates: np.ndarray,
    branching_fracs: np.ndarray,
    xfer_coeffs: np.ndarray,
    t_eval: Optional[np.ndarray] = None,
    prelayer_as_tuple: Optional[
        tuple[float, np.ndarray, list[Callable[[float], float]]]
    ] = None,
    voiding_list: Optional[list[Voiding]] = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Solve the deterministic compartment model.

    Parameters
    ----------
    t_span : tuple[float, float] | list[float]
        2-member sequence. Interval of integration (h).
    initial_nuclei : numpy.ndarray
        Number of nuclei in each compartment in each layer at `t_eval[0]`. Shape (num_layers, num_compartments). `initial_nuclei`[i,j] is for layer i, compartment j.
    trans_rates : numpy.ndarray
        Transition rates (h-1) of nuclides in layers. Shape (num_layers,).
    branching_fracs : numpy.ndarray
        Branching fractions (0 to 1). Shape (num_layers, num_layers). `branching_fracs`[i,j] is for layer j to layer i.
    xfer_coeffs : numpy.ndarray
        Transfer coefficients (h-1) between compartments. Shape (num_layers, num_compartments, num_compartments). `xfer_coeffs`[i,j,k] is for compartment k to compartment j in layer i.
    t_eval : Optional[numpy.ndarray]
        Times (h) at which to solve the model. Must be sorted (ascending). Must be within `t_span`.
    prelayer_as_tuple : Optional[ tuple[float, np.ndarray, list[Callable[[float], float]]] ]
        + Transition rate (h-1) for nuclide in prelayer.
        + Branching fractions (0 to 1). Shape (`num_layers`,).
          `branching_fracs`[i] is for prelayer to layer i in model.
        + Number of prelayer nuclei as a function of time (h) for
          each compartment in the model. Length `num_compartments`.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        + Length num_layers. Element i is the 1D array of
          times (h) at which layer i is solved. If `t_eval`
          is not provided, this is [`t_eval`] * num_layers.
        + Length num_layers. Element i is a 2D array containing
          the solution for layer i. This 2D array has
          shape (num_compartments, len(first_out[i])), where
          first_out is the first element of the return tuple.
    """

    t_layers = []
    nuclei_layers = []
    nuclei_funcs = []

    num_layers, b, c = xfer_coeffs.shape
    assert b == c
    num_compartments = b

    if prelayer_as_tuple is None:
        trans_rate_prelayer = 0
        branching_frac_prelayer = np.zeros(num_layers)
        nuclei_funcs_prelayer = [lambda _: 0] * num_compartments
    else:
        (
            trans_rate_prelayer,
            branching_frac_prelayer,
            nuclei_funcs_prelayer,
        ) = prelayer_as_tuple
    nuclei_funcs.append(nuclei_funcs_prelayer)

    (
        initial_nuclei_new,
        trans_rates_new,
        branching_fracs_new,
        xfer_coeffs_new,
    ) = _include_prelayer(
        initial_nuclei,
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        trans_rate_prelayer,
        branching_frac_prelayer,
    )

    num_layers_new = len(trans_rates_new)

    if voiding_list is None:
        voiding_list = []

    for layer in range(1, num_layers_new):

        layer_voids = _ordered_voids_in_layer(voiding_list, layer - 1)  # prelayer

        t_start = t_span[0]
        initial_nuclei_layer = initial_nuclei_new[layer]

        sol_t_layer = np.array([t_start])  # []
        sol_y_layer = np.reshape(initial_nuclei_layer, (-1, 1))  # [[],[]]
        for void in layer_voids:
            t_end = void.time
            assert t_end <= t_span[1]

            # slice t_eval
            t_eval_interval = (
                None
                if t_eval is None
                else np.append(t_eval[(t_eval >= t_start) & (t_eval < t_end)], [t_end])
            )
            # if None, t_span bounds are always included in the deduced t_eval... I think

            # solve
            sol = solve_ivp(
                _ode_rhs,
                (t_start, t_end),
                initial_nuclei_layer,
                t_eval=t_eval_interval,
                args=(
                    trans_rates_new,
                    branching_fracs_new[layer],
                    xfer_coeffs_new[layer],
                    layer,
                    nuclei_funcs,
                ),
            )

            # record voided nuclei at t_end
            voided_nuclei = sol.y[:, -1] * void.fractions

            # initial conditions for next interval
            initial_nuclei_layer = sol.y[:, -1] - voided_nuclei

            # store soln with modified t_end point
            # already have the initial conditions
            assert sol_t_layer[-1] == sol.t[0]
            assert np.array_equal(sol_y_layer[:, -1], sol.y[:, 0])
            sol_t_layer = np.append(sol_t_layer, sol.t[1:])
            sol_y_layer = np.append(sol_y_layer, sol.y[:, 1:-1], axis=1)
            # modified end point
            sol_y_layer = np.concatenate(
                (sol_y_layer, np.reshape(initial_nuclei_layer, (-1, 1))), axis=1
            )

            # get ready for next interval
            t_start = t_end

        # final interval
        if t_start != t_span[1]:
            t_eval_interval = None if t_eval is None else t_eval[t_eval >= t_start]
            sol = solve_ivp(
                _ode_rhs,
                (t_start, t_span[1]),
                initial_nuclei_layer,
                t_eval=t_eval_interval,
                args=(
                    trans_rates_new,
                    branching_fracs_new[layer],
                    xfer_coeffs_new[layer],
                    layer,
                    nuclei_funcs,
                ),
            )

            # store soln to t_span[1]
            # already have the initial conditions
            print(sol_t_layer)
            print(sol.t)
            exit(1)
            assert sol_t_layer[-1] == sol.t[0]
            assert np.array_equal(sol_y_layer[:, -1], sol.y[:, 0])
            sol_t_layer = np.append(sol_t_layer, sol.t[1:])
            sol_y_layer = np.append(sol_y_layer, sol.y[:, 1:], axis=1)

        t_layers.append(sol_t_layer)
        nuclei_layers.append(sol_y_layer)
        nuclei_funcs.append([interp1d(sol_t_layer, n) for n in sol_y_layer])

    return t_layers, nuclei_layers


def _ode_rhs(
    t: float,
    nuclei: np.ndarray,
    trans_rates: np.ndarray,
    branching_fracs_layer: np.ndarray,
    xfer_coeffs_layer: np.ndarray,
    layer: int,
    nuclei_funcs: list[list[Callable[[float], float]]],
) -> np.ndarray:
    """Right-hand side of the system of ODEs for a layer.

    The func given to scipy.integrate.solve_ivp.

    NB. This function makes no distinction between the layers of the
    compartment model and the prelayer - they are all referred
    to as layers in this docstring. As a result, **num_layers does not
    have its usual meaning.**

    Parameters
    ----------
    t : float
        Part of the calling signature for solve_ivp. Time (h).
    nuclei : np.ndarray
        Part of the calling signature for solve_ivp. Shape (num_compartments,). Number of nuclei in each compartment of layer `layer` at time `t`.
    trans_rates : numpy.ndarray
        Transition rates (h-1) of nuclides in layers. Shape (num_layers,).
    branching_fracs_layer : numpy.ndarray
        Branching fractions (0 to 1). Shape (num_layers,). `branching_fracs_layer`[i] is for layer i to layer `layer`. Zero for i >= `layer`.
    xfer_coeffs_layer : numpy.ndarray
        Transfer coefficients (h-1) between compartments. Shape (num_compartments, num_compartments). `xfer_coeffs`[i,j] is for compartment j to compartment i in layer `layer`.
    layer : int
        Index of layer being solved.
    nuclei_funcs : list[list[Callable[[float], float]]]
        Length `layer`. `nuclei_funcs`[i] is the number of layer i nuclei
        as a function of time (h) for each compartment in model;
        length num_compartments; element at index j is the function for compartment j.

    Returns
    -------
    np.ndarray
        Right-hand side of the system of ODEs for layer `layer`.
    """

    flowin = np.matmul(xfer_coeffs_layer, nuclei)
    flowout = -1 * xfer_coeffs_layer.sum(axis=0) * nuclei
    decay = -1 * trans_rates[layer] * nuclei
    growth = np.dot(
        branching_fracs_layer[:layer] * trans_rates[:layer],
        [[nc(t) for nc in nl] for nl in nuclei_funcs[:layer]],
    )
    return flowin + flowout + decay + growth


def _include_prelayer(
    initial_nuclei: np.ndarray,
    trans_rates: np.ndarray,
    branching_fracs: np.ndarray,
    xfer_coeffs: np.ndarray,
    trans_rate_prelayer: float,
    branching_fracs_prelayer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make parameters for deterministic compartment model
    include the prelayer.

    Parameters
    ----------
    initial_nuclei : numpy.ndarray
        Number of nuclei in each compartment in each layer at `t_eval`[0]. Shape (num_layers, num_compartments). `initial_nuclei`[i,j] is for layer i, compartment j.
    trans_rates : numpy.ndarray
        Transition rates (h-1) of nuclides in layers. Shape (num_layers,).
    branching_fracs : numpy.ndarray
        Branching fractions (0 to 1). Shape (num_layers, num_layers). `branching_fracs`[i,j] is for layer j to layer i.
    xfer_coeffs : numpy.ndarray
        Transfer coefficients (h-1) between compartments. Shape (num_layers, num_compartments, num_compartments). `xfer_coeffs`[i,j,k] is for compartment k to compartment j in layer i.
    trans_rate_prelayer : float
        Transition rate (h-1) for nuclide in prelayer.
    branching_fracs_prelayer : numpy.ndarray
        Branching fractions (0 to 1). Shape (num_layers,).
        `branching_fracs_prelayer`[i] is for prelayer to layer i in model.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        + `initial_nuclei` with 0s for prelayer. Shape (num_layers+1, num_compartment).
        + `trans_rates` with `trans_rate_prelayer` included. Shape (num_layers+1,).
        + `branching_fracs` with `branching_fracs_prelayer` included. Shape (num_layers+1, num_layers+1).
        + `xfer_coeffs` with 0s for prelayer. Shape (num_layers+1, num_compartments, num_compartments).
    """

    trans_rates_new = np.insert(
        trans_rates.astype(float, casting="safe"), 0, trans_rate_prelayer
    )
    branching_fracs_new = _include_prelayer_in_branching_frac(
        branching_fracs, branching_fracs_prelayer
    )
    initial_nuclei_new = np.insert(initial_nuclei, 0, 0, axis=0)
    xfer_coeffs_new = np.insert(xfer_coeffs, 0, 0, axis=0)
    return initial_nuclei_new, trans_rates_new, branching_fracs_new, xfer_coeffs_new


def _include_prelayer_in_branching_frac(
    branching_fracs: np.ndarray, branching_fracs_prelayer: np.ndarray
) -> np.ndarray:
    """Make branching_fracs parameter for deterministic compartment
    model include the prelayer.

    Parameters
    ----------
    branching_fracs : numpy.ndarray
        Branching fractions (0 to 1). Shape (num_layers, num_layers). `branching_fracs`[i,j] is for layer j to layer i.
    branching_fracs_prelayer : numpy.ndarray
        Branching fractions (0 to 1). Shape (num_layers,).
        `branching_fracs_prelayer`[i] is for prelayer to layer i in model.

    Returns
    -------
    np.ndarray
        `branching_fracs` with `branching_fracs_prelayer` included. Shape (num_layers+1, num_layers+1).
    """

    return np.insert(
        np.insert(branching_fracs, 0, 0, axis=0).astype(float, casting="safe"),
        0,
        np.insert(branching_fracs_prelayer, 0, 0),
        axis=1,
    )


def _plot_solved_tacs(
    t_layers: list[np.ndarray],
    nuclei_layers: list[np.ndarray] | np.ndarray,
    trans_rates: np.ndarray,
    layer_names: Optional[list[str]] = None,
    compartment_names: Optional[list[str]] = None,
):
    """Produce plots of time-activity curves or time-nuclei curves.

    Parameters
    ----------
    t_layers : list[numpy.ndarray]
        Length num_layers. `t_layers`[i] is the 1D array of
        times (h) at which layer i is solved.
    nuclei_layers : list[numpy.ndarray] | numpy.ndarray
        Length num_layers. `nuclei_layers`[i] is a 2D array containing
        the solution for layer i. This 2D array has
        shape (num_compartments, len(`t_layers`[i])).
    trans_rates : numpy.ndarray
        Transition rates (h-1) of nuclides in layers. Shape (num_layers,).
    layer_names : Optional[list[str]]
        Names of layers in model. Length num_layers.
    compartment_names : Optional[list[str]]
        Names of compartments in model. Length num_compartments.
    """

    if layer_names is None:
        layer_names = [f"Nuclide {layer+1}" for layer in range(len(trans_rates))]
    if compartment_names is None:
        compartment_names = [f"Compartment {i+1}" for i in range(len(t_layers[0]))]

    for tc, nl, layer_name, trans_rate in zip(
        t_layers, nuclei_layers, layer_names, trans_rates
    ):
        fig, ax = plt.subplots()
        if trans_rate:
            for nc, compartment_name in zip(nl, compartment_names):
                ax.plot(
                    tc,
                    nuclei_to_activity(nc, trans_rate),
                    label=compartment_name,
                )
            ax.set_ylabel("Activity (MBq)")
        else:
            for nc, compartment_name in zip(nl, compartment_names):
                ax.plot(tc, nc, label=compartment_name)
            ax.set_ylabel("Number of nuclei")
        ax.set_xlabel("Time (h)")
        ax.set_title(layer_name)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=tc[0], right=tc[-1])
        plt.legend()
    plt.show()


def _cumulated_activity(
    t_layers: list[np.ndarray],
    nuclei_layers: list[np.ndarray] | np.ndarray,
    trans_rates: np.ndarray,
) -> np.ndarray:
    """Cumulated activity (MBq h).

    Parameters
    ----------
    t_layers : list[numpy.ndarray]
        Length num_layers. `t_layers`[i] is the 1D array of
        times (h) at which layer i is solved.
    nuclei_layers : list[numpy.ndarray] | numpy.ndarray
        Length num_layers. `nuclei_layers`[i] is a 2D array containing
        the solution for layer i. This 2D array has
        shape (num_compartments, len(`t_layers`[i])).
    trans_rates : numpy.ndarray
        Transition rates (h-1) of nuclides in layers. Shape (num_layers,).

    Returns
    -------
    numpy.ndarray
        Shape (num_layers, num_compartments). Element [i,j] is the
        cumulated activity (MBq h) in compartment j during `t_layers`[i].
    """

    num_layers = len(trans_rates)
    assert num_layers == len(t_layers)
    num_compartments = len(nuclei_layers[0])
    ans = np.zeros((num_layers, num_compartments))
    for layer, (tc, nl) in enumerate(zip(t_layers, nuclei_layers)):
        for i, nc in enumerate(nl):
            ans[layer, i] = np.trapz(nuclei_to_activity(nc, trans_rates[layer]), x=tc)
    return ans


def _info_xfer(
    xfer_coeffs: np.ndarray,
    layer_names: Optional[list[str]] = None,
    compartment_names: Optional[list[str]] = None,
) -> str:
    """Get information about transfer coefficients between compartments.

    Parameters
    ----------
    xfer_coeffs : numpy.ndarray
        Transfer coefficients (h-1) between compartments. Shape (num_layers, num_compartments, num_compartments). `xfer_coeffs`[i,j,k] is for compartment k to compartment j in layer i.
    layer_names : Optional[list[str]]
        Names of layers in model. Length num_layers.
    compartment_names : Optional[list[str]]
        Names of compartments in model. Length num_compartments.

    Returns
    -------
    str
        Information.
    """

    num_layers, num_compartments, _ = xfer_coeffs.shape
    if layer_names is None:
        layer_names = [f"Nuclide {layer+1}" for layer in range(num_layers)]
    if compartment_names is None:
        compartment_names = [f"Compartment {i+1}" for i in range(num_compartments)]

    out = ""
    for layer in range(num_layers):
        out += f"Transfer of {layer_names[layer]}:\n"
        for compartment_i in range(num_compartments):
            for compartment_j in range(num_compartments):
                tc = xfer_coeffs[layer, compartment_i, compartment_j]
                if tc:
                    out += f"\N{BULLET} {compartment_names[compartment_j]} \N{RIGHTWARDS ARROW} {compartment_names[compartment_i]}: {tc} /h\n"
    return out.removesuffix("\n")


def _info_growth(
    branching_fracs: np.ndarray, layer_names: Optional[list[str]] = None
) -> str:
    """Get information about the growth of nuclides in layers.

    NB. This function makes no distinction between the layers of the
    compartment model and the prelayer - they are all referred
    to as layers in this docstring. As a result, **num_layers does not
    have its usual meaning.**

    Parameters
    ----------
    branching_fracs : numpy.ndarray
        Branching fractions (0 to 1). Shape (num_layers, num_layers). `branching_fracs`[i,j] is for layer j to layer i.
    layer_names : Optional[list[str]]
        Names of layers in model. Length num_layers.

    Returns
    -------
    str
        Information.
    """

    a, b = branching_fracs.shape
    assert a == b
    num_layers = a
    if layer_names is None:
        layer_names = [f"Nuclide {layer+1}" for layer in range(num_layers)]
    else:
        assert len(layer_names) == num_layers

    out = ""
    for layer in range(1, num_layers):
        if np.sum(branching_fracs[layer]) == 0:
            continue
        layer_name = layer_names[layer]
        out += f"{layer_name} growth from physical decay:\n"
        for parent_name, bf in zip(layer_names[:layer], branching_fracs[layer, :layer]):
            if bf != 0:
                out += f"\N{BULLET} {parent_name} \N{RIGHTWARDS ARROW} {layer_name} with branching fraction {bf}\n"
    return out.removesuffix("\n")
