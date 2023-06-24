import warnings
import matplotlib
import numpy as np
from radcomp.dcm.dcm_internal import (
    _solve_dcm,
    _ode_rhs,
    _include_prelayer,
    _include_prelayer_in_branching_frac,
    _plot_solved_tacs,
    _cumulated_activity,
    _info_xfer,
    _info_growth,
    _valid_dcm_input,
)
from radcomp.common.utils import activity_to_nuclei, nuclei_to_activity
from radcomp.common.voiding import VoidingRule
import pytest


def test_info_xfer():
    xfer_coeffs = np.array([[[0, 1], [2, 0]], [[0, 0], [4, 0]]])
    layer_names = ["nuclide 1", "nuclide 2"]
    compartment_names = ["cmpt 1", "cmpt 2"]
    out1 = _info_xfer(
        xfer_coeffs, layer_names=layer_names, compartment_names=compartment_names
    )
    out2 = _info_xfer(xfer_coeffs)


def test_info_growth():
    branching_fracs = np.array([[0, 0, 0], [0.1, 0, 0], [0.5, 1, 0]])
    layer_names = ["nuclide 1", "nuclide 2", "nuclide 3"]
    out1 = _info_growth(branching_fracs, layer_names=layer_names)
    out2 = _info_growth(branching_fracs)


def test_plot_solved_tacs():
    t_layers = [np.linspace(1, 3, 10), np.linspace(1, 3, 100)]
    nuclei_layers = [np.ones((2, 10)), np.zeros((2, 100))]
    trans_rates = np.array([1, 2])
    layer_names = ["nuclide 1", "nuclide 2"]
    compartment_names = ["cmpt 1", "cmpt 2"]

    backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # don't show figs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        _plot_solved_tacs(
            t_layers,
            nuclei_layers,
            trans_rates,
            layer_names=layer_names,
            compartment_names=compartment_names,
        )
        _plot_solved_tacs(t_layers, nuclei_layers, trans_rates)

    matplotlib.use(backend)


def test_include_prelayer_in_branching_frac():
    branching_fracs = np.array([[0, 0, 0], [0.3, 0, 0], [0.7, 0.4, 0]])
    branching_fracs_prelayer = np.array([0.1, 0.2, 0.6])
    branching_fracs_new = _include_prelayer_in_branching_frac(
        branching_fracs, branching_fracs_prelayer
    )
    assert np.array_equal(
        branching_fracs_new,
        np.array([[0, 0, 0, 0], [0.1, 0, 0, 0], [0.2, 0.3, 0, 0], [0.6, 0.7, 0.4, 0]]),
    )


def test_include_prelayer():
    initial_nuclei = np.array([[1, 0], [0, 3], [2, 1]])
    trans_rates = np.array([2, 1, 0])
    branching_fracs = np.array([[0, 0, 0], [0.3, 0, 0], [0.7, 0.4, 0]])
    xfer_coeffs_l = np.array([[0, 23], [0.2, 0]])
    xfer_coeffs = np.array([xfer_coeffs_l, xfer_coeffs_l * 3, xfer_coeffs_l * 2])
    trans_rate_prelayer = 4.1
    branching_fracs_prelayer = np.array([0.1, 0.2, 0.6])
    voiding_rules = [
        VoidingRule(np.array([1, 2.1]), np.array([[0, 0.2], [0, 0], [1, 0]])),
        VoidingRule(np.array([3.3]), np.array([[1, 0], [0, 0], [0, 0]])),
    ]
    (
        initial_nuclei_new,
        trans_rates_new,
        branching_fracs_new,
        xfer_coeffs_new,
        voiding_rules_new,
    ) = _include_prelayer(
        initial_nuclei,
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        trans_rate_prelayer,
        branching_fracs_prelayer,
        voiding_rules,
    )
    assert np.array_equal(
        initial_nuclei_new, np.array([[0, 0], [1, 0], [0, 3], [2, 1]])
    )
    assert np.array_equal(trans_rates_new, np.array([4.1, 2, 1, 0]))
    assert np.array_equal(
        branching_fracs_new,
        np.array([[0, 0, 0, 0], [0.1, 0, 0, 0], [0.2, 0.3, 0, 0], [0.6, 0.7, 0.4, 0]]),
    )
    assert np.array_equal(
        xfer_coeffs_new,
        np.array(
            [np.zeros((2, 2)), xfer_coeffs_l, xfer_coeffs_l * 3, xfer_coeffs_l * 2]
        ),
    )
    assert voiding_rules_new[0] == VoidingRule(
        np.array([1, 2.1]), np.array([[0, 0], [0, 0.2], [0, 0], [1, 0]])
    )
    assert voiding_rules_new[1] == VoidingRule(
        np.array([3.3]), np.array([[0, 0], [1, 0], [0, 0], [0, 0]])
    )
    assert voiding_rules[0] == VoidingRule(
        np.array([1, 2.1]), np.array([[0, 0.2], [0, 0], [1, 0]])
    )
    assert voiding_rules[1] == VoidingRule(
        np.array([3.3]), np.array([[1, 0], [0, 0], [0, 0]])
    )


def test_ode_rhs():
    trans_rates = np.array([4.1, 2, 1.1, 0])
    branching_fracs = np.array(
        [[0, 0, 0, 0], [0.1, 0, 0, 0], [0.2, 0.3, 0, 0], [0.6, 0.7, 0.4, 0]]
    )
    xfer_coeffs_l = np.array([[0, 23], [0.2, 0]])
    xfer_coeffs = np.array(
        [np.zeros((2, 2)), xfer_coeffs_l, xfer_coeffs_l * 3, xfer_coeffs_l * 2]
    )

    layer = 2  # NB. zero indexed, so 3rd layer
    nuclei_funcs = [
        [lambda t: 3 * np.exp(-2 * t), lambda _: 1],
        [lambda t: 2 * np.exp(-t), lambda t: np.exp(-t) + 5 * np.exp(-3 * t)],
    ]

    branching_fracs_layer = branching_fracs[layer]
    xfer_coeffs_layer = xfer_coeffs[layer]

    # calling signature
    t = 3.1  # h
    nuclei = np.array([4, 2.1])

    flowin = np.array([23 * 3 * 2.1, 0.2 * 3 * 4])
    flowout = np.array([-0.2 * 3 * 4, -23 * 3 * 2.1])
    decay = np.array([-1.1 * 4, -1.1 * 2.1])
    growth = np.array(
        [
            (3 * np.exp(-2 * 3.1) * 4.1 * 0.2) + (2 * np.exp(-3.1) * 2 * 0.3),
            (1 * 4.1 * 0.2) + (np.exp(-3.1) + 5 * np.exp(-3 * 3.1)) * 2 * 0.3,
        ]
    )

    dnucleidt = _ode_rhs(
        t,
        nuclei,
        trans_rates,
        branching_fracs_layer,
        xfer_coeffs_layer,
        layer,
        nuclei_funcs,
    )

    assert np.array_equal(dnucleidt, flowin + flowout + decay + growth)


def test_solve_dcm():
    """
    Unstable nuclide sometimes decays to stable nuclide
    2 compartments, with unstable nuclide able to transfer from one to another

    Layer 1:
    +--------+           +--------+
    |        |           |        |
    |   C1   |           |   C2   |
    |        | --------> |        |
    +--------+    M21    +--------+

    dN11/dt = - (M121 + lambda1) * N11
    dN12/dt = M121 * N11 - lambda1 * N12
    A11(0) = 30 MBq
    N11(0) = 30 * 1e6 * 60 * 60 / 0.1 = 1.08e12
    N12(0) = 0
    M121 = 0.5 h-1
    lambda1 = 0.1 h-1

    Layer 2:
    +--------+           +--------+
    |        |           |        |
    |   C1   |           |   C2   |
    |        |           |        |
    +--------+           +--------+

    dN21/dt = branching_frac21 * lambda1 * N11(t)
    dN22/dt = branching_frac21 * lambda1 * N12(t)
    N21(0) = 0
    N22(0) = 1e10
    branching_frac21 = 0.3

    dy/dt = 3.24e10 * (1 - exp(-0.5 * t)) * exp(-0.1 * t)
    y(0) = 1e10

    """
    # analytical soln
    N11 = lambda t: (1.08e12) * np.exp(-0.6 * t)
    N12 = lambda t: 1.08e12 * (1 - np.exp(-0.5 * t)) * np.exp(-0.1 * t)
    N21 = lambda t: -(3.24e10 / 0.6) * np.exp(-0.6 * t) + (3.24e10 / 0.6)
    N22 = (
        lambda t: (3.24e10)
        * ((1 / 0.6) * np.exp(-0.6 * t) - (1 / 0.1) * np.exp(-0.1 * t))
        + 1e10
        - (3.24e10) * ((1 / 0.6) - (1 / 0.1))
    )

    t_span = (0, 3)
    t_eval = np.linspace(0, 3, 1000)
    initial_nuclei = np.array([[1.08e12, 0], [0.0, 1e10]])
    trans_rates = np.array([0.1, 0])
    branching_fracs = np.array([[0, 0], [0.3, 0]])
    xfer_coeffs = np.array([np.array([[0, 0], [0.5, 0]]), np.zeros((2, 2))])

    t_layers, nuclei_layers, voided_nuclei = _solve_dcm(
        t_span,
        initial_nuclei,
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        t_eval=t_eval,
    )
    soln = [
        np.array([N11(t_layers[0]), N12(t_layers[0])]),
        np.array([N21(t_layers[1]), N22(t_layers[1])]),
    ]
    rel_error11 = (
        100 * np.abs(nuclei_layers[0][0] - N11(t_layers[0])) / N11(t_layers[0])
    )
    assert np.all(rel_error11 < 0.1)
    rel_error12 = (
        100
        * np.abs(nuclei_layers[0][1][1:] - N12(t_layers[0][1:]))
        / N12(t_layers[0][1:])
    )
    assert np.all(rel_error12 < 0.1)
    rel_error21 = (
        100
        * np.abs(nuclei_layers[1][0][1:] - N21(t_layers[1][1:]))
        / N21(t_layers[1][1:])
    )
    assert np.all(rel_error21 < 0.1)
    rel_error22 = (
        100 * np.abs(nuclei_layers[1][1] - N22(t_layers[1])) / N22(t_layers[1])
    )
    assert np.all(rel_error22 < 0.1)
    assert voided_nuclei == []


def test_solve_dcm_prelayer():
    """
    Same as test_solve_dcm() but insert soln to first layer as prelayer

    Unstable nuclide sometimes decays to stable nuclide
    2 compartments, with unstable nuclide able to transfer from one to another

    Layer 1:
    +--------+           +--------+
    |        |           |        |
    |   C1   |           |   C2   |
    |        | --------> |        |
    +--------+    M21    +--------+

    dN11/dt = - (M121 + lambda1) * N11
    dN12/dt = M121 * N11 - lambda1 * N12
    A11(0) = 30 MBq
    N11(0) = 30 * 1e6 * 60 * 60 / 0.1 = 1.08e12
    N12(0) = 0
    M121 = 0.5 h-1
    lambda1 = 0.1 h-1

    Layer 2:
    +--------+           +--------+
    |        |           |        |
    |   C1   |           |   C2   |
    |        |           |        |
    +--------+           +--------+

    dN21/dt = branching_frac21 * lambda1 * N11(t)
    dN22/dt = branching_frac21 * lambda1 * N12(t)
    N21(0) = 0
    N22(0) = 1e10
    branching_frac21 = 0.3
    """
    # analytical soln
    N11 = lambda t: (1.08e12) * np.exp(-0.6 * t)
    N12 = lambda t: 1.08e12 * (1 - np.exp(-0.5 * t)) * np.exp(-0.1 * t)
    N21 = lambda t: -(3.24e10 / 0.6) * np.exp(-0.6 * t) + (3.24e10 / 0.6)
    N22 = (
        lambda t: (3.24e10)
        * ((1 / 0.6) * np.exp(-0.6 * t) - (1 / 0.1) * np.exp(-0.1 * t))
        + 1e10
        - (3.24e10) * ((1 / 0.6) - (1 / 0.1))
    )
    trans_rate_prelayer = 0.1
    branching_frac_prelayer = np.array([0.3])
    nuclei_funcs_prelayer = [N11, N12]
    prelayer_as_tuple = (
        trans_rate_prelayer,
        branching_frac_prelayer,
        nuclei_funcs_prelayer,
    )

    t_span = (0, 3)
    t_eval = np.linspace(0, 3, 1000)
    initial_nuclei = np.array(
        [
            [0.0, 1e10],
        ]
    )
    trans_rates = np.array([0])
    branching_fracs = np.array([[0]])
    xfer_coeffs = np.array(
        [
            np.zeros((2, 2)),
        ]
    )

    t_layers, nuclei_layers, voided_nuclei = _solve_dcm(
        t_span,
        initial_nuclei,
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        t_eval=t_eval,
        prelayer_as_tuple=prelayer_as_tuple,
    )
    rel_error21 = (
        100
        * np.abs(nuclei_layers[0][0][1:] - N21(t_layers[0][1:]))
        / N21(t_layers[0][1:])
    )
    assert np.all(rel_error21 < 0.1)
    rel_error22 = (
        100 * np.abs(nuclei_layers[0][1] - N22(t_layers[0])) / N22(t_layers[0])
    )
    assert np.all(rel_error22 < 0.12)
    assert voided_nuclei == []


def test_solve_dcm_voiding_rules():
    t_span = (0, 72)
    trans_rates = np.array([np.log(2) / 66, np.log(2) / 6])
    initial_nuclei = np.array([[activity_to_nuclei(1e4, trans_rates[0])], [0]])
    branching_fracs = np.array([[0, 0], [0.89, 0]])
    xfer_coeffs = np.array([[[0]], [[0]]])
    voiding_rule = VoidingRule(np.array([24, 48]), np.array([[0], [1]]))
    voiding_rules = [voiding_rule]
    t_max = (trans_rates[0] - trans_rates[1]) ** (-1) * np.log(
        trans_rates[0] / trans_rates[1]
    )

    t_eval = np.sort(
        np.append(
            np.append(np.linspace(0, 72, 1000), voiding_rule.times + t_max), t_max
        ),
    )

    t_layers, nuclei_layers, voided_nuclei = _solve_dcm(
        t_span,
        initial_nuclei,
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        t_eval=t_eval,
        voiding_rules=voiding_rules,
    )

    activity_99mtc = nuclei_to_activity(nuclei_layers[1][0], trans_rates[1])
    activity_99mo = nuclei_to_activity(nuclei_layers[0][0], trans_rates[0])

    mask = t_layers[1] == t_max
    assert np.isclose(activity_99mtc[mask], activity_99mo[mask] * 0.89, rtol=0.00011)

    mask = t_layers[1] == voiding_rules[0].times[0]
    assert np.isclose(activity_99mtc[mask], 0, rtol=0.0001)

    mask = t_layers[1] == voiding_rules[0].times[0] + t_max
    assert np.isclose(activity_99mtc[mask], activity_99mo[mask] * 0.89, rtol=0.0001)

    mask = t_layers[1] == voiding_rules[0].times[1]
    assert np.isclose(activity_99mtc[mask], 0, rtol=0.0001)

    mask = t_layers[1] == voiding_rules[0].times[1] + t_max
    assert np.isclose(activity_99mtc[mask], activity_99mo[mask] * 0.89, rtol=0.00011)

    # void at 24 h
    activity_voided_24h = (
        0.89
        * trans_rates[1]
        / (trans_rates[1] - trans_rates[0])
        * activity_99mo[t_layers[0] == voiding_rules[0].times[0]]
        * (1 - np.exp(-(trans_rates[1] - trans_rates[0]) * (voiding_rules[0].times[0])))
    )[0]
    nuclei_voided_24h = activity_to_nuclei(activity_voided_24h, trans_rates[1])

    # void at 48 h
    activity_voided_48h = (
        0.89
        * trans_rates[1]
        / (trans_rates[1] - trans_rates[0])
        * activity_99mo[t_layers[0] == voiding_rules[0].times[1]]
        * (
            1
            - np.exp(
                -(trans_rates[1] - trans_rates[0])
                * (voiding_rules[0].times[1] - voiding_rules[0].times[0])
            )
        )
    )[0]
    nuclei_voided_48h = activity_to_nuclei(activity_voided_48h, trans_rates[1])

    assert len(voided_nuclei) == 1
    assert voided_nuclei[0].shape == (2, 2, 1)
    assert np.allclose(
        voided_nuclei[0],
        np.array([[[0], [nuclei_voided_24h]], [[0], [nuclei_voided_48h]]]),
        rtol=8e-5,
    )


def test_solve_dcm_voiding_rules_v2():
    t_span = (0, 72)
    trans_rates = np.array([np.log(2) / 66, np.log(2) / 6])
    initial_nuclei = np.array([[activity_to_nuclei(1e4, trans_rates[0])], [0]])
    branching_fracs = np.array([[0, 0], [0.89, 0]])
    xfer_coeffs = np.array([[[0]], [[0]]])
    voiding_rules = [
        VoidingRule(np.array([48]), np.array([[0], [1]])),
        VoidingRule(np.array([24]), np.array([[0], [1]])),
    ]
    t_max = (trans_rates[0] - trans_rates[1]) ** (-1) * np.log(
        trans_rates[0] / trans_rates[1]
    )

    t_eval = np.sort(
        np.append(
            np.append(np.linspace(0, 72, 1000), np.array([24, 48]) + t_max), t_max
        ),
    )

    t_layers, nuclei_layers, voided_nuclei = _solve_dcm(
        t_span,
        initial_nuclei,
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        t_eval=t_eval,
        voiding_rules=voiding_rules,
    )

    activity_99mtc = nuclei_to_activity(nuclei_layers[1][0], trans_rates[1])
    activity_99mo = nuclei_to_activity(nuclei_layers[0][0], trans_rates[0])

    mask = t_layers[1] == t_max
    assert np.isclose(activity_99mtc[mask], activity_99mo[mask] * 0.89, rtol=0.00011)

    mask = t_layers[1] == 24
    assert np.isclose(activity_99mtc[mask], 0, rtol=0.0001)

    mask = t_layers[1] == 24 + t_max
    assert np.isclose(activity_99mtc[mask], activity_99mo[mask] * 0.89, rtol=0.0001)

    mask = t_layers[1] == 48
    assert np.isclose(activity_99mtc[mask], 0, rtol=0.0001)

    mask = t_layers[1] == 48 + t_max
    assert np.isclose(activity_99mtc[mask], activity_99mo[mask] * 0.89, rtol=0.00011)

    # void at 24 h
    activity_voided_24h = (
        0.89
        * trans_rates[1]
        / (trans_rates[1] - trans_rates[0])
        * activity_99mo[t_layers[0] == 24]
        * (1 - np.exp(-(trans_rates[1] - trans_rates[0]) * 24))
    )[0]
    nuclei_voided_24h = activity_to_nuclei(activity_voided_24h, trans_rates[1])

    # void at 48 h
    activity_voided_48h = (
        0.89
        * trans_rates[1]
        / (trans_rates[1] - trans_rates[0])
        * activity_99mo[t_layers[0] == 48]
        * (1 - np.exp(-(trans_rates[1] - trans_rates[0]) * 24))
    )[0]
    nuclei_voided_48h = activity_to_nuclei(activity_voided_48h, trans_rates[1])

    assert len(voided_nuclei) == 2
    assert voided_nuclei[0].shape == (1, 2, 1)
    assert np.allclose(
        voided_nuclei[0],
        np.array([[[0], [nuclei_voided_48h]]]),
        rtol=8e-5,
    )
    assert voided_nuclei[1].shape == (1, 2, 1)
    assert np.allclose(
        voided_nuclei[1],
        np.array([[[0], [nuclei_voided_24h]]]),
        rtol=8e-5,
    )


def test_cumulated_activity():
    N11 = lambda t: 3 * np.exp(-2 * t) + 2 * np.exp(-0.3 * t)
    N12 = lambda t: np.exp(-5 * t)
    N21 = lambda t: np.exp(-2 * t)
    N22 = lambda t: np.exp(-t)
    t_eval = np.linspace(0, 3, 1000)
    t_layers = [t_eval, t_eval]
    nuclei_layers = [
        np.array([N11(t_eval), N12(t_eval)]),
        np.array([N21(t_eval), N22(t_eval)]),
    ]
    trans_rates = np.array([3, 2])
    ans = _cumulated_activity(
        t_layers,
        nuclei_layers,
        trans_rates,
    )

    assert np.allclose(
        ans,
        1e-6
        / (60 * 60)
        * np.array(
            [
                [
                    3 * (3 / 2 * (1 - np.exp(-6)) + 2 / 0.3 * (1 - np.exp(-0.9))),
                    3 * (1 / 5 * (1 - np.exp(-15))),
                ],
                [2 * (1 / 2 * (1 - np.exp(-6))), 2 * (1 - np.exp(-3))],
            ]
        ),
    )

    ans_a = _cumulated_activity(
        t_layers,
        nuclei_layers,
        trans_rates,
        t_end=t_eval[200],
    )
    ans_b = _cumulated_activity(
        t_layers,
        nuclei_layers,
        trans_rates,
        t_start=t_eval[200],
    )
    assert np.allclose(ans_a + ans_b, ans)


def test_valid_dcm_input():
    pass
    """TODO Check some asserts fail when intended.

    t_span = (0, 3)
    t_eval = np.linspace(0, 3, 1000)
    initial_nuclei = np.array([[1.08e12, 0], [0.0, 1e10]])
    trans_rates = np.array([0.1, 0])
    branching_fracs = np.array([[0, 0], [0.3, 0]])
    xfer_coeffs = np.array([np.array([[0, 0], [0.5, 0]]), np.zeros((2, 2))])

    prelayer
    layer_names
    compartment_names
    voiding_rules

    with pytest.raises(AssertionError):
        _valid_dcm_input(
            trans_rates,
            branching_fracs,
            xfer_coeffs,
            initial_nuclei,
            t_eval,
            prelayer,
            layer_names,
            compartment_names,
            voiding_rules,
        )

    """
