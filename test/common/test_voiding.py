from radcomp.common.voiding import Voiding, _VoidLayer, _ordered_voids_in_layer
from radcomp.dcm.dcm_internal import _solve_dcm
import numpy as np


def test_ordered_voids_in_layer_novoiding():
    assert _ordered_voids_in_layer([], 0) == []
    assert _ordered_voids_in_layer([], 1) == []
    assert _ordered_voids_in_layer([], 1203) == []


def test_ordered_voids_in_layer_1voiding():
    voiding = Voiding([6, 3], np.array([[0, 0, 1], [0.5, 0, 0], [0, 0, 0]]))

    ans0 = _ordered_voids_in_layer([voiding], 0)
    assert len(ans0) == 2
    assert ans0[0] == _VoidLayer(3, np.array([0, 0, 1]))
    assert ans0[1] == _VoidLayer(6, np.array([0, 0, 1]))

    ans1 = _ordered_voids_in_layer([voiding], 1)
    assert len(ans1) == 2
    assert ans1[0] == _VoidLayer(3, np.array([0.5, 0, 0]))
    assert ans1[1] == _VoidLayer(6, np.array([0.5, 0, 0]))

    ans2 = _ordered_voids_in_layer([voiding], 2)
    assert len(ans2) == 0


def test_ordered_voids_in_layer_2voiding():
    voiding1 = Voiding([3, 6], np.array([[0, 0, 1], [0.5, 0, 0], [0, 0, 0]]))
    voiding2 = Voiding([1, 7, 8], np.array([[0, 0, 0], [1, 0, 1], [0, 0.4, 0]]))

    ans0 = _ordered_voids_in_layer([voiding1, voiding2], 0)
    assert len(ans0) == 2
    assert ans0[0] == _VoidLayer(3, np.array([0, 0, 1]))
    assert ans0[1] == _VoidLayer(6, np.array([0, 0, 1]))

    ans1 = _ordered_voids_in_layer([voiding1, voiding2], 1)
    assert len(ans1) == 5
    assert ans1[0] == _VoidLayer(1, np.array([1, 0, 1]))
    assert ans1[1] == _VoidLayer(3, np.array([0.5, 0, 0]))
    assert ans1[2] == _VoidLayer(6, np.array([0.5, 0, 0]))
    assert ans1[3] == _VoidLayer(7, np.array([1, 0, 1]))
    assert ans1[4] == _VoidLayer(8, np.array([1, 0, 1]))

    ans2 = _ordered_voids_in_layer([voiding1, voiding2], 2)
    assert len(ans2) == 3
    assert ans2[0] == _VoidLayer(1, np.array([0, 0.4, 0]))
    assert ans2[1] == _VoidLayer(7, np.array([0, 0.4, 0]))
    assert ans2[2] == _VoidLayer(8, np.array([0, 0.4, 0]))


def test_solve_dcm_voiding():
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
    voiding = Voiding([1], np.array([[0, 1], [0, 0]]))
    voiding_list = [voiding]

    t_layers, nuclei_layers = _solve_dcm(
        t_span,
        initial_nuclei,
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        t_eval=t_eval,
        voiding_list=voiding_list,
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
