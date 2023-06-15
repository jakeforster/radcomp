from radcomp.common.voiding import (
    FractionalVoiding,
    _VoidLayer,
    _ordered_voids_in_layer,
)
from radcomp.dcm.dcm_internal import _solve_dcm
import numpy as np


def test_ordered_voids_in_layer_novoiding():
    assert _ordered_voids_in_layer([], 0) == []
    assert _ordered_voids_in_layer([], 1) == []
    assert _ordered_voids_in_layer([], 1203) == []


def test_ordered_voids_in_layer_1voiding():
    voiding = FractionalVoiding(
        np.array([6, 3]), np.array([[0, 0, 1], [0.5, 0, 0], [0, 0, 0]])
    )

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
    voiding1 = FractionalVoiding(
        np.array([3, 6]), np.array([[0, 0, 1], [0.5, 0, 0], [0, 0, 0]])
    )
    voiding2 = FractionalVoiding(
        np.array([1, 7, 8]), np.array([[0, 0, 0], [1, 0, 1], [0, 0.4, 0]])
    )

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
