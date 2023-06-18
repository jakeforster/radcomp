from radcomp.common.voiding import (
    VoidingRule,
    _VoidingEvent,
    _create_time_ordered_voids_for_layer,
)
import numpy as np
import pytest


def test_create_time_ordered_voids_for_layer_novoiding():
    voiding_rules = []
    assert _create_time_ordered_voids_for_layer(voiding_rules, 0) == []
    assert _create_time_ordered_voids_for_layer(voiding_rules, 1) == []
    assert _create_time_ordered_voids_for_layer(voiding_rules, 1203) == []

    voiding_rule = VoidingRule(np.array([]), np.ones((2, 3)))
    voiding_rules = [voiding_rule]
    assert _create_time_ordered_voids_for_layer(voiding_rules, 0) == []
    assert _create_time_ordered_voids_for_layer(voiding_rules, 1) == []
    assert _create_time_ordered_voids_for_layer(voiding_rules, 1203) == []

    voiding_rule = VoidingRule(np.array([1.3, 4]), np.zeros((2, 3)))
    voiding_rules = [voiding_rule]
    assert _create_time_ordered_voids_for_layer(voiding_rules, 0) == []
    assert _create_time_ordered_voids_for_layer(voiding_rules, 1) == []
    with pytest.raises(IndexError):
        _create_time_ordered_voids_for_layer(voiding_rules, 1203)


def test_create_time_ordered_voids_for_layer_1voiding():
    voiding = VoidingRule(
        np.array([6, 3]), np.array([[0, 0, 1], [0.5, 0, 0], [0, 0, 0]])
    )

    ans0 = _create_time_ordered_voids_for_layer([voiding], 0)
    assert len(ans0) == 2
    assert ans0[0] == _VoidingEvent(3, np.array([0, 0, 1]))
    assert ans0[1] == _VoidingEvent(6, np.array([0, 0, 1]))

    ans1 = _create_time_ordered_voids_for_layer([voiding], 1)
    assert len(ans1) == 2
    assert ans1[0] == _VoidingEvent(3, np.array([0.5, 0, 0]))
    assert ans1[1] == _VoidingEvent(6, np.array([0.5, 0, 0]))

    ans2 = _create_time_ordered_voids_for_layer([voiding], 2)
    assert len(ans2) == 0


def test_create_time_ordered_voids_for_layer_2voiding():
    voiding1 = VoidingRule(
        np.array([3, 6]), np.array([[0, 0, 1], [0.5, 0, 0], [0, 0, 0]])
    )
    voiding2 = VoidingRule(
        np.array([1, 7, 8]), np.array([[0, 0, 0], [1, 0, 1], [0, 0.4, 0]])
    )

    ans0 = _create_time_ordered_voids_for_layer([voiding1, voiding2], 0)
    assert len(ans0) == 2
    assert ans0[0] == _VoidingEvent(3, np.array([0, 0, 1]))
    assert ans0[1] == _VoidingEvent(6, np.array([0, 0, 1]))

    ans1 = _create_time_ordered_voids_for_layer([voiding1, voiding2], 1)
    assert len(ans1) == 5
    assert ans1[0] == _VoidingEvent(1, np.array([1, 0, 1]))
    assert ans1[1] == _VoidingEvent(3, np.array([0.5, 0, 0]))
    assert ans1[2] == _VoidingEvent(6, np.array([0.5, 0, 0]))
    assert ans1[3] == _VoidingEvent(7, np.array([1, 0, 1]))
    assert ans1[4] == _VoidingEvent(8, np.array([1, 0, 1]))

    ans2 = _create_time_ordered_voids_for_layer([voiding1, voiding2], 2)
    assert len(ans2) == 3
    assert ans2[0] == _VoidingEvent(1, np.array([0, 0.4, 0]))
    assert ans2[1] == _VoidingEvent(7, np.array([0, 0.4, 0]))
    assert ans2[2] == _VoidingEvent(8, np.array([0, 0.4, 0]))
