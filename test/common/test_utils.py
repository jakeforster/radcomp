import os
import numpy as np
from radcomp.common.utils import (
    activity_to_nuclei,
    nuclei_to_activity,
    read_arrays,
    _save_arrays,
)


def test_activity_to_nuclei():
    """
    N = A / lambda
    A and N can be float or np.ndarray
    1 Bq and transition rate 1 s-1 -> 1 nucleus
    1 s = 1/(60 * 60) h  ->  1 s-1 = 60 * 60 h-1
    """
    # N = A / lambda
    assert activity_to_nuclei(1e-6, 60 * 60) == 1
    assert np.array_equal(
        activity_to_nuclei(np.array([1e-6, 2e-6]), 60 * 60), np.array([1, 2])
    )


def test_nuclei_to_activity():
    """
    A = lambda * N
    A and N can be float or np.ndarray
    1 nucleus and transition rate 1 s-1 -> 1 Bq
    1 s = 1/(60 * 60) h  ->  1 s-1 = 60 * 60 h-1
    """
    assert nuclei_to_activity(1, 60 * 60) == 1e-6
    assert np.array_equal(
        nuclei_to_activity(np.array([1, 2]), 60 * 60), np.array([1e-6, 2e-6])
    )


def test_activity_nuclei_inverse():
    trans_rate = 0.1
    for nuclei in np.linspace(0, 3, 1000):
        assert np.isclose(
            nuclei,
            activity_to_nuclei(nuclei_to_activity(nuclei, trans_rate), trans_rate),
        )


def test_save_load_arrays():
    filepath = "test-save-load-arrays.npz"
    t_eval_in = np.linspace(0, 3)
    activity_eval_in = np.array(
        [[t_eval_in + 1, t_eval_in + 2], [t_eval_in + 20, t_eval_in + 4]]
    )
    nuclei_eval_in = activity_eval_in * 3

    _save_arrays(filepath, t_eval_in, nuclei_eval_in, activity_eval_in)
    t_eval_out, nuclei_eval_out, activity_eval_out = read_arrays(filepath)
    assert np.array_equal(t_eval_in, t_eval_out)
    assert np.array_equal(nuclei_eval_in, nuclei_eval_out)
    assert np.array_equal(activity_eval_in, activity_eval_out)
    os.remove(filepath)
