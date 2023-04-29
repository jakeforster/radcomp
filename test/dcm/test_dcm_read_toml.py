import os
import warnings
import numpy as np

from radcomp.common.utils import activity_to_nuclei
from radcomp.dcm.dcm_read_toml import _dcm_read_toml


def test_dcm_read_toml():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        filepath = os.path.join(
            os.path.dirname(__file__), "tomls-for-testing", "test-read.toml"
        )
        (
            trans_rates,
            branching_fracs,
            xfer_coeffs,
            initial_nuclei,
            layer_names,
            compartment_names,
        ) = _dcm_read_toml(filepath)

        assert np.array_equal(trans_rates, np.array([0.91, 0, 0.2]))
        assert np.array_equal(
            branching_fracs, np.array([[0, 0, 0], [0.1, 0, 0], [0.9, 0, 0]])
        )
        assert np.array_equal(
            xfer_coeffs,
            np.array(
                [
                    [[0, 3.4, 2.1], [4.3, 0, 1.3], [5.8, 9.2, 0]],
                    [[0, 3, 2], [4, 0, 1], [5.1, 9, 0]],
                    [[0, 1, 0], [2, 0, 1.3], [3, 1, 0]],
                ]
            ),
        )
        assert np.array_equal(
            initial_nuclei,
            np.array(
                [
                    activity_to_nuclei(np.array([1, 3.3, 2]), 0.91),
                    [3, 2, 2],
                    activity_to_nuclei(np.array([0, 1.3, 0]), 0.2),
                ]
            ),
        )
        assert all(
            i == j for i, j in zip(layer_names, ["Nuclide A", "Nuclide B", "Nuclide C"])
        )
        assert compartment_names is not None
        assert all(
            i == j for i, j in zip(compartment_names, ["plasma", "kidneys", "lungs"])
        )
