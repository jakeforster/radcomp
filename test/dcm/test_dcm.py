import numpy as np
import matplotlib
import os
import warnings
from radcomp.dcm.dcm import solve_dcm_from_toml
from radcomp.common.utils import read_arrays


def test_dcm():
    # test things not tested in test_dcm_examples
    # lazy so just call things that were tested in test_dcm_funcs
    toml_dir = os.path.join(os.path.dirname(__file__), "tomls-for-testing")
    fp = os.path.join(toml_dir, "test-read.toml")
    model = solve_dcm_from_toml(fp, np.linspace(0, 200))
    assert np.array_equal(
        model.halflife(), np.array([np.log(2) / 0.91, np.inf, np.log(2) / 0.2])
    )

    layer_names = model._get_layer_names()
    layer_names_incl_prelayer = model._get_layer_names_incl_prelayer()
    assert all(i == j for i, j in zip(layer_names_incl_prelayer, ["", *layer_names]))

    backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # don't show figs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model.plot()
    matplotlib.use(backend)

    filepath = "test-save-arrays-dcm.npz"
    model.save_arrays(filepath)
    t_eval_read, nuclei_eval_read, activity_eval_read = read_arrays(filepath)
    assert np.array_equal(model.t_eval, t_eval_read)
    assert np.array_equal(model.nuclei, nuclei_eval_read)
    assert np.array_equal(model.activity(), activity_eval_read)
    os.remove(filepath)

    # tested in test_dcm_funcs
    model.info_xfer()
    model.info_growth()
