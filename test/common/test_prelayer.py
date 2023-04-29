import numpy as np
from radcomp.common.prelayer import Prelayer
from radcomp.common.utils import nuclei_to_activity


def test_prelayer():
    trans_rate = 0.03
    branching_fracs = np.array([0.8])
    n1 = lambda t: (2.4e17) * (np.exp(-0.04 * t) + np.exp(-0.1 * t))
    n2 = lambda t: (3.6e17) * np.exp(-0.05 * t)
    prelayer = Prelayer(
        trans_rate,
        branching_fracs,
        [
            lambda t: nuclei_to_activity(n1(t), trans_rate),
            lambda t: nuclei_to_activity(n2(t), trans_rate),
        ],
        "smth-psma",
    )

    assert prelayer.name == "smth-psma"

    t_eval = np.linspace(0, 200, 1000)
    nuclei_funcs = prelayer.nuclei_funcs()
    # test np.ndarray input
    assert np.allclose(nuclei_funcs[0](t_eval), n1(t_eval))
    assert np.allclose(nuclei_funcs[1](t_eval), n2(t_eval))
    # test float input
    assert np.isclose(nuclei_funcs[0](t_eval[-1]), n1(t_eval[-1]))
    assert np.isclose(nuclei_funcs[1](t_eval[-1]), n2(t_eval[-1]))
