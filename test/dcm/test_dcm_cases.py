import numpy as np
import os

from radcomp.common.utils import nuclei_to_activity
from radcomp.common.prelayer import Prelayer
from radcomp.dcm.dcm import solve_dcm, solve_dcm_from_toml
from radcomp.dcm.dcm_read_toml import _dcm_read_toml

toml_dir = os.path.join(os.path.dirname(__file__), "tomls-for-testing")


def test_const():
    """
    1 stable nuclide, 1 compartment

    dN/dt = 0
    N(0) = 3
    """
    # analytical soln
    N = lambda _: 3

    t_eval = np.linspace(0, 100)

    trans_rates = np.array([0])
    branching_fracs = np.array([[0]])
    xfer_coeffs = np.array([[[0]]])
    initial_nuclei = np.array([[3]])
    model = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
    )
    assert np.allclose(model.nuclei[0], N(t_eval))

    # reading from a toml works too
    fp = os.path.join(toml_dir, "test-stable.toml")
    (
        trans_rates_i,
        branching_fracs_i,
        xfer_coeffs_i,
        initial_nuclei_i,
        _,
        _,
    ) = _dcm_read_toml(fp)
    assert np.array_equal(trans_rates, trans_rates_i)
    assert np.array_equal(branching_fracs, branching_fracs_i)
    assert np.array_equal(xfer_coeffs, xfer_coeffs_i)
    assert np.array_equal(initial_nuclei, initial_nuclei_i)

    model_i = solve_dcm_from_toml(
        fp,
        t_eval,
    )
    assert np.allclose(model_i.nuclei[0], N(t_eval))


def test_const_llzero_nuclei():
    """
    Constant number of nuclei << 1 is OK too

    1 stable nuclide, 1 compartment

    dN/dt = 0
    N(0) = 3
    """
    # analytical soln
    N = lambda _: 1e-20

    t_eval = np.linspace(0, 3, 1000)

    trans_rates = np.array([0])
    branching_fracs = np.array([[0]])
    xfer_coeffs = np.array([[[0]]])
    initial_nuclei = np.array([[1e-20]])
    model = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
    )
    assert np.allclose(model.nuclei[0], N(t_eval))


def test_const_parallel():
    """
    What about multiple stable nuclides in compartments that don't interact?

    2 stable nuclides
    2 compartments, no transfer between

    dN11/dt = 0
    dN12/dt = 0
    dN21/dt = 0
    dN22/dt = 0
    N11(0) = 4
    N12(0) = 13
    N21(0) = 2
    N22(0) = 0
    """
    # analytical soln
    N11 = lambda _: 4
    N12 = lambda _: 13
    N21 = lambda _: 2
    N22 = lambda _: 0

    t_eval = np.linspace(0, 3, 1000)

    trans_rates = np.array([0, 0])
    branching_fracs = np.array([[0, 0], [0, 0]])
    xfer_coeffs = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    initial_nuclei = np.array([[4, 13], [2, 0]])
    model = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
    )
    assert np.allclose(model.nuclei[0][0], np.array([[N11(t_eval)] * 1000]))
    assert np.allclose(model.nuclei[0][1], np.array([[N12(t_eval)] * 1000]))
    assert np.allclose(model.nuclei[1][0], np.array([[N21(t_eval)] * 1000]))
    assert np.allclose(model.nuclei[1][1], np.array([[N22(t_eval)] * 1000]))

    # from a toml?
    fp = os.path.join(toml_dir, "test-stable-parallel.toml")
    (
        trans_rates_i,
        branching_fracs_i,
        xfer_coeffs_i,
        initial_nuclei_i,
        _,
        _,
    ) = _dcm_read_toml(fp)
    assert np.array_equal(trans_rates, trans_rates_i)
    assert np.array_equal(branching_fracs, branching_fracs_i)
    assert np.array_equal(xfer_coeffs, xfer_coeffs_i)
    assert np.array_equal(initial_nuclei, initial_nuclei_i)

    model_i = solve_dcm_from_toml(
        fp,
        t_eval,
    )
    assert np.allclose(model_i.nuclei[0][0], np.array([[N11(t_eval)] * 1000]))
    assert np.allclose(model_i.nuclei[0][1], np.array([[N12(t_eval)] * 1000]))
    assert np.allclose(model_i.nuclei[1][0], np.array([[N21(t_eval)] * 1000]))
    assert np.allclose(model_i.nuclei[1][1], np.array([[N22(t_eval)] * 1000]))


def test_relative_error():
    """
    Investigate relative error for exponential decay

    1 unstable nuclide, 1 compartment

    2 MBq of Tc-99m:
    + trans rate lambda = 0.11552453009332421 h-1
    + 20 half-lives = 120 h
    + 2 MBq -> N(0) = 2 * 1e6 * 60 * 60 / 0.11552453009332421 = 62324425766.40322

    dN/dt = - lambda * N
    N(0) = 62324425766.40322
    lambda = 0.11552453009332421 h-1
    """
    # analytical soln
    N = lambda t: (62324425766.40322) * np.exp(-0.11552453009332421 * t)

    # 20 half-lives (N = 6e4)
    t_eval = np.linspace(0, 120)

    trans_rates = np.array([0.11552453009332421])
    branching_fracs = np.array([[0]])
    xfer_coeffs = np.array([[[0]]])
    initial_nuclei = np.array([[62324425766.40322]])
    model = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
    )
    rel_error = 100 * np.abs(model.nuclei[0][0] - N(t_eval)) / N(t_eval)
    assert np.all(rel_error < 0.8)

    # 40 half-lives (N = 6e-2)
    t_eval = np.linspace(0, 240)
    model = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
    )
    rel_error = 100 * np.abs(model.nuclei[0][0] - N(t_eval)) / N(t_eval)
    assert np.all(rel_error < 1.6)

    # 60 half-lives (N = 5.4e-8) -- now the relative error is getting rough
    t_eval = np.linspace(0, 360)
    model = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
    )
    rel_error = 100 * np.abs(model.nuclei[0][0] - N(t_eval)) / N(t_eval)
    assert 50 < np.max(rel_error) < 60

    # but the activity is always within 0.6% (including an absolute tolerance of 0.01 Bq)
    assert np.allclose(
        model.activity()[0][0],
        nuclei_to_activity(N(t_eval), 0.11552453009332421),
        rtol=6e-3,
    )


def test_relative_error_stable():
    """
    What about the relative error of a stable nuclide growing by the decay of a parent?

    unstable nuclide that sometime decays to a stable nuclide
    1 compartment

    Layer 1:
    + trans rate = 0.1 h-1
    + 20 half-lives = 139 h
    + 80 MBq -> N(0) = 80 * 1e6 * 60 * 60 / 0.1 = 2.88e12

    dN1/dt = - lambda1 * N1
    A1(0) = 80 MBq
    lambda1 = 0.1 h-1
    N1(0) = 2.88e12

    Layer 2:
    dN2/dt = branching_frac21 * lambda1 * N1
    N2(0) = 0
    branching_frac21 = 0.8
    """
    # analytical soln
    N1 = lambda t: (2.88e12) * np.exp(-0.1 * t)
    N2 = lambda t: 2.304e12 * (1 - np.exp(-0.1 * t))

    # about 20 half-lives (N1 = 2.6e6)
    t_eval = np.linspace(0, 139)

    trans_rates = np.array([0.1, 0])
    branching_fracs = np.array([[0, 0], [0.8, 0]])
    xfer_coeffs = np.array([[[0]], [[0]]])
    initial_nuclei = np.array([[2.88e12], [0]])
    model = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
    )
    # rel error is fine for both layers, activity of 1st layer is fine
    rel_error1 = 100 * np.abs(model.nuclei[0][0] - N1(t_eval)) / N1(t_eval)
    assert np.all(rel_error1 < 0.8)
    assert np.allclose(
        model.activity()[0][0],
        nuclei_to_activity(N1(t_eval), 0.1),
        rtol=8e-3,
    )
    rel_error2 = 100 * np.abs(model.nuclei[1][0][1:] - N2(t_eval[1:])) / N2(t_eval[1:])
    assert np.all(rel_error2 < 0.9)

    # what about 60 half-lives (N1=6.7e-4)? Rel errors degraded but not that bad
    t_eval = np.linspace(0, 360)
    model = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
    )
    rel_error1 = 100 * np.abs(model.nuclei[0][0] - N1(t_eval)) / N1(t_eval)
    assert 1.9 < np.max(rel_error1) < 2
    rel_error2 = 100 * np.abs(model.nuclei[1][0][1:] - N2(t_eval[1:])) / N2(t_eval[1:])
    assert 4 < np.max(rel_error2) < 5
    assert np.allclose(
        model.activity()[0][0],
        nuclei_to_activity(N1(t_eval), 0.1),
        rtol=8e-3,
    )


def test_relative_error_ac225():
    """
    This test shows that for 1 MBq of Ac-225,
    even after 20 half-lives,
    the relative error is still less than 0.8%.
    (at which time the activity is 0.95 Bq)

    E.g. 1 MBq of Ac-225:
    + trans rate 0.0028881132523331052 h-1
    + 20 half-lives = 4800 h
    + 1 MBq -> 1 * 1e6 * 60 * 60 / 0.0028881132523331052 = 1246488515328.0645

    1 unstable nuclide, 1 compartment

    dN/dt = - lambda * N
    N(0) = 1246488515328.0645
    lambda = 0.0028881132523331052 h-1
    """
    # analytical soln
    N = lambda t: (1246488515328.0645) * np.exp(-0.0028881132523331052 * t)

    t_eval = np.linspace(0, 4800)

    trans_rates = np.array([0.0028881132523331052])
    branching_fracs = np.array([[0]])
    xfer_coeffs = np.array([[[0]]])
    initial_nuclei = np.array([[1246488515328.0645]])
    model = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
    )
    rel_error = 100 * np.abs(model.nuclei[0][0] - N(t_eval)) / N(t_eval)
    assert np.all(rel_error < 0.8)

    # from a toml
    fp = os.path.join(toml_dir, "test-exp-ac225.toml")
    (
        trans_rates_i,
        branching_fracs_i,
        xfer_coeffs_i,
        initial_nuclei_i,
        _,
        _,
    ) = _dcm_read_toml(fp)
    assert np.array_equal(trans_rates, trans_rates_i)
    assert np.array_equal(branching_fracs, branching_fracs_i)
    assert np.array_equal(xfer_coeffs, xfer_coeffs_i)
    assert np.allclose(initial_nuclei, initial_nuclei_i)

    model_i = solve_dcm_from_toml(
        fp,
        t_eval,
    )
    rel_error = 100 * np.abs(model_i.nuclei[0][0] - N(t_eval)) / N(t_eval)
    assert np.all(rel_error < 0.8)


def test_exp_parallel_layers():
    """
    What about multiple unstable nuclides that do not share a decay chain?

    2 unstable nuclides, first does not decay to second
    1 compartment

    Layer 1:
    dN1/dt = - lambda1 * N1
    A1(0) = 2 MBq
    lambda1 = 3 h-1
    N1(0) = 2 * 1e6 * 60 * 60 / 3 = 2.4e9

    Layer 2:
    dN2/dt = -lambda2 * N2
    A2(0) = 4 MBq
    lambda2 = 0.5 h-1
    N2(0) = 4 * 1e6 * 60 * 60 / 0.5 = 2.88e10
    """
    # analytical soln
    N1 = lambda t: 2.4e9 * np.exp(-3 * t)
    N2 = lambda t: 2.88e10 * np.exp(-0.5 * t)

    t_eval = np.linspace(0, 5, 1000)  # more than 20 half-lives of nuclide 1

    trans_rates = np.array([3, 0.5])
    branching_fracs = np.array([[0, 0], [0, 0]])
    xfer_coeffs = np.array([[[0]], [[0]]])
    initial_nuclei = np.array([[2.4e9], [2.88e10]])
    model = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
    )

    rel_error1 = 100 * np.abs(model.nuclei[0][0] - N1(t_eval)) / N1(t_eval)
    rel_error2 = 100 * np.abs(model.nuclei[1][0] - N2(t_eval)) / N2(t_eval)
    assert np.all(rel_error1 < 0.9)
    assert np.all(rel_error2 < 0.2)
    assert all(
        np.allclose(a, b, rtol=6e-3)
        for a, b in zip(
            model.activity(),
            [
                np.array([nuclei_to_activity(N1(t_eval), 3)]),
                np.array([nuclei_to_activity(N2(t_eval), 0.5)]),
            ],
        )
    )

    # same from toml
    fp = os.path.join(toml_dir, "test-exp-parallel-layers.toml")
    (
        trans_rates_i,
        branching_fracs_i,
        xfer_coeffs_i,
        initial_nuclei_i,
        _,
        _,
    ) = _dcm_read_toml(fp)
    assert np.array_equal(trans_rates, trans_rates_i)
    assert np.array_equal(branching_fracs, branching_fracs_i)
    assert np.array_equal(xfer_coeffs, xfer_coeffs_i)
    assert np.allclose(initial_nuclei, initial_nuclei_i)

    model_i = solve_dcm_from_toml(
        fp,
        t_eval,
    )
    assert all(
        np.allclose(a, b, rtol=6e-3)
        for a, b in zip(
            model_i.activity(),
            [
                np.array([nuclei_to_activity(N1(t_eval), 3)]),
                np.array([nuclei_to_activity(N2(t_eval), 0.5)]),
            ],
        )
    )


def test_eg1():
    """
    2 unstable nuclides, first sometimes decays to second
    1 compartment

    Layer 1:
    dN1/dt = - lambda1 * N1
    A1(0) = 2 MBq
    N1(0) = 2 * 1e6 * 60 * 60 / 0.03 = 2.4e11
    lambda1 = 0.03 h-1

    Layer 2:
    dN2/dt = -lambda2 * N2 + branching_frac21 * lambda1 * N1
    lambda2 = 0.5 h-1
    A2(0) = 1 MBq
    N2(0) = 1 * 1e6 * 60 * 60 / 0.5 = 7.2e9
    branching_frac21 = 0.8
    """
    # analytical soln
    N1 = lambda t: (2.4e11) * np.exp(-0.03 * t)
    N2 = lambda t: (
        (5.76e9 / 0.47) * np.exp(0.47 * t) + (7.2e9 - 5.76e9 / 0.47)
    ) * np.exp(-0.5 * t)

    t_eval = np.linspace(0, 200, 1000)
    fp = os.path.join(toml_dir, "test-eg1.toml")
    model = solve_dcm_from_toml(fp, t_eval)

    assert np.allclose(
        model.activity()[0][0], nuclei_to_activity(N1(t_eval), 0.03), rtol=5e-3
    )
    assert np.allclose(
        model.activity()[1][0], nuclei_to_activity(N2(t_eval), 0.5), rtol=5e-3
    )


def test_eg1_itac():
    """
    Same but first layer supplied as input time activity curve

    2 unstable nuclides, first sometimes decays to second
    1 compartment

    Layer 1:
    dN1/dt = - lambda1 * N1
    A1(0) = 2 MBq
    N1(0) = 2 * 1e6 * 60 * 60 / 0.03 = 2.4e11
    lambda1 = 0.03 h-1

    Layer 2:
    dN2/dt = -lambda2 * N2 + branching_frac21 * lambda1 * N1
    lambda2 = 0.5 h-1
    A2(0) = 1 MBq
    N2(0) = 1 * 1e6 * 60 * 60 / 0.5 = 7.2e9
    branching_frac21 = 0.8
    """
    # analytical soln
    N1 = lambda t: (2.4e11) * np.exp(-0.03 * t)
    N2 = lambda t: (
        (5.76e9 / 0.47) * np.exp(0.47 * t) + (7.2e9 - 5.76e9 / 0.47)
    ) * np.exp(-0.5 * t)

    prelayer = Prelayer(
        0.03, np.array([0.8]), [lambda t: nuclei_to_activity(N1(t), 0.03)]
    )
    assert prelayer.name == "prelayer"
    t_eval = np.linspace(0, 200, 1000)
    fp = os.path.join(toml_dir, "test-eg1-itac.toml")
    model = solve_dcm_from_toml(fp, t_eval, prelayer=prelayer)
    assert np.allclose(
        model.activity()[0][0], nuclei_to_activity(N2(t_eval), 0.5), rtol=7e-4
    )


def test_eg2():
    """
    2 unstable nuclides, first sometimes decays to second
    2 compartments, no transfer between

    Layer 1:
    dN11/dt = - lambda1 * N11
    dN12/dt = - lambda1 * N12
    A11(0) = 2 TBq
    N11(0) = 2 * 1e12 * 60 * 60 / 0.03 = 2.4e17
    A12(0) = 3 TBq
    N12(0) = 3 * 1e12 * 60 * 60 / 0.03 = 3.6e17
    lambda1 = 0.03 h-1

    Layer 2:
    dN21/dt = -lambda2 * N21 + branching_frac21 * lambda1 * N11
    dN22/dt = -lambda2 * N22 + branching_frac21 * lambda1 * N12
    A21(0) = 0.3 TBq
    N21(0) = 0.3 * 1e12 * 60 * 60 / 0.5 =2.16e15
    A22(0) = 4 TBq
    N22(0) = 4 * 1e12 * 60 * 60 / 0.5 = 2.88e16
    lambda2 = 0.5 h-1
    branching_frac21 = 0.8
    """
    # analytical soln
    N11 = lambda t: (2.4e17) * np.exp(-0.03 * t)
    N12 = lambda t: (3.6e17) * np.exp(-0.03 * t)
    N21 = lambda t: (
        (5.76e15 / 0.47) * np.exp(0.47 * t) + (2.16e15 - 5.76e15 / 0.47)
    ) * np.exp(-0.5 * t)
    N22 = lambda t: (
        (8.64e15 / 0.47) * np.exp(0.47 * t) + (2.88e16 - 8.64e15 / 0.47)
    ) * np.exp(-0.5 * t)

    t_eval = np.linspace(0, 200, 1000)
    fp = os.path.join(toml_dir, "test-eg2.toml")
    model = solve_dcm_from_toml(fp, t_eval)
    assert np.allclose(
        model.activity()[0][0], nuclei_to_activity(N11(t_eval), 0.03), rtol=5e-3
    )
    assert np.allclose(
        model.activity()[0][1], nuclei_to_activity(N12(t_eval), 0.03), rtol=5e-3
    )
    assert np.allclose(
        model.activity()[1][0], nuclei_to_activity(N21(t_eval), 0.5), rtol=5e-3
    )
    assert np.allclose(
        model.activity()[1][1], nuclei_to_activity(N22(t_eval), 0.5), rtol=5e-3
    )


def test_eg2_itac():
    """
    Same but first layer supplied as input time activity curve

    2 unstable nuclides, first sometimes decays to second
    2 compartments, no transfer between

    Layer 1:
    dN11/dt = - lambda1 * N11
    dN12/dt = - lambda1 * N12
    A11(0) = 2 TBq
    N11(0) = 2 * 1e12 * 60 * 60 / 0.03 = 2.4e17
    A12(0) = 3 TBq
    N12(0) = 3 * 1e12 * 60 * 60 / 0.03 = 3.6e17
    lambda1 = 0.03 h-1

    Layer 2:
    dN21/dt = -lambda2 * N21 + branching_frac21 * lambda1 * N11
    dN22/dt = -lambda2 * N22 + branching_frac21 * lambda1 * N12
    A21(0) = 0.3 TBq
    N21(0) = 0.3 * 1e12 * 60 * 60 / 0.5 =2.16e15
    A22(0) = 4 TBq
    N22(0) = 4 * 1e12 * 60 * 60 / 0.5 = 2.88e16
    lambda2 = 0.5 h-1
    branching_frac21 = 0.8
    """
    # analytical soln
    N11 = lambda t: (2.4e17) * np.exp(-0.03 * t)
    N12 = lambda t: (3.6e17) * np.exp(-0.03 * t)
    N21 = lambda t: (
        (5.76e15 / 0.47) * np.exp(0.47 * t) + (2.16e15 - 5.76e15 / 0.47)
    ) * np.exp(-0.5 * t)
    N22 = lambda t: (
        (8.64e15 / 0.47) * np.exp(0.47 * t) + (2.88e16 - 8.64e15 / 0.47)
    ) * np.exp(-0.5 * t)

    prelayer = Prelayer(
        0.03,
        np.array([0.8]),
        [
            lambda t: nuclei_to_activity(N11(t), 0.03),
            lambda t: nuclei_to_activity(N12(t), 0.03),
        ],
    )
    t_eval = np.linspace(0, 200, 1000)
    fp = os.path.join(toml_dir, "test-eg2-itac.toml")
    model = solve_dcm_from_toml(fp, t_eval, prelayer=prelayer)
    assert np.allclose(
        model.activity()[0][0], nuclei_to_activity(N21(t_eval), 0.5), rtol=8e-4
    )
    assert np.allclose(
        model.activity()[0][1], nuclei_to_activity(N22(t_eval), 0.5), rtol=9e-4
    )


def test_eg3():
    """
    2 unstable nuclides and 1 stable nuclide, 1st nuclide sometimes decays to 2nd and 2nd nuclide sometimes decays to 3rd
    1 compartment

    Layer 1:
    dN1/dt = - lambda1 * N1
    A1(0) = 2 MBq
    N1(0) = 2 * 1e6 * 60 * 60 / 0.03 = 2.4e11
    lambda1 = 0.03 h-1

    Layer 2:
    dN2/dt = -lambda2 * N2 + branching_frac21 * lambda1 * N1
    lambda_2 = 0.5 h-1
    A2(0) = 1 MBq
    N2(0) = 1 * 1e6 * 60 * 60 / 0.5 = 7.2e9
    branching_frac21 = 0.8

    Layer 3:
    dN3/dt = branching_frac32 * lambda2 * N2
    N3(0) = 0
    branching_frac32 = 0.1
    """
    # analytical soln
    N1 = lambda t: (2.4e11) * np.exp(-0.03 * t)
    N2 = lambda t: (
        (5.76e9 / 0.47) * np.exp(0.47 * t) + (7.2e9 - 5.76e9 / 0.47)
    ) * np.exp(-0.5 * t)
    N3 = lambda t: 0.05 * (
        ((5.76e9 / 0.47) / 0.03 + (7.2e9 - 5.76e9 / 0.47) / 0.5)
        - ((5.76e9 / 0.47) / 0.03) * np.exp(-0.03 * t)
        - ((7.2e9 - 5.76e9 / 0.47) / 0.5) * np.exp(-0.5 * t)
    )

    t_eval = np.linspace(0, 200, 1000)
    fp = os.path.join(toml_dir, "test-eg3.toml")
    model = solve_dcm_from_toml(fp, t_eval)

    assert np.allclose(
        model.activity()[0][0], nuclei_to_activity(N1(t_eval), 0.03), rtol=4e-3
    )
    assert np.allclose(
        model.activity()[1][0], nuclei_to_activity(N2(t_eval), 0.5), rtol=4e-3
    )
    assert np.all(model.activity()[2][0] == 0)
    rel_error3 = 100 * np.abs(model.nuclei[2][0][1:] - N3(t_eval)[1:]) / N3(t_eval)[1:]
    assert np.all(rel_error3 < 0.06)


def test_eg3_itac():
    """
    Same but first layer supplied as input time activity curve

    2 unstable nuclides and 1 stable nuclide, 1st nuclide sometimes decays to 2nd and 2nd nuclide sometimes decays to 3rd
    1 compartment

    Layer 1:
    dN1/dt = - lambda1 * N1
    A1(0) = 2 MBq
    N1(0) = 2 * 1e6 * 60 * 60 / 0.03 = 2.4e11
    lambda1 = 0.03 h-1

    Layer 2:
    dN2/dt = -lambda2 * N2 + branching_frac21 * lambda1 * N1
    lambda_2 = 0.5 h-1
    A2(0) = 1 MBq
    N2(0) = 1 * 1e6 * 60 * 60 / 0.5 = 7.2e9
    branching_frac21 = 0.8

    Layer 3:
    dN3/dt = branching_frac32 * lambda2 * N2
    N3(0) = 0
    branching_frac32 = 0.1
    """
    # analytical soln
    N1 = lambda t: (2.4e11) * np.exp(-0.03 * t)
    N2 = lambda t: (
        (5.76e9 / 0.47) * np.exp(0.47 * t) + (7.2e9 - 5.76e9 / 0.47)
    ) * np.exp(-0.5 * t)
    N3 = lambda t: 0.05 * (
        ((5.76e9 / 0.47) / 0.03 + (7.2e9 - 5.76e9 / 0.47) / 0.5)
        - ((5.76e9 / 0.47) / 0.03) * np.exp(-0.03 * t)
        - ((7.2e9 - 5.76e9 / 0.47) / 0.5) * np.exp(-0.5 * t)
    )

    prelayer = Prelayer(
        0.03,
        np.array([0.8, 0]),
        [
            lambda t: nuclei_to_activity(N1(t), 0.03),
        ],
    )
    t_eval = np.linspace(0, 200, 1000)
    fp = os.path.join(toml_dir, "test-eg3-itac.toml")
    model = solve_dcm_from_toml(fp, t_eval, prelayer=prelayer)
    assert np.allclose(
        model.activity()[0][0], nuclei_to_activity(N2(t_eval), 0.5), rtol=4e-3
    )
    assert np.all(model.activity()[1][0] == 0)
    rel_error3 = 100 * np.abs(model.nuclei[1][0][1:] - N3(t_eval)[1:]) / N3(t_eval)[1:]
    assert np.all(rel_error3 < 0.1)


def test_eg4():
    """
    1 unstable nuclide and two of its daughters, both of which are stable
    1 compartment

    Layer 1:
    dN1/dt = - lambda1 * N1
    A1(0) = 2 MBq
    N1(0) = 2 * 1e6 * 60 * 60 / 0.03 = 2.4e11
    lambda1 = 0.03 h-1

    Layer 2:
    dN2/dt = branching_frac21 * lambda1 * N1
    N2(0) = 0
    branching_frac21 = 0.8

    Layer 3:
    dN3/dt = branching_frac31 * lambda1 * N1
    N3(0) = 0
    branching_frac31 = 0.1
    """
    # analytical soln
    N1 = lambda t: (2.4e11) * np.exp(-0.03 * t)
    N2 = lambda t: (5.76e9 / 0.03) * (1 - np.exp(-0.03 * t))
    N3 = lambda t: (7.2e8 / 0.03) * (1 - np.exp(-0.03 * t))

    t_eval = np.linspace(0, 200, 1000)
    fp = os.path.join(toml_dir, "test-eg4.toml")
    model = solve_dcm_from_toml(fp, t_eval)

    assert np.allclose(
        model.activity()[0][0], nuclei_to_activity(N1(t_eval), 0.03), rtol=4e-3
    )
    assert np.all(model.activity()[1][0] == 0)
    rel_error2 = 100 * np.abs(model.nuclei[1][0][1:] - N2(t_eval)[1:]) / N2(t_eval)[1:]
    assert np.all(rel_error2 < 0.08)
    assert np.all(model.activity()[2][0] == 0)
    rel_error3 = 100 * np.abs(model.nuclei[2][0][1:] - N3(t_eval)[1:]) / N3(t_eval)[1:]
    assert np.all(rel_error3 < 0.08)


def test_eg4_itac():
    """
    Same but first layer supplied as input time activity curve

    1 unstable nuclide and two of its daughters, both of which are stable
    1 compartment

    Layer 1:
    dN1/dt = - lambda1 * N1
    A1(0) = 2 MBq
    N1(0) = 2 * 1e6 * 60 * 60 / 0.03 = 2.4e11
    lambda1 = 0.03 h-1

    Layer 2:
    dN2/dt = branching_frac21 * lambda1 * N1
    N2(0) = 0
    branching_frac21 = 0.8

    Layer 3:
    dN3/dt = branching_frac31 * lambda1 * N1
    N3(0) = 0
    branching_frac31 = 0.1
    """
    # analytical soln
    N1 = lambda t: (2.4e11) * np.exp(-0.03 * t)
    N2 = lambda t: (5.76e9 / 0.03) * (1 - np.exp(-0.03 * t))
    N3 = lambda t: (7.2e8 / 0.03) * (1 - np.exp(-0.03 * t))

    prelayer = Prelayer(
        0.03,
        np.array([0.8, 0.1]),
        [lambda t: nuclei_to_activity(N1(t), 0.03)],
    )
    t_eval = np.linspace(0, 200, 1000)
    fp = os.path.join(toml_dir, "test-eg4-itac.toml")
    model = solve_dcm_from_toml(fp, t_eval, prelayer=prelayer)

    assert np.all(model.activity()[0][0] == 0)
    rel_error2 = 100 * np.abs(model.nuclei[0][0][1:] - N2(t_eval)[1:]) / N2(t_eval)[1:]
    assert np.all(rel_error2 < 0.11)
    assert np.all(model.activity()[1][0] == 0)
    rel_error3 = 100 * np.abs(model.nuclei[1][0][1:] - N3(t_eval)[1:]) / N3(t_eval)[1:]
    assert np.all(rel_error3 < 0.11)


def test_eg5():
    """
    1 stable nuclide
    2 compartments, with transfer from one to another

    +--------+           +--------+
    |        |           |        |
    |   C1   |           |   C2   |
    |        | --------> |        |
    +--------+    M21    +--------+

    dN1/dt = - M21 * N1
    dN2/dt = M21 * N1
    N1(0) = 2e12
    N2(0) = 0
    M21 = 5 h-1
    """
    # analytical soln
    N1 = lambda t: 2e17 * np.exp(-5 * t)
    N2 = lambda t: 2e17 * (1 - np.exp(-5 * t))

    t_eval = np.linspace(0, 3, 1000)
    fp = os.path.join(toml_dir, "test-eg5.toml")
    model = solve_dcm_from_toml(fp, t_eval)

    assert np.all(model.activity()[0][0] == 0)
    assert np.all(model.activity()[0][1] == 0)
    rel_error1 = 100 * np.abs(model.nuclei[0][0] - N1(t_eval)) / N1(t_eval)
    print(rel_error1)
    assert np.all(rel_error1 < 1.1)
    rel_error2 = 100 * np.abs(model.nuclei[0][1][1:] - N2(t_eval)[1:]) / N2(t_eval)[1:]
    assert np.all(rel_error2 < 1.1)


def test_eg6():
    """
    1 stable nuclide
    2 compartments, with transfer both ways

    +--------+    M12    +--------+
    |        | <-------- |        |
    |   C1   |           |   C2   |
    |        | --------> |        |
    +--------+    M21    +--------+

    dN1/dt = - M21 * N1 + M12 * N2
    dN2/dt = - M12 * N2 + M21 * N1
    N1(0) = 2
    N2(0) = 5
    M21 = 5 h-1
    M12 = 4 h-1

    dy1/dt = - 5y1 + 4y2
    dy2/dt = - 4y2 + 5y1
    y1(0) = 2
    y2(0) = 5

    """
    # analytical soln
    N1 = lambda t: 28 / 9 - 10 * np.exp(-9 * t) / 9
    N2 = lambda t: 35 / 9 + 10 * np.exp(-9 * t) / 9

    t_eval = np.linspace(0, 1, 1000)
    fp = os.path.join(toml_dir, "test-eg6.toml")
    model = solve_dcm_from_toml(fp, t_eval)

    assert np.all(model.activity()[0][0] == 0)
    assert np.all(model.activity()[0][1] == 0)
    rel_error1 = 100 * np.abs(model.nuclei[0][0] - N1(t_eval)) / N1(t_eval)
    assert np.all(rel_error1 < 0.04)
    rel_error2 = 100 * np.abs(model.nuclei[0][1] - N2(t_eval)) / N2(t_eval)
    assert np.all(rel_error2 < 0.04)


def test_eg7():
    """
    1 stable nuclide
    3 compartments, with transfer from 1st to 3rd and 2nd to 3rd

    +--------+           +--------+
    |        |           |        |
    |   C1   |           |   C2   |
    |        |           |        |
    +--------+           +--------+
        |                     |
        |                     |
    M31 |                     | M32
        |      +--------+     |
        +----> |        | <---+
               |   C3   |
               |        |
               +--------+

    dN1/dt = - M31 * N1
    dN2/dt = - M32 * N2
    dN3/dt = M31 * N1 + M32 * N2
    N1(0) = 2
    N2(0) = 5
    N3(0) = 2
    M31 = 2 h-1
    M32 = 4 h-1
    """
    # analytical soln
    N1 = lambda t: 2 * np.exp(-2 * t)
    N2 = lambda t: 5 * np.exp(-4 * t)
    N3 = lambda t: 9 - 2 * np.exp(-2 * t) - 5 * np.exp(-4 * t)

    t_eval = np.linspace(0, 1, 1000)
    fp = os.path.join(toml_dir, "test-eg7.toml")
    model = solve_dcm_from_toml(fp, t_eval)

    assert np.all(model.activity()[0][0] == 0)
    assert np.all(model.activity()[0][1] == 0)
    assert np.all(model.activity()[0][2] == 0)
    rel_error1 = 100 * np.abs(model.nuclei[0][0] - N1(t_eval)) / N1(t_eval)
    assert np.all(rel_error1 < 0.04)
    rel_error2 = 100 * np.abs(model.nuclei[0][1] - N2(t_eval)) / N2(t_eval)
    assert np.all(rel_error2 < 0.4)
    rel_error3 = 100 * np.abs(model.nuclei[0][2] - N3(t_eval)) / N3(t_eval)
    assert np.all(rel_error3 < 0.04)


def test_eg8():
    """
    1 stable nuclide
    3 compartments, with transfer from 1st to both 2nd and 3rd
    (use the same params as test_3l1c_b2s2b() -- equivalence of layers and cmpts)

               +--------+
               |        |
        +----- |   C1   | ----+
        |      |        |     |
    M21 |      +--------+     | M31
        |                     |
        |                     |
        v                     v
    +--------+           +--------+
    |        |           |        |
    |   C2   |           |   C3   |
    |        |           |        |
    +--------+           +--------+


    dN1/dt = - (M21 + M31) * N1
    dN2/dt = + M21 * N1
    dN3/dt = + M31 * N1
    N1(0) = 2
    N2(0) = 0
    N3(0) = 0
    M21 = 2.55 h-1
    M31 = 0.45 h-1
    """
    # analytical soln
    N1 = lambda t: 2 * np.exp(-3 * t)
    N2 = lambda t: (5.1 / 3) * (1 - np.exp(-3 * t))
    N3 = lambda t: (0.9 / 3) * (1 - np.exp(-3 * t))

    t_eval = np.linspace(0, 1, 1000)
    fp = os.path.join(toml_dir, "test-eg8.toml")
    model = solve_dcm_from_toml(fp, t_eval)

    assert np.all(model.activity()[0][0] == 0)
    assert np.all(model.activity()[0][1] == 0)
    assert np.all(model.activity()[0][2] == 0)
    rel_error1 = 100 * np.abs(model.nuclei[0][0] - N1(t_eval)) / N1(t_eval)
    assert np.all(rel_error1 < 0.2)
    rel_error2 = 100 * np.abs(model.nuclei[0][1][1:] - N2(t_eval)[1:]) / N2(t_eval)[1:]
    assert np.all(rel_error2 < 0.02)
    rel_error3 = 100 * np.abs(model.nuclei[0][2][1:] - N3(t_eval)[1:]) / N3(t_eval)[1:]
    assert np.all(rel_error3 < 0.02)


def test_eg9():
    """
    1 unstable nuclide
    2 compartments, with transfer both ways

    +--------+    M12    +--------+
    |        | <-------- |        |
    |   C1   |           |   C2   |
    |        | --------> |        |
    +--------+    M21    +--------+

    dN1/dt = - (M21 + lambda) * N1 + M12 * N2
    dN2/dt = - (M12 + lambda) * N2 + M21 * N1
    A1(0) = 2 MBq
    N1(0) = 2 * 1e6 * 60 * 60 / 0.1 = 7.2e10
    N2(0) = 0
    M21 = 5 h-1
    M12 = 4 h-1
    lambda = 0.1 h-1
    """
    # analytical soln
    N1 = lambda t: 4e10 * np.exp(-9.1 * t) + 3.2e10 * np.exp(-0.1 * t)
    N2 = lambda t: -4e10 * np.exp(-9.1 * t) + 4e10 * np.exp(-0.1 * t)

    t_eval = np.linspace(0, 3, 1000)
    fp = os.path.join(toml_dir, "test-eg9.toml")
    model = solve_dcm_from_toml(fp, t_eval)

    assert np.allclose(
        model.activity()[0][0],
        nuclei_to_activity(N1(t_eval), 0.1),
        rtol=2e-3,
    )
    assert np.allclose(
        model.activity()[0][1],
        nuclei_to_activity(N2(t_eval), 0.1),
        rtol=2e-3,
    )


def test_eg10():
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

    t_eval = np.linspace(0, 3, 1000)
    fp = os.path.join(toml_dir, "test-eg10.toml")
    model = solve_dcm_from_toml(fp, t_eval)
    assert np.allclose(
        model.activity()[0][0],
        nuclei_to_activity(N11(t_eval), 0.1),
        rtol=5e-4,
    )
    assert np.allclose(
        model.activity()[0][1],
        nuclei_to_activity(N12(t_eval), 0.1),
        rtol=5e-4,
    )
    rel_error21 = (
        100 * np.abs(model.nuclei[1][0][1:] - N21(t_eval[1:])) / N21(t_eval[1:])
    )
    assert np.all(rel_error21 < 0.05)
    rel_error22 = 100 * np.abs(model.nuclei[1][1] - N22(t_eval)) / N22(t_eval)
    assert np.all(rel_error22 < 0.07)


def test_eg10_itac():
    """
    Same but first layer supplied as input time activity curve

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

    prelayer = Prelayer(
        0.1,
        np.array([0.3]),
        [
            lambda t: nuclei_to_activity(N11(t), 0.1),
            lambda t: nuclei_to_activity(N12(t), 0.1),
        ],
    )
    t_eval = np.linspace(0, 3, 1000)
    fp = os.path.join(toml_dir, "test-eg10-itac.toml")
    model = solve_dcm_from_toml(fp, t_eval, prelayer=prelayer)
    rel_error21 = (
        100 * np.abs(model.nuclei[0][0][1:] - N21(t_eval[1:])) / N21(t_eval[1:])
    )
    assert np.all(rel_error21 < 0.1)
    rel_error22 = 100 * np.abs(model.nuclei[0][1] - N22(t_eval)) / N22(t_eval)
    assert np.all(rel_error22 < 0.12)


def test_eg11():
    """
    2 unstable nuclides
    3 compartments
    there is transfer of the first nuclide between cmpts 1 and 2 in both ways
    there is transfer of the second nuclide from cmpt 2 to 3

    Layer 1:
    +--------+    M112   +--------+
    |        | <-------- |        |
    |   C1   |           |   C2   |
    |        | --------> |        |
    +--------+    M121   +--------+


                         +--------+
                         |        |
                         |   C3   |
                         |        |
                         +--------+

    dN11/dt = - (M121 + lambda1) * N11 + M112 * N12
    dN12/dt = - (M112 + lambda1) * N12 + M121 * N11
    dN13/dt = - lambda * N13
    A11(0) = 2 MBq
    N11(0) = 2 * 1e6 * 60 * 60 / 0.1 = 7.2e10
    N12(0) = 0
    A13(0) = 3 MBq
    N13(0) = 3 * 1e6 * 60 * 60 / 0.1 = 1.08e11
    M121 = 0.5 h-1
    M112 = 0.4 h-1
    lambda1 = 0.1 h-1

    Layer 2:
    +--------+        +--------+
    |        |        |        |
    |   C1   |        |   C2   |
    |        |        |        |
    +--------+        +--------+
                          |
                         M232
                          |
                          V
                      +--------+
                      |        |
                      |   C3   |
                      |        |
                      +--------+

    dN21/dt = - lambda2 * N21 + branching_frac21 * lambda1 * N11
    dN22/dt = - (lambda2 + M232) * N22 + branching_frac21 * lambda1 * N12
    dN23/dt = - lambda2 * N23 + branching_frac21 * lambda1 * N13 + M232 * N22
    A21(0) = 2 MBq
    N21(0) = 2 * 1e6 * 60 * 60 / 0.4 =1.8e10
    A22(0) = 1 MBq
    N22(0) = 1 * 1e6 * 60 * 60 / 0.4 =9e9
    N23(0) = 0
    M232 = 0.8 h-1
    lambda2 = 0.4 h-1
    branching_frac21 = 0.9
    """
    # analytical soln
    N11 = lambda t: 4e10 * np.exp(-1.0 * t) + 3.2e10 * np.exp(-0.1 * t)
    N12 = lambda t: -4e10 * np.exp(-1.0 * t) + 4e10 * np.exp(-0.1 * t)
    N13 = lambda t: 1.08e11 * np.exp(-0.1 * t)
    N21 = (
        lambda t: -6e9 * np.exp(-1.0 * t)
        + 1.44e10 * np.exp(-0.4 * t)
        + 9.6e9 * np.exp(-0.1 * t)
    )
    N22 = (
        lambda t: 261 / 11 * 1e9 * np.exp(-1.2 * t)
        - 1.8e10 * np.exp(-1.0 * t)
        + (360 / 11) * 1e8 * np.exp(-0.1 * t)
    )
    # I cannot find the analytical soln to N23

    t_eval = np.linspace(0, 3, 1000)
    fp = os.path.join(toml_dir, "test-eg11.toml")
    model = solve_dcm_from_toml(fp, t_eval)

    assert np.allclose(
        model.activity()[0][0],
        nuclei_to_activity(N11(t_eval), 0.1),
        rtol=2e-4,
    )
    assert np.allclose(
        model.activity()[0][1],
        nuclei_to_activity(N12(t_eval), 0.1),
        rtol=1e-3,
    )
    assert np.allclose(
        model.activity()[0][2],
        nuclei_to_activity(N13(t_eval), 0.1),
        rtol=2e-4,
    )

    assert np.allclose(
        model.activity()[1][0],
        nuclei_to_activity(N21(t_eval), 0.4),
        rtol=2e-5,
    )
    assert np.allclose(
        model.activity()[1][1],
        nuclei_to_activity(N22(t_eval), 0.4),
        rtol=5e-4,
    )


def test_eg11_itac():
    """
    Same but first layer supplied as input time activity curve

    2 unstable nuclides
    3 compartments
    there is transfer of the first nuclide between cmpts 1 and 2 in both ways
    there is transfer of the second nuclide from cmpt 2 to 3

    Layer 1:
    +--------+    M112   +--------+
    |        | <-------- |        |
    |   C1   |           |   C2   |
    |        | --------> |        |
    +--------+    M121   +--------+


                         +--------+
                         |        |
                         |   C3   |
                         |        |
                         +--------+

    dN11/dt = - (M121 + lambda1) * N11 + M112 * N12
    dN12/dt = - (M112 + lambda1) * N12 + M121 * N11
    dN13/dt = - lambda * N13
    A11(0) = 2 MBq
    N11(0) = 2 * 1e6 * 60 * 60 / 0.1 = 7.2e10
    N12(0) = 0
    A13(0) = 3 MBq
    N13(0) = 3 * 1e6 * 60 * 60 / 0.1 = 1.08e11
    M121 = 0.5 h-1
    M112 = 0.4 h-1
    lambda1 = 0.1 h-1

    Layer 2:
    +--------+        +--------+
    |        |        |        |
    |   C1   |        |   C2   |
    |        |        |        |
    +--------+        +--------+
                          |
                         M232
                          |
                          V
                      +--------+
                      |        |
                      |   C3   |
                      |        |
                      +--------+

    dN21/dt = - lambda2 * N21 + branching_frac21 * lambda1 * N11
    dN22/dt = - (lambda2 + M232) * N22 + branching_frac21 * lambda1 * N12
    dN23/dt = - lambda2 * N23 + branching_frac21 * lambda1 * N13 + M232 * N22
    A21(0) = 2 MBq
    N21(0) = 2 * 1e6 * 60 * 60 / 0.4 =1.8e10
    A22(0) = 1 MBq
    N22(0) = 1 * 1e6 * 60 * 60 / 0.4 =9e9
    N23(0) = 0
    M232 = 0.8 h-1
    lambda2 = 0.4 h-1
    branching_frac21 = 0.9
    """
    # analytical soln
    N11 = lambda t: 4e10 * np.exp(-1.0 * t) + 3.2e10 * np.exp(-0.1 * t)
    N12 = lambda t: -4e10 * np.exp(-1.0 * t) + 4e10 * np.exp(-0.1 * t)
    N13 = lambda t: 1.08e11 * np.exp(-0.1 * t)
    N21 = (
        lambda t: -6e9 * np.exp(-1.0 * t)
        + 1.44e10 * np.exp(-0.4 * t)
        + 9.6e9 * np.exp(-0.1 * t)
    )
    N22 = (
        lambda t: 261 / 11 * 1e9 * np.exp(-1.2 * t)
        - 1.8e10 * np.exp(-1.0 * t)
        + (360 / 11) * 1e8 * np.exp(-0.1 * t)
    )
    # I cannot find the analytical soln to N23

    prelayer = Prelayer(
        0.1,
        np.array([0.9]),
        [
            lambda t: nuclei_to_activity(N11(t), 0.1),
            lambda t: nuclei_to_activity(N12(t), 0.1),
            lambda t: nuclei_to_activity(N13(t), 0.1),
        ],
    )
    t_eval = np.linspace(0, 3, 1000)
    fp = os.path.join(toml_dir, "test-eg11-itac.toml")
    model = solve_dcm_from_toml(fp, t_eval, prelayer=prelayer)
    assert np.allclose(
        model.activity()[0][0],
        nuclei_to_activity(N21(t_eval), 0.4),
        rtol=2e-5,
    )
    assert np.allclose(
        model.activity()[0][1],
        nuclei_to_activity(N22(t_eval), 0.4),
        rtol=5e-4,
    )


def test_linearity_1cmpt():
    # initial activity in only 1 compartment in 1st layer
    # NB linearity requires zero initial nuclei for layers after 1st
    fp = os.path.join(toml_dir, "test-linearity-1cmpt.toml")
    (
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        _,
        _,
    ) = _dcm_read_toml(fp)
    t_eval = np.linspace(0, 100, 1000)
    model = solve_dcm_from_toml(
        fp,
        t_eval,
    )
    model_dbl = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei * 2,
        t_eval,
    )
    assert np.allclose(model.activity() * 2, model_dbl.activity(), rtol=3e-3)


def test_linearity_ncmpt():
    # initial activity in multiple compartment of first layer
    # NB linearity requires zero initial nuclei for layers after 1st
    fp = os.path.join(toml_dir, "test-linearity-ncmpt.toml")
    (
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        _,
        _,
    ) = _dcm_read_toml(fp)
    t_eval = np.linspace(0, 200, 1000)
    model = solve_dcm_from_toml(
        fp,
        t_eval,
    )
    model_dbl = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei * 2,
        t_eval,
    )
    assert np.allclose(model.activity() * 2, model_dbl.activity(), rtol=1e-5)


def test_linearity_itac():
    # works for arbitrary input tacs too
    # NB linearity requires zero initial nuclei for layers after 1st
    fp = os.path.join(toml_dir, "test-linearity-itac.toml")
    (
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        _,
        _,
    ) = _dcm_read_toml(fp)
    A11 = lambda t: 1 * np.exp(-3 * t) + 4 * np.exp(-0.002 * t)
    A12 = lambda t: 0.01 * (np.exp(0.03 * t) - 1)
    A13 = lambda _: 10
    prelayer = Prelayer(
        0.1,
        np.array([1, 0, 0]),
        [
            A11,
            A12,
            A13,
        ],
    )
    t_eval = np.linspace(0, 200, 1000)
    model = solve_dcm_from_toml(
        fp,
        t_eval,
        prelayer=prelayer,
    )
    prelayer_dbl = Prelayer(
        0.1,
        np.array([1, 0, 0]),
        [
            lambda t: A11(t) * 2,
            lambda t: A12(t) * 2,
            lambda t: A13(t) * 2,
        ],
    )
    model_dbl = solve_dcm(
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        t_eval,
        prelayer=prelayer_dbl,
    )
    assert np.allclose(model.activity() * 2, model_dbl.activity(), rtol=1e-5)


def test_time_symmetry():
    """
    The initial state is the state at the beggining of t_eval.
    This test shows that t_eval does not have to start at time zero.

    E.g. 1 MBq of Ac-225:
    + trans rate 0.0028881132523331052 h-1
    + 20 half-lives = 4800 h
    + 1 MBq -> 1 * 1e6 * 60 * 60 / 0.0028881132523331052 = 1246488515328.0645

    1 unstable nuclide, 1 compartment

    dN/dt = - lambda * N
    N(0) = 1246488515328.0645
    lambda = 0.0028881132523331052 h-1
    """
    # analytical soln
    N = lambda t: (1246488515328.0645) * np.exp(-0.0028881132523331052 * t)

    t_eval = np.linspace(0, 4800)
    fp = os.path.join(toml_dir, "test-exp-ac225.toml")
    model = solve_dcm_from_toml(
        fp,
        t_eval,
    )
    assert np.allclose(
        model.activity()[0][0],
        nuclei_to_activity(N(t_eval), 0.0028881132523331052),
        rtol=6e-3,
    )
    # time symmetry
    model_shift = solve_dcm_from_toml(
        fp,
        t_eval + 2313.4,
    )
    assert np.allclose(
        model.activity()[0][0],
        model_shift.activity()[0][0],
    )


def test_voiding_2l():
    """99Mo to 99mTc generator.

    + initially 10 GBq of 99Mo
    + 99mTc is eluted at 0 h, 24 h, 48 h

    99Mo to 99mTc branching fraction is 0.89

    activity of 99mTc reaches its maximum value:
    a = franching_frac * activity 99Mo
    at this time after elution:
    t = (trans rate 99Mo - trans rate 99mTc)^(-1)
    * ln(trans rate 99Mo / trans rate 99mTc )
    """
    pass


def test_voiding_2l2c():
    """99Mo to 99mTc generator, but there is 1 compartment
    representing the generator and another compartment
    that is independent of the first.


    + initially 10 GBq of 99Mo
    + 99mTc is eluted at 0 h, 24 h, 48 h

    99Mo to 99mTc branching fraction is 0.89

    activity of 99mTc reaches its maximum value:
    a = franching_frac * activity 99Mo
    at this time after elution:
    t = (trans rate 99Mo - trans rate 99mTc)^(-1)
    * ln(trans rate 99Mo / trans rate 99mTc )
    """
    pass


def test_voiding_1l2c():
    """99Mo to 99mTc generator, but simulated as transfer
    between compartments in a single layer instead of the
    usual way (i.e. transfer between layers in a single
    compartment).

    + initially 10 GBq of 99Mo
    + 99mTc is eluted at 0 h, 24 h, 48 h

    99Mo to 99mTc branching fraction is 0.89

    activity of 99mTc reaches its maximum value:
    a = franching_frac * activity 99Mo
    at this time after elution:
    t = (trans rate 99Mo - trans rate 99mTc)^(-1)
    * ln(trans rate 99Mo / trans rate 99mTc )
    """
    pass
