import numpy as np

from src.radcomp.common.voiding import Voiding, _VoidLayer, _ordered_voids_in_layer
from src.radcomp.dcm.dcm_internal import _solve_dcm


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

t_span = (0, 3)
t_eval = np.linspace(0, 3, 10)  # try None too
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

print(t_layers)
print(nuclei_layers)
