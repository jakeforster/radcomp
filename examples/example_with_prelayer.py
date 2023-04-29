import numpy as np
import radcomp

"""
This example shows how to add a "prelayer" to the model.
For demonstration, the analytical solution to the first layer of example.toml is provided as the prelayer.
The result is therefore equivalent to example.toml.
"""

# analytical soln (number of nuclei) to first layer of example.toml
n1 = lambda t: 4e10 * np.exp(-1.0 * t) + 3.2e10 * np.exp(-0.1 * t)
n2 = lambda t: -4e10 * np.exp(-1.0 * t) + 4e10 * np.exp(-0.1 * t)
n3 = lambda t: 1.08e11 * np.exp(-0.1 * t)

trans_rate = 0.1  # h-1
branching_fracs = np.array([0.9])
prelayer = radcomp.Prelayer(
    trans_rate,
    branching_fracs,
    [
        lambda t: radcomp.nuclei_to_activity(n1(t), trans_rate),
        lambda t: radcomp.nuclei_to_activity(n2(t), trans_rate),
        lambda t: radcomp.nuclei_to_activity(n3(t), trans_rate),
    ],
    "Radionuclide 1",
)

t_eval = np.linspace(0, 24, 1000)  # h
model = radcomp.solve_dcm_from_toml(
    "example_with_prelayer.toml", t_eval, prelayer=prelayer
)
print(model.info_xfer())
print(model.info_growth())
model.plot()
