import numpy as np
import radcomp

t_eval = np.linspace(0, 72, 1000)  # h
voiding_rules = [
    radcomp.VoidingRule(np.array([24, 48]), np.array([[0, 0], [0, 1]])),
]
model = radcomp.solve_dcm_from_toml(
    "example_voiding.toml", t_eval, voiding_rules=voiding_rules
)

print(model.info_xfer())
print(model.info_growth())

print(model.voided_activity())

model.plot()
