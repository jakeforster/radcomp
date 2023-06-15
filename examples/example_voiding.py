import numpy as np
import radcomp

t_eval = np.linspace(0, 72, 1000)  # h
voiding = [radcomp.FractionalVoiding(np.array([24, 48]), np.array([[0, 0], [0, 1]]))]
model = radcomp.solve_dcm_from_toml("example_voiding.toml", t_eval, voiding=voiding)
print(model.info_xfer())
print(model.info_growth())
model.plot()
