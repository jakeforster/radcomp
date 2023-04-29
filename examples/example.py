import numpy as np
import radcomp

t_eval = np.linspace(0, 24, 1000)  # h
model = radcomp.solve_dcm_from_toml("example.toml", t_eval)
print(model.info_xfer())
print(model.info_growth())
model.plot()
