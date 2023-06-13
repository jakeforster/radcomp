import numpy as np
import radcomp

t_eval = np.linspace(0, 72, 1000)  # h
voiding_list = [radcomp.Voiding([24, 48], np.array([[0], [1]]))]
model = radcomp.solve_dcm_from_toml(
    "example_with_voiding.toml", t_eval, voiding_list=voiding_list
)
print(model.info_xfer())
print(model.info_growth())
model.plot()
