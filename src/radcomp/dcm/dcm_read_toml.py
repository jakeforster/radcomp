import numpy as np
import warnings

try:
    import tomllib
except ModuleNotFoundError:  # for Python < 3.11
    import tomli as tomllib

from radcomp.common.utils import activity_to_nuclei


def _dcm_read_toml(
    filepath: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str] | None]:
    """Read parameters for deterministic compartment model from TOML configuration file.

    Parameters
    ----------
    filepath : str
        Filepath to TOML file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, list[str], list[str] | None]
        + trans_rates : Transition rates (h-1) of nuclides in layers. Shape (num_layers,).
        + branching_fracs : Branching fractions (0 to 1). Shape (num_layers, num_layers). `branching_fracs`[i,j] is for layer j to layer i.
        + xfer_coeffs : Transfer coefficients (h-1) between compartments. Shape (num_layers, num_compartments, num_compartments). `xfer_coeffs`[i,j,k] is for compartment k to compartment j in layer i.
        + initial_nuclei : Number of nuclei in each compartment in each layer at `t_eval[0]`. Shape (num_layers, num_compartments). `initial_nuclei`[i,j] is for layer i, compartment j.
        + layer_names : Names of layers in model.
        + compartment_names : Names of compartments in model.
    """
    with open(filepath, mode="rb") as fp:
        cm = tomllib.load(fp)
    assert "layer" in cm
    layers = cm["layer"]
    num_layers = len(layers)

    if "compartments" in cm:
        compartment_names = cm["compartments"]["names"]
        num_compartments = len(compartment_names)
    else:
        compartment_names = None
        assert "initial_MBq" in layers[0] or "initial_nuclei" in layers[0]
        if "initial_MBq" in layers[0]:
            num_compartments = len(layers[0]["initial_MBq"])
        else:
            num_compartments = len(layers[0]["initial_nuclei"])

    if num_compartments == 1 and any("xfer_coeffs_h-1" in l for l in layers):
        warnings.warn("Ignoring xfer_coeffs_h-1 (1 compartment).")

    trans_rates = np.zeros(num_layers)
    initial_nuclei = np.zeros((num_layers, num_compartments))
    xfer_coeffs = np.zeros((num_layers, num_compartments, num_compartments))
    layer_names = []
    branching_fracs = np.zeros((num_layers, num_layers))
    for i, l in enumerate(layers):
        layer_name = f"Nuclide {i+1}" if "name" not in l else l["name"]
        layer_names.append(layer_name)

        trans_rates[i] = l["trans_rate_h-1"]

        if l["trans_rate_h-1"] != 0:
            initial_MBq_l = np.array(l["initial_MBq"])
            initial_nuclei_l = activity_to_nuclei(initial_MBq_l, l["trans_rate_h-1"])
            if "initial_nuclei" in l:
                warnings.warn(
                    f"Ignoring initial_nuclei for {layer_name}; read initial_MBq instead (unstable)."
                )
        else:
            initial_nuclei_l = np.array(l["initial_nuclei"])
            if "initial_MBq" in l:
                warnings.warn(
                    f"Ignoring initial_MBq for {layer_name}; read initial_nuclei instead (stable)."
                )
        assert initial_nuclei_l.shape == (num_compartments,)
        initial_nuclei[i] = initial_nuclei_l

        if num_compartments > 1:
            xfer_coeffs_l = np.array(l["xfer_coeffs_h-1"])
            assert xfer_coeffs_l.shape == (num_compartments, num_compartments)
            xfer_coeffs[i] = xfer_coeffs_l

        if l["trans_rate_h-1"] == 0 and "branching_fracs" in l:
            warnings.warn(f"Ignoring branching_fracs for {layer_name} (stable).")
        if (i == len(layers) - 1) and "branching_fracs" in l:
            warnings.warn(f"Ignoring branching_fracs for {layer_name} (last layer).")
        if num_layers > 1 and (i != len(layers) - 1) and l["trans_rate_h-1"] != 0:
            assert "branching_fracs" in l
            assert isinstance(l["branching_fracs"], list)
            branching_fracs_l = np.array(l["branching_fracs"])
            assert branching_fracs_l.shape == (num_layers - (i + 1),)
            branching_fracs_l = np.insert(branching_fracs_l, 0, np.zeros(i + 1))
            assert branching_fracs_l.shape == (num_layers,)
            branching_fracs[:, i] = branching_fracs_l

        unknown_keys = [
            key
            for key in l
            if key
            not in [
                "name",
                "trans_rate_h-1",
                "branching_fracs",
                "initial_MBq",
                "initial_nuclei",
                "xfer_coeffs_h-1",
            ]
        ]
        if unknown_keys:
            warnings.warn(
                f"Ignorning unknown key(s) in {layer_name} layer table: {unknown_keys}"
            )

    return (
        trans_rates,
        branching_fracs,
        xfer_coeffs,
        initial_nuclei,
        layer_names,
        compartment_names,
    )
