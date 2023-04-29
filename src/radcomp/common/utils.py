import numpy as np
from typing import TypeVar

T = TypeVar("T", float, np.ndarray)


def activity_to_nuclei(activity: T, trans_rate: float) -> T:
    """Convert activity (MBq) to number of nuclei.

    Parameters
    ----------
    activity : T
        Activity (MBq).
    trans_rate : float
        Transition rate (h-1).

    Returns
    -------
    T
        Number of nuclei.
    """
    return activity * 1e6 * 60 * 60 / trans_rate


def nuclei_to_activity(nuclei: T, trans_rate: float) -> T:
    """Convert number of nuclei to activity (MBq).

    Parameters
    ----------
    nuclei : T
        Number of nuclei.
    trans_rate : float
        Transition rate (h-1).

    Returns
    -------
    T
        Activity (MBq).
    """
    return nuclei * 1e-6 * trans_rate / (60 * 60)


def read_arrays(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a model solution saved to npz file.

    Parameters
    ----------
    filepath : str
        Filepath to npz file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        + t_eval : Times (h). Sorted (ascending). 1D.
        + nuclei : Shape (num_layers, num_compartments, len(t_eval)).
          Element at index [i, j, k] is the number of nuclei in layer i,
          compartment j at element k of t_eval.
        + activity : Shape (num_layers, num_compartments, len(t_eval)).
          Element at index [i, j, k] is the activity (MBq) in layer i,
          compartment j at element k of t_eval.
    """
    assert filepath.endswith(".npz")
    data = np.load(filepath)
    return data["t"], data["nuclei"], data["activity"]


def _save_arrays(
    filepath: str, t_eval: np.ndarray, nuclei: np.ndarray, activity: np.ndarray
) -> None:
    assert filepath.endswith(".npz")
    np.savez(filepath, t=t_eval, nuclei=nuclei, activity=activity)
