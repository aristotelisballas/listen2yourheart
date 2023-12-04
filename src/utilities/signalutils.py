import numpy as np


def analogue2digital(sec: float, fs_hz: float) -> int:
    return np.round(sec * fs_hz)
