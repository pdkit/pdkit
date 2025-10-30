import numpy as np

def frq2mel(f_hz):
    f = np.asarray(f_hz, dtype=float)
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel2frq(mel):
    m = np.asarray(mel, dtype=float)
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)