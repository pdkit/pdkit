import numpy as np

def compute_tkeo(x):
    x = np.asarray(x)
    energy = np.zeros_like(x)
    energy[0] = x[0] ** 2
    energy[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    energy[-1] = x[-1] ** 2
    return energy