import numpy as np
from .filters import frq2mel, mel2frq

def mel_filterbank(n_fft, fs, n_mels, fmin=0.0, fmax=None, power=False, endpoint_taper=True):
    if fmax is None:
        fmax = fs / 2.0

    mel_edges = np.linspace(frq2mel(fmin), frq2mel(fmax), num=n_mels + 2)
    f_edges = mel2frq(mel_edges)
    freqs = np.linspace(0.0, fs / 2.0, num=n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)), dtype=float)

    for m in range(n_mels):
        f_l, f_c, f_u = f_edges[m], f_edges[m + 1], f_edges[m + 2]
        left = (freqs >= f_l) & (freqs <= f_c)
        fb[m, left] = (freqs[left] - f_l) / max(f_c - f_l, np.finfo(float).eps)
        right = (freqs >= f_c) & (freqs <= f_u)
        fb[m, right] = (f_u - freqs[right]) / max(f_u - f_c, np.finfo(float).eps)
        
    if not endpoint_taper:
        fb[0, freqs <= f_edges[1]] = 1.0
        fb[-1, freqs >= f_edges[-2]] = 1.0
    s = fb.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    fb /= s
    fb *= 2.0
    
    return fb