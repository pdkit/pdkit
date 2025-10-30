import numpy as np
from .enframe import compute_enframe
from .melfbank import mel_filterbank
from .spectral import rfft, rdct
from .deltas import delta, delta_delta


def _next_pow2(n: int) -> int:
    return 1 << (int(np.ceil(np.log2(max(1, n)))))


def melcepst(s, fs, w=None, nc=12, p=None, n=None, inc=None, fl=0.0, fh=None):
    s = np.asarray(s, float).reshape(-1)
    w = (w or "").strip()
    if p is None:
        p = int(np.floor(3 * np.log(fs)))
    if n is None:
        n = _next_pow2(int(0.03 * fs))
    n = int(n)
    if inc is None:
        inc = n // 2

    if 'R' in w:
        win = np.ones(n)
    elif 'N' in w:
        win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1, n + 1) / (n + 1))
    else:
        win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))

    frames, tc = compute_enframe(s, win, inc)
    T, frame_len = frames.shape
    NFFT = _next_pow2(frame_len)
    S = rfft(frames, NFFT)
    mag = np.abs(S)
    use_power = 'p' in w
    spec = mag ** 2 if use_power else mag

    if fh is None:
        fh_hz = fs * 0.5
    else:
        fh_hz = fh * fs if fh < 1.0 else fh
    fl_hz = fl * fs if (fl < 1.0) else fl

    fb = mel_filterbank(NFFT, fs, n_mels=p, fmin=fl_hz, fmax=fh_hz,
                        power=use_power, endpoint_taper=('y' not in w))
    melE = np.dot(spec, fb.T)
    pth = np.max(melE) * 1e-20
    melE = np.maximum(melE, pth)
    log_mel = np.log(melE)

    c_full = rdct(log_mel, type=2, norm="ortho")
    want_c0 = '0' in w
    if want_c0:
        ccep = c_full[:, : (nc + 1)]
    else:
        ccep = c_full[:, 1: (nc + 1)]
    cols = [ccep]

    if 'E' in w:
        if use_power:
            pw_total = spec.sum(axis=1)
        else:
            pw_total = (spec ** 2).sum(axis=1)
        pw_total = np.maximum(pw_total, pth)
        loge = np.log(pw_total)[:, None]
        cols = [loge] + cols

    c = np.hstack(cols)

    if 'd' in w:
        d1 = delta(c)
        c = np.hstack([c, d1])
    if 'D' in w:
        if 'd' not in w:
            d1 = delta(c)
        d2 = delta_delta(c)
        c = np.hstack([c, d2])

    return c, tc