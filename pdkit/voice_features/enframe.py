import numpy as np

def compute_enframe(x, win, hop=None):
    x = np.asarray(x, dtype=float).reshape(-1)
    if np.isscalar(win):
        frame_len = int(win)
        w = np.hamming(frame_len)
    else:
        w = np.asarray(win, dtype=float).reshape(-1)
        frame_len = len(w)

    if hop is None:
        hop = frame_len // 2

    if frame_len <= 0 or hop <= 0:
        raise ValueError("frame length and hop must be positive")
    
    n_frames = 1 + max(0, (len(x) - frame_len) // hop)
    pad_needed = (n_frames - 1) * hop + frame_len - len(x)
    if pad_needed > 0:
        x = np.pad(x, (0, pad_needed), mode="constant")

    idx = (np.arange(n_frames)[:, None] * hop) + np.arange(frame_len)[None, :]
    frames = x[idx] * w[None, :]
    tc = idx[:, 0] + (frame_len - 1) / 2.0
    
    return frames, tc