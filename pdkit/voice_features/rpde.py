import numpy as np
from pdkit.voice_features.native.close_ret.close_ret_hook import fast_close_ret


def compute_rpde(x, d=4, tau=50, eps=0.2, tmax=1000, metric="euclidean", standardize=True, step=1):
    if metric == "euclidean" and not standardize and step == 1:
        hist = fast_close_ret(x, m=d, tau=tau, eta=eps)
        
        if hist is not None:
            # Successfully got histogram from C implementation
            if hist.sum() == 0:
                return np.nan
            
            if tmax > 0 and len(hist) > tmax:
                hist = hist[:tmax]
            
            p = hist.astype(np.float64)
            p /= p.sum()
            
            p = p[p > 0]
            
            H = -np.sum(p * np.log(p))
            H_norm = H / np.log(len(hist))
            
            return float(H_norm)
    
    # Fall back to Python implementation
    x = np.asarray(x, dtype=np.float32)
    N = len(x)
    embN_full = N - (d - 1) * tau
    if embN_full <= 1:
        return np.nan

    idx = np.arange(embN_full)[::max(1, int(step))]
    embN = len(idx)
    if embN <= 1:
        return np.nan

    X = np.empty((embN, d), dtype=np.float32)
    for k in range(d):
        Xi = x[k * tau : k * tau + embN_full]
        X[:, k] = Xi[idx]

    if standardize:
        mu = X.mean(axis=0, dtype=np.float64)
        sd = X.std(axis=0, dtype=np.float64) + 1e-6
        X = (X - mu) / sd

    hist = np.zeros(tmax, dtype=np.int64)
    for i in range(embN - 1):
        x_i = X[i]
        W = min(tmax, embN - 1 - i)
        if W <= 0:
            break
        block = X[i+1 : i+1+W]
        if metric == "chebyshev":
            dists = np.max(np.abs(block - x_i), axis=1)
        else:  # euclidean
            diff = block - x_i
            dists = np.sqrt(np.sum(diff * diff, axis=1))
        hits = np.flatnonzero(dists <= eps)
        if hits.size:
            dt = int(hits[0] + 1)
            hist[dt-1] += 1

    if hist.sum() == 0:
        return np.nan

    p = hist.astype(np.float64)
    p /= p.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    H_norm = H / np.log(len(hist))
    return float(H_norm)