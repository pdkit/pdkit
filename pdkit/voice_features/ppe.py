import numpy as np
from scipy.signal import lfilter


def compute_ppe(F0_track: np.ndarray, hc_ref_hz: float = 120.0, ar_order: int = 10) -> float:
    F0_valid = np.asarray(F0_track)
    F0_valid = F0_valid[np.isfinite(F0_valid) & (F0_valid > 0)]
    if F0_valid.size == 0:
        return np.nan
    try:
        logF0 = np.log(F0_valid / float(hc_ref_hz))
        logF0 = logF0[np.isfinite(logF0)]
        if logF0.size < (ar_order + 2):
            return np.nan
        x = logF0 - np.mean(logF0)

        acf_full = np.correlate(x, x, mode='full')
        acf = acf_full[len(x) - 1 : len(x) + ar_order]
        acf = acf / (acf[0] + 1e-12)

        p = ar_order
        R = np.empty((p, p))
        for i in range(p):
            for j in range(p):
                R[i, j] = acf[abs(i - j)]

        r = acf[1 : p + 1]
        a = -np.linalg.solve(R, r)
        A = np.concatenate(([1.0], a))

        sig_filtered = lfilter(A, [1.0], x)
        discard = min(sig_filtered.size, 25)
        sig_filtered = sig_filtered[discard:]

        if sig_filtered.size < 2 or not np.all(np.isfinite(sig_filtered)):
            return np.nan
        
        bins = np.linspace(sig_filtered.min(), sig_filtered.max(), 100)
        counts, _ = np.histogram(sig_filtered, bins=bins, density=False)
        total = counts.sum()

        if total == 0:
            return np.nan
        
        p_hist = counts / total
        p_hist = p_hist[p_hist > 0]

        return float(-np.sum(p_hist * np.log(p_hist)) / np.log(len(p_hist)))
    
    except Exception:
        return np.nan
