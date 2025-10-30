import logging
import numpy as np
from pdkit.voice_features.tkeo import compute_tkeo
from scipy.linalg import toeplitz, solve_toeplitz


def _levinson_durbin(r, order):
    a = np.zeros(order + 1, dtype=np.float64)
    e = r[0]

    if e <= 0:
        a[0] = 1.0
        return a
    
    a[0] = 1.0
    for k in range(1, order + 1):
        acc = r[k]
        for j in range(1, k):
            acc += a[j] * r[k - j]
        refl = -acc / e
        # update
        a_prev = a.copy()
        a[k] = refl
        for j in range(1, k):
            a[j] = a_prev[j] + refl * a_prev[k - j]
        e *= (1.0 - refl * refl)
        if e <= 1e-20:
            break

    return a


def compute_jitter_shimmer(A0, ar_method="levinson", normalize=False, smooth=None, baken_variant: str = "default"):
    A0 = np.asarray(A0, dtype=np.float64)

    if normalize:
        m = np.mean(np.abs(A0[np.isfinite(A0)]))
        if m > 0:
            A0 = A0 / m

    if smooth and len(A0) > 3:
        if smooth == 'median':
            from numpy import median
            k = 3
            sm = []
            for i in range(len(A0)):
                s = max(0, i - k)
                e = min(len(A0), i + k + 1)
                sm.append(median(A0[s:e]))
            A0 = np.asarray(sm)
        elif smooth == 'ma':
            k = 5
            kern = np.ones(k) / k
            A0 = np.convolve(A0, kern, mode='same')

    valid_mask = np.isfinite(A0)
    if not np.any(valid_mask) or len(A0) == 0:
        return np.full(22, np.nan)
    
    A0_valid = A0[valid_mask]
    if len(A0_valid) < 2:
        return np.full(22, np.nan)

    if np.var(A0_valid) < 1e-12:
        return np.full(22, np.nan)
    
    mean_A0 = np.mean(A0_valid)
    if abs(mean_A0) < 1e-12:
        return np.full(22, np.nan)

    measures = []
    abs_diff = np.mean(np.abs(np.diff(A0_valid)))
    measures.append(abs_diff)
    measures.append(100 * abs_diff /    abs(mean_A0))

    pq3 = perq1(A0_valid, 3, ar_method=ar_method, baken_variant=baken_variant)
    measures.extend(pq3)
    pq5 = perq1(A0_valid, 5, ar_method=ar_method, baken_variant=baken_variant)
    measures.extend(pq5)
    pq11 = perq1(A0_valid, 11, ar_method=ar_method, baken_variant=baken_variant)
    measures.extend(pq11)

    measures.append(np.mean(np.abs(A0_valid - mean_A0)))

    if len(A0_valid) > 1:
        ratio = A0_valid[:-1] / (A0_valid[1:] + 1e-12)
        ratio = np.clip(ratio, 1e-12, 1e12)
        log_vals = 20 * np.abs(np.log10(ratio))
        log_vals = log_vals[np.isfinite(log_vals)]
        if len(log_vals) > 0:
            measures.append(np.mean(log_vals))
        else:
            measures.append(np.nan)
    else:
        measures.append(np.nan)

    if len(A0_valid) > 1 and mean_A0 != 0:
        cv = np.mean(np.diff(A0_valid)**2) / (mean_A0**2)
        measures.append(cv)
    else:
        measures.append(np.nan)

    tkeo_95_5 = None
    try:
        tkeo_vals = compute_tkeo(A0_valid)
        if len(tkeo_vals) > 0 and np.any(np.isfinite(tkeo_vals)):
            tkeo_finite = tkeo_vals[np.isfinite(tkeo_vals)]
            if len(tkeo_finite) > 0:
                measures.append(np.mean(np.abs(tkeo_finite)))
                measures.append(np.std(tkeo_finite))
                prc = np.percentile(tkeo_finite, [5, 25, 50, 75, 95])
                measures.extend(prc[:4])
                tkeo_95_5 = prc[3] - prc[0]
            else:
                measures.extend([np.nan] * 5)
        else:
            measures.extend([np.nan] * 5)
    except Exception:
        measures.extend([np.nan] * 5)

    if len(A0_valid) > 0:
        max_val = np.max(A0_valid)
        min_val = np.min(A0_valid)
        if max_val + min_val != 0:
            am_measure = (max_val - min_val) / (max_val + min_val)
            measures.append(am_measure)
        else:
            measures.append(np.nan)
    else:
        measures.append(np.nan)

    if (tkeo_95_5):
        measures.append(tkeo_95_5)
    else:
        measures.append(np.nan)

    if len(measures) < 22:
        measures += [np.nan] * (22 - len(measures))
    elif len(measures) > 22:
        measures = measures[:22]

    logging.debug({
        'mean_A0': mean_A0,
        'len': len(A0_valid),
        'pq3': pq3,
        'pq5': pq5,
            'pq11': pq11,
            'normalize': normalize,
            'smooth': smooth,
            'baken_variant': baken_variant,
        })
    return np.array(measures)

def perq1(time_series, K, ar_method="levinson", baken_variant: str = "default"):
    time_series = np.asarray(time_series, dtype=np.float64)
    N = len(time_series)

    if N <= K:
        return np.nan, np.nan, np.nan

    mean_ts = np.mean(time_series)
    if abs(mean_ts) < 1e-12:
        return np.nan, np.nan, np.nan

    K1 = int(np.floor(K / 2.0 + 0.5))
    K2 = K - K1

    if K1 < 1 or K2 < 0 or K1 + K2 != K:
        return np.nan, np.nan, np.nan

    sum1 = 0.0
    sum2 = 0.0 
    sum2_abswin = 0.0 
    count = 0
    for i in range(K1, N - K2 + 1): 
        c = i - 1
        start = c - K2
        end = c + K2 + 1
        if start < 0 or end > N:
            continue 
        window = time_series[start:end]
        if window.size != K:
            continue
        center_val = time_series[c]
        sum1 += np.mean(np.abs(window - center_val))
        diff_c = np.mean(np.abs(window)) - center_val
        sum2 += diff_c
        sum2_abswin += abs(diff_c)
        count += 1

    denom = (N - K + 1)
    if count == 0 or denom <= 0:
        pq_schoentgen = np.nan
        pq_baken = np.nan
        pq_baken_abswin = np.nan
    else:
        pq_schoentgen = (sum1 / denom) / mean_ts
        pq_baken = (sum2 / denom) / mean_ts
        pq_baken_abswin = (sum2_abswin / denom) / mean_ts

    p = 5
    pq_generalized = np.nan
    pq_baken_alt = np.nan
    ar_coeffs = None

    if N > p + 1:
        ts_centered = (time_series - mean_ts).astype(np.float64)
        r = np.correlate(ts_centered, ts_centered, mode='full')[N - 1:]
        if len(r) >= p + 1:
            try:
                if ar_method == "levinson":
                    a = _levinson_durbin(r, p)
                else:
                    R = toeplitz(r[:p])
                    r_right = r[1:p + 1]
                    if np.linalg.cond(R) > 1e12:
                        raise ValueError("Ill-conditioned R")
                    a_rest = solve_toeplitz((r[:p], r[:p]), r_right)
                    a = np.concatenate(([1.0], a_rest))
                ar_coeffs = a
                sum3 = 0.0
                for i0 in range(p, N):
                    segment = ts_centered[i0 - p:i0 + 1][::-1]
                    sum3 += abs(np.sum(a * segment))
                pq_generalized = (sum3 / (N - p)) / mean_ts
            except Exception:
                pq_generalized = np.nan

    sum2_alt = 0.0
    if count > 0 and denom > 0:
        for i in range(K1, N - K2 + 1):
            c = i - 1
            start = c - K2
            end = c + K2 + 1
            if start < 0 or end > N:
                continue
            window = time_series[start:end]
            if window.size != K:
                continue
            sum2_alt += np.mean(window) - time_series[c]
        pq_baken_alt = (sum2_alt / denom) / mean_ts
    else:
        pq_baken_alt = np.nan

    if baken_variant == 'alt':
        chosen_baken = pq_baken_alt
    elif baken_variant == 'abs':
        chosen_baken = abs(pq_baken) if np.isfinite(pq_baken) else pq_baken
    elif baken_variant == 'abswin':
        chosen_baken = pq_baken_abswin
    elif baken_variant == 'neg':
        chosen_baken = -pq_baken if np.isfinite(pq_baken) else pq_baken
    else:
        chosen_baken = pq_baken
    
    logging.debug(
            pq_schoentgen,
            chosen_baken,
            pq_generalized,
            {
                'K': K,
                'K1': K1,
                'K2': K2,
                'pq_baken_default': pq_baken,
                'pq_baken_alt': pq_baken_alt,
                'pq_baken_abs_aggregate': abs(pq_baken) if np.isfinite(pq_baken) else pq_baken,
                'pq_baken_abswin': pq_baken_abswin,
                'baken_variant_used': baken_variant,
                'ar_method': ar_method,
                'ar_coeffs': ar_coeffs,
            }
        )

    return pq_schoentgen, chosen_baken, pq_generalized
