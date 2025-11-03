import logging
import pywt
import numpy as np
from pdkit.voice_features.tkeo import compute_tkeo


def compute_wavedec_features(data, wname='db8', dec_levels=10, mode='symmetric'):
    data = np.asarray(data)

    data = data[np.isfinite(data)]
    
    if len(data) == 0:
        dummy_data = np.ones(1024)
        dummy_features, dummy_names = compute_wavedec_features(dummy_data, wname, dec_levels, mode)
        return np.full_like(dummy_features, np.nan), dummy_names
    
    max_level = pywt.dwt_max_level(data_len=len(data), filter_len=pywt.Wavelet(wname).dec_len)
    dec_levels = min(dec_levels, max_level)
    features = []
    feature_names = []

    C, L = wavedec(data, wname, dec_levels)
    energy = w_energy(C, L)

    features.append(energy[0])
    features.extend(energy[1])

    feature_names.append('Ea')
    for i in range(len(energy[1])):
        feature_names.append(f'Ed_{i+1}')

    detail_coefficient_expansion(C, L, dec_levels, features, feature_names, "det")
    approx_coefficient_expansion(C, L, dec_levels, wname, features, feature_names, "app")

    log_data = np.log(np.abs(data) + 1e-6)

    log_C, log_L = wavedec(log_data, wname, dec_levels)

    log_energy = w_energy(log_C, log_L)

    features.append(log_energy[0])
    features.extend(log_energy[1])
    feature_names.append('Ea2')
    for i in range(len(log_energy[1])):
        feature_names.append(f'Ed2_{i+1}')

    detail_coefficient_expansion(log_C, log_L, dec_levels, features, feature_names, "det_LT")
    approx_coefficient_expansion(log_C, log_L, dec_levels, wname, features, feature_names, "app_det_LT")

    flat_features = []
    flat_names = []

    for name, value in zip(feature_names, features):
        if isinstance(value, dict): 
            for k, v in value.items():
                if np.isscalar(v):
                    flat_features.append(float(v))
                    flat_names.append(f"{name}->{k}")
        elif isinstance(value, (list, np.ndarray)):
            def flatten_recursive(item):
                if np.isscalar(item):
                    return [float(item)]
                elif isinstance(item, (list, np.ndarray)):
                    result = []
                    for sub_item in item:
                        result.extend(flatten_recursive(sub_item))
                    return result
                else:
                    return [float(item)]
            
            flattened = flatten_recursive(value)
            for i, v in enumerate(flattened):
                flat_features.append(v)
                flat_names.append(f"{name}_{i+1}_coef")
        else:
            flat_features.append(float(value))
            flat_names.append(name)
    
    return np.array(flat_features, dtype=float), flat_names


def wavedec(data, wname, level):
    wavelet = pywt.Wavelet(wname)
    coeffs = pywt.wavedec(data, wavelet, level=level, mode='symmetric')

    C = np.concatenate(coeffs)
    L = [len(c) for c in coeffs]
    L.append(len(data))
    
    return C, np.array(L)


def detcoef(C, L, level):
    n_levels = len(L) - 2
    detail_lengths = L[1:1 + n_levels]
    approx_len = L[0]
    if not (1 <= level <= n_levels):
        raise ValueError("Invalid level specified.")
    
    idx = n_levels - level
    start_index = approx_len + sum(detail_lengths[:idx])
    length = detail_lengths[idx]
    return C[start_index:start_index + length]

def appcoef(C, L, *args):
    return C[0:L[0]]


def appcoef_at_level(C, L, wname, level, mode='symmetric'):
    n_levels = len(L) - 2
    if n_levels <= 0:
        return C
    if not (1 <= level <= n_levels):
        raise ValueError(f"level must be in [1, {n_levels}], got {level}")

    cA_n = C[0:L[0]]
    offset = L[0]
    detail_lengths = L[1:1 + n_levels]
    details = []
    for dl in detail_lengths:
        details.append(C[offset:offset + dl])
        offset += dl

    coeffs_for_recon = [cA_n]
    for idx, cd in enumerate(details):
        level_at_idx = n_levels - idx
        if level_at_idx <= level:
            coeffs_for_recon.append(np.zeros_like(cd))
        else:
            coeffs_for_recon.append(cd)

    wavelet = pywt.Wavelet(wname) if isinstance(wname, str) else wname
    a_level = pywt.waverec(coeffs_for_recon, wavelet, mode=mode)
    return a_level


def w_energy(C, L):
    et = np.sum(np.square(C))
    if et == 0:
        return 0.0, [0.0] * (len(L) - 2)

    level = len(L) - 2
    
    ca = appcoef(C, L)
    ea = 100 * np.sum(np.square(ca)) / et if et > 0 else 0
    
    ed = []
    for i in range(1, level + 1):
        cd = detcoef(C, L, i)
        ed.append(100 * np.sum(np.square(cd)) / et if et > 0 else 0)
        
    return ea, ed


def detail_coefficient_expansion(C, L, dec_levels, features, feature_names, prefix):
    for i in range(1, dec_levels + 1):
        try:
            d = detcoef(C, L, i)
            features.append(shannon_entropy(d))
            features.append(log_energy_entropy(d))
            features.append(np.mean(compute_tkeo(d)))
            features.append(np.std(compute_tkeo(d)))

            feature_names.extend([
                f'{prefix}_entropy_shannon_{i}_coef',
                f'{prefix}_entropy_log_{i}_coef',
                f'{prefix}_TKEO_mean_{i}_coef',
                f'{prefix}_TKEO_std_{i}_coef'
            ])
        except (ValueError, IndexError) as e:
            break


def approx_coefficient_expansion(C, L, dec_levels, wname, features, feature_names, prefix):
    for i in range(1, dec_levels + 1):
        try:
            # Use approximation at the specific level i so features vary by level
            a = appcoef_at_level(C, L, wname, level=i)
            features.append(shannon_entropy(a))
            features.append(log_energy_entropy(a))
            features.append(np.mean(compute_tkeo(a)))
            features.append(np.std(compute_tkeo(a)))

            feature_names.extend([
                f'{prefix}_entropy_shannon_{i}_coef',
                f'{prefix}_entropy_log_{i}_coef',
                f'{prefix}_TKEO_mean_{i}_coef',
                f'{prefix}_TKEO_std_{i}_coef'
            ])
        except (ValueError, IndexError) as e:
            logging.error("Error occurred while processing coefficients:", e)
            break

def shannon_entropy(x):    
    x = np.asarray(x)
    if len(x) == 0: return 0.0
    x_squared = x**2
    total_energy = np.sum(x_squared)
    if total_energy == 0: return 0.0
    p = x_squared / total_energy
    p_nonzero = p[p > 0]
    if len(p_nonzero) == 0: return 0.0
    shannon_ent = -np.sum(p_nonzero * np.log(p_nonzero))
    return shannon_ent / np.log(len(x)) if len(x) > 1 else 0.0

def log_energy_entropy(x):
    x = np.asarray(x)
    if len(x) == 0: return 0.0
    x_squared = x**2
    total_energy = np.sum(x_squared)
    if total_energy == 0: return 0.0
    p = x_squared / total_energy
    p_nonzero = p[p > 0]
    if len(p_nonzero) == 0: return 0.0
    log_energy_ent = np.sum(p_nonzero * np.log(p_nonzero))
    return log_energy_ent / np.log(len(x)) if len(x) > 1 else 0.0