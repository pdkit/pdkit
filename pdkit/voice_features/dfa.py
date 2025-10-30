import numpy as np

def compute_dfa(data, scales=None, fast=True):
    data = np.asarray(data, dtype=np.float64)
    N = len(data)
    
    if scales is None:
        scales = np.arange(50, 201, 20)
    
    # Filter out scales that are too large
    scales = scales[scales < N]
    if len(scales) == 0:
        return np.nan
    
    # Cumulative sum (profile) - centered
    profile = np.cumsum(data - np.mean(data))
    
    if fast:
        fluct = _dfa_vectorized(profile, scales)
    else:
        fluct = _dfa_original(profile, scales)
    
    if len(fluct) < 2:
        return np.nan

    valid_fluct = fluct[fluct > 0]
    valid_scales = scales[:len(valid_fluct)]
    
    if len(valid_fluct) < 2:
        return np.nan
    
    log_scales = np.log10(valid_scales)
    log_fluct = np.log10(valid_fluct)
    
    slope, intercept = np.polyfit(log_scales, log_fluct, 1)
    
    return slope

def _dfa_vectorized(profile, scales):
    fluct = []
    
    for s in scales:
        n_segments = len(profile) // s
        if n_segments == 0:
            continue
            
        # Reshape profile into segments (vectorized)
        end_idx = n_segments * s
        segments = profile[:end_idx].reshape(n_segments, s)
        
        # Vectorized linear detrending
        x = np.arange(s)
        x_mean = np.mean(x)
        x_sq_sum = np.sum(x**2) - s * x_mean**2
        
        # Compute slopes and intercepts for all segments at once
        y_mean = np.mean(segments, axis=1)
        xy_sum = np.sum(segments * x, axis=1) - s * x_mean * y_mean
        slopes = xy_sum / x_sq_sum
        intercepts = y_mean - slopes * x_mean
        
        # Compute trends for all segments
        trends = slopes[:, np.newaxis] * x + intercepts[:, np.newaxis]
        
        # Compute RMS fluctuations (vectorized)
        fluctuations = np.sqrt(np.mean((segments - trends)**2, axis=1))
        
        fluct.append(np.mean(fluctuations))
    
    return np.array(fluct)

def _dfa_original(profile, scales):
    fluct = []
    
    for s in scales:
        n_segments = len(profile) // s
        if n_segments == 0:
            continue
            
        RMS = []
        for v in range(n_segments):
            idx_start = v * s
            idx_end = idx_start + s
            segment = profile[idx_start:idx_end]
            
            # Fit a straight line (1st degree polynomial) to the segment
            C = np.polyfit(np.arange(s), segment, 1)
            fit = np.polyval(C, np.arange(s))
            
            # Compute root mean square error from trend
            RMS.append(np.sqrt(np.mean((segment - fit)**2)))
        
        fluct.append(np.mean(RMS))
    
    return np.array(fluct)
