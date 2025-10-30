import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d
import warnings


class DypsaConfig:
    """Configuration parameters for DYPSA algorithm"""
    def __init__(self):
        self.dy_cpfrac = 0.3
        self.dy_cproj = 0.2
        self.dy_cspurt = -0.45
        self.dy_dopsp = 1
        self.dy_ewdly = 0.0008
        self.dy_ewlen = 0.003
        self.dy_ewtaper = 0.001
        self.dy_fwlen = 0.00045
        self.dy_fxmax = 500
        self.dy_fxmin = 50
        self.dy_fxminf = 60
        self.dy_gwlen = 0.0030
        self.dy_lpcdur = 0.020
        self.dy_lpcn = 2
        self.dy_lpcnf = 0.001
        self.dy_lpcstep = 0.010
        self.dy_nbest = 5
        self.dy_preemph = 50
        self.dy_spitch = 0.2
        self.dy_wener = 0.3
        self.dy_wpitch = 0.5
        self.dy_wslope = 0.1
        self.dy_wxcorr = 0.8
        self.dy_xwlen = 0.01


def v_dypsa(s, fs, config=None, verbose=True, target_fs=None, debug=True):
    """
    Derive glottal closure instances from speech
    
    Parameters:
    -----------
    s : array_like
        Speech signal
    fs : float
        Sampling frequency
    config : DypsaConfig, optional
        Configuration parameters
    verbose : bool, optional
        Print timing information
    target_fs : int, optional
        Target sampling frequency for processing (default None = no resampling)
        Audio will be resampled if fs != target_fs, then GCIs scaled back
    debug : bool, optional
        Print debugging information
        
    Returns:
    --------
    gci : ndarray
        Vector of glottal closure sample numbers (in original fs)
    goi : ndarray
        Vector of glottal opening sample numbers (in original fs)
    """
    import time
    t_start = time.time() if verbose else None
    
    if config is None:
        config = DypsaConfig()
    
    s_orig = np.asarray(s).flatten()
    fs_orig = fs
    
    # Downsample if necessary
    if target_fs is not None and fs != target_fs:
        if verbose:
            print(f"Resampling from {fs}Hz to {target_fs}Hz...")
        from scipy import signal as sig
        num_samples = int(len(s_orig) * target_fs / fs)
        s = sig.resample(s_orig, num_samples)
        fs = target_fs
        scale_factor = fs_orig / fs
    else:
        s = s_orig
        scale_factor = 1.0
    
    # LPC order
    lpcord = int(np.ceil(fs * config.dy_lpcnf + config.dy_lpcn))
    
    # Pre-emphasize input speech
    b = [1, -np.exp(-2 * np.pi * config.dy_preemph / fs)]
    s_used = signal.lfilter(b, 1, s)
    
    # Perform LPC analysis
    ar, e, k = lpcauto(s_used, lpcord, 
                       int(config.dy_lpcstep * fs), 
                       int(config.dy_lpcdur * fs))
    
    if debug:
        print(f"\n=== LPC Analysis ===")
        print(f"LPC order: {lpcord}")
        print(f"Number of frames: {ar.shape[0]}")
        print(f"LPC step: {int(config.dy_lpcstep * fs)} samples")
        print(f"LPC duration: {int(config.dy_lpcdur * fs)} samples")
        print(f"First frame LPC coeffs (first 5): {ar[0, :5]}")
    
    if np.any(np.isinf(ar)):
        warnings.warn('No GCIs returned - infinite prediction coefficients')
        return np.array([]), np.array([])
    
    # Compute prediction residual
    r = lpcifilt(s_used, ar, k, int(config.dy_lpcstep * fs))
    
    if debug:
        print(f"Residual length: {len(r)}")
        print(f"Residual mean: {np.mean(r):.6f}, std: {np.std(r):.6f}")
        print(f"First 10 residual values: {r[:10]}")
    
    # Compute group delay function
    zcr_cand, sew, gdwav, toff = xewgrdel(r, fs, config)
    
    if debug:
        print(f"\n=== Group Delay Analysis ===")
        print(f"Residual length: {len(r)}")
        print(f"Group delay length: {len(gdwav)}")
        print(f"Time offset: {toff}")
        print(f"Initial zero-crossing candidates: {len(zcr_cand)}")
        if len(zcr_cand) > 0:
            print(f"First 10 ZC candidates (before adjustment): {zcr_cand[:10].astype(int)}")
    
    # Note: MATLAB does: gdwav=-[zeros(toff,1); gdwav(1:end-toff)];
    # This prepends toff zeros and removes last toff samples
    if len(gdwav) > toff:
        gdwav_adjusted = np.concatenate([np.zeros(toff), -gdwav[:len(gdwav)-toff]])
    else:
        gdwav_adjusted = -gdwav
    
    if debug:
        print(f"gdwav adjusted length: {len(gdwav_adjusted)}")
        print(f"s_used length: {len(s_used)}")
    
    zcr_cand = np.column_stack([np.round(zcr_cand).astype(int), 
                                 np.ones(len(zcr_cand))])
    sew = 0.5 + sew
    
    # Phase slope projection
    pro_cand = []
    if config.dy_dopsp != 0:
        pro_cand = psp(gdwav_adjusted, fs)
        if debug:
            print(f"Projected candidates: {len(pro_cand)}")
            if len(pro_cand) > 0:
                print(f"First 10 projected: {pro_cand[:10].astype(int)}")
        if len(pro_cand) > 0:
            pro_cand = np.column_stack([pro_cand, np.zeros(len(pro_cand))])
            sew = np.concatenate([sew, np.zeros(len(pro_cand))])
    
    # Sort candidates
    if len(pro_cand) > 0:
        gcic = np.vstack([zcr_cand, pro_cand])
    else:
        gcic = zcr_cand
    
    if debug:
        print(f"Total candidates before sorting: {len(gcic)}")
        print(f"First 10 candidates (unsorted): {gcic[:10, 0].astype(int)}")
    
    sort_idx = np.argsort(gcic[:, 0])
    gcic = gcic[sort_idx]
    sew = sew[sort_idx]
    
    if debug:
        print(f"First 10 candidates (sorted): {gcic[:10, 0].astype(int)}")
    
    # Remove candidates too close to edges
    saf = max([200, config.dy_xwlen * fs / 2 + 1, fs / config.dy_fxminf])
    valid = (gcic[:, 0] > saf) & (gcic[:, 0] < len(gdwav) - saf)
    
    if debug:
        print(f"Candidates after edge filtering (saf={saf:.0f}): {np.sum(valid)} / {len(gcic)}")
    
    gcic = gcic[valid]
    sew = sew[valid]
    
    if len(gcic) == 0:
        warnings.warn('No valid GCI candidates found after filtering')
        return np.array([]), np.array([])
    
    # Compute Frobenius norm
    fnwav = frobfun(s_used, int(config.dy_ewtaper * fs), 
                    int(config.dy_ewlen * fs), 
                    int(config.dy_ewdly * fs))
    
    if debug:
        print(f"Frobenius norm length: {len(fnwav)}")
        print(f"Calling DP with {len(gcic)} candidates...")
    
    # Dynamic programming
    gci = dpgci(gcic, s_used, sew, fnwav, fs, config, debug=debug)
    
    # Estimate glottal opening instants
    goi = simplegci2goi(gci, config.dy_cpfrac)
    
    # Scale back to original sampling rate if resampled
    if scale_factor != 1.0:
        gci = np.round(gci * scale_factor).astype(int)
        goi = np.round(goi * scale_factor).astype(int)
    
    if verbose:
        elapsed = time.time() - t_start
        print(f"DYPSA completed in {elapsed:.2f}s")
        print(f"Found {len(gci)} GCIs ({len(gci)/(len(s_orig)/fs_orig):.1f} per second of audio)")
    
    return gci, goi


def lpcauto(s, p, step, dur):
    """
    LPC analysis using autocorrelation method with Hamming window
    
    Parameters:
    -----------
    s : array_like
        Input signal
    p : int
        Prediction order
    step : int
        Step size in samples
    dur : int
        Window duration in samples
        
    Returns:
    --------
    ar : ndarray
        LPC coefficients [frames x (p+1)]
    e : ndarray
        Prediction errors
    k : ndarray
        Reflection coefficients
    """
    s = np.asarray(s).flatten()
    n = len(s)
    
    # Calculate number of frames
    nframes = int(np.floor((n - dur) / step) + 1)
    
    ar = np.zeros((nframes, p + 1))
    e = np.zeros(nframes)
    k = np.zeros((nframes, p))
    
    window = np.hamming(dur)
    
    for i in range(nframes):
        start = i * step
        end = start + dur
        if end > n:
            break
            
        frame = s[start:end] * window
        
        # Autocorrelation
        r = np.correlate(frame, frame, mode='full')
        r = r[len(frame)-1:len(frame)+p]
        
        # Levinson-Durbin recursion
        if r[0] == 0:
            ar[i, 0] = 1
            continue
            
        a = np.zeros(p + 1)
        a[0] = 1
        e_val = r[0]
        
        for j in range(p):
            if e_val == 0:
                break
                
            k_val = (r[j+1] - np.sum(a[1:j+1] * r[j:0:-1])) / e_val
            k[i, j] = k_val
            
            a_new = a.copy()
            a_new[j+1] = k_val
            for m in range(1, j+1):
                a_new[m] = a[m] - k_val * a[j+1-m]
            a = a_new
            
            e_val = e_val * (1 - k_val**2)
        
        ar[i] = a
        e[i] = e_val
    
    return ar, e, k


def lpcifilt(s, ar, k, step):
    """
    Compute LPC inverse filter (prediction residual)
    
    Parameters:
    -----------
    s : array_like
        Input signal
    ar : ndarray
        LPC coefficients
    k : ndarray
        Reflection coefficients
    step : int
        Step size in samples
        
    Returns:
    --------
    r : ndarray
        Prediction residual
    """
    s = np.asarray(s).flatten()
    nframes = ar.shape[0]
    n = len(s)
    
    r = np.zeros(n)
    
    for i in range(n):
        frame_idx = min(int(i / step), nframes - 1)
        a = ar[frame_idx]
        p = len(a) - 1
        
        val = s[i]
        for j in range(1, min(p + 1, i + 1)):
            val -= a[j] * s[i - j]
        r[i] = val
    
    return r


def xewgrdel(u, fs, config, debug=True):
    """
    Compute group delay using EW method
    
    Parameters:
    -----------
    u : array_like
        Input signal (residual)
    fs : float
        Sampling frequency
    config : DypsaConfig
        Configuration
        
    Returns:
    --------
    tew : ndarray
        Zero crossing positions
    sew : ndarray
        Slopes at zero crossings
    y : ndarray
        Group delay function
    toff : int
        Time offset
    """
    u = np.asarray(u).flatten()
    
    gw = 2 * int(np.floor(config.dy_gwlen * fs / 2)) + 1
    ghw = np.hamming(gw)
    
    # Create weighted window for numerator
    indices = np.arange(gw)
    ghwn = ghw * ((gw - 1) / 2 - indices)
    
    u2 = u ** 2
    yn = signal.convolve(u2, ghwn, mode='same')
    yd = signal.convolve(u2, ghw, mode='same')
    
    # Prevent division by zero
    yd[np.abs(yd) < np.finfo(float).eps] = 10 * np.finfo(float).eps
    
    y_full = yn / yd
    
    # Remove startup transient
    toff = (gw - 1) // 2
    y = y_full[toff:]
    
    # Low pass filter
    fw = 2 * int(np.floor(config.dy_fwlen * fs / 2)) + 1
    if fw > 1:
        daw = np.hamming(fw)
        daw = daw / np.sum(daw)
        y = signal.convolve(y, daw, mode='same')
        toff = toff - (fw - 1) // 2
    
    # Find zero crossings - these are in the y array coordinates
    tew, sew = zerocros(y, mode='n')
    
    # Add back the offset to get positions in original signal coordinates
    tew = tew + toff
    
    if debug:
        print(f"  xewgrdel: toff after filters = {toff}")
        print(f"  xewgrdel: returning {len(tew)} candidates")
        print(f"  xewgrdel: first 5 candidate positions = {tew[:5] if len(tew) > 0 else 'none'}")
    
    return tew, sew, y, toff


def zerocros(x, mode='a'):
    """
    Find zero crossings in a signal
    
    Parameters:
    -----------
    x : array_like
        Input signal
    mode : str
        'p' for positive-going, 'n' for negative-going, 'a' for all
        
    Returns:
    --------
    idx : ndarray
        Indices of zero crossings
    slopes : ndarray
        Slopes at zero crossings
    """
    x = np.asarray(x).flatten()
    
    if mode == 'p':
        crossings = zcrp(x)
    elif mode == 'n':
        crossings = zcrp(-x)
    else:
        crossings = np.concatenate([zcrp(x), zcrp(-x)])
    
    # Find exact zeros
    zeros = np.where((x[:-1] == 0) & (x[1:] != 0))[0]
    
    idx = np.sort(np.concatenate([crossings, zeros])) if len(zeros) > 0 else np.sort(crossings)
    
    # Calculate slopes
    slopes = np.gradient(x)[idx.astype(int)] if len(idx) > 0 else np.array([])
    
    return idx, slopes


def zcrp(x):
    """Find positive-going zero crossings"""
    sign_changes = np.diff(np.sign(x))
    crossings = np.where(sign_changes == -2)[0]
    
    # Find which sample is closer to zero
    if len(crossings) > 0:
        vals = np.column_stack([np.abs(x[crossings]), np.abs(x[crossings + 1])])
        closest = np.argmin(vals, axis=1)
        return crossings + closest
    return np.array([])


def psp(g, fs):
    """
    Calculate phase slope projections
    
    Parameters:
    -----------
    g : array_like
        Group delay function
    fs : float
        Sampling frequency
        
    Returns:
    --------
    z : ndarray
        Projected zero crossing positions
    """
    g = np.asarray(g).flatten()
    
    gdot = np.diff(g)
    gdot = np.append(gdot, 0)
    gdotdot = np.diff(gdot)
    gdotdot = np.append(gdotdot, 0)
    
    # Find turning points using zero crossings of derivative
    turning_points_idx = np.where(np.abs(np.diff(np.sign(gdot))) == 2)[0]
    
    if len(turning_points_idx) == 0:
        return np.array([])
    
    # Filter out first point
    turning_points_idx = turning_points_idx[turning_points_idx > 0]
    
    if len(turning_points_idx) == 0:
        return np.array([])
    
    turning_points = []
    for i, idx in enumerate(turning_points_idx):
        if idx < len(g):
            turning_points.append([i, idx, np.sign(gdotdot[idx]), g[idx]])
    
    if len(turning_points) == 0:
        return np.array([])
    
    turning_points = np.array(turning_points)
    
    # Find negative maxima (peaks below zero)
    mask = (turning_points[:, 2] == -1) & (turning_points[:, 3] < 0) & (turning_points[:, 0] > 0)
    negmaxima = turning_points[mask]
    
    nz = []
    if len(negmaxima) > 0:
        for i in range(len(negmaxima)):
            nmi = int(negmaxima[i, 0])
            if nmi > 0 and nmi < len(turning_points):
                prev_idx = int(turning_points[nmi-1, 1])
                curr_idx = int(turning_points[nmi, 1])
                mid_idx = prev_idx + int(0.5 * (curr_idx - prev_idx))
                if 0 <= mid_idx < len(g):
                    mid_val = g[mid_idx]
                    proj = mid_idx - int(np.round(mid_val))
                    if 0 <= proj < len(g):
                        nz.append(proj)
    
    # Find positive minima (troughs above zero)
    mask = (turning_points[:, 2] == 1) & (turning_points[:, 3] > 0)
    posminima = turning_points[mask]
    
    pz = []
    if len(posminima) > 0:
        for i in range(len(posminima)):
            pmi = int(posminima[i, 0])
            if pmi < len(turning_points) - 1:
                curr_idx = int(turning_points[pmi, 1])
                next_idx = int(turning_points[pmi+1, 1])
                mid_idx = curr_idx + int(0.5 * (next_idx - curr_idx))
                if 0 <= mid_idx < len(g):
                    mid_val = g[mid_idx]
                    proj = mid_idx - int(np.round(mid_val))
                    if 0 <= proj < len(g):
                        pz.append(proj)
    
    z = np.sort(np.concatenate([nz, pz])) if (nz or pz) else np.array([])
    return z


def frobfun(sp, p, m, offset):
    """
    Compute Frobenius norm measure
    
    Parameters:
    -----------
    sp : array_like
        Pre-emphasized speech signal
    p : int
        Prediction order
    m : int
        Window length
    offset : int
        Window offset
        
    Returns:
    --------
    frob : ndarray
        Frobenius norm measure
    """
    sp = np.asarray(sp).flatten()
    
    p = int(np.round(p))
    m = int(np.round(m))
    offset = int(np.round(offset))
    
    w = (p + 1) * np.ones(m + p)
    w[:p] = np.arange(1, p + 1)
    w[m:m+p] = np.arange(p, 0, -1)
    w = w / (p + 1)
    
    frob = signal.convolve(sp**2, w, mode='same')
    
    # Remove initial samples
    trim = int((p + m - 1) / 2) + offset
    if trim < len(frob):
        frob = frob[trim:]
    else:
        frob = np.array([])
    
    return frob


def fnrg(gcic, frob, fs, config):
    """
    Compute Frobenius energy cost
    
    Parameters:
    -----------
    gcic : ndarray
        GCI candidates (positions)
    frob : ndarray
        Frobenius norm
    fs : float
        Sampling frequency
    config : DypsaConfig
        Configuration
        
    Returns:
    --------
    Cfn : ndarray
        Energy costs
    """
    frob = frob.flatten()
    mm = int(np.round(fs / config.dy_fxminf))
    
    # Maximum filter
    mfrob = maximum_filter1d(frob, size=mm, mode='constant', cval=frob[0])
    
    # Avoid division by zero
    mfrob[mfrob == 0] = 1e-10
    
    rfr = frob / mfrob
    
    # Get costs at candidate positions
    indices = np.round(gcic).astype(int)
    indices = np.clip(indices, 0, len(rfr) - 1)
    Cfn = 0.5 - rfr[indices]
    
    return Cfn


def dpgci(gcic, s, Ch, fnwav, fs, config, debug=False):
    """
    Dynamic programming to choose best GCIs
    
    Parameters:
    -----------
    gcic : ndarray
        GCI candidates [N x 2]
    s : ndarray
        Speech signal
    Ch : ndarray
        Phase slope costs
    fnwav : ndarray
        Frobenius norm
    fs : float
        Sampling frequency
    config : DypsaConfig
        Configuration
    debug : bool
        Print debug info
        
    Returns:
    --------
    gci : ndarray
        Selected GCI positions
    """
    s = s.flatten()
    Ncand = len(gcic)
    
    if Ncand == 0:
        return np.array([])
    
    if debug:
        print(f"\n=== DP Starting ===")
        print(f"Processing {Ncand} candidates")
    
    sv2i = -(2 * config.dy_spitch**2)**(-1)
    nxc = int(np.ceil(config.dy_xwlen * fs))
    
    qrmin = int(np.ceil(fs / config.dy_fxmax))
    qrmax = int(np.floor(fs / config.dy_fxmin))
    
    if debug:
        print(f"Pitch range: {qrmin} to {qrmax} samples ({fs/qrmax:.1f} to {fs/qrmin:.1f} Hz)")
    
    Cfn = fnrg(gcic[:, 0], fnwav, fs, config)
    
    # Add start and end states
    gcic = np.vstack([
        [gcic[0, 0] - qrmax - 2, 0],
        gcic,
        [gcic[-1, 0] + qrmax + 2, 0]
    ])
    Cfn = np.concatenate([[0], Cfn, [0]])
    Ch = np.concatenate([[0], Ch, [0]])
    
    # Fixed costs
    g_cr = config.dy_wener * Cfn + config.dy_wslope * Ch + config.dy_cproj * (1 - gcic[:, 1])
    
    g_n = gcic[:, 0]
    g_pr = gcic[:, 1]
    g_sqm = np.zeros(Ncand + 2)
    g_sd = np.zeros(Ncand + 2)
    
    f_pq = np.zeros((Ncand + 2) * config.dy_nbest)
    f_c = np.full((Ncand + 2) * config.dy_nbest, np.inf)
    f_c[0] = 0
    f_f = np.arange((Ncand + 2) * config.dy_nbest)
    f_f[0] = 0
    f_fb = np.zeros(Ncand + 2, dtype=int)  # Initialize to 0 (points to start node)
    f_fb[0] = 0
    f_fb[1] = 0
    fbestc = 0  # Start with 0, not inf
    
    wavix = np.arange(-int(nxc / 2), int(nxc / 2) + 2)
    nx2 = len(wavix)
    sqnx2 = np.sqrt(nx2)
    
    qmin = 2
    paths_found = 0
    voicespurt_starts = 0
    
    # For debugging - sample some costs
    sample_costs = []
    sample_r = min(100, Ncand // 2) if Ncand > 10 else 2
    
    for r in range(2, Ncand + 1):
        r_n = int(g_n[r])
        rix = np.arange(config.dy_nbest * (r - 1), config.dy_nbest * r)
        
        # Find feasible q range (carefully matching MATLAB indexing)
        # MATLAB: qmin=find(g_n(qmin0-1:r-1)<r_n-qrmax); qmin=qmin(end)+qmin0-1;
        qmin0 = qmin
        search_range = g_n[qmin0-1:r]  # g_n(qmin0-1:r-1) in MATLAB 1-indexed = [qmin0-1:r) in Python
        temp = np.where(search_range < r_n - qrmax)[0]
        if len(temp) > 0:
            # temp[-1] gives index within search_range, add offset to get absolute index
            # In MATLAB: qmin(end)+qmin0-1, where qmin(end) is 1-indexed position in slice
            # In Python: temp[-1] is 0-indexed position, so absolute = temp[-1] + qmin0-1 + 1
            qmin = temp[-1] + qmin0
        # else qmin stays as qmin0
        
        # MATLAB: qmax=find(g_n(qmin-1:r-1)<=r_n-qrmin); qmax=qmax(end)+qmin-1;
        search_range = g_n[qmin-1:r]
        temp = np.where(search_range <= r_n - qrmin)[0]
        if len(temp) > 0:
            qmax = temp[-1] + qmin - 1  # Same logic
        else:
            qmax = qmin - 1
        
        # Calculate waveform similarity
        if r_n + wavix[-1] < len(s) and r_n + wavix[0] >= 0:
            sr = s[r_n + wavix]
            wsum = np.sum(sr)
            g_sqm[r] = wsum / sqnx2
            var = sr.T @ sr - wsum**2 / nx2
            g_sd[r] = 1 / np.sqrt(var) if var > 1e-10 else 0
        
        if qmin <= qmax:
            qix = np.arange(qmin, qmax + 1)
            nq = len(qix)
            
            # Cross-correlation cost
            q_cas = np.zeros(nq)
            for qi, q in enumerate(qix):
                q_n = int(g_n[q])
                if q_n + wavix[-1] < len(s) and q_n + wavix[0] >= 0 and g_sd[q] > 0 and g_sd[r] > 0:
                    sq = s[q_n + wavix]
                    # Normalized cross-correlation
                    corr_val = (np.sum(sq * sr) - g_sqm[q] * g_sqm[r]) * g_sd[q] * g_sd[r]
                    # Cost should be negative for good correlation (MATLAB compatibility)
                    # The factor accounts for the window size correction
                    q_cas[qi] = -0.5 * config.dy_wxcorr * corr_val * (nx2 - 1) / (nx2 - 2)
            
            # Pitch deviation cost
            fix = np.arange(1 + (qmin - 1) * config.dy_nbest, (qmax + 1) * config.dy_nbest)
            f_qr_base = r_n - g_n[qix]
            f_qr = np.repeat(f_qr_base, config.dy_nbest)
            
            min_len = min(len(fix), len(f_qr))
            fix = fix[:min_len]
            f_qr = f_qr[:min_len]
            
            f_pr = f_qr + f_pq[fix]
            
            # Avoid division by zero
            denom = f_pr + np.abs(f_qr - f_pq[fix])
            denom[denom == 0] = 1e-10
            
            f_nx = 2 - 2 * f_pr / denom
            f_cp = config.dy_wpitch * (0.5 - np.exp(sv2i * f_nx**2))
            f_cp[f_pq[fix] == 0] = config.dy_cspurt * config.dy_wpitch
            
            # Find N-best paths
            costs = f_c[fix] + f_cp + np.tile(q_cas, config.dy_nbest)[:min_len]
            nbix = np.argsort(costs)[:config.dy_nbest]
            
            f_c[rix] = costs[nbix] + g_cr[r]
            f_f[rix] = fix[nbix]
            f_pq[rix] = f_qr[nbix]
            
            # Debug: sample costs at a specific candidate
            if debug and r == sample_r:
                print(f"\n--- Sample at candidate r={r} (sample {int(r_n)}) ---")
                print(f"  Viable q range: {qmin} to {qmax} ({nq} candidates)")
                print(f"  Best path cost: {f_c[rix[0]]:.3f}")
                print(f"  Fixed cost (g_cr): {g_cr[r]:.3f}")
                print(f"  Sample transition costs (first 3): {costs[nbix[:3]]}")
                print(f"  Cross-corr (first 3 q): {q_cas[:min(3, len(q_cas))]}")
            
            paths_found += 1
            
            # Check for new voicespurt start
            # In MATLAB: cost only includes projection cost, NOT talkspurt cost here
            # The talkspurt cost is added in the pitch period cost calculation
            iNb = rix[-1]
            if qmin >= 2:
                # Cost for starting new spurt (without energy/slope costs per MATLAB comment)
                spurt_cost = f_c[f_fb[qmin-1]] + config.dy_cproj * (1 - gcic[r, 1])
                if spurt_cost < f_c[iNb]:
                    f_f[iNb] = f_fb[qmin-1]
                    f_c[iNb] = spurt_cost
                    f_pq[iNb] = 0
                    voicespurt_starts += 1
            
            # Update best end-of-spurt pointer
            if f_c[rix[0]] < fbestc or fbestc == 0:
                f_fb[r] = rix[0]
                fbestc = f_c[rix[0]]
            else:
                f_fb[r] = f_fb[r-1]
        else:
            # No viable previous candidates - must start new voicespurt
            if qmin >= 2:
                spurt_cost = f_c[f_fb[qmin-1]] + config.dy_cproj * (1 - gcic[r, 1])
                f_c[rix[0]] = spurt_cost
                f_f[rix] = f_fb[qmin-1]
                f_pq[rix] = 0
                voicespurt_starts += 1
            f_fb[r] = f_fb[r-1]
    
    if debug:
        print(f"DP: Found paths for {paths_found}/{Ncand-1} candidates")
        print(f"DP: Created {voicespurt_starts} voicespurt starts")
        print(f"Best end cost: {fbestc:.3f}")
    
    # Traceback
    gci = []
    i = rix[0] - config.dy_nbest
    if i >= 0 and i < len(f_c):
        if i - config.dy_nbest + 1 >= 0 and f_c[i - config.dy_nbest + 1] < f_c[i]:
            i = i - config.dy_nbest + 1
    
    visited = set()
    max_iterations = Ncand + 10
    iterations = 0
    
    while i > 0 and iterations < max_iterations:
        if i in visited:
            if debug:
                print(f"Cycle detected at node {i}")
            break
        visited.add(i)
        
        j = i // config.dy_nbest
        if j >= len(g_n) or j < 0:
            break
            
        gci.append(int(g_n[j]))
        next_i = int(f_f[i])
        
        if next_i == i:
            break
            
        i = next_i
        iterations += 1
    
    if iterations >= max_iterations:
        warnings.warn(f"Traceback exceeded maximum iterations ({max_iterations})")

    gci = np.array(gci[::-1]) if gci else np.array([])
    return gci


def simplegci2goi(gci, pr):
    """
    Estimate glottal opening instants
    
    Parameters:
    -----------
    gci : array_like
        Glottal closure instants
    pr : float
        Closed phase fraction
        
    Returns:
    --------
    goi : ndarray
        Glottal opening instants
    """
    gci = np.round(gci).astype(int)
    
    if len(gci) < 2:
        return np.array([])
    
    diffs = np.diff(gci)
    maxpitch = np.max(signal.medfilt(diffs, 7))
    
    goi = np.zeros(len(gci))
    for kg in range(len(gci) - 1):
        goi[kg] = gci[kg] + min(pr * (gci[kg+1] - gci[kg]), pr * maxpitch)
    
    goi[-1] = gci[-1] + pr * (gci[-1] - gci[-2])
    
    return np.round(goi).astype(int)
dypsa = v_dypsa