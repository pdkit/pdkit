import logging
import numpy as np
import scipy.signal
from pdkit.voice_features.tkeo import compute_tkeo

def log_bb(x):
    """Natural log with zero and negative handling, matching MATLAB log_bb function"""
    x = np.asarray(x)
    x_safe = np.where(x <= 0, 1e-12, x)
    result = np.log(x_safe)
    return np.where(x == 0, 0, result)

def compute_vfer_measure(data, fs, VF_close, VF_open):

    filt_order = 100
    BW = 500
    Fshift = 500
    Fmax = fs/2 - BW - 300

    Fc1 = np.arange(1, Fmax, Fshift)
    Fc2 = Fc1 + BW

    filters = []
    for f1, f2 in zip(Fc1, Fc2):
        nyquist = fs / 2
        low = f1 / nyquist
        high = f2 / nyquist
        if high >= 1.0:
            high = 0.99 
        if low >= high:
            continue
        sos = scipy.signal.butter(N=filt_order//2, Wn=[low, high], btype='bandpass', output='sos')
        filters.append(sos)

    logging.debug(f"Number of filters: {len(filters)}")
    logging.debug(f"Number of GCI cycles: {len(VF_close)-1}")

    NEm = []
    signal_BW_TKEO = []
    signal_BW_SEO = []

    for i in range(len(VF_close)-1):
        tseries = data[VF_close[i]:VF_close[i+1]]
        
        Dwindow = np.hanning(len(tseries))
        segment_sig = tseries * Dwindow
        
        if len(tseries) > 50:
            sig_TKEO = []
            sig_SEO = []
            sigBW = np.zeros((50, len(filters)))
            
            for ii, sos in enumerate(filters):
                thanasis = scipy.signal.sosfilt(sos, segment_sig)
                sigBW[:, ii] = thanasis[:50]
                sig_TKEO.append(np.mean(compute_tkeo(sigBW[:, ii])))
                sig_SEO.append(np.mean(sigBW[:, ii])**2)
            
            Hilb_tr = scipy.signal.hilbert(sigBW, axis=0)
            Hilb_env = np.abs(Hilb_tr)

            correlations = []
            for i in range(Hilb_env.shape[1]):
                for j in range(i, Hilb_env.shape[1]): 
                    c = scipy.signal.correlate(Hilb_env[:, i], Hilb_env[:, j], mode='full')
                    correlations.extend(c)
            max_corr = np.max(correlations)
            NEm.append(max_corr)
                        
            signal_BW_TKEO.append(sig_TKEO)
            signal_BW_SEO.append(sig_SEO)

    if not NEm:
        NEm = [0]
        signal_BW_TKEO = [np.zeros(len(filters))]
        signal_BW_SEO = [np.zeros(len(filters))]

    NEm = np.array(NEm)
    signal_BW_TKEO = np.array(signal_BW_TKEO)
    signal_BW_SEO = np.array(signal_BW_SEO)
    
    VFTKEO = np.mean(signal_BW_TKEO, axis=0)
    VFSEO = np.mean(signal_BW_SEO, axis=0)
    VFlog_SEO = np.mean(np.log(signal_BW_SEO + 1e-12), axis=0)
    
    signal_BW_TKEO2 = np.mean(np.log(signal_BW_TKEO + 1e-12), axis=0)

    VFER = np.zeros(7)
    VFER[0] = np.mean(NEm)
    VFER[1] = np.std(NEm)
    VFER[2] = -np.sum(NEm * log_bb(NEm))
    
    N = len(VFTKEO)
    VFER[3] = np.sum(VFTKEO[:min(5, N)]) / np.sum(VFTKEO[min(6, N)-1:min(10, N)])
    VFER[4] = np.sum(VFSEO[:min(5, N)]) / np.sum(VFSEO[min(6, N)-1:min(10, N)]) 
    VFER[5] = np.sum(signal_BW_TKEO2[min(6, N)-1:min(10, N)]) / np.sum(signal_BW_TKEO2[:min(5, N)])
    VFER[6] = np.sum(VFlog_SEO[min(6, N)-1:min(10, N)]) / np.sum(VFlog_SEO[:min(5, N)])

    return VFER
