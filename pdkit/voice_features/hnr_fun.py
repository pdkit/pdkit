import numpy as np
import scipy.signal

def compute_hnr_fun(data, fs):
    f0min = 50
    f0max = 500

    tstep = int(0.01 * fs)
    window_size = int(0.08 * fs)
    steps = (len(data) - window_size) // tstep

    HNR_dB = []
    NHR = []

    for i in range(steps):
        segment = data[i*tstep:i*tstep+window_size]
        segment = segment - np.mean(segment)
        Dwindow = np.hanning(len(segment))
        seg_windowed = segment * Dwindow

        ACF = scipy.signal.correlate(seg_windowed, seg_windowed, mode='full')
        ACF = ACF[len(ACF)//2:]

        low_lim = int(np.ceil(fs/f0max))
        up_lim = int(np.floor(fs/f0min))

        if len(ACF) < up_lim:
            continue

        peak_idx = np.argmax(ACF[low_lim:up_lim]) + low_lim

        r = ACF[peak_idx] / ACF[0] if ACF[0] != 0 else 0

        if r <= 0 or r >= 1:
            continue
            
        HNR_dB.append(10 * np.log10(r / (1 - r + 1e-6)))
        NHR.append((1 - r) / (r + 1e-6))

    if len(HNR_dB) == 0:
        hnr_mean, hnr_std = np.nan, np.nan
    else:
        hnr_mean, hnr_std = np.nanmean(HNR_dB), np.nanstd(HNR_dB)
        
    if len(NHR) == 0:
        nhr_mean, nhr_std = np.nan, np.nan
    else:
        nhr_mean, nhr_std = np.nanmean(NHR), np.nanstd(NHR)

    return np.array([hnr_mean, hnr_std]), np.array([nhr_mean, nhr_std])
