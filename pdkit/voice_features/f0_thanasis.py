import numpy as np
import scipy.signal

def compute_f0_thanasis(data, fs, x=None, tstep=None, f0min=50, f0max=500):  
    data = np.asarray(data)
    if x is None:
        x = int(0.04 * fs)
    if tstep is None:
        tstep = int(0.01 * fs)

    steps = int(np.round((len(data) - x) / tstep))
    f0 = np.zeros(steps)
    
    f0_times = np.arange(x/(2*fs), (steps+2)*tstep/fs + tstep/fs, tstep/fs)
    f0_times = f0_times * 1000

    for i in range(steps):
        tseries = data[i * tstep : i * tstep + x]
        tseries = tseries - np.mean(tseries)
        N = len(tseries)
        alpha = 2.5 
        std_matlab = (N - 1) / (2 * alpha)
        Dwindow = scipy.signal.windows.gaussian(N, std=std_matlab)
        segment_sig = tseries * Dwindow

        # Guard against silent/constant segments
        if np.all(segment_sig == 0) or np.max(np.abs(segment_sig)) < 1e-8:
            f0[i] = np.nan
            continue

        acf = np.correlate(segment_sig, segment_sig, mode='full')
        acf_max = np.max(acf)
        if acf_max == 0 or not np.isfinite(acf_max):
            f0[i] = np.nan
            continue
        acf = acf / acf_max
        acf2 = acf[len(segment_sig) - 1:]

        aa = np.fft.fft(segment_sig)
        aa = np.fft.ifft(np.abs(aa)**2).real
        aa_max = np.max(aa)
        if aa_max == 0 or not np.isfinite(aa_max):
            f0[i] = np.nan
            continue
        aa = aa / aa_max

        ACF_Dwindow = np.correlate(Dwindow, Dwindow, mode='full')
        ACF_Dwindow_max = np.max(ACF_Dwindow)
        if ACF_Dwindow_max == 0 or not np.isfinite(ACF_Dwindow_max):
            f0[i] = np.nan
            continue
        ACF_Dwindow = ACF_Dwindow / ACF_Dwindow_max
        ACF_Dwindow2 = ACF_Dwindow[len(Dwindow) - 1:]

        bb = np.fft.fft(Dwindow)
        bb = np.fft.ifft(np.abs(bb)**2).real
        bb_max = np.max(bb)
        if bb_max == 0 or not np.isfinite(bb_max):
            f0[i] = np.nan
            continue
        bb = bb / bb_max
        ACF_signal = acf2 / ACF_Dwindow2
        ACF_signal = ACF_signal[:len(ACF_signal)//3]

        rho = aa / bb
        rho = rho[:len(rho)//2]
        rho_max = np.max(rho)
        if rho_max == 0 or not np.isfinite(rho_max):
            f0[i] = np.nan
            continue
        rho = rho / rho_max

        # Guard against illusion harmonics
        low_lim = int(np.ceil(fs/f0max))
        up_lim = int(np.floor(fs/f0min))
        d1 = np.sort(rho)[::-1]
        d2 = np.argsort(rho)[::-1]

        # Find first valid peak within limits
        m = 1
        while m < len(d2):
            mm = d2[m]
            if low_lim <= mm <= up_lim:
                break
            m += 1
        
        if m >= len(d2):
            f0[i] = np.nan
            continue
        mm = d2[m]
        
        # Approach with interpolation
        dt = 1/fs
        if low_lim < mm < up_lim and mm > 0 and mm < len(rho)-1:
            tmax1 = 0.5 * (rho[mm+1] - rho[mm-1])
            tmax2 = 2*rho[mm] - rho[mm-1] - rho[mm+1]
            if tmax2 != 0:
                tmax = dt * (mm + tmax1/tmax2)
            else:
                tmax = mm * dt
        else:
            # Guard against limit findings
            tmax = mm * dt
        f0[i] = 1/tmax if tmax != 0 else np.nan

    return f0, f0_times