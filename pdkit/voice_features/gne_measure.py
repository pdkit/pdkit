import numpy as np
import scipy.signal
import librosa
from pdkit.voice_features.tkeo import compute_tkeo

def compute_gne_measure(data, fs):
    filt_order = 100
    new_fs = 10000
    x = int(0.03 * new_fs)
    tstep = int(0.01 * new_fs)
    BW = 1000
    Fshift = 500

    if fs != new_fs:
        try:
            fs_i = int(fs)
            gcd = np.gcd(int(new_fs), fs_i) if fs_i > 0 else 1
            up = int(new_fs // gcd) if gcd else int(new_fs)
            down = int(fs_i // gcd) if gcd else fs_i
            data = scipy.signal.resample_poly(data, up, down)
        except Exception:
            data = scipy.signal.resample(data, int(len(data) * new_fs / fs))


    Fc1 = np.arange(1, new_fs / 2 - BW - 500, Fshift)
    Fc2 = Fc1 + BW

    taps = []
    numtaps = filt_order + 1
    for f1, f2 in zip(Fc1, Fc2):
        h = scipy.signal.firwin(numtaps=numtaps, cutoff=[f1, f2], pass_zero=False, fs=new_fs, window='hamming')
        fc = (f1 + f2) / 2 
        w, H = scipy.signal.freqz(h, worN=[2*np.pi*fc/new_fs])
        gain = abs(H[0])
        if gain > 1e-12:
            h = h / gain
        taps.append(h)

    GNEm = []
    frames_tkeo = []
    frames_energy = []

    steps = int((len(data) - x) / tstep) if len(data) >= x else - 1
    for i in range(steps + 1):
        start = i * tstep
        segment = data[start:start + x]
        
        window = np.hanning(x)
        segment = segment * window
        
        try:
            a = librosa.lpc(segment, order=13)
        except Exception as ex:
            a = np.zeros(14)
            a[0] = 1.0

        b = np.concatenate(([0.0], -a[1:])) if a.shape[0] > 1 else np.array([0.0])
        est_x = scipy.signal.lfilter(b, [1.0], segment)
        e = segment - est_x
        r = scipy.signal.correlate(e, e, mode='full')

        mid = (r.size - 1) // 2
        LPE = r[mid:]

        if LPE.size and LPE[0] != 0:
            LPE = LPE / LPE[0]

        cols = []
        for k, h in enumerate(taps):
            filtered = scipy.signal.lfilter(h, [1.0], LPE)
            cols.append(filtered)

        sigBW = np.stack(cols, axis=1)

        tkeo_means = []
        energy_means = []
        for k in range(sigBW.shape[1]):
            s = sigBW[:, k]
            tk = compute_tkeo(s)
            tkeo_means.append(float(np.mean(tk) if tk.size else 0.0))
            energy_means.append(float(np.mean(s) ** 2))

        frames_tkeo.append(np.array(tkeo_means))
        frames_energy.append(np.array(energy_means))

        env = np.abs(scipy.signal.hilbert(sigBW, axis=0))
        
        max_corr = 0.0
        cross_corrs = []
        nb = env.shape[1]
        for p in range(nb):
            for q in range(nb):
                c = scipy.signal.correlate(env[:, p], env[:, q], mode='full')
                if c.size:
                    mc = float(np.max(c))
                    cross_corrs.append(mc)
                    if mc > max_corr:
                        max_corr = mc

        GNEm.append(max_corr)
        

    if frames_tkeo:
        frames_tkeo = np.stack(frames_tkeo, axis=0)  
        frames_energy = np.stack(frames_energy, axis=0)
    else:
        frames_tkeo = np.zeros((0, len(Fc1)))
        frames_energy = np.zeros((0, len(Fc1)))

    gnTKEO = np.mean(frames_tkeo, axis=0)
    gnSEO = np.mean(frames_energy, axis=0)

    signal_BW_TKEO2 = np.mean(np.log(np.maximum(frames_tkeo, 1e-12)), axis=0)
    signal_energy2 = np.mean(np.log(np.maximum(frames_energy, 1e-12)), axis=0)


    def safe_ratio(num, den):
        num_s = float(np.sum(num))
        den_s = float(np.sum(den))
        if abs(den_s) < 1e-18:
            return 0.0
        return num_s / den_s
    GNE = [
        float(np.mean(GNEm) if GNEm else 0.0),
        float(np.std(GNEm) if GNEm else 0.0),
        safe_ratio(gnTKEO[:2], gnTKEO[-4:]),
        safe_ratio(gnSEO[:2], gnSEO[-4:]),
        safe_ratio(signal_BW_TKEO2[-4:], signal_BW_TKEO2[:2]),
        safe_ratio(signal_energy2[-4:], signal_energy2[:2]),
    ]
    
    return np.array(GNE)
