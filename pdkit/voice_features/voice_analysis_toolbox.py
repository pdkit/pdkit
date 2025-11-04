import logging
import numpy as np
import librosa
from scipy.signal import resample

from pdkit.voice_features import (
    compute_f0_thanasis, compute_wavedec_features, compute_dfa, compute_glottis_quotient, 
    compute_gne_measure, compute_hnr_fun, compute_imf_measure, compute_jitter_shimmer, 
    compute_rpde, compute_ppe, compute_vfer_measure, dypsa, melcepst 
)

def voice_analysis(data, fs=None, f0_alg='SWIPE'):
    logging.debug("Starting voice analysis...")

    if isinstance(data, str):
        data, fs = librosa.load(data, sr=None)
    elif fs is None:
        raise ValueError("If input is a vector, sampling frequency fs must be provided.")

    data = data - np.mean(data)
    data = data / (np.max(np.abs(data)) + 1e-6)

    F0, _ = compute_f0_thanasis(data, fs)

    try:
        vf_close, vf_open = dypsa(data, fs)
        logging.debug(f"DYPSA detected {len(vf_close)} GCIs and {len(vf_open)} GOIs.")
        A0 = np.zeros(len(vf_close))
        for i in range(len(vf_close) - 1):
            A0[i] = max(np.abs(data[vf_open[i]:vf_close[i+1]]))

    except Exception as e:
        logging.error(f"DYPSA GCI detection failed: {e}")
        frame_len = int(0.01 * fs)
        if frame_len <= 0:
            return np.array([np.max(np.abs(data))])
        A0 = []
        for i in range(0, len(data) - frame_len + 1, frame_len):
            frame = data[i:i + frame_len]
            A0.append(np.max(np.abs(frame)))
        
    f0_valid_mask = ~np.isnan(F0)
    voice_activity_ratio = np.mean(f0_valid_mask)

    jitter_feats = compute_jitter_shimmer(F0)
    shimmer_feats = compute_jitter_shimmer(A0)

    hnr_feats, nhr_feats = compute_hnr_fun(data, fs)
    
    if len(vf_close) > 2 and len(vf_open) > 2:
        try:
            GQ_feats = compute_glottis_quotient(vf_close, vf_open, fs)
            logging.debug(f"GQ features computed using DYPSA: {GQ_feats}")
        except Exception as e:
            logging.error(f"DYPSA-based GQ calculation failed: {e}")
            GQ_feats = np.array([np.nan, np.nan, np.nan])
    else:
        logging.warning("Insufficient DYPSA GCIs/GOIs for GQ calculation")
        GQ_feats = np.array([np.nan, np.nan, np.nan])
    

    if len(vf_close) > 2 and len(vf_open) > 2:
        try:
            VFER_feats = compute_vfer_measure(data, fs, vf_close, vf_open)
            logging.debug(f"VFER features computed using DYPSA: {VFER_feats}")
        except Exception as e:
            logging.error(f"DYPSA-based VFER calculation failed: {e}")
            VFER_feats = np.array([np.nan]*7)
    else:
        logging.warning("Insufficient DYPSA GCIs/GOIs for VFER calculation")
        VFER_feats = np.array([np.nan]*7)

    GNE_feats = compute_gne_measure(data, fs)

    PPE_feat = compute_ppe(F0)

    melcepst_features, tc = melcepst(data, fs, 'E0dD', nc=12)
    logging.debug(f"MELCEPST features computed: {melcepst_features.shape} (frames x features)")

    log_energy = melcepst_features[:, 0]          # First column is log energy
    mfcc_coeffs = melcepst_features[:, 1:14]      # Next 13 are MFCC coefficients 0-12
    mfcc_deltas = melcepst_features[:, 14:28]     # All 14 deltas (log_energy + 13 MFCCs)
    mfcc_delta_deltas = melcepst_features[:, 28:42]  # All 14 delta-deltas
    
    mfcc = np.vstack([log_energy.reshape(1, -1), mfcc_coeffs.T])
    mfcc_delta = mfcc_deltas.T
    mfcc_delta_2 = mfcc_delta_deltas.T
    
    all_mfcc = np.vstack([mfcc, mfcc_delta, mfcc_delta_2])
    mfcc_mean = np.mean(all_mfcc, axis=1)
    mfcc_std = np.std(all_mfcc, axis=1)    

    
    if len(data) > 1000000:
        dfa_downsample = 4
        dfa_data = data[::dfa_downsample]
        logging.debug(f"Long signal detected for DFA ({len(data)} samples), using {dfa_downsample}x downsampling")
    elif len(data) > 500000:
        dfa_downsample = 2
        dfa_data = data[::dfa_downsample]
        logging.debug(f"Medium signal detected for DFA ({len(data)} samples), using {dfa_downsample}x downsampling")
    else:
        dfa_data = data
    
    dfa_value = compute_dfa(dfa_data)
    DFA_feat = 1/(1+np.exp(-dfa_value))

    data_resampled = resample(data, int(len(data) * 25000 / fs))
    RPDE_feat = compute_rpde(data_resampled, d=4, tau=50, eps=0.2, tmax=1000)

    if len(data) > 800000:
        downsample_factor = 8
        logging.debug(f"Very long signal detected ({len(data)} samples), using downsample factor {downsample_factor} for IMF")
    elif len(data) > 400000:
        downsample_factor = 6
        logging.debug(f"Long signal detected ({len(data)} samples), using downsample factor {downsample_factor} for IMF")
    elif len(data) > 200000:
        downsample_factor = 4
        logging.debug(f"Medium-long signal detected ({len(data)} samples), using downsample factor {downsample_factor} for IMF")
    elif len(data) > 100000:
        downsample_factor = 3
        logging.debug(f"Medium signal detected ({len(data)} samples), using downsample factor {downsample_factor} for IMF")
    elif len(data) > 50000:
        downsample_factor = 2
        logging.debug(f"Short-medium signal detected ({len(data)} samples), using downsample factor {downsample_factor} for IMF")
    else:
        downsample_factor = None
        logging.debug(f"Short signal detected ({len(data)} samples), using no downsampling for IMF")
   
    IMF_feats = compute_imf_measure(data, downsample_factor=downsample_factor)
    wavelet_feats, wavelet_names = compute_wavedec_features(F0)

    measures_vector = np.concatenate([
        np.ravel(jitter_feats),
        np.ravel(shimmer_feats),
        np.ravel(hnr_feats),
        np.ravel(nhr_feats),
        np.ravel([GQ_feats]),
        np.ravel(GNE_feats),
        np.ravel([VFER_feats]),
        np.ravel(mfcc_mean),
        np.ravel(mfcc_std),
        np.ravel(wavelet_feats),
        np.ravel([PPE_feat]),
        np.ravel([DFA_feat]),
        np.ravel([RPDE_feat]),
        np.ravel(IMF_feats),
        np.ravel([voice_activity_ratio])
    ])
    
    measures_vector = np.real(measures_vector)
    
    measures_names = []

    jitter_shimmer_labels = [
        "F0_abs_dif", "F0_dif_percent", "F0_PQ3_classical_Schoentgen", "F0_PQ3_classical_Baken",
        "F0_PQ3_generalised_Schoentgen", "F0_PQ5_classical_Schoentgen", "F0_PQ5_classical_Baken",
        "F0_PQ5_generalised_Schoentgen", "F0_PQ11_classical_Schoentgen", "F0_PQ11_classical_Baken",
        "F0_PQ11_generalised_Schoentgen", "F0_abs0th_perturb", "F0_DB", "F0_CV",
        "F0_TKEO_mean", "F0_TKEO_std", "F0_TKEO_prc5", "F0_TKEO_prc25", "F0_TKEO_prc75", "F0_TKEO_prc95",
        "F0_FM", "F0range_5_95_perc"
    ]

    measures_names += [f"Jitter->{name}" for name in jitter_shimmer_labels]

    measures_names += [f"Shimmer->{name}" for name in jitter_shimmer_labels]
    measures_names += ["HNR_mean", "HNR_std"]
    measures_names += ["NHR_mean", "NHR_std"]
    measures_names += ["GQ->prc5_95", "GQ->std_cycle_open", "GQ->std_cycle_close"]
    measures_names += [f"GNE->{name}" for name in ["mean", "std", "SNR_TKEO", "SNR_SEO", "NSR_TKEO", "NSR_SEO"]]
    measures_names += [f"VFER->{name}" for name in ["mean", "std", "entropy", "SNR_TKEO", "SNR_SEO", "NSR_TKEO", "NSR_SEO"]]
    
    # MFCC features: 42 total (14 static + 14 deltas + 14 delta-deltas)
    # Static: log_energy + c0-c12 (14 features)
    measures_names += ["mean_Log energy"] + [f"mean_MFCC_{i} coef" for i in range(13)]
    # Deltas: delta_log_energy + delta_c0-c12 (14 features)  
    measures_names += ["mean_delta log energy"] + [f"mean_{i} delta" for i in range(13)]
    # Delta-deltas: delta_delta_log_energy + delta_delta_c0-c12 (14 features)
    measures_names += ["mean_delta delta log energy"] + [f"mean_{i} delta-delta" for i in range(13)]

    # Same structure for std
    measures_names += ["std_Log energy"] + [f"std_MFCC_{i} coef" for i in range(13)]
    measures_names += ["std_delta log energy"] + [f"std_{i} delta" for i in range(13)]
    measures_names += ["std_delta delta log energy"] + [f"std_{i} delta-delta" for i in range(13)]

    measures_names += wavelet_names
    measures_names += ["PPE", "DFA", "RPDE"]
    measures_names += [f"IMF->{name}" for name in ["SNR_SEO", "SNR_TKEO", "SNR_entropy", "NSR_SEO", "NSR_TKEO", "NSR_entropy"]]
    measures_names += ["Voice_activity_ratio"]

    return measures_vector, measures_names, F0


def ordinal(n):
    if n == 0:
        return "0th"
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def estimate_f0(data, fs, f0_alg='SWIPE', f0min=50, f0max=500):
    """
    Fundamental frequency estimation dispatcher:
    - Uses SHRP if available
    - Otherwise SWIPE
    - Falls back to Thanasis method
    """
    f0_alg = f0_alg.upper()

    if f0_alg == 'SHRP':
        logging.debug("SHRP not available. Falling back to F0_Thanasis.")
        F0, _ = compute_f0_thanasis(data, fs, f0min=f0min, f0max=f0max)
        
    elif f0_alg == 'SWIPE':
        try:

            logging.debug("Using SWIPE algorithm")
            _f0, t = librosa.pyin(data, hop_length=int(round(0.01*fs)), fmin=f0min, fmax=f0max, sr=fs)
            F0 = np.nan_to_num(_f0, nan=0.0)
        except ImportError:
            logging.debug("SWIPE (via librosa) not available. Falling back to F0_Thanasis.")
            F0, _ = compute_f0_thanasis(data, fs, f0min=f0min, f0max=f0max)

    else:
        logging.debug("No F0 estimation algorithm specified. Falling back to F0_Thanasis.")
        F0, _ = compute_f0_thanasis(data, fs, f0min=f0min, f0max=f0max)
    return F0

