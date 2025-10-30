import logging
import numpy as np
from pdkit.voice_features.tkeo import compute_tkeo
from pdkit.voice_features.native.libeemd.eemd_hook import fast_eemd

def log_bb(x):
    x = np.asarray(x)
    x_safe = np.where(x <= 0, 1e-12, x)
    result = np.log(x_safe)
    return np.where(x == 0, 0, result)

def compute_imf_measure(data, downsample_factor=None):  
    if downsample_factor is not None and len(data) > 50000:
        data = data[::downsample_factor]
    
    original_len = len(data) if downsample_factor is None else len(data) * downsample_factor
    
    if original_len > 2000000:
        S_number, num_siftings = 4, 500
        logging.debug(f"Very long signal: using fast EMD parameters (S={S_number}, sift={num_siftings})")
    elif original_len > 1000000:
        S_number, num_siftings = 6, 1000
        logging.debug(f"Long signal: using medium EMD parameters (S={S_number}, sift={num_siftings})")
    elif original_len > 500000:
        S_number, num_siftings = 10, 2000
        logging.debug(f"Medium signal: using standard EMD parameters (S={S_number}, sift={num_siftings})")
    else:
        S_number, num_siftings = 15, 5000
        logging.debug(f"Short signal: using high-quality EMD parameters (S={S_number}, sift={num_siftings})")

    imf = fast_eemd(data, S_number=S_number, num_siftings=num_siftings)
    
    if imf.ndim == 1:
        logging.warning("fast_eemd returned 1D array, reshaping to 2D")
        imf = imf[np.newaxis, :]
    
    IMF_dec = imf.T
    IMF_dec2 = log_bb(IMF_dec)
    N, M = IMF_dec.shape
    
    IMF_decEnergy = np.zeros(M)
    IMF_decTKEO = np.zeros(M) 
    IMF_decEntropia = np.zeros(M)
    IMF_decEnergy2 = np.zeros(M)
    IMF_decTKEO2 = np.zeros(M)
    IMF_decEntropia2 = np.zeros(M)
    
    for i in range(M):
        IMF_decEnergy[i] = np.abs(np.mean(IMF_dec[:, i]**2))
        IMF_decTKEO[i] = np.abs(np.mean(compute_tkeo(IMF_dec[:, i])))
        imf_abs = np.abs(IMF_dec[:, i])
        IMF_decEntropia[i] = np.abs(np.mean(-np.sum(imf_abs * log_bb(imf_abs))))
        IMF_decEnergy2[i] = np.abs(np.mean(IMF_dec2[:, i]**2))
        IMF_decTKEO2[i] = np.abs(np.mean(compute_tkeo(IMF_dec2[:, i])))
        imf2_abs = np.abs(IMF_dec2[:, i])
        IMF_decEntropia2[i] = np.abs(np.mean(-np.sum(imf2_abs * log_bb(imf2_abs))))
    try:
        E1 = np.sum(IMF_decEnergy[3:]) / np.sum(IMF_decEnergy[:3])
        T1 = np.sum(IMF_decTKEO[3:]) / np.sum(IMF_decTKEO[:3])
        H1 = np.sum(IMF_decEntropia[3:]) / np.sum(IMF_decEntropia[:3])

        E2 = np.abs(np.sum(IMF_decEnergy2[:2]) / np.sum(IMF_decEnergy2[3:]))
        T2 = np.abs(np.sum(IMF_decTKEO2[:2]) / np.sum(IMF_decTKEO2[2:]))
        H2 = np.sum(IMF_decEntropia2[:2]) / np.sum(IMF_decEntropia2[2:])
        
    except (ZeroDivisionError, IndexError):
        E1 = T1 = H1 = E2 = T2 = H2 = np.nan
    
    return np.array([E1, T1, H1, E2, T2, H2])