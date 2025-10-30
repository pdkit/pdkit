import numpy as np

def compute_glottis_quotient(VF_close, VF_open, fs, f0min=50, f0max=500, flag=True):
    VF_close = np.asarray(VF_close, dtype=float)
    VF_open = np.asarray(VF_open, dtype=float)

    cycle_open = np.abs(VF_open[1:] - VF_close[:-1]).astype(float)
    cycle_closed = np.abs(VF_open[:-1] - VF_close[:-1]).astype(float)

    if flag:
        low_lim = fs / f0max
        up_lim = fs / f0min
        N = len(cycle_open)

        mask_open = (cycle_open > up_lim) | (cycle_open < low_lim)
        mask_closed = (cycle_closed > up_lim) | (cycle_closed < low_lim)
        
        cycle_open[mask_open] = np.nan
        cycle_closed[mask_closed] = np.nan
    

    prc1 = np.nanpercentile(cycle_open, [5, 95])
    cycle_open_range_5_95_perc = prc1[1] - prc1[0]
    prc2 = np.nanpercentile(cycle_closed, [5, 95])
    cycle_closed_range_5_95_perc = prc2[1] - prc2[0]

    GQ = np.zeros(3)
    total_range = cycle_open_range_5_95_perc + cycle_closed_range_5_95_perc
    if total_range > 0:
        GQ[0] = cycle_open_range_5_95_perc / total_range
    else:
        GQ[0] = np.nan
    GQ[1] = np.nanstd(cycle_open, ddof=1)
    GQ[2] = np.nanstd(cycle_closed, ddof=1)

    return GQ
