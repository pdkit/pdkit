import numpy as np
from scipy import signal

def delta(c, width=2):
    nf, nc = c.shape
    
    vf = np.array([4, 3, 2, 1, 0, -1, -2, -3, -4]) / 60.0

    ww = 4 
    pad_top = np.tile(c[0:1, :], (ww, 1))
    pad_bottom = np.tile(c[-1:, :], (ww, 1))
    cx = np.vstack([pad_top, c, pad_bottom])
    

    cx_flat = cx.flatten('F')
    vx_flat = signal.lfilter(vf, [1.0], cx_flat)
    vx = vx_flat.reshape((nf + 8, nc), order='F')
    vx = vx[8:, :]
    
    return vx


def delta_delta(c):
    nf, nc = c.shape
    
    vf = np.array([4, 3, 2, 1, 0, -1, -2, -3, -4]) / 60.0
    ww = 4
    pad_top = np.tile(c[0:1, :], (ww, 1))
    pad_bottom = np.tile(c[-1:, :], (ww, 1))
    cx = np.vstack([pad_top, c, pad_bottom])
    cx_flat = cx.flatten('F')
    vx_flat = signal.lfilter(vf, [1.0], cx_flat)
    vx = vx_flat.reshape((nf + 8, nc), order='F')
    vx = vx[8:, :]
    
    af = np.array([1, 0, -1]) / 2.0
    ww_dd = 1
    pad_top_dd = np.tile(vx[0:1, :], (ww_dd, 1))
    pad_bottom_dd = np.tile(vx[-1:, :], (ww_dd, 1))
    cx_dd = np.vstack([pad_top_dd, vx, pad_bottom_dd])
    cx_dd_flat = cx_dd.flatten('F')
    ax_flat = signal.lfilter(af, [1.0], cx_dd_flat)
    ax = ax_flat.reshape((nf + 2, nc), order='F')
    ax = ax[2:, :]
    
    return ax