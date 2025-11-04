"""
Python wrapper for the close_ret C library using ctypes.
Provides fast computation of close return histograms for RPDE analysis.
"""
import ctypes
import numpy as np
import os
import platform
from importlib import resources


def _bin_lib_name():
    """Get platform-specific library name."""
    sysname = platform.system()
    if sysname == "Windows":
        return "libclose_ret.dll"
    elif sysname == "Darwin":
        return "libclose_ret.dylib"
    else:
        return "libclose_ret.so"


def _resolve_libclose_ret_path():
    """
    Prefer the packaged binary under voice_features/_bin/.
    If not found (editable/dev), try next to this file as a fallback.
    """
    libname = _bin_lib_name()

    # Try multiple search strategies
    candidates = []

    # 1. packaged location using importlib.resources (works in installed packages)
    try:
        import voice_features  # noqa
        with resources.as_file(resources.files("voice_features") / "_bin" / libname) as p:
            candidates.append(str(p))
    except Exception:
        pass

    # 2. Check relative to voice_features package location (works in editable installs)
    try:
        import voice_features
        import sys
        # Get all possible voice_features module locations
        if 'voice_features' in sys.modules and hasattr(sys.modules['voice_features'], '__path__'):
            for path in sys.modules['voice_features'].__path__:
                candidates.append(os.path.join(path, '_bin', libname))
    except Exception:
        pass

    # 3. dev fallback: same directory as this hook (useful in local builds)
    here = os.path.dirname(__file__)
    candidates.append(os.path.join(here, libname))

    # Try each candidate
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # last resort: let the OS loader try PATH / LD_LIBRARY_PATH
    return libname  # may still work if user added folder to PATH


def _load_native_close_ret(lib_path: str):
    """Load the close_ret library and set up function signatures."""
    close_ret_lib = ctypes.CDLL(lib_path)

    # Set up the close_ret function signature
    # long close_ret(
    #     const double *input,
    #     unsigned long N,
    #     unsigned long m,
    #     unsigned long tau,
    #     double eta,
    #     double *output
    # )
    close_ret_lib.close_ret.restype = ctypes.c_long
    close_ret_lib.close_ret.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ]

    def fast_close_ret(x, m=4, tau=50, eta=0.2):
        # Convert input to contiguous float64 array
        x = np.ascontiguousarray(x, dtype=np.float64)
        N = len(x)
        
        # Calculate expected embedded space size
        if N < (m - 1) * tau + 1:
            return None  # Signal too short
            
        embed_elements = N - (m - 1) * tau
        
        # Allocate output array
        output = np.zeros(embed_elements, dtype=np.float64)
        
        # Call C function
        result = close_ret_lib.close_ret(
            x,
            ctypes.c_ulong(N),
            ctypes.c_ulong(m),
            ctypes.c_ulong(tau),
            ctypes.c_double(eta),
            output
        )
        
        # Check for errors
        if result < 0:
            return None
            
        return output

    return fast_close_ret


# Try to load the native library, fallback to Python implementation
try:
    _lib_path = _resolve_libclose_ret_path()
    fast_close_ret = _load_native_close_ret(_lib_path)
except Exception as e:
    # Define a pure Python fallback if needed, or just return None
    def fast_close_ret(x, m=4, tau=50, eta=0.2):
        """Fallback - returns None to signal Python implementation should be used."""
        return None  # Signal to caller that native impl not available
