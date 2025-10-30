import ctypes
import numpy as np
import os
import platform
from importlib import resources

try:
    from PyEMD import EMD as _PyEMD
except Exception:
    _PyEMD = None


def _bin_lib_name():
    sysname = platform.system()
    if sysname == "Windows":
        return "libeemd.dll"
    elif sysname == "Darwin":
        return "libeemd.dylib"
    else:
        return "libeemd.so"


def _resolve_libeemd_path():
    """
    Prefer the packaged binary under voice_features/_bin/.
    If not found (editable/dev), try next to this file as a fallback.
    """
    libname = _bin_lib_name()

    # packaged location: voice_features/_bin/libeemd.*
    # __package__ here should be "voice_features.native.libeemd"
    # so we reference the top-level package explicitly:
    try:
        import voice_features  # noqa
        with resources.as_file(resources.files("voice_features") / "_bin" / libname) as p:
            if p.exists():
                return str(p)
    except Exception:
        pass

    # dev fallback: same directory as this hook (useful in local builds)
    here = os.path.dirname(__file__)
    candidate = os.path.join(here, libname)
    if os.path.exists(candidate):
        return candidate

    # last resort: let the OS loader try PATH / LD_LIBRARY_PATH
    return libname  # may still work if user added folder to PATH


def _load_native_fast_eemd(lib_path: str):
    # CDLL is fine cross-platform; WinDLL isn’t needed unless you need stdcall.
    eemd_lib = ctypes.CDLL(lib_path)

    # --- signatures from libeemd's C API ---
    # int eemd(const double* input, size_t N, double* output, size_t M,
    #          unsigned ensemble_size, double noise_strength,
    #          unsigned S_number, unsigned num_siftings, unsigned long rng_seed);
    eemd_lib.eemd.restype = ctypes.c_int
    eemd_lib.eemd.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_double,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_ulong,
    ]

    # size_t emd_num_imfs(size_t N);
    if hasattr(eemd_lib, "emd_num_imfs"):
        eemd_lib.emd_num_imfs.restype = ctypes.c_size_t
        eemd_lib.emd_num_imfs.argtypes = [ctypes.c_size_t]
        _has_num_imfs = True
    else:
        _has_num_imfs = False

    # int ceemdan(...)  (optional, but keep signature correct if you use it elsewhere)
    if hasattr(eemd_lib, "ceemdan"):
        eemd_lib.ceemdan.restype = ctypes.c_int
        eemd_lib.ceemdan.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_double,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_ulong,
        ]

    def fast_eemd(
        signal,
        ensemble_size: int = 1,
        noise_strength: float = 0.0,
        S_number: int = 4,
        num_siftings: int = 2000,
        rng_seed: int = 0,
        max_imfs: int = 14,
    ):
        """
        EEMD via libeemd.
        Returns IMFs as (M, N) to match your downstream code.
        """
        x = np.ascontiguousarray(signal, dtype=np.float64)
        N = x.size

        if _has_num_imfs:
            M_calc = int(eemd_lib.emd_num_imfs(N))
            M = max(1, min(M_calc, max_imfs))
        else:
            # Conservative heuristic if symbol missing
            M = min(int(np.floor(np.log2(max(N, 2)))), max_imfs)

        out = np.zeros(N * M, dtype=np.float64)

        err = eemd_lib.eemd(
            x, N, out, M,
            ctypes.c_uint(max(1, int(ensemble_size))),
            float(noise_strength),
            ctypes.c_uint(max(1, int(S_number))),
            ctypes.c_uint(max(1, int(num_siftings))),
            ctypes.c_ulong(max(0, int(rng_seed))),
        )
        if err != 0:
            # Non-zero error code: return a single IMF of zeros (shape compatible)
            return np.zeros((1, N), dtype=np.float64)

        return out.reshape(M, N)

    return fast_eemd

try:
    _lib_path = _resolve_libeemd_path()
    fast_eemd = _load_native_fast_eemd(_lib_path)
except Exception as e:
    def fast_eemd(signal, **kwargs):
        if _PyEMD is None:
            raise RuntimeError(
                f"Native libeemd failed to load ({e}) and PyEMD is not installed."
            )
        return _PyEMD().emd(np.asarray(signal, dtype=float))
