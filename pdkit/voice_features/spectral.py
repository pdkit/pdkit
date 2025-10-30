from scipy.fft import rfft as _rfft, irfft as _irfft, dct as _dct, idct as _idct

def rfft(x, n):
    return _rfft(x, n=n, axis=-1)

def irfft(X, n):
    return _irfft(X, n=n, axis=-1)

def rdct(y, type=2, norm="ortho"):
    return _dct(y, type=type, norm=norm, axis=-1)

def irdct(c, type=2, norm="ortho"):
    return _idct(c, type=type, norm=norm, axis=-1)