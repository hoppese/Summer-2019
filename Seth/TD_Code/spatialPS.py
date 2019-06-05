import numpy as np
import math
import scipy

def du_dx_spec(u,dx):
    return spec_deriv(u)

def fft_coeffs(u):
    return np.conj(2.0*np.fft.fft(u,len(u))) / len(u)

def inverse_fft(coeffs):
    return np.fft.ifft(np.conj(coeffs),len(coeffs))*len(coeffs)/2.0

def spec_deriv(u):
    N   = len(u)
    ck  = fft_coeffs(u)
    ckp = 1j*np.zeros(N)

    ckp[0:N//2]  = -2*np.pi*1j*np.arange(0, N//2, 1)
    ckp[N//2+1:] = -2*np.pi*1j*np.arange(-N//2 + 1, 0, 1)
    ckp = ckp*ck
    return np.real(inverse_fft(ckp))
