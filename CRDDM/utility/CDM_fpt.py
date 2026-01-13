import numpy as np

from numba import jit
from CRDDM.utility.helpers import iv_numba, ive_numba

from CRDDM.utility.Constants import zeros_0 as zeros
from CRDDM.utility.Constants import JVZ1 as JVZ

# The firs-passage time distribution of zero-drift process for small RTs
@jit(nopython=True)
def short_t_fpt_z(t, x):
    term1 = ((1 - x)*(1 + t)**2) / (np.sqrt(x + t) * t**1.5)
    term2 = np.exp(-.5*(1-x)**2/t -  0.5*zeros[0]**2*t)
    return term1*term2

# The firs-passage time distribution of zero-drift process
@jit(nopython=True)
def long_t_fpt_z(t, threshold, sigma=1):
    fpt_z = np.zeros(t.shape)
    for i in range(t.shape[0]):
        series = np.sum((zeros/JVZ) * np.exp(-(zeros**2 * sigma**2)/(2*threshold**2)*t[i]))
        fpt_z[i] = sigma**2/threshold**2 * series
    return fpt_z


@jit(nopython=False)
def k(threshold, decay, t, q, sigma=2):
    da = -2*decay * (threshold - decay*t)
    return 0.5 * (q - 0.5*sigma - da)

@jit(nopython=False)
def psi(threshold, decay, t, z, tau, q, sigma=2):
    kk = k(threshold, decay, t, q, sigma)
    
    a = (threshold - decay*t)**2
    da = -2*decay * (threshold - decay*t)
    
    if 2*np.sqrt(a*z)/(sigma*(t-tau))<=700:
        term1 = 1./(sigma*(t - tau)) * np.exp(- (a + z)/(sigma*(t-tau)))
        term2 = (a/z)**(0.5*(q-sigma)/sigma)
        term3 = da - (a/(t-tau)) + kk
        term4 = iv_numba(q/sigma-1, 2*np.sqrt(a*z)/(sigma*(t-tau)))
        term5 = (np.sqrt(a*z)/(t-tau)) * iv_numba(q/sigma, 2*np.sqrt(a*z)/(sigma*(t-tau)))
    else:
        term1 = 1./(sigma*(t - tau))
        term2 = (a/z)**(0.5*(q-sigma)/sigma)
        term3 = da - (a/(t-tau)) + kk
        term4 = ive_numba(q/sigma-1, (a + z)/(sigma*(t-tau)))
        term5 = (np.sqrt(a*z)/(t-tau)) * ive_numba(q/sigma, (a + z)/(sigma*(t-tau)))
    
    return term1 * term2 * (term3 * term4 + term5)

@jit(nopython=False)
def ie_fpt(threshold, decay, q, z, sigma=2, dt=0.1, T_max=2):
    g = np.zeros((int(T_max/dt)+2,))
    T = np.zeros((int(T_max/dt)+2,))
    if threshold - decay*dt > 0:
        g[1] = -2*psi(threshold, decay, dt, z, 0, q, sigma)
    T[1] = dt
    
    for n in range(2, int(T_max/dt)+2):
        if threshold - decay*(n*dt) <= 0:
            T[n] = n*dt
            continue
        
        s = -2 * psi(threshold, decay, n*dt, z, 0, q, sigma)

        for j in range(1, n):
            if threshold - decay*(j*dt) <= 0:
                continue
            
            s += 2 * dt * g[j] * psi(threshold, decay, n*dt, (threshold - decay*(j*dt))**2, j*dt, q, sigma)

        g[n] = s
        T[n] = n*dt

    return g, T