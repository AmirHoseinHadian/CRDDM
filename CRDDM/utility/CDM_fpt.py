import numpy as np
import pandas as pd

from numba import jit
from scipy.special import iv, ive
from scipy.interpolate import interp1d

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


def k(a, da, t, q, sigma=2):
    return 0.5 * (q - 0.5*sigma - da(t))

def psi(a, da, t, z, tau, q, sigma=2):
    kk = k(a, da, t, q, sigma)
    
    if 2*np.sqrt(a(t)*z)/(sigma*(t-tau))<=700:
        term1 = 1./(sigma*(t - tau)) * np.exp(- (a(t) + z)/(sigma*(t-tau)))
        term2 = (a(t)/z)**(0.5*(q-sigma)/sigma)
        term3 = da(t) - (a(t)/(t-tau)) + kk
        term4 = iv(q/sigma-1, 2*np.sqrt(a(t)*z)/(sigma*(t-tau)))
        term5 = (np.sqrt(a(t)*z)/(t-tau)) * iv(q/sigma, 2*np.sqrt(a(t)*z)/(sigma*(t-tau)))
    else:
        term1 = 1./(sigma*(t - tau))
        term2 = (a(t)/z)**(0.5*(q-sigma)/sigma)
        term3 = da(t) - (a(t)/(t-tau)) + kk
        term4 = ive(q/sigma-1, (a(t) + z)/(sigma*(t-tau)))
        term5 = (np.sqrt(a(t)*z)/(t-tau)) * ive(q/sigma, (a(t) + z)/(sigma*(t-tau)))
    
    return term1 * term2 * (term3 * term4 + term5)

def ie_fpt(a, da, q, z, sigma=2, dt=0.1, T_max=2):
    g = [0]
    T = [0]
    g.append(-2*psi(a, da, dt, z, 0, q, sigma))
    T.append(dt)
    
    for n in range(2, int(T_max/dt)+2):
        s = -2 * psi(a, da, n*dt, z, 0, q, sigma)

        for j in range(1, n):
            if a(j*dt) == 0:
                continue
            
            s += 2 * dt * g[j] * psi(a, da, n*dt, a(j*dt), j*dt, q, sigma)

        g.append(s)
        T.append(n*dt)
        
    g = np.asarray(g)
    T = np.asarray(T)
    
    gt = interp1d(T, g)
    return gt