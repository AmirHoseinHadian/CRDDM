import numpy as np
import pandas as pd
from numba import jit

@jit(nopython=True)
def simulate_PSDM_trial(threshold, ndt, mu, sigma=1, dt=0.001):
    x = np.zeros((3,))
    muz = mu[0]
    eta = mu[1]
    
    norm_mu = np.sqrt(eta**2 + muz**2)
    theta_mu = np.arctan2(eta, muz)
    
    rt = 0
    rphi = np.pi/4 # it is not important (just a dummpy value)
    mux = norm_mu * np.sin(theta_mu) * np.cos(rphi)
    muy = norm_mu * np.sin(theta_mu) * np.sin(rphi)
    while np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) < threshold:
        x[0] += mux*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1)
        x[1] += muy*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1)
        x[2] += muz*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1)
        
        rt += dt
    theta = np.arctan2(np.sqrt(x[0]**2 + x[1]**2), x[2])    
    
    return rt+ndt, theta