import numpy as np
import pandas as pd
from numba import jit

from scipy.special import jv, iv
from scipy.special import jn_zeros

zeros = jn_zeros(1, 100)
JVZ = jv(2, zeros)

@jit(nopython=True)
def simulate_HSDM_trial(threshold, drift_vec, ndt, s_v=0, s_a=0, s_t=0, sigma=1, dt=0.001):
    '''
    input:
        threshold: a positive floating number
        drift_vec: drift vector; a four-dimensional array
        ndt: a positive floating number
        s_v: standard deviation of drift rate variability
        s_a: range of threshold variability
        s_t: range of non-decision time variability
        sigma: standard deviation of the diffusion process
        dt: time step for the simulation
    returns:
        rt: response time
        theta: response angle between [-pi, pi]
    '''
    x = np.zeros((4,))
    
    rt = 0

    if s_a>0:
        threshold_t = threshold + (2*s_a*np.random.rand() - s_a)
    else:
        threshold_t = threshold

    if s_t>0:
        ndt_t = ndt + (2*s_t*np.random.rand() - s_t)
    else:
        ndt_t = ndt

    if s_v>0:
        mu_t = drift_vec + s_v*np.random.randn(4)
    else:
        mu_t = drift_vec

    while np.linalg.norm(x) < threshold_t:
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(4)
        rt += dt
    
    theta1 = np.arctan2(np.sqrt(x[3]**2 + x[2]**2 + x[1]**2), x[0])
    theta2 = np.arctan2(np.sqrt(x[3]**2 + x[2]**2), x[1])
    theta3 = np.arctan2(x[3], x[2])

    return ndt_t+rt, (theta1, theta2, theta3)