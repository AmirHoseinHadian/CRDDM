import numpy as np
import pandas as pd
from numba import jit

@jit(nopython=True)
def simulate_SDM_trial(threshold, drift_vec, ndt, s_v=0, s_a=0, s_t=0, sigma=1, dt=0.001):
    '''
    input:
        threshold: a positive floating number
        drift_vec: drift vector; a three-dimensional array
        ndt: a positive floating number
    returns:
        rt: response time
        theta: response angle between [-pi, pi]
    '''
    x = np.zeros((3,))
    
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
        mu_t = drift_vec + s_v*np.random.randn(3)
    else:
        mu_t = drift_vec

    while np.linalg.norm(x) < threshold_t:
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
        rt += dt
    
    theta1 = np.arctan2(np.sqrt(x[2]**2 + x[1]**2), x[0])
    theta2 = np.arctan2(x[2], x[1])

    return ndt_t+rt, (theta1, theta2)