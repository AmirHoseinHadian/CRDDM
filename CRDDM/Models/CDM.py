import numpy as np
import pandas as pd
from numba import jit

from scipy.special import jv, iv
from scipy.special import jn_zeros

zeros = jn_zeros(0, 100)
JVZ = jv(1, zeros)

@jit(nopython=True)
def simulate_CDM_trial(threshold, drift_vec, ndt, s_v=0, s_a=0, s_t=0, sigma=1, dt=0.001):
    '''
    input:
        threshold: a positive floating number
        drift_vec: drift vector; a two-dimensional array
        ndt: a positive floating number
    returns:
        rt: response time
        theta: response angle between [-pi, pi]
    '''
    x = np.zeros((2,))
    
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
        mu_t = drift_vec + s_v*np.random.randn(2)
    else:
        mu_t = drift_vec
    
    while np.sqrt(x[0]**2 + x[1]**2) < threshold_t:
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(2)
        rt += dt
    
    theta = np.arctan2(x[1], x[0]) 
    return ndt_t+rt, theta


def rng(threshold, drift_vec, ndt, s_v=0, s_a=0, s_t=0, sigma=1, dt=0.001, n_sample=1):    
    RT = np.empty((n_sample,))
    Choice = np.empty((n_sample,))

    for n in range(n_sample):
        RT[n], Choice[n] = simulate_CDM_trial(threshold, drift_vec.astype(np.float64), ndt, 
                                              s_v=s_v, s_a=s_a, s_t=s_t, sigma=sigma, dt=dt)
    
    return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response'])


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

def response_time_pdf(t, threshold, drift_vec, sigma=1):
    kappa = threshold * np.linalg.norm(drift_vec)
    normalized_term = iv(0, kappa)
    girsanov_term = np.exp(-0.5 * (drift_vec[0]**2+ drift_vec[1]**2) * t)
    zero_drift_fpt = long_t_fpt_z(t, threshold, sigma=sigma)
    return normalized_term * girsanov_term * zero_drift_fpt

def response_pdf(theta, threshold, drift_vec):
    drift_angle = np.arctan2(drift_vec[1], drift_vec[0])
    kappa = threshold * np.linalg.norm(drift_vec)
    return 0.5/np.pi * np.exp(kappa * np.cos(theta - drift_angle)) / iv(0, kappa)

@jit(nopython=True)
def joint_lpdf(rt, theta, threshold, drift_vec, ndt, sigma=1):
    tt = np.maximum(rt - ndt, 0)
    s = tt/threshold**2
    
    s0 = 0.002
    s1 = 0.02
    w = np.minimum(np.maximum((s - s0) / (s1 - s0), 0), 1)
    
    # first-passage time density of zero drift process
    fpt_lt = long_t_fpt_z(tt, threshold, sigma=sigma)
    fpt_st = 1/threshold**2 * short_t_fpt_z(tt/threshold**2, 0.1**8/threshold**2)   
    fpt_z =  (1 - w) * fpt_st + w * fpt_lt
    
    fpt_z = np.maximum(fpt_z, 0.1**14)

    mu_dot_x0 = drift_vec[0]*np.cos(theta)
    mu_dot_x1 = drift_vec[1]*np.sin(theta)

    term1 = threshold * (mu_dot_x0 + mu_dot_x1)
    term2 = 0.5 * (drift_vec[0]**2+ drift_vec[1]**2) * tt

    log_density = term1 - term2 + np.log(fpt_z) - np.log(2*np.pi)
    
    log_density[rt - ndt < 0] = np.log(0.1**14)
    log_density = np.maximum(log_density, np.log(0.1**14))
        
    return log_density