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

def rng(threshold, drift_vec, ndt, s_v=0, s_a=0, s_t=0, sigma=1, dt=0.001, n_sample=1):    
    RT = np.empty((n_sample,))
    Choice = np.empty((n_sample, 3))

    for n in range(n_sample):
        RT[n], Choice[n, :] = simulate_HSDM_trial(threshold, drift_vec.astype(np.float64), ndt, 
                                                  s_v=s_v, s_a=s_a, s_t=s_t, sigma=sigma, dt=dt)
    
    return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response1', 'response2', 'response3'])

# The firs-passage time distribution of zero-drift process for small RTs
@jit(nopython=True)
def short_t_fpt_z(t, x):
    term1 = ((1 - x)*(1 + t)**3) / ((x + t) * np.sqrt(x + t) * t**1.5)
    term2 = np.exp(-.5*(1-x)**2/t -  0.5*zeros[0]**2*t)
    return term1*term2

# The firs-passage time distribution of zero-drift process
@jit(nopython=True)
def long_t_fpt_z(t, threshold, sigma=1):
    fpt_z = np.zeros(t.shape)
    for i in range(t.shape[0]):
        series = np.sum((zeros**2/JVZ) * np.exp(-(zeros**2 * sigma**2)/(2*threshold**2)*t[i]))
        fpt_z[i] = 0.5*sigma**2/threshold**2 * series
    return fpt_z


@jit(nopython=True)
def joint_lpdf(rt, theta, threshold, drift_vec, ndt, s_v=0, s_a=0, s_t=0, sigma=1):
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


    # Girsanov: no drift variability
    if s_v == 0:
        mu_dot_x0 = drift_vec[0]*np.cos(theta[:, 0])
        mu_dot_x1 = drift_vec[1]*np.sin(theta[:, 0])*np.cos(theta[:, 1]) 
        mu_dot_x2 = drift_vec[2]*np.sin(theta[:, 0])*np.sin(theta[:, 1])*np.cos(theta[:, 2])
        mu_dot_x3 = drift_vec[3]*np.sin(theta[:, 0])*np.sin(theta[:, 1])*np.sin(theta[:, 2])
        term1 = threshold * (mu_dot_x0 + mu_dot_x1 + mu_dot_x2 + mu_dot_x3)
        term2 = 0.5 * np.linalg.norm(drift_vec, 2)**2 * tt
        
        log_density = term1 - term2 + np.log(fpt_z) - np.log(2*np.pi)
    else:
        pass # to be implemented later

    log_density[rt - ndt <= 0] = np.log(0.1**14)
    log_density = np.maximum(log_density, np.log(0.1**14))
        
    return log_density