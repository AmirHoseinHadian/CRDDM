import numpy as np
import pandas as pd
from numba import jit
from scipy.special import iv

from CRDDM.utility.Constants import zeros_0 as zeros
from CRDDM.utility.Constants import JVZ1 as JVZ

@jit(nopython=True)
def simulate_CDM_trial(threshold, drift_vec, ndt, s_v=0, s_a=0, s_t=0, sigma=1, dt=0.001):
    '''
    input:
        threshold: a positive floating number
        drift_vec: drift vector; a two-dimensional array
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

    while np.linalg.norm(x) < threshold_t:
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(2)
        rt += dt
    
    theta = np.arctan2(x[1], x[0]) 
    return ndt_t+rt, theta

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

class CDM:
    '''
    Circular Diffusion Model
    '''

    def __init__(self):
        self.name = 'CDM'


    def rng(self, threshold, drift_vec, ndt, s_v=0, s_a=0, s_t=0, sigma=1, dt=0.001, n_sample=1):    
        RT = np.empty((n_sample,))
        Choice = np.empty((n_sample,))

        for n in range(n_sample):
            RT[n], Choice[n] = simulate_CDM_trial(threshold, drift_vec.astype(np.float64), ndt, 
                                                  s_v=s_v, s_a=s_a, s_t=s_t, sigma=sigma, dt=dt)
        
        return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response'])

    def response_time_pdf(self, t, threshold, drift_vec, sigma=1):
        kappa = threshold * np.linalg.norm(drift_vec)
        normalized_term = iv(0, kappa)
        girsanov_term = np.exp(-0.5 * (drift_vec[0]**2+ drift_vec[1]**2) * t)
        zero_drift_fpt = long_t_fpt_z(t, threshold, sigma=sigma)
        return normalized_term * girsanov_term * zero_drift_fpt

    def response_pdf(self,theta, threshold, drift_vec):
        drift_angle = np.arctan2(drift_vec[1], drift_vec[0])
        kappa = threshold * np.linalg.norm(drift_vec)
        return 0.5/np.pi * np.exp(kappa * np.cos(theta - drift_angle)) / iv(0, kappa)


    def joint_lpdf(self, rt, theta, threshold, drift_vec, ndt, s_v=0, s_a=0, s_t=0, sigma=1):
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
            mu_dot_x0 = drift_vec[0]*np.cos(theta)
            mu_dot_x1 = drift_vec[1]*np.sin(theta)

            term1 = threshold * (mu_dot_x0 + mu_dot_x1)
            term2 = 0.5 * (drift_vec[0]**2+ drift_vec[1]**2) * tt

            log_density = term1 - term2 + np.log(fpt_z) - np.log(2*np.pi)
        else:
            s_v2 = s_v**2
            x0 =  threshold*np.cos(theta)
            x1 =  threshold*np.sin(theta)
            fixed = 1/(np.sqrt(s_v2 * tt + 1))
            exponent0 = -0.5*drift_vec[0]**2/s_v2 + 0.5*(x0 * s_v2 + drift_vec[0])**2 / (s_v2 * (s_v2 * tt + 1))
            exponent1 = -0.5*drift_vec[1]**2/s_v2 + 0.5*(x1 * s_v2 + drift_vec[1])**2 / (s_v2 * (s_v2 * tt + 1))

            # the joint density of choice and RT for the full process
            log_density = 2*np.log(fixed) + exponent0 + exponent1 + np.log(fpt_z) - np.log(2*np.pi)

        log_density[rt - ndt <= 0] = np.log(0.1**14)
        log_density = np.maximum(log_density, np.log(0.1**14))
            
        return log_density