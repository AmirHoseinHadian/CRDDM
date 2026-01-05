import numpy as np
import pandas as pd
from numba import jit
from scipy.special import iv

@jit(nopython=True)
def simulate_PSDM_trial(threshold, drift_vec, ndt, sigma=1, dt=0.001):
    x = np.zeros((3,))
    muz = drift_vec[0]
    eta = drift_vec[1]
    
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

def rng(threshold, drift_vec, ndt,sigma=1, dt=0.001, n_sample=1):    
    RT = np.empty((n_sample,))
    Choice = np.empty((n_sample,))

    for n in range(n_sample):
        RT[n], Choice[n] = simulate_PSDM_trial(threshold, drift_vec.astype(np.float64), ndt,
                                               sigma=sigma, dt=dt)
    
    return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response'])

# The firs-passage time distribution of zero-drift process for small RTs
@jit(nopython=True)
def short_t_fpt_z(t, x):
    term1 = ((1 - x)*(1 + t)**2.5) / ((x + t) * t**1.5)
    term2 = np.exp(-.5*(1-x)**2/t -  0.5*np.pi**2*t)
    return term1*term2

# The firs-passage time distribution of zero-drift process
@jit(nopython=True)
def long_t_fpt_z(t, threshold, sigma=1, max_n=500):
    fpt_z = np.zeros(t.shape)
    for n in range(1, max_n):
        s = (-1)**(n+1) * n**2 * np.exp(- (n**2 * np.pi**2 * t)/(2*threshold**2))
        fpt_z += np.pi**2/threshold**2 * s
    return fpt_z

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

    norm_mu = np.linalg.norm(drift_vec, 2)
    theta_mu = np.arctan2(drift_vec[1], drift_vec[0])

    # Girsanov: no drift variability
    if s_v == 0:
        term1 = np.exp(threshold*norm_mu*np.cos(theta_mu)*np.cos(theta))
        term2 = iv(0, threshold*norm_mu*np.sin(theta_mu)*np.sin(theta))
        term3 = -0.5 * norm_mu**2 * tt

        log_density = np.log(2*np.pi) + np.log(term1) + np.log(term2) + term3 + np.log(fpt_z)

    else:
        pass # to be implemented later

    log_density[rt - ndt <= 0] = np.log(0.1**14)
    log_density = np.maximum(log_density, np.log(0.1**14))
        
    return log_density