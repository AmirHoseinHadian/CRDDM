import numpy as np
import pandas as pd
from numba import jit

from CRDDM.utility.fpts import sdm_short_t_fpt_z, sdm_long_t_fpt_z, ie_fpt

@jit(nopython=True)
def simulate_SDM_trial(threshold, drift_vec, ndt, decay=0, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    input:
        threshold: a positive floating number
        drift_vec: drift vector; a three-dimensional array
        ndt: a positive floating number
        decay: decay rate of the collapsing boundary
        s_v: standard deviation of drift rate variability
        s_t: range of non-decision time variability
        sigma: standard deviation of the diffusion process
        dt: time step for the simulation
    returns:
        rt: response time in seconds
        theta: a tuple of response angles (theta1, theta2); theta 1 between [0, pi] and theta2 between [-pi, pi]
    '''
    x = np.zeros((3,))
    
    rt = 0

    if s_t>0:
        ndt_t = ndt + (2*s_t*np.random.rand() - s_t)
    else:
        ndt_t = ndt

    if s_v>0:
        mu_t = drift_vec + s_v*np.random.randn(3)
    else:
        mu_t = drift_vec

    while np.linalg.norm(x) < threshold - decay*rt:
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
        rt += dt
    
    theta1 = np.arctan2(np.sqrt(x[2]**2 + x[1]**2), x[0])
    theta2 = np.arctan2(x[2], x[1])

    return ndt_t+rt, (theta1, theta2)


class FixedThresholdSDM:
    '''
    Spherical Diffusion Model with fixed boundaries
    '''

    def __init__(self):
        self.name = 'Spherical Diffusion Model with fixed boundaries'

    def simulate(self, threshold, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):    
        RT = np.empty((n_sample,))
        Choice = np.empty((n_sample, 2))

        for n in range(n_sample):
            RT[n], Choice[n, :] = simulate_SDM_trial(threshold, drift_vec.astype(np.float64), ndt, 
                                                     s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        
        return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response1', 'response2'])

    def joint_lpdf(self, rt, theta, threshold, drift_vec, ndt, s_v=0, s_t=0, sigma=1):
        tt = np.maximum(rt - ndt, 0)
        s = tt/threshold**2
        
        s0 = 0.002
        s1 = 0.02
        w = np.minimum(np.maximum((s - s0) / (s1 - s0), 0), 1)
        
        # first-passage time density of zero drift process
        fpt_lt = sdm_long_t_fpt_z(tt, threshold, sigma=sigma)
        fpt_st = 1/threshold**2 * sdm_short_t_fpt_z(tt/threshold**2, 0.1**8/threshold**2)   
        fpt_z =  (1 - w) * fpt_st + w * fpt_lt
        fpt_z = np.maximum(fpt_z, 0.1**14)


        # Girsanov: no drift variability
        if s_v == 0:
            mu_dot_x0 = drift_vec[0]*np.cos(theta[:, 0])
            mu_dot_x1 = drift_vec[1]*np.sin(theta[:, 0])*np.cos(theta[:, 1]) 
            mu_dot_x2 = drift_vec[2]*np.sin(theta[:, 0])*np.sin(theta[:, 1])
            term1 = threshold * (mu_dot_x0 + mu_dot_x1 + mu_dot_x2)
            term2 = 0.5 * np.linalg.norm(drift_vec, 2)**2 * tt
            
            log_density = term1 - term2 + np.log(fpt_z) - np.log(2*np.pi)
        else:
            pass # to be implemented later

        log_density[rt - ndt <= 0] = np.log(0.1**14)
        log_density = np.maximum(log_density, np.log(0.1**14))
            
        return log_density


class CollapsingThresholdSDM:
    def __init__(self):
        self.name = 'Spherical Diffusion Model with collapsing boundaries'
    
    def simulate(self, threshold, decay, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):
        RT = np.empty((n_sample,))
        Choice = np.empty((n_sample, 2))

        for n in range(n_sample):
            RT[n], Choice[n, :] = simulate_SDM_trial(threshold, drift_vec.astype(np.float64), ndt, 
                                                     decay=decay, s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        
        return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response1', 'response2'])

    def joint_lpdf(self, rt, theta, threshold, decay, drift_vec, ndt, s_v=0, s_t=0, sigma=1):
        tt = np.maximum(rt - ndt, 0)

        T_max = min(rt.max(), threshold/decay)
        g_z, T = ie_fpt(threshold, decay, 3, 0.000001, dt=0.02, T_max=T_max)
        
        fpt_z = np.interp(tt, T, g_z)
        fpt_z = np.maximum(fpt_z, 0.1**14)

        # Girsanov: no drift variability
        if s_v == 0:
            mu_dot_x0 = drift_vec[0]*np.cos(theta[:, 0])
            mu_dot_x1 = drift_vec[1]*np.sin(theta[:, 0])*np.cos(theta[:, 1]) 
            mu_dot_x2 = drift_vec[2]*np.sin(theta[:, 0])*np.sin(theta[:, 1])
            term1 = (threshold - decay*tt) * (mu_dot_x0 + mu_dot_x1 + mu_dot_x2)
            term2 = 0.5 * np.linalg.norm(drift_vec, 2)**2 * tt

            log_density = term1 - term2 + np.log(fpt_z) - np.log(2*np.pi)

        log_density[rt - ndt <= 0] = np.log(0.1**14)
        log_density = np.maximum(log_density, np.log(0.1**14))
            
        return log_density