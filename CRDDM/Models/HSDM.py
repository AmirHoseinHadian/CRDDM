import numpy as np
import pandas as pd
from scipy.special import iv

from CRDDM.utility.simulators import simulate_HSDM_trial
from CRDDM.utility.fpts import hsdm_short_t_fpt_z, hsdm_long_t_fpt_z, ie_fpt


class FixedThresholdHSDM:
    '''
    Hyper-Spherical Diffusion Model with fixed boundaries
    '''

    def __init__(self):
        self.name = 'Hyper-Spherical Diffusion Model with fixed boundaries'

    def simulate(self, threshold, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):    
        RT = np.empty((n_sample,))
        Choice = np.empty((n_sample, 3))

        for n in range(n_sample):
            RT[n], Choice[n, :] = simulate_HSDM_trial(threshold, drift_vec.astype(np.float64), ndt, 
                                                      s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        
        return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response1', 'response2', 'response3'])
    
    def response_time_pdf(self, t, threshold, drift_vec, sigma=1):
        kappa = threshold * np.linalg.norm(drift_vec)
        normalized_term = iv(1, kappa)/kappa
        girsanov_term = np.exp(-0.5 * (drift_vec[0]**2 + drift_vec[1]**2 + drift_vec[2]**2 + drift_vec[3]**2) * t)
        zero_drift_fpt = hsdm_long_t_fpt_z(t, threshold, sigma=sigma)
        return normalized_term * girsanov_term * zero_drift_fpt
    
    def joint_lpdf(self, rt, theta, threshold, drift_vec, ndt, s_v=0, s_t=0, sigma=1):
        tt = np.maximum(rt - ndt, 0)
        s = tt/threshold**2
        
        s0 = 0.002
        s1 = 0.02
        w = np.minimum(np.maximum((s - s0) / (s1 - s0), 0), 1)
        
        # first-passage time density of zero drift process
        fpt_lt = hsdm_long_t_fpt_z(tt, threshold, sigma=sigma)
        fpt_st = 1/threshold**2 * hsdm_short_t_fpt_z(tt/threshold**2, 0.1**8/threshold**2)   
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
            s_v2 = s_v**2
            x3 =  threshold*np.cos(theta[:0])
            x2 =  threshold*np.sin(theta[:, 0])*np.cos(theta[:, 1])
            x1 =  threshold*np.sin(theta[:, 0])*np.sin(theta[:, 1])*np.cos(theta[:, 2])
            x0 =  threshold*np.sin(theta[:, 0])*np.sin(theta[:, 1])*np.sin(theta[:, 2])
            fixed = 1/(np.sqrt(s_v2 * tt + 1))
            exponent0 = -0.5*drift_vec[0]**2/s_v2 + 0.5*(x0 * s_v2 + drift_vec[0])**2 / (s_v2 * (s_v2 * tt + 1))
            exponent1 = -0.5*drift_vec[1]**2/s_v2 + 0.5*(x1 * s_v2 + drift_vec[1])**2 / (s_v2 * (s_v2 * tt + 1))
            exponent2 = -0.5*drift_vec[2]**2/s_v2 + 0.5*(x2 * s_v2 + drift_vec[2])**2 / (s_v2 * (s_v2 * tt + 1))
            exponent3 = -0.5*drift_vec[3]**2/s_v2 + 0.5*(x3 * s_v2 + drift_vec[3])**2 / (s_v2 * (s_v2 * tt + 1))

            # the joint density of choice and RT for the full process
            log_density = 4*np.log(fixed) + exponent0 + exponent1 + exponent2 + exponent3 + np.log(fpt_z) - np.log(2*np.pi)
        log_density[rt - ndt <= 0] = np.log(0.1**14)
        log_density = np.maximum(log_density, np.log(0.1**14))
            
        return log_density
    

class CollapsingThresholdHSDM:
    def __init__(self):
        self.name = 'Hyper-Spherical Diffusion Model with collapsing boundaries'

    def simulate(self, threshold, decay, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):
        RT = np.empty((n_sample,))
        Choice = np.empty((n_sample, 3))

        for n in range(n_sample):
            RT[n], Choice[n, :] = simulate_HSDM_trial(threshold, drift_vec.astype(np.float64), ndt, 
                                                      decay=decay, s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        
        return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response1', 'response2', 'response3'])
    
    def response_time_pdf(self, t, threshold, decay, drift_vec, sigma=1):
        kappa = (threshold - decay * t) * np.linalg.norm(drift_vec)
        normalized_term = 2*iv(1, kappa)/kappa
        girsanov_term = np.exp(-0.5 * np.linalg.norm(drift_vec)**2 * t)

        gz, T = ie_fpt(threshold, decay, 4, 0.000001, dt=0.02, T_max=t.max())
        zero_drift_fpt = np.interp(t, T, gz)
        
        return normalized_term * girsanov_term * zero_drift_fpt
    
    def joint_lpdf(self, rt, theta, threshold, decay, drift_vec, ndt, s_v=0, s_t=0, sigma=1):
        tt = np.maximum(rt - ndt, 0)

        T_max = min(rt.max(), threshold/decay)
        g_z, T = ie_fpt(threshold, decay, 4, 0.000001, dt=0.02, T_max=T_max)
        
        fpt_z = np.interp(tt, T, g_z)
        fpt_z = np.maximum(fpt_z, 0.1**14)

        # Girsanov: no drift variability
        if s_v == 0:
            mu_dot_x0 = drift_vec[0]*np.cos(theta[:, 0])
            mu_dot_x1 = drift_vec[1]*np.sin(theta[:, 0])*np.cos(theta[:, 1]) 
            mu_dot_x2 = drift_vec[2]*np.sin(theta[:, 0])*np.sin(theta[:, 1])*np.cos(theta[:, 2])
            mu_dot_x3 = drift_vec[3]*np.sin(theta[:, 0])*np.sin(theta[:, 1])*np.sin(theta[:, 2])
            term1 = (threshold - decay*tt) * (mu_dot_x0 + mu_dot_x1 + mu_dot_x2 + mu_dot_x3)
            term2 = 0.5 * np.linalg.norm(drift_vec, 2)**2 * tt

            log_density = term1 - term2 + np.log(fpt_z) - np.log(2*np.pi)
        else:
            pass # to be implemented later

        log_density[rt - ndt <= 0] = np.log(0.1**14)
        log_density = np.maximum(log_density, np.log(0.1**14))
            
        return log_density