import numpy as np
import pandas as pd

from scipy.special import iv

from CRDDM.utility.simulators import simulate_PSDM_trial
from CRDDM.utility.fpts import sdm_short_t_fpt_z, sdm_long_t_fpt_z, ie_fpt

class fixedThresholdPSDM:
    '''
    Spherical Diffusion Model with fixed boundaries
    '''

    def __init__(self):
        self.name = 'Spherical Diffusion Model with fixed boundaries'

    def simulate(self, threshold, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):    
        RT = np.empty((n_sample,))
        Choice = np.empty((n_sample,))

        for n in range(n_sample):
            RT[n], Choice[n] = simulate_PSDM_trial(threshold, drift_vec.astype(np.float64), ndt,
                                                   s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        
        return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response'])

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
    

class collapsingThresholdPSDM:
    def __init__(self):
        self.name = 'Spherical Diffusion Model with collapsing boundaries'

    def simulate(self, threshold, decay, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):    
        RT = np.empty((n_sample,))
        Choice = np.empty((n_sample,))

        for n in range(n_sample):
            RT[n], Choice[n] = simulate_PSDM_trial(threshold, drift_vec.astype(np.float64), ndt,
                                                   decay=decay, s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        
        return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response'])
    
    def joint_lpdf(self, rt, theta, threshold, decay, drift_vec, ndt, s_v=0, s_t=0, sigma=1):
        tt = np.maximum(rt - ndt, 0)

        T_max = min(rt.max(), threshold/decay)
        g_z, T = ie_fpt(threshold, decay, 3, 0.000001, dt=0.02, T_max=T_max)
        
        fpt_z = np.interp(tt, T, g_z)
        fpt_z = np.maximum(fpt_z, 0.1**14)

        norm_mu = np.linalg.norm(drift_vec, 2)
        theta_mu = np.arctan2(drift_vec[1], drift_vec[0])

        # Girsanov: no drift variability
        if s_v == 0:
            term1 = np.exp((threshold - decay*tt)*norm_mu*np.cos(theta_mu)*np.cos(theta))
            term2 = iv(0, (threshold - decay*tt)*norm_mu*np.sin(theta_mu)*np.sin(theta))
            term3 = -0.5 * norm_mu**2 * tt

            log_density = np.log(2*np.pi) + np.log(term1) + np.log(term2) + term3 + np.log(fpt_z)

        else:
            pass # to be implemented later

        log_density[rt - ndt <= 0] = np.log(0.1**14)
        log_density = np.maximum(log_density, np.log(0.1**14))
            
        return log_density