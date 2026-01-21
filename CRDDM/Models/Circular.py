import numpy as np
import pandas as pd
from scipy.special import iv

from CRDDM.utility.simulators import simulate_CDM_trial
from CRDDM.utility.fpts import cdm_short_t_fpt_z, cdm_long_t_fpt_z, ie_fpt_linear, ie_fpt_exponential, ie_fpt_hyperbolic
    

class CircularDiffusionModel:
    '''
    Circular Diffusion Model
    '''

    def __init__(self, threshold_dynamic='fixed'):
        '''
        Parameters
        ----------
        threshold_dynamic : str, optional
            The type of threshold collapse ('fixed', 'linear', 'exponential', or 'hyperbolic'), default is 'fixed'
        '''
        self.name = 'Circular Diffusion Model'
        self.threshold_dynamic = threshold_dynamic


    def simulate(self, drift_vec, ndt, threshold, decay=0, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):
        '''
        Simulate data from the Circular Diffusion Model with collapsing boundaries

        Parameters
        ----------
        drift_vec : array-like, shape (2,)
            The drift vector [drift_x, drift_y]
        ndt : float
            The non-decision time
        threshold : float
            The initial decision threshold
        decay : float
            The decay rate of the threshold (default is 0)
        s_v : float, optional
            The standard deviation of drift variability (default is 0)
        s_t : float, optional
            The standard deviation of non-decision time variability (default is 0)
        sigma : float, optional
            The diffusion coefficient (default is 1)
        dt : float, optional
            The time step for simulation (default is 0.001)
        n_sample : int, optional
            The number of samples to simulate (default is 1)

        Returns
        -------
        pd.DataFrame
            A DataFrame containing simulated response times and choice angles
        '''
        RT = np.empty((n_sample,))
        Choice = np.empty((n_sample,))

        for n in range(n_sample):
            RT[n], Choice[n] = simulate_CDM_trial(threshold, drift_vec.astype(np.float64), ndt, 
                                                  threshold_dynamic=self.threshold_dynamic,
                                                  decay=decay, s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        
        return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response'])


    def joint_lpdf(self, rt, theta, drift_vec, ndt, threshold, decay=0, s_v=0, s_t=0, sigma=1):
        '''
        Compute the joint log-probability density function of response time and choice angle

        Parameters
        ----------
        rt : array-like
            Response times
        theta : array-like
            Choice angles in radians
        drift_vec : array-like, shape (2,)
            The drift vector [drift_x, drift_y]
        ndt : float
            The non-decision time
        threshold : float
            The initial decision threshold
        decay : float
            The decay rate of the threshold (default is 0)
        s_v : float, optional
            The standard deviation of drift variability (default is 0)
        s_t : float, optional
            The standard deviation of non-decision time variability (default is 0)
        sigma : float, optional
            The diffusion coefficient (default is 1)

        Returns
        -------
        array-like
            The joint log-probability density evaluated at (rt, theta) with same shape as rt and theta
        '''
        tt = np.maximum(rt - ndt, 0)

        # first-passage time density of zero drift process
        if self.threshold_dynamic == 'fixed':
            a = threshold
            s = tt/threshold**2
            s0 = 0.002
            s1 = 0.02
            w = np.minimum(np.maximum((s - s0) / (s1 - s0), 0), 1)
            
            fpt_lt = cdm_long_t_fpt_z(tt, threshold, sigma=sigma)
            fpt_st = 1/threshold**2 * cdm_short_t_fpt_z(tt/threshold**2, 0.1**8/threshold**2)   
            fpt_z =  (1 - w) * fpt_st + w * fpt_lt
        elif self.threshold_dynamic == 'linear':
            a = threshold - decay*tt
            T_max = min(rt.max(), threshold/decay)
            g_z, T = ie_fpt_linear(threshold, decay, 2, 0.000001, dt=0.02, T_max=T_max)
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'exponential':
            a = threshold * np.exp(-decay*tt)
            g_z, T = ie_fpt_exponential(threshold, decay, 2, 0.000001, dt=0.02, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'hyperbolic':
            a = threshold / (1 + decay*tt)
            g_z, T = ie_fpt_hyperbolic(threshold, decay, 2, 0.000001, dt=0.02, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)

        fpt_z = np.maximum(fpt_z, 0.1**14)

        # Girsanov:
        if s_v == 0:
            # No drift variability
            mu_dot_x0 = drift_vec[0] * np.cos(theta)
            mu_dot_x1 = drift_vec[1] * np.sin(theta)

            term1 = a * (mu_dot_x0 + mu_dot_x1)
            term2 = 0.5 * (drift_vec[0]**2 + drift_vec[1]**2) * tt

            log_density = term1 - term2 + np.log(fpt_z) - np.log(2*np.pi)
        else:
            # With drift variability            
            s_v2 = s_v**2
            x0 =  a * np.cos(theta)
            x1 =  a * np.sin(theta)
            fixed = 1/(np.sqrt(s_v2 * tt + 1))
            exponent0 = -0.5*drift_vec[0]**2/s_v2 + 0.5*(x0 * s_v2 + drift_vec[0])**2 / (s_v2 * (s_v2 * tt + 1))
            exponent1 = -0.5*drift_vec[1]**2/s_v2 + 0.5*(x1 * s_v2 + drift_vec[1])**2 / (s_v2 * (s_v2 * tt + 1))

            log_density = 2*np.log(fixed) + exponent0 + exponent1 + np.log(fpt_z) - np.log(2*np.pi)

        log_density[rt - ndt <= 0] = np.log(0.1**14)
        log_density = np.maximum(log_density, np.log(0.1**14))
            
        return log_density

    