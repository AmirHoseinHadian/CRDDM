import numpy as np
import pandas as pd
from scipy.special import iv

from CRDDM.utility.simulators import simulate_SDM_trial, simulate_custom_threshold_SDM_trial
from CRDDM.utility.simulators import simulate_PSDM_trial, simulate_custom_threshold_PSDM_trial
from CRDDM.utility.fpts import sdm_short_t_fpt_z, sdm_long_t_fpt_z, ie_fpt_linear, ie_fpt_exponential, ie_fpt_hyperbolic, ie_fpt_custom


class SphericalDiffusionModel:
    '''
    Spherical Diffusion Model
    '''
    def __init__(self, threshold_dynamic='fixed'):
        '''
        Parameters
        ----------
        threshold_dynamic : str, optional
            The type of threshold collapse ('fixed', 'linear', 'exponential', or 'hyperbolic'), default is 'fixed'
        '''
        self.name = 'Spherical Diffusion Model'
        self.threshold_dynamic = threshold_dynamic
    
    def simulate(self, drift_vec, ndt, threshold=1, decay=0, threshold_function=None, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):
        '''
        Simulate data from the Spherical Diffusion Model

        Parameters
        ----------
        drift_vec : array-like
            Drift vector; a three-dimensional array
        ndt : float
            Non-decision time; a positive floating number
        threshold : float
            Decision threshold; a positive floating number (default is 1)
        decay : float, optional
            Decay rate of the collapsing boundary (default is 0)
        threshold_function : callable, if threshold_dynamic is 'custom'
            A function that takes time t and returns the threshold at time t
        s_v : float, optional
            The standard deviation of drift variability (default is 0)
        s_t : float, optional
            The standard deviation of non-decision time variability (default is 0)
        sigma : float, optional
            The diffusion coefficient (default is 1)
        dt : float, optional
            Time step for the simulation (default is 0.001)
        n_sample : int, optional
            Number of samples to simulate (default is 1)

        Returns
        -------
        pd.DataFrame
            A DataFrame containing simulated response times and choice angles
        '''
        RT = np.empty((n_sample,))
        Choice = np.empty((n_sample, 2))

        if self.threshold_dynamic != 'custom':
            for n in range(n_sample):
                RT[n], Choice[n, :] = simulate_SDM_trial(threshold, drift_vec.astype(np.float64), ndt, 
                                                        threshold_dynamic=self.threshold_dynamic, 
                                                        decay=decay, s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        else:
            for n in range(n_sample):
                RT[n], Choice[n, :] = simulate_custom_threshold_SDM_trial(threshold_function,
                                                                          drift_vec.astype(np.float64), ndt, 
                                                                          s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        
        return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response1', 'response2'])

    def joint_lpdf(self, rt, theta, drift_vec, ndt, threshold, decay=0, threshold_function=None, dt_threshold_function=None, s_v=0, s_t=0, sigma=1):
        '''
        Compute the joint log-probability density function of response time and choice angles
        
        Parameters
        ----------
        rt : array-like, shape (n_samples,)
            The response times
        theta : array-like, shape (n_samples, 2)
            The choice angles in spherical coordinates (theta1, theta2)
        drift_vec : array-like, shape (3,)
            The drift rates in each dimension
        ndt : float
            The non-decision time
        threshold : float
            The decision threshold (default is 1)
        decay : float, optional
            The threshold decay rate (default is 0)
        threshold_function : callable, if threshold_dynamic is 'custom'
            A function that takes time t and returns the threshold at time t
        dt_threshold_function : callable, if threshold_dynamic is 'custom'
            A function that takes time t and returns the derivative of the threshold at time t
        s_v : float, optional
            The standard deviation of drift variability (default is 0)
        s_t : float, optional
            The standard deviation of non-decision time variability (default is 0)
        sigma : float, optional
            The diffusion coefficient (default is 1)

        Returns
        -------
        log_density : array-like, shape (n_samples,)
            The joint log-probability density of response time and choice angles
        '''

        tt = np.maximum(rt - ndt, 0)

        # first-passage time density of zero drift process
        if self.threshold_dynamic == 'fixed':
            a = threshold
            s = tt/threshold**2
            s0 = 0.002
            s1 = 0.02
            w = np.minimum(np.maximum((s - s0) / (s1 - s0), 0), 1)
            fpt_lt = sdm_long_t_fpt_z(tt, threshold, sigma=sigma)
            fpt_st = 1/threshold**2 * sdm_short_t_fpt_z(tt/threshold**2, 0.1**8/threshold**2)   
            fpt_z =  (1 - w) * fpt_st + w * fpt_lt
        elif self.threshold_dynamic == 'linear':
            a = threshold - decay*tt
            T_max = min(rt.max(), threshold/decay)
            g_z, T = ie_fpt_linear(threshold, decay, 3, 0.000001, dt=0.02, T_max=T_max)
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'exponential':
            a = threshold * np.exp(-decay*tt)
            g_z, T = ie_fpt_exponential(threshold, decay, 3, 0.000001, dt=0.02, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'hyperbolic':
            a = threshold / (1 + decay*tt)
            g_z, T = ie_fpt_hyperbolic(threshold, decay, 3, 0.000001, dt=0.02, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'custom':
            threshold_function2 = lambda t: threshold_function(t)**2
            dt_threshold_function2 = lambda t: 2 * dt_threshold_function(t) * threshold_function(t)
            a = threshold_function(tt)
            g_z, T = ie_fpt_custom(threshold_function2, dt_threshold_function2, 3, 0.000001, dt=0.02, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)

        fpt_z = np.maximum(fpt_z, 0.1**14)

        # Girsanov:
        if s_v == 0:
            # No drift variability
            mu_dot_x0 = drift_vec[0]*np.cos(theta[:, 0])
            mu_dot_x1 = drift_vec[1]*np.sin(theta[:, 0])*np.cos(theta[:, 1]) 
            mu_dot_x2 = drift_vec[2]*np.sin(theta[:, 0])*np.sin(theta[:, 1])
            term1 = a * (mu_dot_x0 + mu_dot_x1 + mu_dot_x2)
            term2 = 0.5 * np.linalg.norm(drift_vec, 2)**2 * tt
            
            log_density = term1 - term2 + np.log(fpt_z) - np.log(4*np.pi)
        else:
            # With drift variability
            s_v2 = s_v**2
            x2 =  a * np.cos(theta[:, 0])
            x1 =  a * np.sin(theta[:, 0])*np.cos(theta[:, 1]) 
            x0 =  a * np.sin(theta[:, 0])*np.sin(theta[:, 1])
            fixed = 1/(np.sqrt(s_v2 * tt + 1))
            exponent0 = -0.5*drift_vec[0]**2/s_v2 + 0.5*(x0 * s_v2 + drift_vec[0])**2 / (s_v2 * (s_v2 * tt + 1))
            exponent1 = -0.5*drift_vec[1]**2/s_v2 + 0.5*(x1 * s_v2 + drift_vec[1])**2 / (s_v2 * (s_v2 * tt + 1))
            exponent2 = -0.5*drift_vec[2]**2/s_v2 + 0.5*(x2 * s_v2 + drift_vec[2])**2 / (s_v2 * (s_v2 * tt + 1))

            log_density = 3*np.log(fixed) + exponent0 + exponent1 + exponent2 + np.log(fpt_z) - np.log(4*np.pi)

        log_density[rt - ndt <= 0] = np.log(0.1**14)
        log_density = np.maximum(log_density, np.log(0.1**14))
            
        return log_density
    


class ProjectedSphericalDiffusionModel:
    '''
    Projected Spherical Diffusion Model
    '''
    def __init__(self, threshold_dynamic='fixed'):
        '''
        Parameters
        ----------
        threshold_dynamic : str, optional
            The type of threshold collapse ('fixed', 'linear', 'exponential', or 'hyperbolic'), default is 'fixed'
        '''
        self.name = 'Projected Spherical Diffusion Model'
        self.threshold_dynamic = threshold_dynamic

    def simulate(self, drift_vec, ndt, threshold=1, decay=0, threshold_function=None, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):
        '''
        Simulate response times and choices from the Projected Spherical Diffusion Model

        Parameters
        ----------
        drift_vec : array-like
            The drift vector should be of shape (2,)
        ndt : float
            The non-decision time
        threshold : float
            The decision threshold (default is 1)
        decay : float, optional
            The threshold decay rate (default is 0)
        threshold_function : callable, if threshold_dynamic is 'custom'
            A function that takes time t and returns the threshold at time t
        s_v : float, optional
            The standard deviation of drift variability (default is 0)
        s_t : float, optional
            The standard deviation of non-decision time variability (default is 0)
        sigma : float, optional
            The diffusion coefficient (default is 1)
        dt : float, optional
            Time step for the simulation (default is 0.001)
        n_sample : int, optional
            Number of samples to simulate (default is 1)

        Returns
        -------
        pd.DataFrame
            A DataFrame containing simulated response times and choice angles
        '''    
        RT = np.empty((n_sample,))
        Choice = np.empty((n_sample,))

        if self.threshold_dynamic != 'custom':
            for n in range(n_sample):
                RT[n], Choice[n] = simulate_PSDM_trial(threshold, drift_vec.astype(np.float64), ndt,
                                                       threshold_dynamic=self.threshold_dynamic,
                                                       decay=decay, s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        else:
            for n in range(n_sample):
                RT[n], Choice[n] = simulate_custom_threshold_PSDM_trial(threshold_function,
                                                                        drift_vec.astype(np.float64), ndt, 
                                                                        s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)

        return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response'])
    
    def joint_lpdf(self, rt, theta, drift_vec, ndt, threshold, decay=0, threshold_function=None, dt_threshold_function=None, s_v=0, s_t=0, sigma=1):
        '''
        Compute the joint log-probability density function of response time and choice angle

        Parameters
        ----------
        rt : array-like, shape (n_samples,)
            The response times
        theta : array-like, shape (n_samples,)
            The choice angles in radians
        drift_vec : array-like, shape (2,)
            The drift vector [drift_x, drift_y]
        ndt : float
            The non-decision time
        threshold : float
            The decision threshold (default is 1)
        decay : float, optional
            The threshold decay rate (default is 0)
        threshold_function : callable, if threshold_dynamic is 'custom'
            A function that takes time t and returns the threshold at time t
        dt_threshold_function : callable, if threshold_dynamic is 'custom'
            A function that takes time t and returns the derivative of the threshold at time t
        s_v : float, optional
            The standard deviation of drift variability (default is 0)
        s_t : float, optional
            The standard deviation of non-decision time variability (default is 0)
        sigma : float, optional
            The diffusion coefficient (default is 1)

        Returns
        -------
        log_density : array-like, shape (n_samples,)
            The joint log-probability density of response time and choice angle
        '''
        
        tt = np.maximum(rt - ndt, 0)

        # first-passage time density of zero drift process
        if self.threshold_dynamic == 'fixed':
            a = threshold
            s = tt/threshold**2
            s0 = 0.002
            s1 = 0.02
            w = np.minimum(np.maximum((s - s0) / (s1 - s0), 0), 1)
            fpt_lt = sdm_long_t_fpt_z(tt, threshold, sigma=sigma)
            fpt_st = 1/threshold**2 * sdm_short_t_fpt_z(tt/threshold**2, 0.1**8/threshold**2)   
            fpt_z =  (1 - w) * fpt_st + w * fpt_lt
        elif self.threshold_dynamic == 'linear':
            a = threshold - decay*tt
            T_max = min(rt.max(), threshold/decay)
            g_z, T = ie_fpt_linear(threshold, decay, 3, 0.000001, dt=0.02, T_max=T_max)
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'exponential':
            a = threshold * np.exp(-decay*tt)
            g_z, T = ie_fpt_exponential(threshold, decay, 3, 0.000001, dt=0.02, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'hyperbolic':
            a = threshold / (1 + decay*tt)
            g_z, T = ie_fpt_hyperbolic(threshold, decay, 3, 0.000001, dt=0.02, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'custom':
            threshold_function2 = lambda t: threshold_function(t)**2
            dt_threshold_function2 = lambda t: 2 * dt_threshold_function(t) * threshold_function(t)
            a = threshold_function(tt)
            g_z, T = ie_fpt_custom(threshold_function2, dt_threshold_function2, 3, 0.000001, dt=0.02, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)

        fpt_z = np.maximum(fpt_z, 0.1**14)

        norm_mu = np.linalg.norm(drift_vec, 2)
        theta_mu = np.arctan2(drift_vec[1], drift_vec[0])

        # Girsanov:
        if s_v == 0:
            # No drift variability
            term1 = np.exp(a * norm_mu * np.cos(theta_mu) * np.cos(theta))
            term2 = iv(0, a * norm_mu * np.sin(theta_mu) * np.sin(theta))
            term3 = -0.5 * norm_mu**2 * tt

            log_density = np.log(2*np.pi) + np.log(term1) + np.log(term2) + term3 + np.log(fpt_z)
        else:
            # With drift variability
            s_v2 = s_v**2
            c1 = a * np.sin(theta) * s_v2
            c2 = 2*s_v2 * (s_v2 * tt + 1)
            term1 = 2*np.pi * iv(0, 2*c1 * drift_vec[1]/c2)
            term2 = (1/(np.sqrt(s_v2 * tt + 1)))**3
            p1 = (c1**2 + drift_vec[1]**2)/c2
            p2 = (a * np.cos(theta) * s_v2 + drift_vec[0])**2 / c2
            p3 = (norm_mu**2)/(2*s_v2)
            term3 = np.exp(p1 + p2 - p3)

        log_density[rt - ndt <= 0] = np.log(0.1**14)
        log_density = np.maximum(log_density, np.log(0.1**14))
            
        return log_density