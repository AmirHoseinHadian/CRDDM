import numpy as np
import pandas as pd
from scipy.special import iv

from CRDDM.utility.simulators import simulate_HSDM_trial, simulate_custom_threshold_HSDM_trial
from CRDDM.utility.simulators import simulate_PHSDM_trial, simulate_custom_threshold_PHSDM_trial
from CRDDM.utility.fpts import hsdm_short_t_fpt_z, hsdm_long_t_fpt_z, ie_fpt_linear, ie_fpt_exponential, ie_fpt_hyperbolic, ie_fpt_custom


class HyperSphericalDiffusionModel:
    '''
    Hyper-Spherical Diffusion Model
    '''
    def __init__(self, threshold_dynamic='fixed'):
        '''
        Parameters
        ----------
        threshold_dynamic : str, optional
            The type of threshold collapse ('fixed', 'linear', 'exponential', 'hyperbolic', or 'custom'), default is 'fixed'
        '''
        self.name = 'Hyper-Spherical Diffusion Model'
        
        if threshold_dynamic in ['fixed', 'linear', 'exponential', 'hyperbolic', 'custom']:
            self.threshold_dynamic = threshold_dynamic
        else:
            raise ValueError("\'threshold_dynamic\' must be one of \'fixed\', \'linear\', \'exponential\', \'hyperbolic\', or \'custom\'. However, got \'{}\'".format(threshold_dynamic))

    def simulate(self, drift_vec, ndt, threshold=1, decay=0, threshold_function=None, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):
        '''
        Simulate data from the Hyper-Spherical Diffusion Model

        Parameters
        ----------
        drift_vec : array-like, shape (4,)
            The drift rates in each dimension
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
        Choice = np.empty((n_sample, 3))

        if drift_vec.ndim == 1:
            drift_vec = drift_vec * np.ones((n_sample, 4))
        elif drift_vec.shape[0] != n_sample:
            raise ValueError("Number of rows in drift_vec must be equal to n_sample")
        
        if type(ndt) is float or type(ndt) is int:
            ndt = np.full((n_sample,), ndt)
        elif len(ndt) != n_sample:
            raise ValueError("Length of ndt must be equal to n_sample")
        
        if type(threshold) is float or type(threshold) is int:
            threshold = np.full((n_sample,), threshold)
        elif len(threshold) != n_sample:
            raise ValueError("Length of threshold must be equal to n_sample")
        
        if type(decay) is float or type(decay) is int:
            decay = np.full((n_sample,), decay)
        elif len(decay) != n_sample:
            raise ValueError("Length of decay must be equal to n_sample")
        
        if threshold_function is None and self.threshold_dynamic == 'custom':
            raise ValueError("threshold_function must be provided when threshold_dynamic is 'custom'")
        
        if threshold_function is not None and self.threshold_dynamic != 'custom':
            raise ValueError("threshold_function should be None when threshold_dynamic is not 'custom'")
        
        if s_v < 0:
            raise ValueError("s_v must be non-negative")
        if s_t < 0:
            raise ValueError("s_t must be non-negative")

        if self.threshold_dynamic != 'custom':
            for n in range(n_sample):
                RT[n], Choice[n, :] = simulate_HSDM_trial(threshold[n], drift_vec[n, :].astype(np.float64), ndt[n],
                                                          threshold_dynamic=self.threshold_dynamic, 
                                                          decay=decay[n], s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        else:
            for n in range(n_sample):
                RT[n], Choice[n, :] = simulate_custom_threshold_HSDM_trial(threshold_function,
                                                                           drift_vec[n, :].astype(np.float64), ndt[n], 
                                                                           s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response1', 'response2', 'response3'])

    def joint_lpdf(self, rt, theta, drift_vec, ndt, threshold, decay=0, threshold_function=None, dt_threshold_function=None, s_v=0, s_t=0, sigma=1):
        '''
        Compute the joint log-probability density function of response time and choice angles

        Parameters
        ----------
        rt : array-like, shape (n_samples,)
            The response times
        theta : array-like, shape (n_samples, 3)
            The choice angles in spherical coordinates (theta1, theta2, theta3)
        drift_vec : array-like, shape (4,) or (n_samples, 4)
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

        if drift_vec.ndim == 1:
            drift_vec = np.array(drift_vec).reshape(1, -1)

        if drift_vec.shape[1] != 4 or drift_vec.ndim != 2:
            raise ValueError("drift_vec must have shape (4,) or (n_samples, 4)")

        tt = np.maximum(rt - ndt, 0)

        # first-passage time density of zero drift process
        if self.threshold_dynamic == 'fixed':
            s = tt/threshold**2
            s0 = 0.002
            s1 = 0.02
            w = np.minimum(np.maximum((s - s0) / (s1 - s0), 0), 1)
            fpt_lt = hsdm_long_t_fpt_z(tt, threshold, sigma=sigma)
            fpt_st = 1/threshold**2 * hsdm_short_t_fpt_z(tt/threshold**2, 0.1**8/threshold**2)   
            fpt_z =  (1 - w) * fpt_st + w * fpt_lt
        elif self.threshold_dynamic == 'linear':
            a = threshold - decay*tt
            T_max = min(rt.max(), threshold/decay)
            g_z, T = ie_fpt_linear(threshold, decay, 4, 0.000001, dt=0.02, T_max=T_max)
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'exponential':
            a = threshold * np.exp(-decay*tt)
            g_z, T = ie_fpt_exponential(threshold, decay, 4, 0.000001, dt=0.02, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'hyperbolic':
            a = threshold / (1 + decay*tt)
            g_z, T = ie_fpt_hyperbolic(threshold, decay, 4, 0.000001, dt=0.02, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'custom':
            threshold_function2 = lambda t: threshold_function(t)**2
            dt_threshold_function2 = lambda t: 2 * dt_threshold_function(t) * threshold_function(t)
            a = threshold_function(tt)
            g_z, T = ie_fpt_custom(threshold_function2, dt_threshold_function2, 4, 0.000001, dt=0.02, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)

        fpt_z = np.maximum(fpt_z, 0.1**14)

        # Girsanov:
        if s_v == 0:
            # No drift variability
            mu_dot_x0 = drift_vec[:, 0]*np.cos(theta[:, 0])
            mu_dot_x1 = drift_vec[:, 1]*np.sin(theta[:, 0])*np.cos(theta[:, 1]) 
            mu_dot_x2 = drift_vec[:, 2]*np.sin(theta[:, 0])*np.sin(theta[:, 1])*np.cos(theta[:, 2])
            mu_dot_x3 = drift_vec[:, 3]*np.sin(theta[:, 0])*np.sin(theta[:, 1])*np.sin(theta[:, 2])
            term1 = a * (mu_dot_x0 + mu_dot_x1 + mu_dot_x2 + mu_dot_x3)
            term2 = 0.5 * (drift_vec[:, 0]**2 + drift_vec[:, 1]**2 + drift_vec[:, 2]**2 + drift_vec[:, 3]**2) * tt

            log_density = term1 - term2 + np.log(fpt_z) - np.log(2*np.pi)
        else:
            # With drift variability
            s_v2 = s_v**2
            x3 =  a * np.cos(theta[:, 0])
            x2 =  a * np.sin(theta[:, 0])*np.cos(theta[:, 1])
            x1 =  a * np.sin(theta[:, 0])*np.sin(theta[:, 1])*np.cos(theta[:, 2])
            x0 =  a * np.sin(theta[:, 0])*np.sin(theta[:, 1])*np.sin(theta[:, 2])
            fixed = 1/(np.sqrt(s_v2 * tt + 1))
            exponent0 = -0.5*drift_vec[:, 0]**2/s_v2 + 0.5*(x0 * s_v2 + drift_vec[:, 0])**2 / (s_v2 * (s_v2 * tt + 1))
            exponent1 = -0.5*drift_vec[:, 1]**2/s_v2 + 0.5*(x1 * s_v2 + drift_vec[:, 1])**2 / (s_v2 * (s_v2 * tt + 1))
            exponent2 = -0.5*drift_vec[:, 2]**2/s_v2 + 0.5*(x2 * s_v2 + drift_vec[:, 2])**2 / (s_v2 * (s_v2 * tt + 1))
            exponent3 = -0.5*drift_vec[:, 3]**2/s_v2 + 0.5*(x3 * s_v2 + drift_vec[:, 3])**2 / (s_v2 * (s_v2 * tt + 1))

            # the joint density of choice and RT for the full process
            log_density = 4*np.log(fixed) + exponent0 + exponent1 + exponent2 + exponent3 + np.log(fpt_z) - np.log(2*np.pi)

        log_density[rt - ndt <= 0] = np.log(0.1**14)
        log_density = np.maximum(log_density, np.log(0.1**14))
            
        return log_density
    

class ProjectedHyperSphericalDiffusionModel:
    '''
    Projected Hyper-Spherical Diffusion Model
    '''
    def __init__(self, threshold_dynamic='fixed'):
        self.name = 'Projected Hyper-Spherical Diffusion Model'
        
        if threshold_dynamic in ['fixed', 'linear', 'exponential', 'hyperbolic', 'custom']:
            self.threshold_dynamic = threshold_dynamic
        else:
            raise ValueError("\'threshold_dynamic\' must be one of \'fixed\', \'linear\', \'exponential\', \'hyperbolic\', or \'custom\'. However, got \'{}\'".format(threshold_dynamic))

    def simulate(self, drift_vec, ndt, threshold=1, decay=0, threshold_function=None, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):
        pass # To be implemented

    def joint_lpdf(self, rt, theta, threshold, decay, drift_vec, ndt, s_v=0, s_t=0, sigma=1):
        pass # To be implemented