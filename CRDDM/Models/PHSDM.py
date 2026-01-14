import numpy as np
import pandas as pd

from CRDDM.utility.simulators import simulate_PSDM_trial
from CRDDM.utility.fpts import hsdm_short_t_fpt_z, hsdm_long_t_fpt_z, ie_fpt

class fixedThresholdPHSDM:
    '''
    Hyper-Spherical Diffusion Model with fixed boundaries
    '''

    def __init__(self):
        self.name = 'Hyper-Spherical Diffusion Model with fixed boundaries'

    def simulate(self, threshold, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):    
        pass  # to be implemented later
    def joint_lpdf(self, rt, theta, threshold, drift_vec, ndt, s_v=0, s_t=0, sigma=1):
        pass  # to be implemented later

class collapsingThresholdPHSDM:
    def __init__(self):
        self.name = 'Hyper-Spherical Diffusion Model with collapsing boundaries'
    
    def simulate(self, threshold, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):    
        pass  # to be implemented later
    
    def joint_lpdf(self, rt, theta, threshold, drift_vec, ndt, s_v=0, s_t=0, sigma=1):
        pass  # to be implemented later
