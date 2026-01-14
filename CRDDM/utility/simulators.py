import numpy as np
from numba import jit

@jit(nopython=True)
def simulate_CDM_trial(threshold, drift_vec, ndt, decay=0, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    input:
        threshold: a positive floating number
        drift_vec: drift vector; a two-dimensional array
        ndt: a positive floating number
        decay: decay rate of the collapsing boundary
        s_v: standard deviation of drift rate variability
        s_t: range of non-decision time variability
        sigma: standard deviation of the diffusion process
        dt: time step for the simulation
    returns:
        rt: response time in seconds
        theta: response angle between [-pi, pi]
    '''
    x = np.zeros((2,))
    
    rt = 0

    if s_t>0:
        ndt_t = ndt + (2*s_t*np.random.rand() - s_t)
    else:
        ndt_t = ndt

    if s_v>0:
        mu_t = drift_vec + s_v*np.random.randn(2)
    else:
        mu_t = drift_vec

    while np.linalg.norm(x) < threshold - decay*rt:
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(2)
        rt += dt
    
    theta = np.arctan2(x[1], x[0]) 
    return ndt_t+rt, theta


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

@jit(nopython=True)
def simulate_HSDM_trial(threshold, drift_vec, ndt, decay=0, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    input:
        threshold: a positive floating number
        drift_vec: drift vector; a four-dimensional array
        ndt: a positive floating number
        decay: decay rate of the collapsing boundary
        s_v: standard deviation of drift rate variability
        s_t: range of non-decision time variability
        sigma: standard deviation of the diffusion process
        dt: time step for the simulation
    returns:
        rt: response time in seconds
        theta: a tuple of response angles (theta1, theta2, theta3); theta1 and theta2 between [0, pi], and theta3 between [-pi, pi]
    '''
    x = np.zeros((4,))
    
    rt = 0

    if s_t>0:
        ndt_t = ndt + (2*s_t*np.random.rand() - s_t)
    else:
        ndt_t = ndt

    if s_v>0:
        mu_t = drift_vec + s_v*np.random.randn(4)
    else:
        mu_t = drift_vec

    while np.linalg.norm(x) < threshold - decay*rt:
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(4)
        rt += dt
    
    theta1 = np.arctan2(np.sqrt(x[3]**2 + x[2]**2 + x[1]**2), x[0])
    theta2 = np.arctan2(np.sqrt(x[3]**2 + x[2]**2), x[1])
    theta3 = np.arctan2(x[3], x[2])

    return ndt_t+rt, (theta1, theta2, theta3)


@jit(nopython=True)
def simulate_PSDM_trial(threshold, drift_vec, ndt, decay=0, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    input:
        threshold: a positive floating number
        drift_vec: drift vector; a two-dimensional array
        ndt: a positive floating number
        decay: decay rate of the collapsing boundary
        s_v: standard deviation of drift rate variability
        s_t: range of non-decision time variability
        sigma: standard deviation of the diffusion process
        dt: time step for the simulation
    returns:
        rt: response time in seconds
        theta: response angle between [0, pi]
    '''
    x = np.zeros((3,))
    muz = drift_vec[0]
    eta = drift_vec[1]
    
    norm_mu = np.sqrt(eta**2 + muz**2)
    theta_mu = np.arctan2(eta, muz)
    
    rt = 0
    rphi = np.pi/4 # it is not important (just a dummpy value)
    mux = norm_mu * np.sin(theta_mu) * np.cos(rphi)
    muy = norm_mu * np.sin(theta_mu) * np.sin(rphi)

    mu = np.array([mux, muy, muz])
    mmut = np.zeros((3,))

    if s_v>0:
        mu_t = mu + s_v*np.random.randn(3)
    else:
        mu_t = mu

    if s_t>0:
        ndt_t = ndt + (2*s_t*np.random.rand() - s_t)
    else:
        ndt_t = ndt

    while np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) < threshold - decay*rt:
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
        
        rt += dt
    theta = np.arctan2(np.sqrt(x[0]**2 + x[1]**2), x[2])    
    
    return ndt_t+rt, theta