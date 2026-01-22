# Quick Start: Circular Diffusion Model

This tutorial demonstrates how to simulate data from a **Circular Diffusion Model (CDM)** and recover model parameters using likelihood-based estimation with CRDDM.

---

## 1. Import required packages

```python
import numpy as np

from scipy.optimize import differential_evolution
from CRDDM.Models.Circular import CircularDiffusionModel
```

---

## 2. Create a circular diffusion model

```python
model = CircularDiffusionModel(threshold_dynamic="fixed")
```

---

## 3. Simulate continuous-response decision data

```python
# Ground-truth parameters
threshold = 1.0
drift_vector = np.array([1.0, 0.0])  # drift in x and y directions
ndt = 0.25                           # non-decision time

# Simulate data
sim_data = model.simulate(
    drift_vector=drift_vector,
    ndt=ndt,
    threshold=threshold,
    n_sample=1000
)

sim_data.head()
```

The simulated dataset contains:
- `rt`: response times
- `response`: continuous angular responses on the circle in radian

---

## 4. Define the likelihood function

We estimate parameters by **maximizing the joint likelihood** of response times and continuous responses.

```python
def negative_log_likelihood(params, rt, theta, model):
    threshold = params[0]
    ndt = params[1]
    drift_vec = np.array([params[2], params[3]])

    logpdf = model.joint_lpdf(rt, theta, drift_vec, ndt, threshold)
    return -np.sum(logpdf)
```

---

## 5. Estimate parameters using differential evolution

```python
# Parameter bounds
bounds = [
    (0.05, 5.0),   # threshold
    (0.0, 2.0),    # non-decision time
    (-5.0, 5.0),   # drift_x
    (-5.0, 5.0),   # drift_y
]

param_names = ["threshold", "ndt", "drift_x", "drift_y"]

result = differential_evolution(
    negative_log_likelihood,
    bounds=bounds,
    args=(sim_data["rt"].values, sim_data["response"].values, model),
)
```

---

## 6. Inspect recovered parameters

```python
for name, value in zip(param_names, result.x):
    print(f"{name}: {value:.3f}")
```

The recovered parameters should be close to the values used for simulation, demonstrating correct likelihood evaluation and parameter recovery.

---

## Notes

- This example uses a **fixed decision threshold**.
- Other threshold dynamics and response spaces (e.g., spherical or hyperspherical models) follow the same workflow.
