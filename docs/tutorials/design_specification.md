# Design Specification: Condition-Specific Parameters (Speed–Accuracy Tradeoff)

This tutorial shows how to implement **design specifications** in CRDDM—i.e., how experimental
conditions can be mapped to **different model parameters** within a single likelihood function.

In the example below (based on the accompanying notebook), we fit a **Circular Diffusion Model (CDM)**
to a task with:

- a **speed vs. accuracy** instruction manipulation (`isSpeed`),
- a **difficulty** manipulation (`jitter` ∈ {15, 30, 45}), which modulates the drift magnitude.

We implement a model in which:

- the **decision threshold** differs between speed and accuracy blocks,
- the **drift magnitude** differs across jitter levels,
- the **drift angle** and **non-decision time (NDT)** are shared across conditions.

---

## 1. Import required packages

```python
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.optimize import differential_evolution

from CRDDM.utility.datasets import load_kvam2019
from CRDDM.Models.Circular import CircularDiffusionModel
```

---

## 2. Load the dataset

```python
data = load_kvam2019()
data.head()
```

This dataset includes:

- `rt`: response time,
- `deviation`: continuous response deviation (angular),
- `isSpeed`: 1 for speed instruction, 0 for accuracy instruction,
- `jitter`: difficulty manipulation (15, 30, 45),
- `Participant`: participant identifier.

---

## 3. Define the likelihood with design-specific parameters

### Model specification

- **Two thresholds:** `threshold_speed` and `threshold_accuracy`
- **One NDT:** `ndt`
- **One drift direction:** parameterized by a drift angle `drift_angle`
- **Three drift magnitudes:** one per jitter level (15, 30, 45)

```python
def negative_log_likelihood(params, rt, theta, isSpeed, jitter, model):
    threshold_speed = params[0]
    threshold_accuracy = params[1]
    ndt = params[2]
    drift_angle = params[3]

    # Drift direction is shared across conditions
    drift_direction = np.array([np.cos(drift_angle), np.sin(drift_angle)])

    # Drift magnitude depends on jitter
    drift_magnitudes = np.empty(rt.shape)
    drift_magnitudes[jitter == 15] = params[4]
    drift_magnitudes[jitter == 30] = params[5]
    drift_magnitudes[jitter == 45] = params[6]

    # Trial-wise drift vectors
    drift_vectors = np.outer(drift_magnitudes, drift_direction)

    # Condition-specific thresholds (speed vs accuracy)
    logpdf_speed = model.joint_lpdf(
        rt[isSpeed == 1],
        theta[isSpeed == 1],
        drift_vectors[isSpeed == 1, :],
        ndt,
        threshold_speed,
    )

    logpdf_accuracy = model.joint_lpdf(
        rt[isSpeed == 0],
        theta[isSpeed == 0],
        drift_vectors[isSpeed == 0, :],
        ndt,
        threshold_accuracy,
    )

    return -np.sum(logpdf_speed) - np.sum(logpdf_accuracy)
```

---

## 4. Set parameter bounds and fit the model per participant

```python
param_names = [
    "threshold_speed",
    "threshold_accuracy",
    "ndt",
    "drift_angle",
    "drift_magnitude15",
    "drift_magnitude30",
    "drift_magnitude45",
]

bounds = [
    (0.05, 5.0),        # threshold_speed
    (0.05, 5.0),        # threshold_accuracy
    (0.0, 1.0),         # ndt
    (-np.pi, np.pi),    # drift_angle
    (0.0, 8.0),         # drift_magnitude15
    (0.0, 8.0),         # drift_magnitude30
    (0.0, 8.0),         # drift_magnitude45
]

model = CircularDiffusionModel(threshold_dynamic="fixed")

estimation_rows = []

for sbj in tqdm(data.Participant.unique()):
    sbj_data = data[data["Participant"] == sbj].reset_index(drop=True)

    res = differential_evolution(
        negative_log_likelihood,
        bounds=bounds,
        args=(
            sbj_data["rt"].values,
            sbj_data["deviation"].values,
            sbj_data["isSpeed"].values,
            sbj_data["jitter"].values,
            model,
        ),
    )

    k = len(res.x)
    n = sbj_data.shape[0]
    nlpdf = res.fun

    row = {"Participant": sbj, "nlpdf": nlpdf, "AIC": 2 * nlpdf + 2 * k, "BIC": 2 * nlpdf + k * np.log(n)}
    row.update(dict(zip(param_names, res.x)))
    estimation_rows.append(row)

estimation_data = pd.DataFrame(estimation_rows)
estimation_data.head()
```

---

## 5. Model-based prediction (posterior predictive simulation)

We now simulate data from the fitted model and compare summary statistics between observed
and predicted data.

```python
n_sample_condition = 200
predictions = []

for sbj in tqdm(data.Participant.unique()):
    prms = estimation_data[estimation_data["Participant"] == sbj].iloc[0]

    drift_dir = np.array([np.cos(prms["drift_angle"]), np.sin(prms["drift_angle"])])

    for jitter in [15, 30, 45]:
        drift_mag = prms[f"drift_magnitude{jitter}"]
        drift_vec = drift_mag * drift_dir

        # Speed condition
        speed_df = model.simulate(
            drift_vec,
            prms["ndt"],
            threshold=prms["threshold_speed"],
            n_sample=n_sample_condition,
        )
        speed_df["isSpeed"] = 1
        speed_df["jitter"] = jitter
        speed_df["Participant"] = sbj

        # Accuracy condition
        acc_df = model.simulate(
            drift_vec,
            prms["ndt"],
            threshold=prms["threshold_accuracy"],
            n_sample=n_sample_condition,
        )
        acc_df["isSpeed"] = 0
        acc_df["jitter"] = jitter
        acc_df["Participant"] = sbj

        predictions.append(pd.concat([speed_df, acc_df], ignore_index=True))

model_prediction = pd.concat(predictions, ignore_index=True)
model_prediction.head()
```

---

## 6. Compare summary measures

A common approach is to compare mean RT and mean absolute deviation across conditions:

```python
data = data.copy()
data["absoluteDeviation"] = np.abs(data["deviation"])

model_prediction = model_prediction.copy()
model_prediction["absoluteDeviation"] = np.abs(model_prediction["response"])

observed_summary = data.groupby(["jitter", "isSpeed"])[["rt", "absoluteDeviation"]].mean()
predicted_summary = model_prediction.groupby(["jitter", "isSpeed"])[["rt", "absoluteDeviation"]].mean()

observed_summary, predicted_summary
```

---

## Notes

- This pattern—mapping design factors to different model parameters—scales to more complex designs
  (e.g., stimulus-specific drifts, participant-level covariates, hierarchical extensions).
- The key idea is to construct **trial-wise parameters** (like `drift_vectors`) and to route trials
  to condition-specific parameters (like thresholds) inside the likelihood.