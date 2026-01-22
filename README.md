# CRDDM:

DRDDM is a Python package for modeling continuous-report decision tasks using **Continuous Response Diffusion Decision Models (CRDDM)**. 

The package provides fast and numerically stable likelihood evaluation for continuous-response diffusion decision models using the integral equation method proposed by [Hadian Rasanan et al.,(2025)](https://doi.org/10.3758/s13428-025-02810-3). CRDDM supports a wide range of continuous response scale, that can be employed in experimental studies including:
- Bounded one-dimensional scales (e.g., arcs or sliders),
- Circular scales (e.g., color wheels),
- Two-dimensional scales (e.g., 2D planes).

CRDDM is designed for researchers in cognitive science, mathematical psychology, and neuroscience who work with diffusion models of continuous responses.

---

## Install
### Install via `pip`
The package can be installed via pip:
```bash
pip istall crddm
```
### Install from source
Alternatively, clone or download the source code and install locally:
```bash
python -m setup.py
```

---

## Dependencies
CRDDM requires the following Python packages:

- `numpy`
- `scipy`
- `pandas`
- `numba`

All dependencies are installed automatically when using `pip`.

---

### Conda environment (suggested)
If you have Andaconda or miniconda installed and you would like to create a separate environment:

```bash
conda create --n crddm python=3 numpy scipy pandas numba
conda activate crddm
pip install crddm
```

## Documentation

The latest documentation can be found here: **amirhoseinhadian.github.io/crddm/**

---

## Selected References

For background on diffusion models for continuous response tasks and the estimation methods implemented in CRDDM, see:

- Hadian Rasanan, A. H., Evans, N. J., Amani Rad, J., & Rieskamp, J. (2025). Parameter estimation of hyper-spherical diffusion models with a time-dependent threshold: An integral equation method. Behavior research methods, 57(10), 283. https://doi.org/10.3758/s13428-025-02810-3

- Smith, P. L. (2016). Diffusion theory of decision making in continuous report. Psychological Review, 123 (4), 425–451, https://doi.org/10.1037/rev0000023

- Smith, P.L., & Corbett, E.A. (2019). Speeded multielement decision-making as diffusion in a hypersphere: Theory and application to double-target detection. Psychonomic Bulletin & Review, 26, https://doi.org/10.3758/s13423-018-1491-0
