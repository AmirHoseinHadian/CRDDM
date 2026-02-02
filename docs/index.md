# CRDDM

DRDDM is a Python package for modeling continuous-report decision tasks using **Continuous Response Diffusion Decision Models (CRDDM)**. 

The package provides fast and numerically stable likelihood evaluation for continuous-response diffusion decision models using the integral equation method proposed by [Hadian Rasanan et al.,(2025)](https://doi.org/10.3758/s13428-025-02810-3). CRDDM supports a wide range of continuous response scale, that can be employed in experimental studies including:

- Bounded one-dimensional scales (e.g., arcs or sliders),
- Circular scales (e.g., color wheels),
- Two-dimensional scales (e.g., 2D planes).

![Examples of continuous-report decision tasks.](imgs/Continuous_tasks.png "Examples of continuous-report decision tasks.")


![Diffusion models of continuous response tasks.](imgs/Diffusion_models.png "Diffusion models of continuous response tasks.")
CRDDM is designed for researchers in cognitive science, mathematical psychology, and neuroscience who work with diffusion models of continuous responses.


## What this documentation contains
- Installation
- A quickstart tutorial
- API reference generated from docstrings


## Credits

This package was developed by me, Amir Hosein Hadian Rasanan,
with getting support from Dr. Nathan J Evans and Prof. Dr. Jörg Rieskamp. 

When using this package or part of the code for your own research, I ask you to cite us:

> Hadian Rasanan, A. H., Evans, N. J., and Rieskamp, J. (in prepration). Modeling Continuous-response Decisions with Multi-dimensional Diffusion Decision Models: A Tutorial

*Also don't forget to cite the original paper for each model.* 

- **Circular Diffusion Model**: Smith, P. L. (2016). Diffusion theory of decision making in continuous report. *Psychological Review*, 123(4), 425–451. [https://doi.org/10.1037/rev0000023](https://doi.org/10.1037/rev0000023)
- **Hyper-spherical Diffusion Model**: Smith, P. L., & Corbett, E. A. (2019). Speeded multielement decision-making as diffusion in a hypersphere: Theory and application to double-target detection. *Psychonomic Bulletin & Review*, 26(1), 127-162. [https://doi.org/10.3758/s13423-018-1491-0](https://doi.org/10.3758/s13423-018-1491-0)
- **Projected Spherical Diffusion Model**: Hadian Rasanan, A. H., Olschewski, S., & Rieskamp, J. (in prepration) The Projected Spherical Diffusion Model: A Theory of Evidence Accumulation for Continuous Estimation Tasks
- **Multi-dimensional Diffusion Models with Collapsing Decision Threshold**: Hadian Rasanan, A. H., Evans, N. J., Amani Rad, J., & Rieskamp, J. (2025). Parameter estimation of hyper-spherical diffusion models with a time-dependent threshold: An integral equation method. *Behavior Research Methods*, 57(10), 283. [https://doi.org/10.3758/s13428-025-02810-3](https://doi.org/10.3758/s13428-025-02810-3)