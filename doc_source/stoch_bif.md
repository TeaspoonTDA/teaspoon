# Stochastic P-Bifurcation Detection

Stochastic P-bifurcations are points of topological changes in the joint probability density function (PDF) of a stochastic system. Please cite the papers [“A topological framework for identifying phenomenological bifurcations in stochastic dynamical systems”](https://doi.org/10.1007/s11071-024-09289-1) and [“Topological Detection of Phenomenological Bifurcations with Unreliable Kernel Densities”](https://doi.org/10.48550/arXiv.2401.16563) when using these functions. These modules can be used to detect P-bifurcation given

- [Analytical Densities](#analytical-density)

## Analytical Density

Given the analytical PDFs, the homological bifurcation plot can be generated with the module below

```{eval-rst}
.. automodule:: teaspoon.SP.StochasticP
   :members: analytical_homological_bifurcation_plot
```

### Example

The following example plots an analytical bifurcation plot for a set of PDFs where the system shifts from a monostability to a limit cycle:

```python
import numpy as np
from teaspoon.SP.StochasticP import analytical_homological_bifurcation_plot
X, Y = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
factors = np.linspace(-1,1,20)

PDFs = []
for h in factors:

    p = np.exp(-0.5*((X**2+Y**2)**2 + h*(X**2 + Y**2)))
    PDFs.append(p)

M = analytical_homological_bifurcation_plot(PDFs, bifurcation_parameters=factors, dimensions=[1], filter=0.02, maxEps=1, numStops=100, plotting=True)
```

The output for this example is

```{image} figures/analytical_homological_plot.png
:alt: Analytical Homological Plot
:width: 300px
:align: left
```
