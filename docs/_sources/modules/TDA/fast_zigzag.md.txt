# Fast Zigzag

These modules help generate the input file and read the output file for [fast-zigzag software (fzz)](https://github.com/TDA-Jyamiti/fzz). 

```{eval-rst}
.. warning:: 
   Because `fzz` is not pip-installable, it is not automatically included when you install teaspoon.  In order to use this submodule, you need to manually install `fzz` first. 

   Note also that currently the way this system works is to write a file that is read into `fzz`, and the output file is then read back into python. This is not ideal, so in the future we hope to find a more efficient pipeline.  
```

## Generate Input File & Read Output 

Given a list of point clouds, the input file can be generated using the following module

```{eval-rst}
.. automodule:: teaspoon.TDA.fast_zigzag
   :members: generate_input_file
```

and the output can be read and plotted by the following module

```{eval-rst}
.. automodule:: teaspoon.TDA.fast_zigzag
   :members: plot_output_zigzag
```

### Example

The following example plots the zigzag persistence for point clouds such that the rips complex turns from a blob to a circle:

```python
import numpy as np
from teaspoon.TDA.fast_zigzag import generate_input_file, plot_output_zigzag

point_clouds = []
t = np.linspace(0, 2*np.pi, 8)[:-1]
point_clouds.append(np.vstack((1*np.cos(t), 1*np.sin(t))).T)
point_clouds.append(np.vstack((21*np.cos(t), 21*np.sin(t))).T)

inserts, deletes = generate_input_file(point_clouds, filename='output', radius=19, n_perm=25, plotting=False)

os.system(f"./fzz output")

filename = 'output_pers'
plot_output_zigzag(filename, inserts, deletes, plotH2=False)

```

The point clouds with the peristence diagram 

```{image}fast_zigzag.png
:alt: Point Clouds with zigzag persistence diagram
:width: 100%
:align: center
```

<br></br>