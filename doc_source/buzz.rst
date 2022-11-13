Bifurcations using ZigZag (BuZZ)
====================================

This module contains code for the Bifurcations using ZigZag (BuZZ) method, which uses zigzag persistence to detect parameter values in a dynamical system which have a Hopf bifurcation. The paper summarizing this methodology is "`Using Zigzag Persistent Homology to Detect Hopf Bifurcations in Dynamical Systems <https://www.mdpi.com/1999-4893/13/11/278>`_". A basic example showing the functionality for some synthetic time series data can be found below.

.. automodule:: teaspoon.TDA.BuZZ
   :members: PtClouds

Example
########

First, import necessary modules, create some example data, and plot it::

  import numpy as np
  import matplotlib.pyplot as plt
  from teaspoon.TDA.Draw import drawDgm
  from teaspoon.TDA.BuZZ import PtClouds
  from teaspoon.SP.tsa_tools import takens

  t = np.linspace(0, 6*np.pi+1, 50)

  # Amplitudes of sine waves
  amps = [0.1,0.5,1,1.5,2,1.5,1,0.5,0.1]

  ts_list = []
  ptcloud_list = []
  for a in amps:
      y = a*np.sin(t) + (0.1*np.random.randint(-100,100,len(t))/100)

      # Compute sine wave and add noise uniformly distributed in [-0.1, 0.1]
      ts_list.append(y)

      # Compute time delay embedding point clouds
      ptcloud_list.append(takens(y, n=2, tau=4))


Which produces the following figure:

  .. image:: figures/buzz_initialdata.png
    :width: 500 px


Next, set up the zigzag with the point cloud data, and plot::

  # Setup zigzag with point clouds
  ZZ = PtClouds(ptcloud_list, num_landmarks=25, verbose=True)

  # Plot zigzag of point clouds
  ZZ.plot_ZZ_PtClouds()

Which produces the following figure:

  .. image:: figures/buzz_pc_zigzag.png
    :width: 500 px

Lastly, compute zigzag persistence, plot the zigzag of resulting Rips complexes and plot the persistence diagrams::

  # Compute zigzag persistence
  ZZ.run_Zigzag(r=0.85)

  # Plot zigzag of Rips complexes
  ZZ.plot_ZZ_Cplx()

  # Plot zigzag persistence diagram
  drawDgm(ZZ.zz_dgms[0]) # 0-dimensional diagram
  drawDgm(ZZ.zz_dgms[1]) # 1-dimensional diagram

Which produces the following figures:

  .. image:: figures/buzz_rips_zigzag.png
    :width: 500 px

  .. image:: figures/buzz_pd.png
    :width: 300 px
