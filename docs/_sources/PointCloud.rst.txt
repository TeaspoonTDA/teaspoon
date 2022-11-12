.. _Point_Cloud:

Point Cloud Data Generation Module Documentation
=========================================================

This module is used to generate point cloud data sampled from various underlying shapes.
Specifically, the possible shapes are:

- Annulus
- Circle
- Clusters
- Cube
- Sphere
- Torus

Additionally, this module can also generate random persistence diagrams.

.. automodule:: teaspoon.MakeData.PointCloud
    :members:


Examples
#########

The following is an example generating point clouds sampled from different
underlying shapes::

  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt

  from teaspoon.MakeData.PointCloud import Torus, Annulus, Cube, Clusters, Sphere

  numPts = 500
  seed = 0

  # Generate Torus
  t = Torus(N=numPts,seed = seed)

  # Generate Annulus
  a = Annulus(N=numPts,seed = seed)

  # Generate Sphere
  s = Sphere(N = numPts, noise = .05,seed = seed)

  # Generate Cube
  c = Cube(N=numPts,seed = seed)

  # Generate 3 clusters
  cl = Clusters(centers=np.array( [ [0,0], [0,2], [2,0]  ]), N = numPts, seed = seed, sd = .05)

  # Generate 3 clusters of 3 clusters
  centers = np.array( [ [0,0], [0,1.5], [1.5,0]  ])
  theta = np.pi/4
  centersUp = np.dot(centers,np.array([(np.sin(theta),np.cos(theta)),(np.cos(theta),-np.sin(theta))])) + [0,5]
  centersUpRight = centers + [3,5]
  centers = np.concatenate( (centers,  centersUp, centersUpRight))
  clcl = Clusters(centers=centers, N = numPts, sd = .05,seed = seed)

  fig, axes = plt.subplots(2, 3, figsize=(9,6))

  axes[0,0].scatter(t[:,0], t[:,1], c= t[:,2], cmap='Blues')
  axes[0,0].set_title('Torus')

  axes[0,1].scatter(a[:,0], a[:,1], cmap='Blues')
  axes[0,1].set_title('Annulus')

  axes[0,2].scatter(c[:,0], c[:,1], cmap='Blues')
  axes[0,2].set_title('Cube')

  axes[1,0].scatter(cl[:,0], cl[:,1], cmap='Blues')
  axes[1,0].set_title('3Clusters')

  axes[1,1].scatter(clcl[:,0], clcl[:,1], cmap='Blues')
  axes[1,1].set_title('3Clusters of 3Clusters')

  axes[1,2].scatter(s[:,0], s[:,1], c= s[:,2], cmap='Blues')
  axes[1,2].set_title('Sphere')

  plt.tight_layout()

  plt.show()

Where the output for this example is:

.. figure:: figures/PCs.png

The following is an example generating a data set of persistence diagrams
computed from point clouds sampled from different underlying shapes. In this
case, the persistence diagrams correspond to the point clouds shown in the above
example::

  from teaspoon.MakeData import PointCloud
  import pandas as pd
  import matplotlib.pyplot as plt

  df = PointCloud.testSetManifolds(numDgms = 1, numPts = 500, seed=0)

  fig, axes = plt.subplots(2,3,figsize=(9,6), sharex=True, sharey=True)
  axes = axes.ravel()
  for i in df.index:
    axes[i].scatter(df.loc[i,'Dgm0'][:,0], df.loc[i,'Dgm0'][:,1],label='$H_0$')
    axes[i].scatter(df.loc[i,'Dgm1'][:,0], df.loc[i,'Dgm1'][:,1],label='$H_1$')
    axes[i].plot([0,2], [0,2],c='k',linestyle='--')
    axes[i].set_title(df.loc[i,'trainingLabel'])
    axes[i].legend(loc=4)
  plt.show()

Where the output for this example is:

.. figure:: figures/PointCloudEx.png
