Drawing
===========

.. automodule:: teaspoon.TDA.Draw
   :members: drawDgm,drawPtCloud

Examples
########

The following example plots a persistence diagram::

	from ripser import ripser
	from teaspoon.TDA.Draw import drawDgm
	from teaspoon.MakeData.PointCloud import Torus

	numPts = 500
	seed = 0
	
	# Generate Torus
	t = Torus(N=numPts,seed = seed)

	# Compute persistence diagrams
	PD1 = ripser(t,2)['dgms'][1]

	# Plot the diagram
	drawDgm(PD1)

The output for this example is

.. figure:: figures/persistence_diagram_plot_example.png
   :scale: 20 %
   
   
The following example plots a point cloud::

	from ripser import ripser
	from teaspoon.TDA.Draw import drawPtCloud
	from teaspoon.MakeData.PointCloud import Torus

	numPts = 500
	seed = 0

	# Generate Torus
	T = Torus(N=numPts,seed = seed)

	# Plot the point cloud
	drawPtCloud(T)

The output for this example is

.. figure:: figures/point_cloud_plot_example.png
   :scale: 20 %
