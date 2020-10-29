Distances
=========

.. automodule:: teaspoon.TDA.Distance
   :members: dgmDist_Hera

Example
#######

The following example computes the distance between two persistence diagrams::

	from ripser import ripser
	from teaspoon.MakeData.PointCloud import Torus, Annulus, Cube, Clusters, Sphere
	from teaspoon.TDA.Distance import dgmDist_Hera
	numPts = 500
	seed = 0

	# Generate Torus
	t = Torus(N=numPts,seed = seed)

	# Generate Annulus
	a = Annulus(N=numPts,seed = seed)

	# Compute persistence diagrams
	PD1 = ripser(t,2)['dgms'][1]
	PD2 = ripser(a,1)['dgms'][1]

	# Distance between diagrams
	dgmDist_Hera(PD1,PD2,wassDeg='Bottleneck')

Output for this example is::

	0.6366935