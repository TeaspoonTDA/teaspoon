.. _featurization:

=============
Featurization
=============
This documentation includes five different persistence diagram featurization methods.
These are persistence landscapes, persistence images, Carlsson Coordinates, kernel method, and signature of paths.

.. _persistence_landscapes:

Persistence Landscapes
----------------------

Landscape class
~~~~~~~~~~~~~~~

.. autoclass:: teaspoon.ML.feature_functions.PLandscape
    :members: 
    :undoc-members:
    :private-members:
    :special-members: __init__

**Example:** In this example, we do not specify which landscape function we want specifically. Therefore, algorihtms returns a warning to user if desired landscape points is wanted. 
    	
	>>> import teaspoon.ML.feature_functions as Ff
	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> # generate persistence diagrams
	>>> df = testSetManifolds(numDgms = 50, numPts = 100) 
	>>> Diagrams_H1 = df['Dgm1']
	>>> # Compute the persistence landscapes 
	>>> PLC = Ff.PLandscape(Diagrams_H1[0])
	>>> print(PLC.PL_number)
	18
	>>> print(PLC.AllPL)
	    Landscape Number                                             Points
	0                1.0  [[0.3154523968696594, 0.0], [0.320831686258316...
	1                2.0  [[0.7809883952140808, 0.0], [0.790885269641876...
	2                3.0  [[0.9872134923934937, 0.0], [1.046982824802398...
	3                4.0  [[1.0011025667190552, 0.0], [1.053927361965179...
	4                5.0  [[1.010329246520996, 0.0], [1.05854070186615, ...
	5                6.0  [[1.0254381895065308, 0.0], [1.066095173358917...
	6                7.0  [[1.0484610795974731, 0.0], [1.077606618404388...
	7                8.0  [[1.1333231925964355, 0.0], [1.148026943206787...
	8                9.0  [[1.153172254562378, 0.0], [1.1579514741897583...
	9               10.0  [[1.1831640005111694, 0.0], [1.264368951320648...
	10              11.0  [[1.188441276550293, 0.0], [1.26700758934021, ...
	11              12.0  [[1.193040132522583, 0.0], [1.269307017326355,...
	12              13.0  [[1.1967955827713013, 0.0], [1.229873955249786...
	13              14.0  [[1.2031320333480835, 0.0], [1.233042180538177...
	14              15.0  [[1.2164264917373657, 0.0], [1.239689409732818...
	15              16.0  [[1.2490906715393066, 0.0], [1.256021499633789...
	16              17.0  [[1.2782702445983887, 0.0], [1.304370105266571...
	17              18.0  [[1.2977339029312134, 0.0], [1.314101934432983...
	>>> print(PLC.DesPL)
	Warning: Desired landscape numbers were not specified.
	>>> fig = PLC.PLandscape_plot(PLC.AllPL['Points'])

Output of the plotting functions is:

.. figure:: figures/All_Landscapes.png
   :align: center
   :scale: 10 %

   All landscape functions for the given persistence diagram
   
If user specify the desired landscapes, output will be:

	>>> PLC  = PLandscape(Diagrams_H1[0],[2,3])
	>>> print(PLC.DesPL)
	[array([[0.7809884 , 0.        ],
	       [0.79088527, 0.00989687],
	       [0.80078214, 0.        ],
	       [0.98659766, 0.        ],
	       [1.08002311, 0.09342545],
	       [1.08727556, 0.086173  ],
	       [1.30410457, 0.303002  ],
	       [1.30871791, 0.29838866],
	       [1.39603722, 0.38570797],
	       [1.7817452 , 0.        ]])
	 array([[0.98721349, 0.        ],
	       [1.04698282, 0.05976933],
	       [1.05392736, 0.0528248 ],
	       [1.08727556, 0.086173  ],
	       [1.0918889 , 0.08155966],
	       [1.30871791, 0.29838866],
	       [1.32778382, 0.27932274],
	       [1.38544387, 0.33698279],
	       [1.43779945, 0.2846272 ],
	       [1.46482104, 0.31164879],
	       [1.77646983, 0.        ]])]
	>>> PLC.PLandscape_plot(points['Points'])

Output of the plotting functions is:
	
.. figure:: figures/Des_Landscapes.png
   :align: center
   :scale: 25 %

   Chosen landscape functions for the given persistence diagram

.. _PB_Landscape:
   
Parameter bucket for landscapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
   
.. autoclass:: teaspoon.ML.Base.LandscapesParameterBucket
    :members:
    :undoc-members:
    :private-members:
    :special-members: __init__

**Example:** If user does not provide classification labels, parameter bucket will return a warning as shown below.
 
	>>> from teaspoon.ML.Base import LandscapesParameterBucket
	>>> from sklearn.svm import LinearSVC
	>>> from termcolor import colored
	>>> params = LandscapesParameterBucket()
	>>> params.clf_model = LinearSVC
	>>> params.test_size =0.5
	>>> params.Labels = None
	>>> params.PL_Number = [2]
	>>> print(params)
	Variables in parameter bucket
	-----------------------------
	clf_model : <class 'sklearn.svm._classes.LinearSVC'>
	feature_function : <function F_Landscape at 0x000001F6AB9F6558>
	PL_Number : [2]
	Labels : None
	test_size : 0.5
	-----------------------------
	Warning: Classification labels are missing.

Feature matrix generation  
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _F_Matrix:

.. figure:: figures/Mesh_Generation_Example.png
   :align: center
   :scale: 15 %

   Feature matrix generation steps explained with an simple example.

:numref:`F_Matrix` explains the steps for generation feature matrix using persistence landscapes. There are three persistence landscape sets for three different persistence diagram. 
We choose one landscape function among them. In the example above, second landscape function is selected and plotted for each landscape set.
The plot in the third column includes all selected landscape functions. 
In other words, we plot all selected landscapes in same figure.
The next step is to find the mesh points using node points of landscapes. 
Node points are projected on x-axis.
The red dots in the plot represent these projections.
Then, we sort these points (red dots) and remove the duplicates if there is any.
Resulting array will be our mesh and it is used to obtain features.
The mesh points is shown in :numref:`Mesh` with red dots.
There may not be corresponding y value for each mesh points in selected landscape functions so we use linear interpolation to find these values.
Then, these y values become the feature for each landscape functions, and they can be used in classification.

.. _Mesh:

.. figure:: figures/Mesh_Points.png
   :align: center
   :scale: 20 %

   Mesh obtained using second landscape function for the example provided in :numref:`F_Matrix`.

.. automodule:: teaspoon.ML.feature_functions
    :members: F_Landscape
    :undoc-members:
    :private-members:
    :special-members:

.. _persistence_images:
	
Persistence Images
------------------
	
.. currentmodule:: teaspoon.ML.feature_functions
.. autofunction:: F_Image
	

**Example**::
	
	import teaspoon.ML.feature_functions as Ff
	from teaspoon.MakeData.PointCloud import testSetManifolds
	
	# generate persistence diagrams
	df = testSetManifolds(numDgms = 50, numPts = 100) 
	Diagrams_H1= df['Dgm1'].sort_index().values

	TF_Learning = False
	D_Img=[1,75]
	plot=False
	feature_PI = Ff.F_Image(Diagrams_H1,0.01,0.15,plot,TF_Learning, D_Img)
	# if user wants to plot images
	plot=True
	feature_PI = Ff.F_Image(Diagrams_H1,0.01,0.15,plot,TF_Learning, D_Img)
	fig = feature_PI['figures']

The algorithm will return two images as shown in :numref:`PI_Example`.

.. _PI_Example:

.. figure:: figures/PI_Example.png
   :align: center
   :scale: 20 %
   
   Example persistence images.    

.. _carlsson_coordinates:

Carlsson Coordinates
--------------------

.. currentmodule:: teaspoon.ML.feature_functions
.. autofunction:: F_CCoordinates

**Example:**
	
	>>> import teaspoon.ML.feature_functions as Ff
	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> # generate persistence diagrams
	>>> df = testSetManifolds(numDgms = 50, numPts = 100) 
	>>> Diagrams_H1= df['Dgm1'].sort_index().values
	>>> FN=3
	>>> FeatureMatrix,TotalNumComb,CombList = Ff.F_CCoordinates(Diagrams_H1,FN)
	>>> print(TotalNumComb)
	7
	>>> print(CombList)
	[[1. 0. 0. 0. 0.]
	 [2. 0. 0. 0. 0.]
	 [3. 0. 0. 0. 0.]
	 [1. 2. 0. 0. 0.]
	 [1. 3. 0. 0. 0.]
	 [2. 3. 0. 0. 0.]
	 [1. 2. 3. 0. 0.]]	


.. _path_signatures:


Path Signatures
---------------

.. currentmodule:: teaspoon.ML.feature_functions
.. autofunction:: F_PSignature	 

**Example**::
	
	import numpy as np
	import teaspoon.ML.feature_functions as Ff
	from teaspoon.MakeData.PointCloud import testSetManifolds
	# generate persistence diagrams
	df = testSetManifolds(numDgms = 1, numPts = 100) 
	Diagrams_H1= df['Dgm1'].sort_index().values
	#compute persistence landscapes
	PerLand=np.ndarray(shape=(6),dtype=object)
	for i in range(0, 6):
	    Land=Ff.PLandscape(Diagrams_H1[i])
	    PerLand[i]=Land.AllP
	#choose landscape number for which feature matrix will be computed
	L_number = [2]
	#compute feature matrix
	feature_PS = Ff.F_PSignature(PerLand,L_number)

Kernel Method
-------------

.. currentmodule:: teaspoon.ML.feature_functions
.. autofunction:: KernelMethod

**Example:**

	>>> import teaspoon.ML.feature_functions as Ff
	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> # generate persistence diagrams
	>>> df = testSetManifolds(numDgms = 1, numPts = 100) 
	>>> Diagrams_H1 = df['Dgm1']
	>>> #compute kernel between two persistence diagram
	>>> sigma=0.25
	>>> kernel = Ff.KernelMethod(Diagrams_H1[0], Diagrams_H1[1], sigma)
	>>> print(kernel)
	1.6310484200361053