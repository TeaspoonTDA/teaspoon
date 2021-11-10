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
    	
	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> from teaspoon.ML import feature_functions as Ff
	>>> # generate persistence diagrams
	>>> df = testSetManifolds(numDgms = 50, numPts = 100) 
	>>> Diagrams_H1 = df['Dgm1']
	>>> # Compute the persistence landscapes 
	>>> PLC = PLandscape(Diagrams_H1[0])
	>>> print(PLC.PL_number)
	15
	>>> print(PLC.AllPL)
		Landscape Number                                             Points
	0                1.0  [[0.5428633093833923, 0.0], [0.580724596977233...
	1                2.0  [[0.571907639503479, 0.0], [0.5952467620372772...
	2                3.0  [[0.9977497458457947, 0.0], [1.132654219865799...
	3                4.0  [[0.9980520009994507, 0.0], [1.132805347442627...
	4                5.0  [[1.0069313049316406, 0.0], [1.137244999408722...
	5                6.0  [[1.01357901096344, 0.0], [1.0538994073867798,...
	6                7.0  [[1.078373670578003, 0.0], [1.0862967371940613...
	7                8.0  [[1.1089069843292236, 0.0], [1.188232839107513...
	8                9.0  [[1.114268183708191, 0.0], [1.1280720233917236...
	9               10.0  [[1.1168304681777954, 0.0], [1.129353165626525...
	10              11.0  [[1.1619293689727783, 0.0], [1.214744031429290...
	11              12.0  [[1.1846998929977417, 0.0], [1.226129293441772...
	12              13.0  [[1.2282723188400269, 0.0], [1.247915506362915...
	13              14.0  [[1.2527109384536743, 0.0], [1.260134816169738...
	14              15.0  [[1.2588499784469604, 0.0], [1.263204336166381...
	>>> print(PLC.DesPL)
	Warning: Desired landscape numbers were not specified.
	>>> fig = PLC.PLandscape_plot(PLC.AllPL['Points'])

Output of the plotting functions is:

.. figure:: figures/All_Landscapes.png
   :align: center
   :scale: 30 %

   All landscape functions for the given persistence diagram
   
If user specify the desired landscapes, output will be:

	>>> PLC  = PLandscape(Diagrams_H1[0],[2,3])
	>>> print(PLC.DesPL)
	[array([[0.57190764, 0.        ],
			[0.59524676, 0.02333912],
			[0.61858588, 0.        ],
			[0.69152009, 0.        ],
			[0.70559016, 0.01407006],
			[0.71966022, 0.        ],
			[0.8154344 , 0.        ],
			[0.83258173, 0.01714733],
			[0.84972906, 0.        ],
			[0.96607411, 0.        ],
			[1.19829136, 0.23221725],
			[1.21428031, 0.21622831],
			[1.23277295, 0.23472095],
			[1.28820044, 0.17929345],
			[1.31611174, 0.20720476],
			[1.32007349, 0.20324302],
			[1.39760172, 0.28077126],
			[1.50310916, 0.17526382],
			[1.54805887, 0.22021353],
			[1.62611502, 0.14215738],
			[1.65717965, 0.17322201],
			[1.76435941, 0.06604224],
			[1.81276023, 0.11444306],
			[1.9272033 , 0.        ]]) 
	array([[0.99774975, 0.        ],
			[1.13265422, 0.13490447],
			[1.13280535, 0.13475335],
			[1.21428031, 0.21622831],
			[1.21871996, 0.21178865],
			[1.22592431, 0.21899301],
			[1.27691215, 0.16800517],
			[1.28820044, 0.17929345],
			[1.29216218, 0.17533171],
			[1.32007349, 0.20324302],
			[1.34069568, 0.18262082],
			[1.34636515, 0.1882903 ],
			[1.34829241, 0.18636304],
			[1.3942554 , 0.23232603],
			[1.47721338, 0.14936805],
			[1.50310916, 0.17526382],
			[1.58116531, 0.09720767],
			[1.62611502, 0.14215738],
			[1.66337371, 0.10489869],
			[1.68506891, 0.12659389],
			[1.75498998, 0.05667281],
			[1.76435941, 0.06604224],
			[1.83040166, 0.        ]])]
	>>> fig = PLC.PLandscape_plot(PLC.AllPL['Points'],[2,3])

Output of the plotting functions is:
	
.. figure:: figures/Des_Landscapes.png
   :align: center
   :scale: 25 %

   Chosen landscape functions for the given persistence diagram

.. _PB_Landscape:
   
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
	
	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> from teaspoon.ML import feature_functions as Ff

	>>> # generate persistence diagrams
	>>> df = testSetManifolds(numDgms=50, numPts=100)
	>>> Diagrams_H1 = df['Dgm1'].sort_index().values

	>>> D_Img = []
	>>> plot = False
	>>> PS = 0.01
	>>> var = 0.01
	>>> feature_PI = Ff.F_Image(Diagrams_H1, PS, var, plot, D_Img=[], pers_imager = None,training=True)['F_Matrix']
	>>> # if user wants to plot images
	>>> plot = True
	>>> D_Img = [1,5]
	>>> feature_PI = Ff.F_Image(Diagrams_H1, PS, var, plot, D_Img=D_Img, pers_imager = None,training=True)
	>>> fig = feature_PI['figures']

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

**Example**::
	

	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> from teaspoon.ML import feature_functions as Ff

	>>> # generate persistence diagrams
	>>> df = testSetManifolds(numDgms=50, numPts=100)
	>>> Diagrams_H1 = df['Dgm1'].sort_index().values

	>>> # compute feature matrix
	>>> FN = 3
	>>> FeatureMatrix, TotalNumComb, CombList = Ff.F_CCoordinates(Diagrams_H1, FN)


.. _template_functions:

Template Functions
------------------

.. currentmodule:: teaspoon.ML.feature_functions
.. autofunction:: tent

.. currentmodule:: teaspoon.ML.feature_functions
.. autofunction:: interp_polynomial


**Example**::

	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> from teaspoon.ML import feature_functions as fF
	>>> from teaspoon.ML import Base
	>>> import numpy as np
	>>> import pandas as pd

	>>> # generate persistence diagrams
	>>> df = testSetManifolds(numDgms=50, numPts=100)
	>>> listOfG = []
	>>> dgm_col = ['Dgm0', 'Dgm1']
	>>> allDgms = pd.concat((df[label] for label in dgm_col))

	>>> # parameter bucket to set template function parameters
	>>> params = Base.ParameterBucket()
	>>> params.feature_function = fF.interp_polynomial
	>>> params.k_fold_cv=5
	>>> params.d = 20
	>>> params.makeAdaptivePartition(allDgms, meshingScheme=None)
	>>> params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial

	>>> # compute features
	>>> for dgmColLabel in dgm_col:
	>>> 	feature = Base.build_G(df[dgmColLabel], params)
	>>> 	listOfG.append(feature)
	>>> feature = np.concatenate(listOfG, axis=1) 







.. _path_signatures:

Path Signatures
---------------

.. currentmodule:: teaspoon.ML.feature_functions
.. autofunction:: F_PSignature	 

**Example**::
	
	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> from teaspoon.ML import feature_functions as fF
	>>> import numpy as np

	>>> # generate persistence diagrams
	>>> df = testSetManifolds(numDgms=1, numPts=100)
	>>> Diagrams_H1 = df['Dgm1'].sort_index().values
	
	>>> # compute persistence landscapes
	>>> PerLand = np.ndarray(shape=(6), dtype=object)
	>>> for i in range(0, 6):
	>>> 	Land = fF.PLandscape(Diagrams_H1[i])
	>>> 	PerLand[i] = Land.AllPL
	
	>>> # choose landscape number for which feature matrix will be computed
	>>> L_number = [2]
	>>> # compute feature matrix
	>>> feature_PS = fF.F_PSignature(PerLand, L_number)

.. _Kernel_Method:

Kernel Method
-------------

.. currentmodule:: teaspoon.ML.feature_functions
.. autofunction:: KernelMethod
 
**Example**::

	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> from teaspoon.ML import feature_functions as fF
	>>> # generate persistence diagrams
	>>> df = testSetManifolds(numDgms=1, numPts=100)
	>>> Diagrams_H1 = df['Dgm1']
	>>> # compute kernel between two persistence diagram
	>>> sigma = 0.25
	>>> kernel = fF.KernelMethod(Diagrams_H1[0], Diagrams_H1[1], sigma)
	>>> print(kernel)
	1.6310484200361053