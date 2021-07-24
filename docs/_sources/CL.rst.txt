==============
Classification
==============

In :ref:`featurization`, we provide documentation on how to generate feature matrix using persistence diagrams with five different techniques.
This page includes the documentation of the classification codes and some examples.

Persistence Landscapes
----------------------

.. automodule:: teaspoon.ML.PD_Classification
    :members: CL_PL
    :undoc-members:
    :private-members:
    :special-members:

**Example:** We classify chatter in time series obtained from cutting signals in turning.
Persistence diagrams and landscapes are computed beforehand.
The time series data is available in :cite:`Khasawneh2019`.

**Example:**

	>>> import numpy as np
	>>> from teaspoon.ML.feature_functions import PLandscape
	>>> import teaspoon.ML.PD_Classification as PD_CL
	>>> from teaspoon.ML.Base import LandscapesParameterBucket
	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> from sklearn.svm import SVC
	>>> # generate persistence diagrams
	>>> df = testSetManifolds(numDgms = 10, numPts = 100) 
	>>> Diagrams_H1= df['Dgm1'].sort_index().values
	>>> Labels = df['trainingLabel'].sort_index().values
	>>> #parameters for classification
	>>> params = LandscapesParameterBucket()
	>>> params.clf_model = SVC
	>>> params.test_size = 0.33
	>>> params.Labels = Labels
	>>> params.PL_Number = [1,2,3,4,5,6,7,8]
	>>> print(params)
	Variables in parameter bucket
	-----------------------------
	clf_model : <class 'sklearn.svm._classes.SVC'>
	feature_function : <function F_Landscape at 0x000001A97957BD38>
	PL_Number : [1, 2, 3, 4, 5, 6, 7, 8]
	test_size : 0.33
	-----------------------------
	>>> # Compute the persistence landscapes 
	>>> PerLand=np.ndarray(shape=(60),dtype=object)
	>>> for i in range(0, 60):
	>>>     Land=PLandscape(Diagrams_H1[i])
	>>>     PerLand[i]=Land.AllPL
	>>> # Perform classification
	>>> result = PD_CL.CL_PL(PerLand,params)
	Landscapes used in feature matrix generation: [1, 2, 3, 4, 5, 6, 7, 8]
	Test set score: 0.58
	Test set deviation: 0.12288205727444508
	Training set score: 0.7449999999999999
	Training set deviation: 0.05099019513592784
	Total elapsed time: 0.6263787746429443

.. _CL_PB:

Parameter Bucket for Classification
-----------------------------------

.. currentmodule:: teaspoon.ML.Base
.. autoclass:: CL_ParameterBucket
   :special-members: __init__



Persistence Images
------------------

.. currentmodule:: teaspoon.ML.PD_Classification
.. autofunction:: CL_PI


**Example:**

	>>> import numpy as np
	>>> import teaspoon.ML.feature_functions as Ff
	>>> import teaspoon.ML.PD_Classification as CL_PD
	>>> from teaspoon.ML.Base import CL_ParameterBucket
	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> from sklearn.svm import SVC
	>>> # generate two sets of persistence diagrams
	>>> df_1 = testSetManifolds(numDgms = 10, numPts = 100)
	>>> df_2 = testSetManifolds(numDgms = 10, numPts = 100)
	>>> Diagrams_H1_1= df_1['Dgm1'].sort_index().values
	>>> Diagrams_H1_2= df_2['Dgm1'].sort_index().values
	>>> Labels_1 = df_1['trainingLabel'].sort_index().values
	>>> Labels_2 = df_2['trainingLabel'].sort_index().values
	>>> # classification without using transfer learning
	>>> TF_Learning = False
	>>> plot = False
	>>> D_Img=[]
	>>> #classification parameters
	>>> params = CL_ParameterBucket()
	>>> params.clf_model = SVC
	>>> params.test_size =0.33
	>>> params.Labels = Labels_1
	>>> print(params)
	Variables in parameter bucket
	-----------------------------
	clf_model : <class 'sklearn.svm._classes.SVC'>
	test_size : 0.33
	TF_Learning : False
	-----------------------------	
	>>> F_Images =Ff.F_Image(Diagrams_H1_1,0.1,0.10,plot,TF_Learning,D_Img)
	>>> results = CL_PD.CL_PI(F_Images['F_Matrix'],params)
	Test set score: 0.63
	Test set deviation: 0.1268857754044952
	Training set score: 0.7675
	Training set deviation: 0.05921359641163505
	Total elapsed time: 0.04411721229553223
	>>> # classification using transfer learning
	>>> # compute the feature matrix for second set of persistence diagrams
	>>> TF_Learning =True
	>>> F_Images_2 = Ff.F_Image(Diagrams_H1_1,0.1,0.10,plot,TF_Learning,D_Img,Diagrams_H1_2)
	>>> params = CL_ParameterBucket()
	>>> params.clf_model = SVC
	>>> params.test_size =0.33
	>>> params.training_labels = Labels_1
	>>> params.test_labels = Labels_2
	>>> params.TF_Learning  = True
	>>> print(params)
	Variables in parameter bucket
	-----------------------------
	clf_model : <class 'sklearn.svm._classes.SVC'>
	test_size : 0.33
	TF_Learning : True
	-----------------------------
	>>> results = CL_PD.CL_PI(F_Images_2['F_train'],params,F_Images_2['F_test'])
	Test set score: 0.6666666666666667
	Test set deviation: 0.06734350297014738
	Training set score: 0.6925000000000001
	Training set deviation: 0.04616546328154847
	Total elapsed time: 0.06018877029418945



Carlsson Coordinates
--------------------

.. currentmodule:: teaspoon.ML.PD_Classification
.. autofunction:: CL_CC


**Example:**

	>>> import numpy as np
	>>> import teaspoon.ML.feature_functions as Ff
	>>> import teaspoon.ML.PD_Classification as CL_PD
	>>> from teaspoon.ML.Base import CL_ParameterBucket
	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> from sklearn.svm import SVC
	>>> # generate two sets of persistence diagrams
	>>> df_1 = testSetManifolds(numDgms = 10, numPts = 100)
	>>> df_2 = testSetManifolds(numDgms = 10, numPts = 100)
	>>> Diagrams_H1_1= df_1['Dgm1'].sort_index().values
	>>> Diagrams_H1_2= df_2['Dgm1'].sort_index().values
	>>> # labels
	>>> Labels_1 = df_1['trainingLabel'].sort_index().values
	>>> Labels_2 = df_2['trainingLabel'].sort_index().values
	>>> # parameters used in classification without transfer learning
	>>> params = CL_ParameterBucket()
	>>> params.clf_model = SVC
	>>> params.test_size =0.33
	>>> params.Labels = Labels_1
	>>> params.TF_Learning =False
	>>> params.FN = 5
	>>> print(params)
	Variables in parameter bucket
	-----------------------------
	clf_model : <class 'sklearn.svm._classes.SVC'>
	test_size : 0.33
	TF_Learning : False
	FN : 5
	-----------------------------
	>>> # classification on one persistence diagram set
	>>> results = CL_PD.CL_CC(Diagrams_H1_1,params)
	Number of combinations: 31
	Highest accuracy among all combinations:
	Test set score: 0.635
	Test set deviation: 0.11191514642799695
	Training set score: 0.7325
	Training set deviation: 0.053677276383959714
	Total elapsed time: 0.2777674198150635
	>>> # parameters used in classification with transfer learning
	>>> params = CL_ParameterBucket()
	>>> params.clf_model = SVC
	>>> params.test_size =0.33
	>>> params.training_labels = Labels_1
	>>> params.test_labels = Labels_2
	>>> params.TF_Learning =True
	>>> params.FN = 5
	>>> print(params)
	Variables in parameter bucket
	-----------------------------
	clf_model : <class 'sklearn.svm._classes.SVC'>
	test_size : 0.33
	TF_Learning : True
	FN : 5
	-----------------------------
	>>> results = CL_PD.CL_CC(Diagrams_H1_1,params,Diagrams_H1_2)
	Number of combinations: 31
	Highest accuracy among all combinations:
	Test set score: 0.6976190476190476
	Test set deviation: 0.05639390134441434
	Training set score: 0.735
	Training set deviation: 0.04769696007084728
	Total elapsed time: 0.2907731533050537

Path Signatures
---------------

.. currentmodule:: teaspoon.ML.PD_Classification
.. autofunction:: CL_PS

**Example:**

	>>> import numpy as np
	>>> import teaspoon.ML.feature_functions as Ff
	>>> import teaspoon.ML.PD_Classification as CL_PD
	>>> from teaspoon.ML.Base import CL_ParameterBucket
	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> from sklearn.svm import SVC
	>>> # generate two sets of persistence diagrams
	>>> df_1 = testSetManifolds(numDgms = 5, numPts = 100)
	>>> df_2 = testSetManifolds(numDgms = 5, numPts = 100)
	>>> Diagrams_H1_1= df_1['Dgm1'].sort_index().values
	>>> Diagrams_H1_2= df_2['Dgm1'].sort_index().values
	>>> # labels
	>>> Labels_1 = df_1['trainingLabel'].sort_index().values
	>>> Labels_2 = df_2['trainingLabel'].sort_index().values
	>>> #compute persistence landscapes for both sets of persistence diagram
	>>> PerLand1=np.ndarray(shape=(60),dtype=object)
	>>> PerLand2=np.ndarray(shape=(60),dtype=object)
	>>> for i in range(0, 30):
	>>>     Land=Ff.PLandscape(Diagrams_H1_1[i])
	>>>     PerLand1[i]=Land.AllPL
	>>>     Land=Ff.PLandscape(Diagrams_H1_2[i])
	>>>     PerLand2[i]=Land.AllPL
	>>> # compute features using first landscapes
	>>> features1 = Ff.F_PSignature(PerLand1,L_Number=[1])
	>>> features2 = Ff.F_PSignature(PerLand2,L_Number=[1])
	>>> # traditional classification
	>>> # adjust parameters
	>>> params = CL_ParameterBucket()
	>>> params.clf_model = SVC
	>>> params.test_size =0.33
	>>> params.Labels = Labels1
	>>> params.TF_Learning = False
	>>> print(params)
	Variables in parameter bucket
	-----------------------------
	clf_model : <class 'sklearn.svm._classes.SVC'>
	test_size : 0.33
	TF_Learning : False
	-----------------------------
	>>> results = CL_PS(features,params)
	Test set score: 0.05
	Test set deviation: 0.15000000000000002
	Training set score: 0.575
	Training set deviation: 0.13919410907075053
	Total elapsed time: 0.009609222412109375
	>>> #transfer learning
	>>> params = CL_ParameterBucket()
	>>> params.clf_model = SVC
	>>> params.test_size =0.33
	>>> params.training_labels = Labels1
	>>> params.test_labels = Labels2
	>>> params.TF_Learning = True
	>>> print(params)
	Variables in parameter bucket
	-----------------------------
	clf_model : <class 'sklearn.svm._classes.SVC'>
	test_size : 0.33
	TF_Learning : True
	-----------------------------
	>>> results = CL_PS(features1,params,features2)
	Test set score: 0.4444444444444445
	Test set deviation: 0.09938079899999067
	Training set score: 0.5625
	Training set deviation: 0.0625
	Total elapsed time: 0.009023904800415039

Kernel Method
-------------  

.. currentmodule:: teaspoon.ML.PD_Classification
.. autofunction:: CL_KM 
  
**Example:**

	>>> import numpy as np
	>>> import teaspoon.ML.feature_functions as Ff
	>>> from teaspoon.ML.PD_Classification import CL_KM
	>>> from teaspoon.ML.Base import CL_ParameterBucket
	>>> from teaspoon.MakeData.PointCloud import testSetManifolds
	>>> # generate two sets of persistence diagrams
	>>> df_1 = testSetManifolds(numDgms = 5, numPts = 100)
	>>> Diagrams_H1_1= df_1['Dgm1'].sort_index().values
	>>> Labels_1 = df_1['trainingLabel'].sort_index().values
	>>> #convert string labels into integers ones 
	>>> Labels_ = np.zeros((len(Diagrams_H1_1)))
	>>> for i in range(len(Diagrams_H1_1)):
	>>>     if Labels_1[i]=='Torus':
	>>>         Labels_[i]=0
	>>>     elif Labels_1[i]=='Annulus':
	>>>         Labels_[i]=1
	>>>     elif Labels_1[i]=='Cube':
	>>>         Labels_[i]=2   
	>>>     elif Labels_1[i]=='3Cluster':
	>>>         Labels_[i]=3     
	>>>     elif Labels_1[i]=='3Clusters of 3Clusters':  
	>>>         Labels_[i]=4           
	>>>     elif Labels_1[i]=='Sphere':  
	>>>         Labels_[i]=5 
	>>> params = CL_ParameterBucket()
	>>> params.test_size =0.33
	>>> params.Labels = Labels_
	>>> params.sigma = 0.25
	>>> results = CL_KM(Diagrams_H1_1,params)
	Test set score: 23.333333333333332
	Test set deviation: 9.428090415820632
	Total elapsed time: 23.333333333333332
