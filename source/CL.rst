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

	>>> import pickle
	>>> import numpy as np
	>>> from teaspoon.ML.PD_Classification import CL_PL
	>>> from teaspoon.ML.Base import import LandscapesParameterBucket
	>>> from sklearn.svm import SVC
	>>> with open("Persistence_Landscapes_3.5inch_chatter.txt", "rb") as fp:
			PL = pickle.load(fp)
	>>> labels = pd.read_csv("Labels_3p5inch.csv")
	>>> params = LandscapesParameterBucket()
	>>> params.clf_model = SVC
	>>> params.test_size =0.33
	>>> params.Labels = labels
	>>> params.PL_Number = [1,2,3]
	>>> print(params)
	Variables in parameter bucket
	-----------------------------
	clf_model : <class 'sklearn.svm._classes.SVC'>
	feature_function : <function F_Landscape at 0x000001F97A549708>
	PL_Number : [1, 2, 3]
	test_size : 0.33
	-----------------------------
	>>> result = CL_PL(PL,params,labels)
	Landscapes used in feature matrix generation: [1, 2, 3]
	Test set score: 0.8913043478260869
	Test set deviation: 0.09770523936627927
	Training set score: 0.925
	Training set deviation: 0.017750567445242397
	Total elapsed time: 0.713921070098877
	
	
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
	
	>>> import pandas as pd
	>>> from teaspoon.ML.PD_Classification import CL_PI
	>>> from teaspoon.ML.Base import CL_ParameterBucket
	>>> from sklearn.svm import SVC
	>>> from ML.feature_functions import F_Image
	>>> #traditional classification
	>>> PD1 = pd.read_csv("Persistence_Diagram_Turning_Chatter_2inch.csv")
	>>> labels1 = pd.read_csv("Labels_2inch.csv")
	>>> PD2 = pd.read_csv("Persistence_Diagram_Turning_Chatter_4p5inch.csv")
	>>> labels2 = pd.read_csv("Labels_4p5inch.csv")
	>>> params = CL_ParameterBucket()
	>>> params.clf_model = SVC
	>>> params.test_size =0.33
	>>> params.Labels = labels1
	>>> print(params)
	Variables in parameter bucket
	-----------------------------
	clf_model : <class 'sklearn.svm._classes.SVC'>
	test_size : 0.33
	TF_Learning : False
	-----------------------------
	>>> params = CL_ParameterBucket()
	>>> TF_Learning = False
	>>> plot = False
	>>> D_Img=[]
	>>> F_Images = F_Image(PD1,0.1,0.10,plot,TF_Learning,D_Img)
	>>> results = CL_PI(F_Images['F_Matrix'],params)
	Test set score: 0.8362244897959183
	Test set deviation: 0.02514535006194208
	Training set score: 0.8269521410579346
	Training set deviation: 0.013981550716753569
	Total elapsed time: 9.852953910827637
	>>> #transfer learning
	>>> params = CL_ParameterBucket()
	>>> params.clf_model = SVC
	>>> params.test_size =0.33
	>>> params.training_labels = labels1
	>>> params.test_labels = labels2
	>>> params.TF_Learning =True
	>>> params.FN = 5
	>>> print(params)	
	Variables in parameter bucket
	-----------------------------
	clf_model : <class 'sklearn.svm._classes.SVC'>
	test_size : 0.33
	TF_Learning : True
	-----------------------------
	>>> TF_Learning = True
	>>> F_Images = F_Image(PD1,0.1,0.10,plot,TF_Learning,D_Img,PD2)
	>>> results = CL_PI(F_Images['F_train'],params,F_Images['F_test'])
	Test set score: 0.6419354838709678
	Test set deviation: 0.019819686657168545
	Training set score: 0.8299748110831235
	Training set deviation: 0.006875236304260034
	Total elapsed time: 9.155236959457397
	
Carlsson Coordinates
--------------------

.. currentmodule:: teaspoon.ML.PD_Classification
.. autofunction:: CL_CC


**Example:** 

	>>> import pandas as pd
	>>> from teaspoon.ML.Base import CL_ParameterBucket
	>>> from teaspoon.ML.PD_Classification import CL_CC
	>>> from sklearn.svm import SVC
	>>> #traditional classification
	>>> PD1 = pd.read_csv("Persistence_Diagram_Turning_Chatter_2inch.csv")
	>>> labels1 = pd.read_csv("Labels_2inch.csv")
	>>> PD2 = pd.read_csv("Persistence_Diagram_Turning_Chatter_4p5inch.csv")
	>>> labels2 = pd.read_csv("Labels_4p5inch.csv")
	>>> params = CL_ParameterBucket()
	>>> params.clf_model = SVC
	>>> params.test_size =0.33
	>>> params.Labels = labels1
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
	>>> results = CL_CC(PD1,params)
	Number of combinations: 31
	Highest accuracy among all combinations:
	Test set score: 0.8612244897959185
	Test set deviation: 0.023358210494407318
	Training set score: 0.8780856423173804
	Training set deviation: 0.009302864137339745
	Total elapsed time: 3.000014066696167

	
Path Signatures
---------------

.. currentmodule:: teaspoon.ML.PD_Classification
.. autofunction:: CL_PS

**Example:** 
	
	>>> from teaspoon.MakeData import PointCloud as gpc
	>>> from teaspoon.ML.Base import CL_ParameterBucket
	>>> from teaspoon.ML.PD_Classification import CL_PS
	>>> import teaspoon.ML.feature_functions as fF
	>>> # generate persistence diagrams
	>>> df1 = gpc.testSetManifolds(numDgms = 5, numPts = 30) 
	>>> Diagrams1 = df1['Dgm1'].values
	>>> Labels1 = df1['trainingLabel']
        >>> 
	>>> df2 = gpc.testSetManifolds(numDgms = 5, numPts = 30) 
	>>> Diagrams2 = df2['Dgm1'].values
	>>> Labels2 = df2['trainingLabel']
	>>> # compute features using first landscapes
	>>> features1 = fF.F_PSignature(PerLand1,L_Number=[1])
	>>> features2 = fF.F_PSignature(PerLand2,L_Number=[1])
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
	Test set score: 0.38
	Test set deviation: 0.13999999999999999
	Training set score: 0.6199999999999999
	Training set deviation: 0.07141428428542848
	Total elapsed time: 0.014008522033691406
	>>> #transfer learning 
	>>> params = CL_ParameterBucket()
	>>> params.clf_model = SVC
	>>> params.test_size =0.33
	>>> params.training_labels = Labels1
	>>> params.test_labels = Labels2
	>>> params.TF_Learning = True
	>>> print(params)
	-----------------------------
	clf_model : <class 'sklearn.svm._classes.SVC'>
	test_size : 0.33
	TF_Learning : True
	-----------------------------
	>>> results = CL_PS(features1,params,features2)
	Test set score: 0.4666666666666666
	Test set deviation: 0.10168645954315535
	Training set score: 0.5599999999999999
	Training set deviation: 0.09949874371066199
	Total elapsed time: 0.015541315078735352
	
	
Kernel Method
-------------

.. currentmodule:: teaspoon.ML.PD_Classification
.. autofunction:: CL_KM

**Example:** For this method, we use simple persistence diagram set since the methods is computationally expensive.

	>>> from teaspoon.ML.PD_Classification import CL_KM
	>>> from teaspoon.ML.PD_ParameterBucket import CL_ParameterBucket
	>>> PD=[]
	>>> # simple persistence diagram
	>>> PD.append(np.array([(1,5),(1,4),(2,14),(3,4),(4,8.1),(6,7),(7,8.5),(9,12)]))
	>>> PD.append(np.array([(1.2,3.6),(2,2.6),(2,7),(3.7,4),(5,7.3),(5.5,7),(9,11),(9,12),(12,17)]))
	>>> PD.append(np.array([(2,8),(3,4),(5.6,7)]))
	>>> params = CL_ParameterBucket()
	>>> params.test_size =0.33
	>>> params.Labels = labels
	>>> params.sigma = 0.25
	>>> results = CL_KM(a,params)
	Test set score: 66.66666666666667
	Test set deviation: 47.14045207910317
	Total elapsed time: 0.016573667526245117

