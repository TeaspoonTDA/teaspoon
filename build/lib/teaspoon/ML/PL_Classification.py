# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 22:00:03 2020

@author: yesillim
"""

import teaspoon.ML.feature_functions as Ff
import teaspoon.ML.PD_Classification as PD_CL
from teaspoon.ML.Base import LandscapesParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.svm import SVC

# generate persistence diagrams
df = testSetManifolds(numDgms = 10, numPts = 100) 
Diagrams_H1= df['Dgm1'].sort_index().values
Labels = df['trainingLabel'].sort_index().values
#parameters for classification
params = LandscapesParameterBucket()
params.clf_model = SVC
params.test_size = 0.33
params.Labels = Labels
params.PL_Number = [2]
print(params)
# Perform classification
result = PD_CL.CL_PL(PL,params,labels)