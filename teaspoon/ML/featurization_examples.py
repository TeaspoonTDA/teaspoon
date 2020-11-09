# -*- coding: utf-8 -*-
"""
This file provides examples for featurization functions in machine learning moduel of teaspoon.
"""

#%% -------------- Persistence Landscapes Class -------------------------------

import teaspoon.ML.feature_functions as Ff
from teaspoon.MakeData.PointCloud import testSetManifolds

# generate persistence diagrams
df = testSetManifolds(numDgms = 50, numPts = 100) 
Diagrams_H1 = df['Dgm1']

# Compute the persistence landscapes 
PLC = Ff.PLandscape(Diagrams_H1[0])
print(PLC.PL_number)
print(PLC.AllPL)
print(PLC.DesPL)
fig = PLC.PLandscape_plot(PLC.AllPL['Points'])
# fig.savefig('All_Landscapes.png',bbox_inches = 'tight', dpi=300)


PLC  = Ff.PLandscape(Diagrams_H1[0],[2,3])
print(PLC.PL_number)
print(PLC.DesPL)
fig = PLC.PLandscape_plot(PLC.AllPL['Points'],[2,3])
fig.show()
# fig.savefig('Des_Landscapes.png',bbox_inches = 'tight', dpi=300)

#%% ----------- Parameter Bucket for Persistence Land ------------------------

from teaspoon.ML.Base import LandscapesParameterBucket
from sklearn.svm import LinearSVC
from termcolor import colored
params = LandscapesParameterBucket()
params.clf_model = LinearSVC
params.test_size =0.33
params.Labels = None
params.PL_Number = [2]
print(params)


import numpy as np
import teaspoon.ML.feature_functions as Ff
from teaspoon.MakeData.PointCloud import testSetManifolds

# generate persistence diagrams
df = testSetManifolds(numDgms = 50, numPts = 100) 
Diagrams_H1= df['Dgm1'].sort_index().values

# Compute the persistence landscapes  for first and second persistence diagrams
PerLand=np.ndarray(shape=(2),dtype=object)
for i in range(0, 2):
    Land=Ff.PLandscape(Diagrams_H1[i])
    PerLand[i]=Land.AllPL

feature, Sorted_mesh = Ff.F_Landscape(PerLand,params)

#%% ---------------------- Persistence Images -------------------------------

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

fig[0].savefig('PI_Example_1.png',bbox_inches = 'tight', dpi=300)
fig[1].savefig('PI_Example_2.png',bbox_inches = 'tight', dpi=300)

#%% ---------------------- Carlsson Coordinates -------------------------------

import teaspoon.ML.feature_functions as Ff
from teaspoon.MakeData.PointCloud import testSetManifolds

# generate persistence diagrams
df = testSetManifolds(numDgms = 50, numPts = 100) 
Diagrams_H1= df['Dgm1'].sort_index().values

#compute feature matrix
FN=3
FeatureMatrix,TotalNumComb,CombList = Ff.F_CCoordinates(Diagrams_H1,FN)
print(TotalNumComb)
print(CombList)

#%% ---------------------- Path Signatures -----------------------------------

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

#%% ---------------------- Kernel Method -----------------------------------
import teaspoon.ML.feature_functions as Ff
from teaspoon.MakeData.PointCloud import testSetManifolds
# generate persistence diagrams
df = testSetManifolds(numDgms = 1, numPts = 100) 
Diagrams_H1 = df['Dgm1']
#compute kernel between two persistence diagram
sigma=0.25
kernel = Ff.KernelMethod(Diagrams_H1[0], Diagrams_H1[1], sigma)
print(kernel)






