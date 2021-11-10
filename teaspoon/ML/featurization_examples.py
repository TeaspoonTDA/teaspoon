# -*- coding: utf-8 -*-
"""
This file provides examples for featurization functions in machine learning moduel of teaspoon.
"""

# %% -------------- Persistence Landscapes Class -------------------------------

from teaspoon.MakeData.PointCloud import testSetManifolds
from teaspoon.ML import feature_functions as Ff

# generate persistence diagrams
df = testSetManifolds(numDgms=50, numPts=100)
Diagrams_H1 = df['Dgm1']

# Compute the persistence landscapes
PLC = Ff.PLandscape(Diagrams_H1[0])
print(PLC.PL_number)
print(PLC.AllPL)
print(PLC.DesPL)

fig = PLC.PLandscape_plot(PLC.AllPL['Points'])
# fig.savefig('All_Landscapes.png',bbox_inches = 'tight', dpi=300)


PLC = Ff.PLandscape(Diagrams_H1[0], [2, 3])
print(PLC.PL_number)
print(PLC.DesPL)
fig = PLC.PLandscape_plot(PLC.AllPL['Points'], [2, 3])
fig.show()
# fig.savefig('Des_Landscapes.png',bbox_inches = 'tight', dpi=300)

# %% ----------- Parameter Bucket for Persistence Land ------------------------

params = LandscapesParameterBucket()
params.clf_model = LinearSVC
params.test_size = 0.33
params.Labels = None
params.PL_Number = [2]
print(params)


# generate persistence diagrams
df = testSetManifolds(numDgms=50, numPts=100)
Diagrams_H1 = df['Dgm1'].sort_index().values

# Compute the persistence landscapes  for first and second persistence diagrams
PerLand = np.ndarray(shape=(2), dtype=object)
for i in range(0, 2):
    Land = Ff.PLandscape(Diagrams_H1[i])
    PerLand[i] = Land.AllPL

feature, Sorted_mesh = Ff.F_Landscape(PerLand, params)

# %% ---------------------- Persistence Images -------------------------------
from teaspoon.MakeData.PointCloud import testSetManifolds
from teaspoon.ML import feature_functions as Ff

# generate persistence diagrams
df = testSetManifolds(numDgms=50, numPts=100)
Diagrams_H1 = df['Dgm1'].sort_index().values

D_Img = []
plot = False
PS = 0.01
var = 0.01
feature_PI = Ff.F_Image(Diagrams_H1, PS, var, plot, D_Img=[], pers_imager = None,training=True)['F_Matrix']
# if user wants to plot images
plot = True
D_Img = [1,5]
feature_PI = Ff.F_Image(Diagrams_H1, PS, var, plot, D_Img=D_Img, pers_imager = None,training=True)
fig = feature_PI['figures']

fig[0].savefig('PI_Example_1.png', bbox_inches='tight', dpi=300)
fig[1].savefig('PI_Example_2.png', bbox_inches='tight', dpi=300)

# %% ---------------------- Carlsson Coordinates -------------------------------
from teaspoon.MakeData.PointCloud import testSetManifolds
from teaspoon.ML import feature_functions as Ff

# generate persistence diagrams
df = testSetManifolds(numDgms=50, numPts=100)
Diagrams_H1 = df['Dgm1'].sort_index().values

# compute feature matrix
FN = 3
FeatureMatrix, TotalNumComb, CombList = Ff.F_CCoordinates(Diagrams_H1, FN)
print(TotalNumComb)
print(CombList)

#%% ------------------------- Template Functions -----------------------------
from teaspoon.MakeData.PointCloud import testSetManifolds
from teaspoon.ML import feature_functions as fF
from teaspoon.ML import Base
import numpy as np
import pandas as pd

# generate persistence diagrams
df = testSetManifolds(numDgms=50, numPts=100)
listOfG = []
dgm_col = ['Dgm0', 'Dgm1']
allDgms = pd.concat((df[label] for label in dgm_col))

# parameter bucket to set template function parameters
params = Base.ParameterBucket()
params.feature_function = fF.interp_polynomial
params.k_fold_cv=5
params.d = 20
params.makeAdaptivePartition(allDgms, meshingScheme=None)
params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial

# compute features
for dgmColLabel in dgm_col:
    feature = Base.build_G(df[dgmColLabel], params)
    listOfG.append(feature)
feature = np.concatenate(listOfG, axis=1) 
# %% ---------------------- Path Signatures -----------------------------------
from teaspoon.MakeData.PointCloud import testSetManifolds
from teaspoon.ML import feature_functions as fF
import numpy as np

# generate persistence diagrams
df = testSetManifolds(numDgms=1, numPts=100)
Diagrams_H1 = df['Dgm1'].sort_index().values
# compute persistence landscapes
PerLand = np.ndarray(shape=(6), dtype=object)
for i in range(0, 6):
    Land = fF.PLandscape(Diagrams_H1[i])
    PerLand[i] = Land.AllPL
# choose landscape number for which feature matrix will be computed
L_number = [2]
# compute feature matrix
feature_PS = fF.F_PSignature(PerLand, L_number)

# %% ---------------------- Kernel Method -----------------------------------
from teaspoon.MakeData.PointCloud import testSetManifolds
from teaspoon.ML import feature_functions as fF
# generate persistence diagrams
df = testSetManifolds(numDgms=1, numPts=100)
Diagrams_H1 = df['Dgm1']
# compute kernel between two persistence diagram
sigma = 0.25
kernel = fF.KernelMethod(Diagrams_H1[0], Diagrams_H1[1], sigma)
print(kernel)
