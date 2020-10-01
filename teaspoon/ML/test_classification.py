# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:43:11 2018

@author: khasawn3
"""

# this file tests the classification code
# cd to the teaspoon folder, load the package and import the needed modules
import numpy as np
#import teaspoon.MakeData.PointCloud as gpc
from MakeData import PointCloud as gpc
from ML.PD_Classification import getPercentScore
from ML.Base import ParameterBucket
import ML.feature_functions as fF
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, RidgeClassifierCV

# to reload a module
from imp import reload
reload(fF)
reload(gpc)

#----------------------------------
# define a parameters bucket
#----------------------------------
params = ParameterBucket()

#----------------------------------
# generate a data frame for testing
#----------------------------------

#--- Test classification
# df = gpc.testSetClassification()
# dgmColLabel = 'Dgm'
# params.clfClass = RidgeClassifierCV



#--- Test regression
# df = gpc.testSetRegressionBall()
# dgmColLabel = 'Dgm'
# params.clfClass = RidgeCV


#--- Test manifold classification
df = gpc.testSetManifolds(numDgms = 50, numPts = 100) # numpTs=200
# dgmColLabel = ['Dgm0', 'Dgm1']
dgmColLabel = ['Dgm1']
params.clfClass = RidgeClassifierCV
params.TF_Learning=False
#print('\n', params, '\n')

# get a diagram
#Dgm = df['Dgm'][0]

#----------------------------------
# Find bounding box
#----------------------------------
params.setBoundingBox(df[dgmColLabel], pad = .05)
params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial
params.useAdaptivePart = False
#----------------------------------
# define the number of base points
#----------------------------------
params.d = 10


#----------------------------------
# choose a feature function
#----------------------------------
params.feature_function = fF.interp_polynomial
#params.feature_function = fF.tent

#----------------------------------
# Run the experiment
#----------------------------------
num_runs = 10
yy = np.zeros((num_runs))
for i in np.arange(num_runs):
	xx = getPercentScore(df,
					labels_col = 'trainingLabel',
					dgm_col = dgmColLabel,
					params = params,
					normalize = False,
					verbose = False
					)
	yy[i] = xx['score']

print('\navg success rate = {}\nStdev = {}'.format(np.mean(yy), np.std(yy)))

kernel = stats.gaussian_kde(yy)
values = np.linspace(yy.min(), yy.max(), 1000)
# plotting
#plt.plot(values, kernel(values), '.')
#plt.show()
