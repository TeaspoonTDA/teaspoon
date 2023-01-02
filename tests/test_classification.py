# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:43:11 2018

@author: khasawn3
"""

# this file tests the classification code
# cd to the teaspoon folder, load the package and import the needed modules
import numpy as np
#import teaspoon.MakeData.PointCloud as gpc
import teaspoon.MakeData.PointCloud as gpc
from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML.Base import ParameterBucket
import teaspoon.ML.feature_functions as fF
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, RidgeClassifierCV

import unittest

import warnings




class TestManifolds(unittest.TestCase):

    def test_learning_Manifolds(self):

        """
        From Liz, 7/30/21
        This is a hot mess.  Something was wrong with a deprication warning and now it's just a mess in here

        set the name of this function to test_learning_Manifolds to get the unittest to find it again
        """

        # ----------------------------------
        # define a parameters bucket
        # ----------------------------------
        params = ParameterBucket()

        # ----------------------------------
        # generate a data frame for testing
        # ----------------------------------

        # --- Test classification
        # df = gpc.testSetClassification()
        # dgmColLabel = 'Dgm'
        # params.clfClass = RidgeClassifierCV


        # --- Test regression
        # df = gpc.testSetRegressionBall()
        # dgmColLabel = 'Dgm'
        # params.clfClass = RidgeCV


        # --- Test manifold classification
        df = gpc.testSetManifolds(numDgms=5, numPts=100)  # numpTs=200
        # dgmColLabel = ['Dgm0', 'Dgm1']
        dgmColLabel = ['Dgm1']
        params.clfClass = RidgeClassifierCV
        params.TF_Learning = False
        #print('\n', params, '\n')

        # get a diagram
        #Dgm = df['Dgm'][0]

        # ----------------------------------
        # Find bounding box
        # ----------------------------------
        params.setBoundingBox(df[dgmColLabel], pad=.05)
        params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial
        params.useAdaptivePart = False
        # ----------------------------------
        # define the number of base points
        # ----------------------------------
        params.d = 10


        # ----------------------------------
        # choose a feature function
        # ----------------------------------
        params.feature_function = fF.interp_polynomial
        #params.feature_function = fF.tent

        # ----------------------------------
        # Run the experiment
        # ----------------------------------
        num_runs = 1
        yy = np.zeros((num_runs))
        for i in np.arange(num_runs):
            xx = getPercentScore(df,
                                 labels_col='trainingLabel',
                                 dgm_col=dgmColLabel,
                                 params=params,
                                 normalize=False,
                                 verbose=False
                                 )
            #print(xx[0][0]['Annulus']['f1-score'])
            yy[i] = xx[0][0]['Annulus']['f1-score']

        # print('\navg success rate = {}\nStdev = {}'.format(np.mean(yy), np.std(yy)))

        # kernel = stats.gaussian_kde(yy)
        # values = np.linspace(yy.min(), yy.max(), 1000)
        # plotting
        #plt.plot(values, kernel(values), '.')
        # plt.show()


        self.assertAlmostEqual(np.mean(yy), 0.8, delta = 0.3)



if __name__ == '__main__':
    unittest.main()
