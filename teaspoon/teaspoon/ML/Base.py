import sys
import os
import teaspoon.ML.feature_functions as fF
import time
import numpy as np
import pandas as pd
import itertools
from teaspoon.TDA import Persistence as pP
from teaspoon.SP.adaptivePart import Partitions
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, RidgeClassifierCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn.svm import LinearSVC, NuSVC, SVC
"""
This is the main code for running ML code in teaspoon.

Here, we start with an instance of the `ParameterBucket` class. The intention of this
object is to keep all determined parameters in one easy to use object. A new
`ParameterBucket` subclass can be defined to inform any featurization method of interest.

"""


# For instance, a simple example of using tent functions as defined in *Approximating
# Continuous Functions on Persistence Diagrams Using Template Functions* (Perea, Munch,
# Khasawneh 2018) is shown below.

# import teaspoon.ML.Base as Base
# import teaspoon.MakeData.PointCloud as gPC
# import teaspoon.ML.feature_functions as fF
# from sklearn.linear_model import RidgeClassifierCV

# params = Base.TentParameters(clf_model = RidgeClassifierCV,
#                              feature_function = fF.tent,
#                               test_size = .33,
#                               seed = 48824,
#                               d = 10,
#                               delta = 1,
#                               epsilon = 0
#                              )

# DgmsDF = gPC.testSetClassification(N = 20,
#                                   numDgms = 50,
#                                   muRed = (1,3),
#                                   muBlue = (2,5),
#                                   sd = 1,
#                                    seed = 48824
#                                   )

# out = Base.getPercentScore(DgmsDF,dgm_col = 'Dgm', labels_col = 'trainingLabel', params = params )


"""
.. module: Base
"""


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', '..', 'teaspoon'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'teaspoon'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'teaspoon', 'ML'))
sys.path.insert(0, os.path.join(os.path.dirname(
    __file__), '..', '..', 'teaspoon', 'ML'))
sys.path.insert(0, os.path.join(os.path.dirname(
    __file__), '..', '..', 'teaspoon', 'TSP'))


class ParameterBucket(object):
    def __init__(self,
                 clf_model=RidgeClassifierCV,
                 feature_function=fF.tent,
                 k_fold_cv=10,
                 TF_Learning=False,
                 param_tuning=False,
                 seed=None,
                 PL_Number=None,
                 parToTune=None,
                 sigma=None,
                 L_number=None,
                 pixel_size=0.01,
                 var=0.15,
                 FN=5,
                 **kwargs):
        """
        Parameters
        ----------
        clf_model : classification model, optional
           Classifier. The default is RidgeClassifierCV.
        feature_function : function, optional
            Feature function that is used to extract features from persistence diagrams.
            This package currenly support template functions, persistence landscapes,
            persistence images, Carlsson Coordinates, path signatures of landscapes, and 
            kernel method. The default is fF.tent.
        k_fold_cv : int, optional
            The number of folds for k-fold cross validation.The default is 10.
        TF_Learning : boolean, optional
            User sets this parameter to True if training and test set diagrams are different but
            related. The default is False.
        param_tuning : boolean, optional
            This enables hyperparameter tuning using GridSearch algorithm for selected classifier, when it is set to True.
            User also needs to pass a list of a dict that enables includes the parameters to tune and their range. The default is False.
        seed : int, optional
            The random state number for stratified k-fold. Pass None if you dont want to fix it.. The default is None.
        PL_Number : list, optional
            The list of integers that includes landscape numbers. These landscape numbers are used to extract features. This parameter is only
            needed when feature function is persistence landscape. If user does not provide this parameter, algoritm will warn user. The default is None.
        parToTune : list, optional
            User needs to pass the list of parameters to tune for selected classifier. Otherwise, algorithm will warn the user.
            This parameter is only required when param_tuning is set to True. The default is None.
        sigma : int, optional
            Kernel scale for kernel method. The default is None.
        L_number : list, optional
            The list of landscape numbers that will be used to extract features using path signatures. The default is None.
        pixel_size : float, optional
            The size of the pixels for persistence images. The default is 0.01.
        var : float, optional
            Variance of the Gaussian distribution. The default is 0.15.
        FN : int, optional
            The number of coordinates user wants to use for Carlsson Coordinates. Maximum value for this parameter is 5. The default is 5.
        **kwargs : 
            Any additional parameters.


        """

        # Parameters that used to be included. Documentation left here to figure out stuff for later.
        #
        # @param d, delta, epsilon
        #     The bounding box for the persistence diagram in the (birth, lifetime) coordinates is [0,d * delta] x [epsilon, d* delta + epsilon].  In the usual coordinates, this creates a parallelogram.
        # @param maxPower
        #     The maximum degree used for the monomial combinations of the tent functions.  Testing suggests we usually want this to be 1.  Increasing causes large increase in number of features.
        # @param boundingBoxMatrix
        # 	Not yet implemented.  See self.findBoundingBox()

        self.clf_model = clf_model
        self.feature_function = feature_function
        self.seed = seed
        self.k_fold_cv = k_fold_cv
        self.TF_Learning = TF_Learning
        self.param_tuning = param_tuning
        self.pixel_size = pixel_size
        self.var = var
        self.parToTune = parToTune
        self.FN = FN
        self.PL_Number = PL_Number
        self.L_number = L_number
        self.sigma = sigma

        self.__dict__.update(kwargs)

    def __str__(self):
        """!
        @brief Nicely prints all currently set values in the ParameterBucket.
        """
        attrs = vars(self)
        output = ''
        output += 'Variables in parameter bucket\n'
        output += '---\n'
        for key in attrs.keys():
            output += str(key) + ' : '
            output += str(attrs[key]) + '\n'
        output += '---\n'
        return output

    def makeAdaptivePartition(self, DgmsPD, type='BirthDeath', meshingScheme='DV'):
        '''
        Combines all persistence diagrams in the series together, then generates an adaptive partition mesh and includes it in the parameter bucket as self.partitions

        The partitions can be viewed using self.partitions.plot()
        TODO: This can't handle infinite points in the diagram yet
        '''

        # TODO Deal with infinite points???
        try:
            AllDgms = []
            for label in DgmsPD.columns:
                DgmsSeries = DgmsPD[label]
                AllDgms.extend(list(DgmsSeries))

            AllPoints = np.concatenate(AllDgms)
        except:
            # you had a series to start with
            AllPoints = np.concatenate(list(DgmsPD))

        # Remove inifinite points

        AllPoints = pP.removeInfiniteClasses(AllPoints)

        x = AllPoints[:, 0]
        y = AllPoints[:, 1]
        if type == 'BirthDeath':
            life = y-x
        else:
            life = y
        fullData = np.column_stack((x, life))

        self.partitions = Partitions(
            data=fullData, meshingScheme=meshingScheme)

    def setBoundingBox(self, DgmsPD, pad=0):
        """!@brief Sets a bounding box in the birth-lifetime planeself.

        @param DgmsPD a pd.Series or pd.DataFrame with a persistence diagram in each entry.
        @param pad is the additional padding desired outside of the points in the diagrams.




        Sets
        `self.boundingBox`
        to be a dictionary with two keys, 'birthAxis' and 'lifetimeAxis', each outputing
        a tuple of length 2 so that all points in all diagrams (written in (birth,lifetime) coordinates) in the series are contained in the box `self.birthAxis X self.lifetimeAxis`.
        If `pad` is non-zero, the boundaries of the bounding box on all sides except the one touching the diagonal 		are at least `pad` distance away from the closest point.


        """
        if isinstance(DgmsPD, pd.Series):
            topPers = pP.maxPersistenceSeries(DgmsPD)
            bottomPers = pP.minPersistenceSeries(DgmsPD)
            topBirth = max(DgmsPD.apply(pP.maxBirth))
            bottomBirth = min(DgmsPD.apply(pP.minBirth))
        elif isinstance(DgmsPD, pd.DataFrame):  # Assumes that you handed me a dataframe
            topPers = []
            bottomPers = []
            topBirth = []
            bottomBirth = []
            for label in DgmsPD.columns:
                D = DgmsPD[label]
                topPers.append(pP.maxPersistenceSeries(D))
                bottomPers.append(pP.minPersistenceSeries(D))
                topBirth.append(max(D.apply(pP.maxBirth)))
                bottomBirth.append(min(D.apply(pP.minBirth)))
            topPers = max(topPers)
            bottomPers = min(bottomPers)
            topBirth = max(topBirth)
            bottomBirth = min(bottomBirth)
        else:
            print('You gave me a', type(DgmsPD))
            print(
                'This function requires a pandas Series or DataFrame full of persistence diagrams.')

        # print(topPers,bottomPers,topBirth,bottomBirth)

        self.boundingBox = {}
        self.boundingBox['birthAxis'] = (bottomBirth - pad, topBirth + pad)
        self.boundingBox['lifetimeAxis'] = (bottomPers/2, topPers + pad)

    def testEnclosesDgms(self, DgmSeries):
        '''!
        @brief Tests to see if the parameters enclose the persistence diagrams in the DgmSeries

        @returns boolean

        @todo Change this to work with self.boundingbox instead of d, delta, and epsilon
        '''

        # Height of parallelogram; equivalently maximum lifetime enclosed
        height = self.d * self.delta + self.epsilon
        width = self.d * self.delta

        minBirth = pP.minBirthSeries(DgmSeries)
        if minBirth < 0:
            print("This code assumes positive birth times.")
            return False

        maxBirth = pP.maxBirthSeries(DgmSeries)
        if maxBirth > width:
            print('There are birth times outside the bounding box.')
            return False

        minPers = pP.minPersistenceSeries(DgmSeries)
        if minPers < self.epsilon:
            print('There are points below the epsilon shift.')
            return False

        maxPers = pP.maxPersistenceSeries(DgmSeries)
        if maxPers > height:
            print('There are points above the box.')
            return False

        return True


# A new type of parameter ParameterBucket
#
# This is the type specially built for tents
class InterpPolyParameters(ParameterBucket):

    def __init__(self, d=3,
                 useAdaptivePart=False,
                 meshingScheme='DV',
                 jacobi_poly='cheb1',
                 clf_model=RidgeClassifierCV,
                 test_size=.33,
                 seed=None,
                 maxPower=1,
                 **kwargs
                 ):
        # Set all the necessary parameters for tents function
        # TODO

        self.feature_function = fF.interp_polynomial
        self.partitions = None
        self.jacobi_poly = jacobi_poly
        self.d = d
        self.useAdaptivePart = useAdaptivePart  # This should be boolean
        self.meshingScheme = meshingScheme
        self.clf_model = clf_model
        self.seed = seed
        self.test_size = test_size
        self.maxPower = maxPower
        self.__dict__.update(kwargs)

    def check(self):
        # Check for all the parameters required for tents function
        # TODO
        print("This hasn't been made yet. Ask me later.")
        pass


# A new type of parameter ParameterBucket
#
# This is the type specially built for tents
class TentParameters(ParameterBucket):

    def __init__(self, d=10, delta=1, epsilon=0,
                 clf_model=RidgeClassifierCV,
                 test_size=.33,
                 seed=None,
                 maxPower=1,
                 **kwargs):
        # Set all the necessary parameters for tents function
        self.feature_function = fF.tent
        self.useAdaptivePart = False

        self.d = d
        self.delta = delta
        self.epsilon = epsilon
        self.clf_model = clf_model
        self.seed = seed
        self.test_size = test_size
        self.maxPower = maxPower
        self.__dict__.update(kwargs)

    def check(self):
        # Check for all the parameters required for tents function
        # TODO
        print("This hasn't been made yet. Ask me later.")
        pass

    def chooseDeltaEpsWithPadding(self, DgmsPD, pad=0):
        """!@brief Sets delta and epsilon for tent function mesh

        @param DgmsSeries is pd.series consisting of persistence diagrams
        @param pad is the additional padding outside of the points in the diagrams

        This code assumes that self.d has been set.

        The result is to set self.delta\f$=\delta\f$ and self.epsilon\f$=\epsilon\f$ so that the bounding box for the persistence diagram in the (birth, lifetime) coordinates is
        \f[  [0,d \cdot \delta] \, \times \, [\epsilon, d \cdot \delta + \epsilon].  \f]
        In the usual coordinates, this creates a parallelogram.


        """
        if isinstance(DgmsPD, pd.DataFrame):
            AllDgms = []
            for label in DgmsPD.columns:
                DgmsSeries = DgmsPD[label]
                AllDgms.extend(list(DgmsSeries))

            DgmsSeries = pd.Series(AllDgms)

        elif isinstance(DgmsPD, pd.Series):
            DgmSeries = DgmsPD

        topPers = pP.maxPersistenceSeries(DgmsSeries)
        bottomPers = pP.minPersistenceSeries(DgmsSeries)
        topBirth = max(DgmsSeries.apply(pP.maxBirth))

        height = max(topPers, topBirth)

        bottomBirth = min(DgmsSeries.apply(pP.minBirth))

        if bottomBirth < 0:
            print(
                'This code assumes that birth time is always positive\nbut you have negative birth times....')
            print('Your minimum birth time was', bottomBirth)

        epsilon = bottomPers/2

        delta = (height + pad - epsilon) / self.d

        self.delta = delta
        self.epsilon = epsilon


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# --------ML on diagrams using featurization ------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


# Applies the passed featurization function to all diagrams in the series and outputs the feature matrix
# @param DgmSeries : pd.Series.
#	The structure holding the persistence diagrams.
# @param params : ParameterBucket.
# 	A parameter bucket used for calculations.
def build_G(DgmSeries, params):

    # if not hasattr(params,'partitions'):
    # 	print('You have to have a partition bucket set in the')
    # 	print('params.  I should probably tell you how to do')
    # 	print('this here.  Exiting...')
    # 	return

    def applyFeaturization(x): return params.feature_function(x, params=params)
    G = np.array(list(DgmSeries.apply(applyFeaturization)))

    # Include powers if necessary
    try:
        if params.maxPower > 1:
            poly = PolynomialFeatures(params.maxPower)
            G = poly.fit_transform(G)
    except:
        pass
    return G


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# --------ML on diagrams using featurization ------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

# Main function to run ML with featurization on persistence diagrams.
# Takes data frame DgmsDF and specified persistence diagram column labels,
# computes the G matrix using build_G.
# Does classification using labels from labels_col in the data frame.
# Returns trained model.
#
# 	@param DgmsDF
# 		A pandas data frame containing, at least, a column of
# 		diagrams and a column of labels
# 	@param labels_col
# 		A string.  The label for the column in DgmsDF containing
# 		the training labels.
# 	@param dgm_col
# 		The label(s) for the column containing the diagrams given as a string or list of strings.
# 	@param params
# 		A class of type ParameterBucket
# 		Should store:
# 			- **d**:
# 				An integer, the number of elements for griding up
# 				the x and y axis of the diagram.  Will result in
# 				d*(d+1) tent functions
# 			- **delta**, **epsilon**:
# 				Controls location and width of mesh elements for x and y axis of the
# 				diagram.
# 			- **clfClass**:
# 				The class which will be used for classification.  Currently tested
#				using `sklearn.RidgeClassifierCV` and `sklearn.RidgeCV`.
#
# @return
# 	The classifier object. Coefficients can be found from clf.coef_
def ML_via_featurization(DgmsDF,
                         labels_col='trainingLabel',
                         dgm_col='Dgm1',
                         params=TentParameters(),
                         normalize=False,
                         verbose=True
                         ):

    clf = params.clf_model()

    if verbose:
        print('Training estimator.')
        startTime = time.time()

    # check to see if only one column label was passed. If so, turn it into a list.
    if type(dgm_col) == str:
        dgm_col = [dgm_col]

    if verbose:
        print('Making G...')

    listOfG = []
    for dgmColLabel in dgm_col:
        G = build_G(DgmsDF[dgmColLabel], params)
        listOfG.append(G)

    G = np.concatenate(listOfG, axis=1)

    numFeatures = np.shape(G)[1]

    # Normalize G
    if normalize:
        G = scale(G)

    if verbose:
        print('Number of features used is', numFeatures, '...')

    clf.fit(G, list(DgmsDF[labels_col]))

    if verbose:
        print('Checking score on training set...')

    score = clf.score(G, list(DgmsDF[labels_col]))

    if verbose:
        print('Score on training set: ' + str(score) + '.\n')

    clf.trainingScore = score

    return clf


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ---------------Get percent score-----------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


# Main testing function for classification or regression methods.
# Does train/test split, creates classifier, and returns score on test.
#
# 	@param DgmsDF
# 		A pandas data frame containing, at least, a column of
# 		diagrams and a column of labels
# 	@param labels_col
# 		A string.  The label for the column in DgmsDF containing the training labels.
# 	@param dgm_col
# 		A string or list of strings giving the label for the column containing the diagrams.
# 	@param params
# 		A class of type ParameterBucket.
# 		Should store at least:
#			- **featureFunction**:
#				The function use for featurizing the persistence diagrams. Should take in
#				a diagram and a ParameterBucket and output a vector of real numbers as features.
# 			- **clfClass**:
# 				The model which will be used for classification.  Currently tested
#				using `sklearn.RidgeClassifierCV` and `sklearn.RidgeCV`.
#			- **seed**:
#				None if we don't want to mess with the seed for the train_test_split function. Else, pass integer.
#			- **test_split**:
#				The percentage of the data to be reserved for the test part of the train/test split.
#
# 	@return
#		Returned as a dictionary of entries:
# 		- **score**
# 			The percent correct when predicting on the test set.
# 		- **DgmsDF**
# 			The original data frame passed back with a column labeled
# 			'Prediction' added with the predictions gotten for the
# 			test set. Data points in the training set will have an
# 			entry of NaN
# 		- **clf**
# 			The fitted model
#
