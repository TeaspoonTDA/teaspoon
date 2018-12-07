## @package teaspoon.ML.Base
# Basic code used for doing ML on persistence diagrams
# by featurization.  Includes... @todo More description
#


from teaspoon.Misc import printPrettyTime
from teaspoon.ML import feature_functions as fF
from teaspoon.TDA import Persistence as pP
from teaspoon.TSP.adaptivePart import Partitions

import time
import numpy as np
import pandas as pd


from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, RidgeClassifierCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.preprocessing import scale, PolynomialFeatures
from scipy.special import comb
import itertools


class ParameterBucket(object):
	def __init__(self, description = '',
					clf_model = RidgeClassifierCV,
					feature_function = fF.tent,
					test_size = .33,
					seed = None,
					**kwargs):
					# d=10,
					# delta = 1,
					# epsilon = 0,
					# maxPower = 1,
					#
					# feature_function=None,
					# boundingBoxMatrix = None):
		"""!@brief Creates a new ParameterBucket object.

		This object is being used to keep track of all the parameters needed
		for the tents ML featurization.

		Parameters that are included in the ParameterBucket initially:

		@param description
			A description, has no effect on code. This can be set on initialization.
		@param clf_model
			The choice of tool used for classification or regression, passed as the function.  This code has been tested using `sklearn` functions `RidgeClassiferCV` for classification and `RidgeCV` for regression.
		@param feature_function
			The function you want to use for featurization.  This should be a function that takes as inputs a diagram and a ParameterBucket, and returns a vector of features. The default is ML.feature_functions.tents()
		@param test_size
			A number in \f$[0,1]\f$.  Gives the percentage of data points to be reserved for the testing set if this is being used for a train/test split experiment.  Otherwise, ignored.
		@param seed
			The seed for the pseudo-random number generator.  Pass None if you don't want it fixed; otherwise, pass an integer.
		@param kwargs
			Any leftover inputs are stored as attributes. Some common attributes used elsewhere are `d`, `delta`, and `epsilon` to describe the mesh. If its set, `boundingbox` keeps track of a box which encloses all points in all diagrams in a particular series; see setBoundingBox().

		"""


		# Parameters that used to be included. Documentation left here to figure out stuff for later.
		#
		# @param d, delta, epsilon
		#     The bounding box for the persistence diagram in the (birth, lifetime) coordinates is [0,d * delta] x [epsilon, d* delta + epsilon].  In the usual coordinates, this creates a parallelogram.
		# @param maxPower
		#     The maximum degree used for the monomial combinations of the tent functions.  Testing suggests we usually want this to be 1.  Increasing causes large increase in number of features.
		# @param boundingBoxMatrix
		# 	Not yet implemented.  See self.findBoundingBox()

		self.description = description
		self.clf_model = clf_model
		self.seed = seed
		self.test_size = test_size
		self.feature_function = feature_function
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
			output += str(attrs[key])+ '\n'
		output += '---\n'
		return output


	def makeAdaptivePartition(self, DgmsPD, type = 'BirthDeath',meshingScheme = 'DV'):
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

		x = AllPoints[:,0]
		y = AllPoints[:,1]
		if type == 'BirthDeath':
			life = y-x
		else:
			life = y
		fullData = np.column_stack((x,life))

		self.partitions = Partitions(data = fullData, meshingScheme = meshingScheme)

	def setBoundingBox(self,DgmsPD,pad = 0):
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
		elif isinstance(DgmsPD, pd.DataFrame): #Assumes that you handed me a dataframe
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
			print('This function requires a pandas Series or DataFrame full of persistence diagrams.')

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
		if minBirth <0:
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



## A new type of parameter ParameterBucket
#
# This is the type specially built for tents
class InterpPolyParameters(ParameterBucket):

	def __init__(self, d = 3,
					useAdaptivePart = False,
					meshingScheme = 'DV',
					jacobi_poly = 'cheb1',
				clf_model = RidgeClassifierCV,
				test_size = .33,
				seed = None,
				maxPower = 1,
				**kwargs
					):
		# Set all the necessary parameters for tents function
		# TODO

		self.feature_function = fF.interp_polynomial
		self.partitions = None
		self.jacobi_poly = jacobi_poly
		self.d = d
		self.useAdaptivePart = useAdaptivePart #This should be boolean
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




## A new type of parameter ParameterBucket
#
# This is the type specially built for tents
class TentParameters(ParameterBucket):

	def __init__(self, d = 10, delta = 1, epsilon = 0,
				clf_model = RidgeClassifierCV,
				test_size = .33,
				seed = None,
				maxPower = 1,
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





	def chooseDeltaEpsWithPadding(self, DgmsPD, pad = 0):
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

		height = max(topPers,topBirth)

		bottomBirth = min(DgmsSeries.apply(pP.minBirth))


		if bottomBirth < 0:
			print('This code assumes that birth time is always positive\nbut you have negative birth times....')
			print('Your minimum birth time was', bottomBirth)

		epsilon = bottomPers/2

		delta = (height + pad - epsilon) / self.d


		self.delta = delta
		self.epsilon = epsilon



#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#--------ML on diagrams using featurization ------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------


## Applies the passed featurization function to all diagrams in the series and outputs the feature matrix
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

	applyFeaturization = lambda x: params.feature_function(x,params = params)
	G = np.array(list(DgmSeries.apply(applyFeaturization )))

	# Include powers if necessary
	try:
		if params.maxPower>1:
			poly = PolynomialFeatures(params.maxPower)
			G = poly.fit_transform(G)
	except:
		pass
	return G


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#--------ML on diagrams using featurization ------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

## Main function to run ML with featurization on persistence diagrams.
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
			labels_col = 'trainingLabel',
			dgm_col = 'Dgm1',
			params = TentParameters(),
			normalize = False,
			verbose = True
			):


	clf = params.clf_model()

	if verbose:
		print('Training estimator.')
		startTime = time.time()

	#check to see if only one column label was passed. If so, turn it into a list.
	if type(dgm_col) == str:
		dgm_col = [dgm_col]

	if verbose:
		print('Making G...')

	listOfG = []
	for dgmColLabel in dgm_col:
		G = build_G(DgmsDF[dgmColLabel],params)
		listOfG.append(G)

	G = np.concatenate(listOfG,axis = 1)

	numFeatures = np.shape(G)[1]

	# Normalize G
	if normalize:
		G = scale(G)


	if verbose:
		print('Number of features used is', numFeatures,'...')

	clf.fit(G,list(DgmsDF[labels_col]))

	if verbose:
		print('Checking score on training set...')

	score = clf.score(G,list(DgmsDF[labels_col]))

	if verbose:
		print('Score on training set: ' + str(score) + '.\n')

	clf.trainingScore = score


	return clf




#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#---------------Get percent score-----------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------




## Main testing function for classification or regression methods.
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
def getPercentScore(DgmsDF,
					labels_col = 'trainingLabel',
					dgm_col = 'Dgm1',
					params = TentParameters(),
					normalize = False,
					verbose = True
					):

	if verbose:
		print('---')
		print('Beginning experiment.')
		print(params)


	#check to see if only one column label was passed. If so, turn it into a list.
	if type(dgm_col) == str:
		dgm_col = [dgm_col]

	# Run actual train/test experiment using sklearn
	D_train, D_test, L_train,L_test = train_test_split(DgmsDF,
													DgmsDF[labels_col],
													test_size=params.test_size,
													random_state = params.seed
													)

	# Get the portions of the test data frame with diagrams and concatenate into giant series:
	allDgms = pd.concat((D_train[label] for label in dgm_col))

	if params.useAdaptivePart == True:
		# Hand the series to the makeAdaptivePartition function
		params.makeAdaptivePartition(allDgms, meshingScheme = 'DV')
	else:
		# TODO this should work for both interp and tents but doesn't yet
		params.makeAdaptivePartition(allDgms, meshingScheme = None)

	#--------Training------------#
	if verbose:
		print('Using ' + str(len(L_train)) + '/' + str(len(DgmsDF)) + ' to train...')
	clf = ML_via_featurization(D_train,
					labels_col = labels_col,
					dgm_col = dgm_col,
					params = params,
					normalize = normalize,
					verbose = verbose)

	#--------Testing-------------#
	if verbose:
		print('Using ' + str(len(L_test)) + '/' + str(len(DgmsDF)) + ' to test...')
	listOfG = []
	for dgmColLabel in dgm_col:
		G = build_G(D_test[dgmColLabel],params)
		listOfG.append(G)

	G = np.concatenate(listOfG,axis = 1)

	# Normalize G
	if normalize:
		G = scale(G)


	# Compute predictions and add to DgmsDF data frame
	L_predict = pd.Series(clf.predict(G),index = L_test.index)
	DgmsDF['Prediction'] = L_predict

	# Compute score
	score = clf.score(G,list(L_test))
	if verbose:
		print('Score on testing set: ' + str(score) +"...\n")

		print('Finished with train/test experiment.')

	output = {}
	output['score'] = score
	output['DgmsDF'] = DgmsDF
	output['clf'] = clf

	return output
