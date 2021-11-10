"""

This file provides examples for classification using featurization methods 
available in machine learning moduel of teaspoon.

"""
# %%  -------------- Persistence Landscapes -------------------------------


"""
This example includes classification of persistence diagrams using persistence 
landscapes. In this example, landscapes are not precomputed. They are computed in
during classification.
"""

#########---------------Diagrams to Landsapes ---------------------------#####

from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# generate persistence diagrams
DgmsDF = testSetManifolds(numDgms=20, numPts=100)
labels_col='trainingLabel'
dgm_col='Dgm1'

# convert categorical labels into integers
label_encoder = LabelEncoder()
x = DgmsDF[labels_col]
y = label_encoder.fit_transform(x)
DgmsDF[labels_col] = y

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.F_Landscape
params.PL_Number = [1,2]
params.k_fold_cv=5
params.clf_model = SVC
c_report_train,c_report_test=getPercentScore(DgmsDF,
                                            labels_col='trainingLabel',
                                            dgm_col='Dgm1',
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None)

#%% parameter turning for persistence landscapes
import numpy as np
from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# generate persistence diagrams
DgmsDF = testSetManifolds(numDgms=20, numPts=100)
labels_col='trainingLabel'
dgm_col='Dgm1'

# convert categorical labels into integers
label_encoder = LabelEncoder()
x = DgmsDF[labels_col]
y = label_encoder.fit_transform(x)
DgmsDF[labels_col] = y

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.F_Landscape
params.PL_Number = [1,2]
params.k_fold_cv=5
params.clf_model = SVC
params.param_tuning = True

gamma_range = np.logspace(-3, 3, num=10)
lambda_range = np.logspace(-3, 3, num=10)
params.parToTune = [] # the list that contains the paramters to tune for each classifier
params.parToTune.append({'C': lambda_range, 'kernel': ('rbf','sigmoid'),'gamma':gamma_range}) # SVM paramters

c_report_train,c_report_test=getPercentScore(DgmsDF,
                                            labels_col='trainingLabel',
                                            dgm_col='Dgm1',
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None)


#%%#########---------------Precomputed Landscapes---------------------------#####
"""
If user provides the precomputed persistence landscapes, "precomputed" parameter
needs to be set to True so that algorithm will treat the given inputs as persistence landscapes.
"""
from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# generate persistence diagrams
DgmsDF = testSetManifolds(numDgms=20, numPts=100)
labels_col='trainingLabel'
dgm_col='Dgm1'

# convert categorical labels into integers
label_encoder = LabelEncoder()
x = DgmsDF[labels_col]
y = label_encoder.fit_transform(x)
DgmsDF[labels_col] = y


# compute the persistence diagrams for given diagrams
PL = np.ndarray(shape=(len(DgmsDF)), dtype=object)

# compute persistence landscape for training set 
for i in range(len(DgmsDF)):
    PLs = fF.PLandscape(DgmsDF[dgm_col][i])
    PL[i] = PLs.AllPL
    
# conver the landscapes into dataframe to be consistent with data structure in the classification algorithm
PL = pd.DataFrame(PL)
PL[labels_col] = DgmsDF[labels_col]
PL = PL.rename(columns={0: "PerLand"})

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.F_Landscape
params.PL_Number = [1]
params.k_fold_cv=5
params.clf_model = SVC
c_report_train,c_report_test=getPercentScore(PL,
                                            labels_col='trainingLabel',
                                            dgm_col="PerLand",
                                            params=params,
                                            precomputed = True,
                                            saving = False,
                                            saving_path = None)

#%%########-----------Diagrams to Landsapes (Transfer Learning)----------#####

"""
This example includes classification of persistence diagrams using persistence 
landscapes. In this example, landscapes are not precomputed. They are computed in
during classification. Two sets of persistence diagrams are provided by user and 
we apply transfer learning between these two sets.

"""

from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# generate persistence diagrams
DgmsDF_train = testSetManifolds(numDgms=20, numPts=100)
DgmsDF_test = testSetManifolds(numDgms=20, numPts=100)

labels_col='trainingLabel'
dgm_col='Dgm1'

# convert categorical labels into integers
label_encoder = LabelEncoder()
x_train,x_test = DgmsDF_train[labels_col],DgmsDF_test[labels_col]
y_train = label_encoder.fit_transform(x_train)
y_test = label_encoder.fit_transform(x_test)
DgmsDF_train[labels_col],DgmsDF_test[labels_col] = y_train,y_test




# set classification parameters
params = ParameterBucket()
params.feature_function = fF.F_Landscape
params.PL_Number = [1]
params.k_fold_cv=5
params.clf_model = SVC
params.TF_Learning=True
c_report_train,c_report_test=getPercentScore(DgmsDF_train,
                                            labels_col='trainingLabel',
                                            dgm_col='Dgm1',
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None,
                                            DgmsDF_test = DgmsDF_test)



#%%########-----Precomputed Landscapes (Transfer Learning)---------------#####

from ML.PD_Classification import getPercentScore
from ML import feature_functions as fF
from ML.Base import ParameterBucket
from MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# generate persistence diagrams
DgmsDF_train = testSetManifolds(numDgms=20, numPts=100)
DgmsDF_test = testSetManifolds(numDgms=20, numPts=100)

labels_col='trainingLabel'
dgm_col='Dgm1'

# convert categorical labels into integers
label_encoder = LabelEncoder()
x_train,x_test = DgmsDF_train[labels_col],DgmsDF_test[labels_col]
y_train = label_encoder.fit_transform(x_train)
y_test = label_encoder.fit_transform(x_test)
DgmsDF_train[labels_col],DgmsDF_test[labels_col] = y_train,y_test


# compute the persistence diagrams for given diagrams
PL_train = np.ndarray(shape=(len(DgmsDF_train)), dtype=object)
PL_test = np.ndarray(shape=(len(DgmsDF_test)), dtype=object)

# compute persistence landscape for training set 
for i in range(len(DgmsDF_train)):
    PLs = fF.PLandscape(DgmsDF_train[dgm_col][i])
    PL_train[i] = PLs.AllPL
# compute persistence landscape for training set 
for i in range(len(DgmsDF_test)):
    PLs = fF.PLandscape(DgmsDF_test[dgm_col][i])
    PL_test[i] = PLs.AllPL
    
# convert the landscapes into dataframe to be consistent with data structure in the classification algorithm
PL_train, PL_test = pd.DataFrame(PL_train),pd.DataFrame(PL_test)
PL_train[labels_col], PL_test[labels_col] = DgmsDF_train[labels_col], DgmsDF_test[labels_col]
PL_train = PL_train.rename(columns={0: "PerLand"})
PL_test = PL_test.rename(columns={0: "PerLand"})

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.F_Landscape
params.PL_Number = [1]
params.k_fold_cv=5
params.clf_model = SVC
params.TF_Learning = True
c_report_train,c_report_test=getPercentScore(PL_train,
                                            labels_col='trainingLabel',
                                            dgm_col='PerLand',
                                            params=params,
                                            precomputed = True,
                                            saving = False,
                                            saving_path = None,
                                            DgmsDF_test=PL_test)

# if user wants to tune parameters for the selected classifier
params.param_tuning = True
gamma_range = np.logspace(-3, 3, num=10)
lambda_range = np.logspace(-3, 3, num=10)
params.parToTune = [] # the list that contains the paramters to tune for each classifier
params.parToTune.append({'C': lambda_range, 'kernel': ('rbf','sigmoid'),'gamma':gamma_range}) # SVM paramters
c_report_train,c_report_test=getPercentScore(PL_train,
                                            labels_col='trainingLabel',
                                            dgm_col='PerLand',
                                            precomputed = True,
                                            params=params,
                                            saving_path = None,
                                            DgmsDF_test=PL_test)




# %%  -------------- Persistence Images ------------------
from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# generate persistence diagrams
DgmsDF = testSetManifolds(numDgms=20, numPts=100)
labels_col='trainingLabel'
dgm_col='Dgm1'

# convert categorical labels into integers
label_encoder = LabelEncoder()
x = DgmsDF[labels_col]
y = label_encoder.fit_transform(x)
DgmsDF[labels_col] = y

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.F_Image
params.k_fold_cv=5
params.clf_model = SVC
c_report_train,c_report_test=getPercentScore(DgmsDF,
                                            labels_col='trainingLabel',
                                            dgm_col='Dgm1',
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None)

#%%-------------- Persistence Images (Transfer learning) ------------------
from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np

# generate persistence diagrams
DgmsDF_train = testSetManifolds(numDgms=20, numPts=100)
DgmsDF_test = testSetManifolds(numDgms=20, numPts=100)

labels_col='trainingLabel'
dgm_col='Dgm1'

# convert categorical labels into integers
label_encoder = LabelEncoder()
x_train,x_test = DgmsDF_train[labels_col],DgmsDF_test[labels_col]
y_train = label_encoder.fit_transform(x_train)
y_test = label_encoder.fit_transform(x_test)
DgmsDF_train[labels_col],DgmsDF_test[labels_col] = y_train,y_test

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.F_Image
params.k_fold_cv=5
params.clf_model = SVC
params.TF_Learning = True
c_report_train,c_report_test=getPercentScore(DgmsDF_train,
                                            labels_col='trainingLabel',
                                            dgm_col='Dgm1',
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None,
                                            DgmsDF_test = DgmsDF_test)

#%%  if user wants to tune parameters for the selected classifier
import numpy as np
from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# generate persistence diagrams
DgmsDF = testSetManifolds(numDgms=20, numPts=100)
labels_col='trainingLabel'
dgm_col='Dgm1'

# convert categorical labels into integers
label_encoder = LabelEncoder()
x = DgmsDF[labels_col]
y = label_encoder.fit_transform(x)
DgmsDF[labels_col] = y

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.F_Image
params.k_fold_cv=5
params.clf_model = SVC
params.param_tuning = True

# parameters to tune and their ranges
gamma_range = np.logspace(-3, 3, num=5)
lambda_range = np.logspace(-3, 3, num=5)
params.parToTune = [] # the list that contains the paramters to tune for each classifier
params.parToTune.append({'C': lambda_range,'gamma':gamma_range}) # SVM paramters

# perform classification
c_report_train,c_report_test=getPercentScore(DgmsDF,
                                            labels_col='trainingLabel',
                                            dgm_col='Dgm1',
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None)


# %%  -------------- Carlsson Coordinates -------------------------------
from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# generate persistence diagrams
DgmsDF = testSetManifolds(numDgms=20, numPts=100)
labels_col='trainingLabel'
dgm_col='Dgm1'

# convert categorical labels into integers
label_encoder = LabelEncoder()
x = DgmsDF[labels_col]
y = label_encoder.fit_transform(x)
DgmsDF[labels_col] = y

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.F_CCoordinates
params.k_fold_cv=5
params.FN =3
params.clf_model = SVC
c_report_train,c_report_test=getPercentScore(DgmsDF,
                                            labels_col='trainingLabel',
                                            dgm_col='Dgm1',
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None)


#%%  -------------- Carlsson Coordinates (TF_Learning) --------------------------
from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np

# generate persistence diagrams
DgmsDF_train = testSetManifolds(numDgms=20, numPts=100)
DgmsDF_test = testSetManifolds(numDgms=20, numPts=100)

labels_col='trainingLabel'
dgm_col='Dgm1'

# convert categorical labels into integers
label_encoder = LabelEncoder()
x_train,x_test = DgmsDF_train[labels_col],DgmsDF_test[labels_col]
y_train = label_encoder.fit_transform(x_train)
y_test = label_encoder.fit_transform(x_test)
DgmsDF_train[labels_col],DgmsDF_test[labels_col] = y_train,y_test

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.F_CCoordinates
params.k_fold_cv=5
params.TF_Learning=True
params.FN = 5
params.clf_model = SVC
c_report_train,c_report_test=getPercentScore(DgmsDF_train,
                                            labels_col='trainingLabel',
                                            dgm_col='Dgm1',
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None,
                                            DgmsDF_test = DgmsDF_test)



#%% if user wants to tune parameters for the selected classifier
import numpy as np
from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# generate persistence diagrams
DgmsDF = testSetManifolds(numDgms=20, numPts=100)
labels_col='trainingLabel'
dgm_col='Dgm1'

# convert categorical labels into integers
label_encoder = LabelEncoder()
x = DgmsDF[labels_col]
y = label_encoder.fit_transform(x)
DgmsDF[labels_col] = y

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.F_CCoordinates
params.k_fold_cv=5
params.FN =3
params.clf_model = SVC
params.param_tuning = True

# parameters to tune and their range
gamma_range = np.logspace(-3, 3, num=5)
lambda_range = np.logspace(-3, 3, num=5)
params.parToTune = [] # the list that contains the paramters to tune for each classifier
params.parToTune.append({'C': lambda_range,'gamma':gamma_range}) # SVM paramters

#perform classification
c_report_train,c_report_test=getPercentScore(DgmsDF,
                                            labels_col='trainingLabel',
                                            dgm_col='Dgm1',
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None)



#%% Path Signatures

from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# generate persistence diagrams
DgmsDF = testSetManifolds(numDgms=2, numPts=100)
labels_col='trainingLabel'
dgm_col='Dgm1'

# convert categorical labels into integers
label_encoder = LabelEncoder()
x = DgmsDF[labels_col]
y = label_encoder.fit_transform(x)
DgmsDF[labels_col] = y

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.F_PSignature
params.k_fold_cv=2
params.L_number = [1]
params.clf_model = SVC
c_report_train,c_report_test=getPercentScore(DgmsDF,
                                            labels_col='trainingLabel',
                                            dgm_col='Dgm1',
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None)



# %%  -------------- Kernel Method  -------------------------------

from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# generate persistence diagrams
DgmsDF = testSetManifolds(numDgms=5, numPts=100)
labels_col='trainingLabel'
dgm_col=['Dgm1']

# convert categorical labels into integers
label_encoder = LabelEncoder()
x = DgmsDF[labels_col]
y = label_encoder.fit_transform(x)
DgmsDF[labels_col] = y

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.KernelMethod
params.k_fold_cv=5
params.sigma = 0.25
params.clf_model = SVC
c_report_train,c_report_test=getPercentScore(DgmsDF,
                                            labels_col='trainingLabel',
                                            dgm_col='Dgm1',
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None)



#%% Template Functions

from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# generate persistence diagrams
DgmsDF = testSetManifolds(numDgms=20, numPts=100)
labels_col='trainingLabel'
dgm_col=['Dgm0','Dgm1']

# convert categorical labels into integers
label_encoder = LabelEncoder()
x = DgmsDF[labels_col]
y = label_encoder.fit_transform(x)
DgmsDF[labels_col] = y

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.interp_polynomial
params.k_fold_cv=5
params.d = 20
params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial
params.useAdaptivePart = False
params.clf_model = SVC
params.TF_Learning = False

# perform classification
c_report_train,c_report_test=getPercentScore(DgmsDF,
                                            labels_col='trainingLabel',
                                            dgm_col=dgm_col,
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None)


#%% template functions (transfer learning)

from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np

# generate persistence diagrams
DgmsDF_train = testSetManifolds(numDgms=20, numPts=100)
DgmsDF_test = testSetManifolds(numDgms=20, numPts=100)

labels_col='trainingLabel'
dgm_col=['Dgm0']

# convert categorical labels into integers
label_encoder = LabelEncoder()
x_train,x_test = DgmsDF_train[labels_col],DgmsDF_test[labels_col]
y_train = label_encoder.fit_transform(x_train)
y_test = label_encoder.fit_transform(x_test)
DgmsDF_train[labels_col],DgmsDF_test[labels_col] = y_train,y_test


# set classification parameters
params = ParameterBucket()
params.feature_function = fF.interp_polynomial
params.k_fold_cv=5
params.d = 20
params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial
params.useAdaptivePart = False
params.clf_model = SVC
params.TF_Learning = True

# perform classification
c_report_train,c_report_test=getPercentScore(DgmsDF_train,
                                            labels_col='trainingLabel',
                                            dgm_col=dgm_col,
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None,
                                            DgmsDF_test = DgmsDF_test)


#%% parameter tuning using persistence images
import numpy as np
from teaspoon.ML.PD_Classification import getPercentScore
from teaspoon.ML import feature_functions as fF
from teaspoon.ML.Base import ParameterBucket
from teaspoon.MakeData.PointCloud import testSetManifolds
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# generate persistence diagrams
DgmsDF = testSetManifolds(numDgms=20, numPts=100)
labels_col='trainingLabel'
dgm_col=['Dgm1']

# convert categorical labels into integers
label_encoder = LabelEncoder()
x = DgmsDF[labels_col]
y = label_encoder.fit_transform(x)
DgmsDF[labels_col] = y

# set classification parameters
params = ParameterBucket()
params.feature_function = fF.interp_polynomial
params.k_fold_cv=5
params.d = 20
params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial
params.useAdaptivePart = False
params.clf_model = SVC
params.TF_Learning = False
params.param_tuning = True

# parameters to tune and their ranges
gamma_range = np.logspace(-3, 3, num=5)
lambda_range = np.logspace(-3, 3, num=5)
params.parToTune = [] # the list that contains the paramters to tune for each classifier
params.parToTune.append({'C': lambda_range,'gamma':gamma_range}) # SVM paramters

# perform classification
c_report_train,c_report_test=getPercentScore(DgmsDF,
                                            labels_col='trainingLabel',
                                            dgm_col='Dgm1',
                                            params=params,
                                            precomputed = False,
                                            saving = False,
                                            saving_path = None)

