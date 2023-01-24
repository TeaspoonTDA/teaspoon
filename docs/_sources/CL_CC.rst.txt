Carlsson Coordinates
--------------------

In this section, we provide examples for classification of persistence diagrams using Carlsson Coordinates
provided in :ref:`carlsson_coordinates`. 
In below example, user provide a set of persistence diagrams in a Pandas dataframe inluding the labels of each
persistence diagram. 
Then, classification parameters are selected and persistence diagrams are classified. 
In addition, user can choose the number of coordinates to be used in feature matrix generation. 

::

    >>> from teaspoon.ML.PD_Classification import getPercentScore
    >>> from teaspoon.ML import feature_functions as fF
    >>> from teaspoon.ML.Base import ParameterBucket
    >>> from teaspoon.MakeData.PointCloud import testSetManifolds
    >>> from sklearn.preprocessing import LabelEncoder
    >>> from sklearn.svm import SVC

    >>> # generate persistence diagrams
    >>> DgmsDF = testSetManifolds(numDgms=20, numPts=100)
    >>> labels_col='trainingLabel'
    >>> dgm_col='Dgm1'

    >>> # convert categorical labels into integers
    >>> label_encoder = LabelEncoder()
    >>> x = DgmsDF[labels_col]
    >>> y = label_encoder.fit_transform(x)
    >>> DgmsDF[labels_col] = y

    >>> # set classification parameters
    >>> params = ParameterBucket()
    >>> params.feature_function = fF.F_CCoordinates
    >>> params.k_fold_cv=5
    >>> params.FN =3
    >>> params.clf_model = SVC
    >>> c_report_train,c_report_test=getPercentScore(DgmsDF,
    >>>                                             labels_col='trainingLabel',
    >>>                                             dgm_col='Dgm1',
    >>>                                             params=params,
    >>>                                             precomputed = False,
    >>>                                             saving = False,
    >>>                                             saving_path = None)

    Beginning experiments

    Run Number: 1
    Test set acc.: 0.625 
    Training set acc.: 0.698
    ------------------------------
    Run Number: 2
    Test set acc.: 0.583 
    Training set acc.: 0.677
    ------------------------------
    Run Number: 3
    Test set acc.: 0.542 
    Training set acc.: 0.656
    ------------------------------
    Run Number: 4
    Test set acc.: 0.667 
    Training set acc.: 0.667
    ------------------------------
    Run Number: 5
    Test set acc.: 0.583 
    Training set acc.: 0.688
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.600
    Standard deviation: 0.042

    Training Set 
    ---------
    Average accuracy: 0.677
    Standard deviation: 0.015

    For more metrics, see the outputs.

Transfer learning between two sets of persistence diagrams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Machine learning module of teaspoon also provides user with transfer learning option.
When it is enabled, user can train and test a classifier on two different sets of persistence diagrams.
In this example, we first generate two sets of persistence diagrams.
Categorical labels of the diagrams are converted into the integers.
In the last step, we set classification parameters and perform the classification using SVM.

::

    >>> from teaspoon.ML.PD_Classification import getPercentScore
    >>> from teaspoon.ML import feature_functions as fF
    >>> from teaspoon.ML.Base import ParameterBucket
    >>> from teaspoon.MakeData.PointCloud import testSetManifolds
    >>> from sklearn.preprocessing import LabelEncoder
    >>> from sklearn.svm import SVC
    >>> import numpy as np

    >>> # generate persistence diagrams
    >>> DgmsDF_train = testSetManifolds(numDgms=20, numPts=100)
    >>> DgmsDF_test = testSetManifolds(numDgms=20, numPts=100)

    >>> labels_col='trainingLabel'
    >>> dgm_col='Dgm1'

    >>> # convert categorical labels into integers
    >>> label_encoder = LabelEncoder()
    >>> x_train,x_test = DgmsDF_train[labels_col],DgmsDF_test[labels_col]
    >>> y_train = label_encoder.fit_transform(x_train)
    >>> y_test = label_encoder.fit_transform(x_test)
    >>> DgmsDF_train[labels_col],DgmsDF_test[labels_col] = y_train,y_test

    >>> # set classification parameters
    >>> params = ParameterBucket()
    >>> params.feature_function = fF.F_CCoordinates
    >>> params.k_fold_cv=5
    >>> params.TF_Learning=True
    >>> params.FN = 5
    >>> params.clf_model = SVC
    >>> c_report_train,c_report_test=getPercentScore(DgmsDF_train,
    >>>                                             labels_col='trainingLabel',
    >>>                                             dgm_col='Dgm1',
    >>>                                             params=params,
    >>>                                             precomputed = False,
    >>>                                             saving = False,
    >>>                                             saving_path = None,
    >>>                                             DgmsDF_test = DgmsDF_test)    

    Beginning experiments

    Run Number: 1
    Test set acc.: 0.688 
    Training set acc.: 0.708
    ------------------------------
    Run Number: 2
    Test set acc.: 0.708 
    Training set acc.: 0.719
    ------------------------------
    Run Number: 3
    Test set acc.: 0.656 
    Training set acc.: 0.708
    ------------------------------
    Run Number: 4
    Test set acc.: 0.771 
    Training set acc.: 0.708
    ------------------------------
    Run Number: 5
    Test set acc.: 0.667 
    Training set acc.: 0.729
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.698
    Standard deviation: 0.041

    Training Set 
    ---------
    Average accuracy: 0.715
    Standard deviation: 0.008

    For more metrics, see the outputs.

Hyperparameter tuning 
~~~~~~~~~~~~~~~~~~~~~
Our package also provides user with hyperparameter tuning. 
When it is enabled, user is expected to provide the parameters and their range in a dictionary to tune parameters.
Algorithm implements `GridSearchCV
<https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_. 

::

    >>> import numpy as np
    >>> from teaspoon.ML.PD_Classification import getPercentScore
    >>> from teaspoon.ML import feature_functions as fF
    >>> from teaspoon.ML.Base import ParameterBucket
    >>> from teaspoon.MakeData.PointCloud import testSetManifolds
    >>> from sklearn.preprocessing import LabelEncoder
    >>> from sklearn.svm import SVC

    >>> # generate persistence diagrams
    >>> DgmsDF = testSetManifolds(numDgms=20, numPts=100)
    >>> labels_col='trainingLabel'
    >>> dgm_col='Dgm1'

    >>> # convert categorical labels into integers
    >>> label_encoder = LabelEncoder()
    >>> x = DgmsDF[labels_col]
    >>> y = label_encoder.fit_transform(x)
    >>> DgmsDF[labels_col] = y

    >>> # set classification parameters
    >>> params = ParameterBucket()
    >>> params.feature_function = fF.F_CCoordinates
    >>> params.k_fold_cv=5
    >>> params.FN =3
    >>> params.clf_model = SVC
    >>> params.param_tuning = True

    >>> # parameters to tune and their range
    >>> gamma_range = np.logspace(-3, 3, num=5)
    >>> lambda_range = np.logspace(-3, 3, num=5)
    >>> params.parToTune = [] # the list that contains the parameters to tune for each classifier
    >>> params.parToTune.append({'C': lambda_range,'gamma':gamma_range}) # SVM paramters

    >>> #perform classification
    >>> c_report_train,c_report_test=getPercentScore(DgmsDF,
    >>>                                             labels_col='trainingLabel',
    >>>                                             dgm_col='Dgm1',
    >>>                                             params=params,
    >>>                                             precomputed = False,
    >>>                                             saving = False,
    >>>                                             saving_path = None) 

    Beginning experiments

    Run Number: 1
    Test set acc.: 0.750 
    Training set acc.: 0.844
    ------------------------------
    Run Number: 2
    Test set acc.: 0.750 
    Training set acc.: 0.844
    ------------------------------
    Run Number: 3
    Test set acc.: 0.750 
    Training set acc.: 0.854
    ------------------------------
    Run Number: 4
    Test set acc.: 0.875 
    Training set acc.: 0.823
    ------------------------------
    Run Number: 5
    Test set acc.: 0.792 
    Training set acc.: 0.844
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.783
    Standard deviation: 0.049

    Training Set 
    ---------
    Average accuracy: 0.842
    Standard deviation: 0.010

    For more metrics, see the outputs.    




