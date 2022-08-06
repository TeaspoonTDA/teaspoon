Persistence Images
------------------
In this section, we provide examples for classification using persistence images.
As mentioned in :ref:`persistence_images` section, we use `Persim
<https://persim.scikit-tda.org/en/latest/notebooks/Classification%20with%20persistence%20images.html>`_ library
to compute persistence images. 
This library can automatically capture the bounds of the image. Our algorithm implements that option, so user
do not need to give boundaries for life time and birth time of the images.

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
    >>> params.feature_function = fF.F_Image
    >>> params.k_fold_cv=5
    >>> params.clf_model = SVC

    >>> # perform classification
    >>> c_report_train,c_report_test=getPercentScore(DgmsDF,
    >>>                                             labels_col='trainingLabel',
    >>>                                             dgm_col='Dgm1',
    >>>                                             params=params,
    >>>                                             precomputed = False,
    >>>                                             saving = False,
    >>>                                             saving_path = None)
    Beginning experiments

    Run Number: 1
    Test set acc.: 0.583 
    Training set acc.: 0.688
    ------------------------------
    Run Number: 2
    Test set acc.: 0.750 
    Training set acc.: 0.823
    ------------------------------
    Run Number: 3
    Test set acc.: 0.708 
    Training set acc.: 0.771
    ------------------------------
    Run Number: 4
    Test set acc.: 0.583 
    Training set acc.: 0.688
    ------------------------------
    Run Number: 5
    Test set acc.: 0.625 
    Training set acc.: 0.708
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.650
    Standard deviation: 0.068

    Training Set 
    ---------
    Average accuracy: 0.735
    Standard deviation: 0.053

    For more metrics, see the outputs.

Transfer learning between two sets of persistence diagrams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

User can enable transfer learning option to train and test a classifier using two different sets of 
persistence diagrams. The first diagram is treated as the training set, while the second diagram is considered
as test set. 
For given two sets of diagrams, algorithm computes their images and generates feature matrices.
Then, supervised classification is performed with respect defined parameters.

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
    >>> params.feature_function = fF.F_Image
    >>> params.k_fold_cv=5
    >>> params.clf_model = SVC
    >>> params.TF_Learning = True
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
    Test set acc.: 0.656 
    Training set acc.: 0.667
    ------------------------------
    Run Number: 2
    Test set acc.: 0.719 
    Training set acc.: 0.719
    ------------------------------
    Run Number: 3
    Test set acc.: 0.833 
    Training set acc.: 0.844
    ------------------------------
    Run Number: 4
    Test set acc.: 0.750 
    Training set acc.: 0.771
    ------------------------------
    Run Number: 5
    Test set acc.: 0.812 
    Training set acc.: 0.844
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.754
    Standard deviation: 0.064

    Training Set 
    ---------
    Average accuracy: 0.769
    Standard deviation: 0.070

    For more metrics, see the outputs. 

Hyperparameter tuning
~~~~~~~~~~~~~~~~~~~~~

Our package also provides user with hyperparameter tuning. 
When it is enabled, user is expected to provide the parameters and their range in a dictionary to tune parameters.
Algorithm implements `GridSearchCV
<https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_. 

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
    >>> params.feature_function = fF.F_Image
    >>> params.k_fold_cv=5
    >>> params.clf_model = SVC
    >>> params.param_tuning = True

    >>> # parameters to tune and their ranges
    >>> gamma_range = np.logspace(-3, 3, num=5)
    >>> lambda_range = np.logspace(-3, 3, num=5)
    >>> params.parToTune = [] # the list that contains the parameters to tune for each classifier
    >>> params.parToTune.append({'C': lambda_range,'gamma':gamma_range}) # SVM paramters

    >>> # perform classification
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
    Test set acc.: 0.958 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 2
    Test set acc.: 0.958 
    Training set acc.: 0.990
    ------------------------------
    Run Number: 3
    Test set acc.: 1.000 
    Training set acc.: 0.990
    ------------------------------
    Run Number: 4
    Test set acc.: 0.958 
    Training set acc.: 0.990
    ------------------------------
    Run Number: 5
    Test set acc.: 1.000 
    Training set acc.: 1.000
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.975
    Standard deviation: 0.020

    Training Set 
    ---------
    Average accuracy: 0.994
    Standard deviation: 0.005

    For more metrics, see the outputs.    