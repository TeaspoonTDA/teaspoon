Template Functions
------------------

In this section, we provide classification examples using :ref:`template_functions` approach. For more details, please refer to 
Ref. :cite:`8 <Perea2019>`. We perform classification using SVM. User can select different classifier models as well.
Multiple dimensions of persistence diagrams can be selected to generate features matrices. This option is only valid for
template function.

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
    >>> dgm_col=['Dgm0','Dgm1']

    >>> # convert categorical labels into integers
    >>> label_encoder = LabelEncoder()
    >>> x = DgmsDF[labels_col]
    >>> y = label_encoder.fit_transform(x)
    >>> DgmsDF[labels_col] = y

    >>> # set classification parameters
    >>> params = ParameterBucket()
    >>> params.feature_function = fF.interp_polynomial
    >>> params.k_fold_cv=10
    >>> params.d = 20
    >>> params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial
    >>> params.useAdaptivePart = False
    >>> params.clf_model = SVC
    >>> params.TF_Learning = False

    >>> # perform classification
    >>> c_report_train,c_report_test=getPercentScore(DgmsDF,
    >>>                                             labels_col='trainingLabel',
    >>>                                             dgm_col=dgm_col,
    >>>                                             params=params,
    >>>                                             precomputed = False,
    >>>                                             saving = False,
    >>>                                             saving_path = None)

    Beginning experiments

    Run Number: 1
    Test set acc.: 1.000 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 2
    Test set acc.: 0.958 
    Training set acc.: 0.979
    ------------------------------
    Run Number: 3
    Test set acc.: 0.917 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 4
    Test set acc.: 1.000 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 5
    Test set acc.: 1.000 
    Training set acc.: 1.000
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.975
    Standard deviation: 0.033

    Training Set 
    ---------
    Average accuracy: 0.996
    Standard deviation: 0.008

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
    f>> rom teaspoon.ML import feature_functions as fF
    >>> from teaspoon.ML.Base import ParameterBucket
    >>> from teaspoon.MakeData.PointCloud import testSetManifolds
    >>> from sklearn.preprocessing import LabelEncoder
    >>> from sklearn.svm import SVC
    >>> import numpy as np

    >>> # generate persistence diagrams
    >>> DgmsDF_train = testSetManifolds(numDgms=20, numPts=100)
    >>> DgmsDF_test = testSetManifolds(numDgms=20, numPts=100)

    >>> labels_col='trainingLabel'
    >>> dgm_col=['Dgm0']

    >>> # convert categorical labels into integers
    >>> label_encoder = LabelEncoder()
    >>> x_train,x_test = DgmsDF_train[labels_col],DgmsDF_test[labels_col]
    >>> y_train = label_encoder.fit_transform(x_train)
    >>> y_test = label_encoder.fit_transform(x_test)
    >>> DgmsDF_train[labels_col],DgmsDF_test[labels_col] = y_train,y_test


    >>> # set classification parameters
    >>> params = ParameterBucket()
    >>> params.feature_function = fF.interp_polynomial
    >>> params.k_fold_cv=5
    >>> params.d = 20
    >>> params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial
    >>> params.useAdaptivePart = False
    >>> params.clf_model = SVC
    >>> params.TF_Learning = True

    >>> # perform classification
    >>> c_report_train,c_report_test=getPercentScore(DgmsDF_train,
    >>>                                             labels_col='trainingLabel',
    >>>                                             dgm_col=dgm_col,
    >>>                                             params=params,
    >>>                                             precomputed = False,
    >>>                                             saving = False,
    >>>                                             saving_path = None,
    >>>                                             DgmsDF_test = DgmsDF_test)

    Beginning experiments

    Run Number: 1
    Test set acc.: 0.823 
    Training set acc.: 0.865
    ------------------------------
    Run Number: 2
    Test set acc.: 0.823 
    Training set acc.: 0.885
    ------------------------------
    Run Number: 3
    Test set acc.: 0.844 
    Training set acc.: 0.844
    ------------------------------
    Run Number: 4
    Test set acc.: 0.854 
    Training set acc.: 0.854
    ------------------------------
    Run Number: 5
    Test set acc.: 0.854 
    Training set acc.: 0.865
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.840
    Standard deviation: 0.014

    Training Set 
    ---------
    Average accuracy: 0.863
    Standard deviation: 0.014

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
    >>> dgm_col=['Dgm1']

    >>> # convert categorical labels into integers
    >>> label_encoder = LabelEncoder()
    >>> x = DgmsDF[labels_col]
    >>> y = label_encoder.fit_transform(x)
    >>> DgmsDF[labels_col] = y

    >>> # set classification parameters
    >>> params = ParameterBucket()
    >>> params.feature_function = fF.interp_polynomial
    >>> params.k_fold_cv=5
    >>> params.d = 20
    >>> params.jacobi_poly = 'cheb1'  # choose the interpolating polynomial
    >>> params.useAdaptivePart = False
    >>> params.clf_model = SVC
    >>> params.TF_Learning = False
    >>> params.param_tuning = True

    >>> # parameters to tune and their ranges
    >>> gamma_range = np.logspace(-3, 3, num=5)
    >>> lambda_range = np.logspace(-3, 3, num=5)
    >>> params.parToTune = [] # the list that contains the paramters to tune for each classifier
    >>> params.parToTune.append({'C': lambda_range,'gamma':gamma_range}) # SVM paramters

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
    Test set acc.: 1.000 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 2
    Test set acc.: 0.958 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 3
    Test set acc.: 0.917 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 4
    Test set acc.: 1.000 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 5
    Test set acc.: 1.000 
    Training set acc.: 1.000
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.975
    Standard deviation: 0.033

    Training Set 
    ---------
    Average accuracy: 1.000
    Standard deviation: 0.000

    For more metrics, see the outputs.