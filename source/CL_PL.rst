
Persistence Landscapes
----------------------
The code block below provides an example for classification using persistence landscapes. 
We generate persistence diagrams using *testSetManifolds* function from :ref:`Point_Cloud`.
Splitting diagrams into test set and training set is performed using `StratifiedKFold
<https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html>`_ function.
Number of folds are selected by user. Then, SVM with default hyperparameters is used to classify these persistence diagrams.

::

    >>> # import libraries
    >>> from teaspoon.ML.PD_Classification import getPercentScore
    >>> from ML import feature_functions as fF
    >>> from ML.Base import ParameterBucket
    >>> from MakeData.PointCloud import testSetManifolds
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
    >>> params.feature_function = fF.F_Landscape
    >>> params.PL_Number = [1,2]
    >>> params.k_fold_cv=5
    >>> params.clf_model = SVC

    >>> # perform classification
    >>> c_report_train,c_report_test=getPercentScore(DgmsDF,
    >>>                                            labels_col='trainingLabel',
    >>>                                            dgm_col='Dgm1',
    >>>                                            params=params,
    >>>                                            precomputed = False,
    >>>                                            saving = False,
    >>>                                            saving_path = None)

    Beginning experiments

    Run Number: 1
    Test set acc.: 1.000 
    Training set acc.: 0.990
    ------------------------------
    Run Number: 2
    Test set acc.: 0.958 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 3
    Test set acc.: 0.958 
    Training set acc.: 0.990
    ------------------------------
    Run Number: 4
    Test set acc.: 0.958 
    Training set acc.: 0.990
    ------------------------------
    Run Number: 5
    Test set acc.: 1.000 
    Training set acc.: 0.990
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.975
    Standard deviation: 0.020

    Training Set 
    ---------
    Average accuracy: 0.992
    Standard deviation: 0.004

    For more metrics, see the outputs.

Precomputed Landscapes
~~~~~~~~~~~~~~~~~~~~~~

User can also feed precomputed persistence landscapes to classification algorithm to save computation time.
In this case, the first input to *getPercentScore* function will be trated as landscapes, and algorithm will
not spend time on computing the landscapes.
To enable this option, *precomputed* needs to be set to True.

::

    >>> from teaspoon.ML.PD_Classification import getPercentScore
    >>> from teaspoon.ML import feature_functions as fF
    >>> from teaspoon.ML.Base import ParameterBucket
    >>> from teaspoon.MakeData.PointCloud import testSetManifolds
    >>> from sklearn.preprocessing import LabelEncoder
    >>> from sklearn.svm import SVC
    >>> import numpy as np
    >>> import pandas as pd

    >>> # generate persistence diagrams
    >>> DgmsDF = testSetManifolds(numDgms=20, numPts=100)
    >>> labels_col='trainingLabel'
    >>> dgm_col='Dgm1'

    >>> # convert categorical labels into integers
    >>> label_encoder = LabelEncoder()
    >>> x = DgmsDF[labels_col]
    >>> y = label_encoder.fit_transform(x)
    >>> DgmsDF[labels_col] = y


    >>> # compute the persistence diagrams for given diagrams
    >>> PL = np.ndarray(shape=(len(DgmsDF)), dtype=object)

    >>> # compute persistence landscape for training set 
    >>> for i in range(len(DgmsDF)):
    >>>    PLs = fF.PLandscape(DgmsDF[dgm_col][i])
    >>>    PL[i] = PLs.AllPL
        
    >>> # convert the landscapes into dataframe to be consistent with data structure 
    >>> # in the classification algorithm
    >>> PL = pd.DataFrame(PL)
    >>> PL[labels_col] = DgmsDF[labels_col]
    >>> PL = PL.rename(columns={0: "PerLand"})

    >>> # set classification parameters
    >>> params = ParameterBucket()
    >>> params.feature_function = fF.F_Landscape
    >>> params.PL_Number = [1]
    >>> params.k_fold_cv=5
    >>> params.clf_model = SVC
    
    >>> # perform classification
    >>> c_report_train,c_report_test=getPercentScore(PL,
    >>>                                             labels_col='trainingLabel',
    >>>                                             dgm_col="PerLand",
    >>>                                             params=params,
    >>>                                             precomputed = True,
    >>>                                             saving = False,
    >>>                                             saving_path = None)

    Beginning experiments

    User provided precomputed landscapes, we are working on generating feature matrices...

    Run Number: 1
    Test set acc.: 0.917 
    Training set acc.: 0.990
    ------------------------------
    Run Number: 2
    Test set acc.: 0.792 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 3
    Test set acc.: 0.917 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 4
    Test set acc.: 0.917 
    Training set acc.: 0.979
    ------------------------------
    Run Number: 5
    Test set acc.: 1.000 
    Training set acc.: 0.990
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.908
    Standard deviation: 0.067

    Training Set 
    ---------
    Average accuracy: 0.992
    Standard deviation: 0.008

    For more metrics, see the outputs.


Transfer learning between two sets of persistence diagrams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Machine learning module of teaspoon also provides user with transfer learning option.
When it is enabled, user can train and test a classifier on two different sets of persistence diagrams
or persistence landscapes. 
In this example, we first generate two sets of persistence diagrams and then compute their persistence 
landscapes. 
Categorical labels of the diagrams (or landscapes) are converted into the integers.
In the last step, we set classification parameters and perform the classification using SVM.

::

    >>> from teaspoon.ML.PD_Classification import getPercentScore
    >>> from teaspoon.ML import feature_functions as fF
    >>> from teaspoon.ML.Base import ParameterBucket
    >>> from teaspoon.MakeData.PointCloud import testSetManifolds
    >>> from sklearn.preprocessing import LabelEncoder
    >>> from sklearn.svm import SVC
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
    >>> params.feature_function = fF.F_Landscape
    >>> params.PL_Number = [1]
    >>> params.k_fold_cv=5
    >>> params.clf_model = SVC
    >>> params.TF_Learning=True

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
    Test set acc.: 0.917 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 2
    Test set acc.: 0.938 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 3
    Test set acc.: 0.917 
    Training set acc.: 0.990
    ------------------------------
    Run Number: 4
    Test set acc.: 0.938 
    Training set acc.: 1.000
    ------------------------------
    Run Number: 5
    Test set acc.: 0.958 
    Training set acc.: 1.000
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.933
    Standard deviation: 0.016

    Training Set 
    ---------
    Average accuracy: 0.998
    Standard deviation: 0.004

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

    # set classification parameters
    >>> params = ParameterBucket()
    >>> params.feature_function = fF.F_Landscape
    >>> params.PL_Number = [1,2]
    >>> params.k_fold_cv=5
    >>> params.clf_model = SVC
    >>> params.param_tuning = True

    >>> gamma_range = np.logspace(-3, 3, num=10)
    >>> lambda_range = np.logspace(-3, 3, num=10)
    >>> params.parToTune = [] # the list that contains the paramters to tune for each classifier
    >>> params.parToTune.append({'C': lambda_range, 'kernel': ('rbf','sigmoid'),'gamma':gamma_range}) # SVM paramters
    
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
    Test set acc.: 0.792 
    Training set acc.: 0.896
    ------------------------------
    Run Number: 2
    Test set acc.: 0.583 
    Training set acc.: 0.802
    ------------------------------
    Run Number: 3
    Test set acc.: 0.750 
    Training set acc.: 0.844
    ------------------------------
    Run Number: 4
    Test set acc.: 0.792 
    Training set acc.: 0.885
    ------------------------------
    Run Number: 5
    Test set acc.: 0.958 
    Training set acc.: 0.906
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.775
    Standard deviation: 0.120

    Training Set 
    ---------
    Average accuracy: 0.867
    Standard deviation: 0.039

    For more metrics, see the outputs.  