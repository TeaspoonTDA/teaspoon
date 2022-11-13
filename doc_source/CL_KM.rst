Kernel Method for Persistence Diagrams
--------------------------------------

In this section, we provide classification example that uses the precomputed kernels for persistence diagrams.
The detailed information about this approach is provided in :ref:`Kernel_Method`. 
Transfer learning and parameter tuning is not available for this approach. 

::

    >>> from teaspoon.ML.PD_Classification import getPercentScore
    >>> from teaspoon.ML import feature_functions as fF
    >>> from teaspoon.ML.Base import ParameterBucket
    >>> from teaspoon.MakeData.PointCloud import testSetManifolds
    >>> from sklearn.preprocessing import LabelEncoder
    >>> from sklearn.svm import SVC

    >>> # generate persistence diagrams
    >>> DgmsDF = testSetManifolds(numDgms=5, numPts=100)
    >>> labels_col='trainingLabel'
    >>> dgm_col=['Dgm1']

    >>> # convert categorical labels into integers
    >>> label_encoder = LabelEncoder()
    >>> x = DgmsDF[labels_col]
    >>> y = label_encoder.fit_transform(x)
    >>> DgmsDF[labels_col] = y

    >>> # set classification parameters
    >>> params = ParameterBucket()
    >>> params.feature_function = fF.KernelMethod
    >>> params.k_fold_cv=5
    >>> params.sigma = 0.25
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
    Test set acc.: 0.167 
    Training set acc.: 0.542
    ------------------------------
    Run Number: 2
    Test set acc.: 0.500 
    Training set acc.: 0.667
    ------------------------------
    Run Number: 3
    Test set acc.: 0.667 
    Training set acc.: 0.750
    ------------------------------
    Run Number: 4
    Test set acc.: 0.167 
    Training set acc.: 0.792
    ------------------------------
    Run Number: 5
    Test set acc.: 0.667 
    Training set acc.: 0.792
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.433
    Standard deviation: 0.226

    Training Set 
    ---------
    Average accuracy: 0.708
    Standard deviation: 0.095

    For more metrics, see the outputs.       