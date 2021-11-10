Path Signatures of Persistence Landscapes
-----------------------------------------

In this section, we provide a classification example for path signature approach. 
Path signatures of selected landscapes functions are used to generate feature matrices. 
Then, we perform classification using SVM.

::

    >>> from ML.PD_Classification import getPercentScore
    >>> from ML import feature_functions as fF
    >>> from ML.Base import ParameterBucket
    >>> from MakeData.PointCloud import testSetManifolds
    >>> from sklearn.preprocessing import LabelEncoder
    >>> from sklearn.svm import SVC

    >>> # generate persistence diagrams
    >>> DgmsDF = testSetManifolds(numDgms=2, numPts=100)
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
    Test set acc.: 0.500 
    Training set acc.: 0.833
    ------------------------------
    Run Number: 2
    Test set acc.: 0.333 
    Training set acc.: 0.833
    ------------------------------

    Finished with training/testing experiments

    Test Set 
    ---------
    Average accuracy: 0.417
    Standard deviation: 0.083

    Training Set 
    ---------
    Average accuracy: 0.833
    Standard deviation: 0.000

    For more metrics, see the outputs.   

.. note:: This approach uses symbolic toolbox of Python. Therefore, its speed is slow compared to other approaches. We will make improvements to speed up the computation soon.  