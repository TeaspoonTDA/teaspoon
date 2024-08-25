

import teaspoon.ML.Base as Base
import teaspoon.ML.feature_functions as fF
import time
import numpy as np
import os
import sys
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import classification_report
from scipy.special import comb
from itertools import combinations
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings("ignore")

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, os.path.dirname(__file__))
# sys.path.insert(0, os.path.join(os.path.dirname(
#     __file__), '..', '..', 'teaspoon', 'ML'))
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'teaspoon', 'ML'))

"""
.. module::PD_Classification
"""


def train_test_split(DgmsDF, labels_col, dgm_col, params, *args):
    """
    This function splits given dataframe of diagrams and labels into training and test using
    StratifiedK-fold cross validation.

    Parameters
    ----------
    DgmsDF : dataframe
        Dataframe that includes diagrams and labels 
    labels_col : str
        Name of the column that stores the labels of the diagrams
    dgm_col : str
        Name of the columns that stores the persistence diagrams 
    params : ParameterBucket object
        Object that includes featurization and classification parameters
    *args : 
        When transfer learning is applied, user needs to pass test set diagrams in a different 
        dataframe.

    Returns
    -------
    training_set_dgm : list
        The list includes training set diagrams for each fold
    test_set_dgm : list
        The list includes test set diagrams for each fold
    training_set_label : list
        The list includes training set labels for each fold
    test_set_label : list
        The list includes test set labels for each fold

    """
    # generate the lists that store test set and training set diagrams/landscapes
    test_set_dgm = []
    training_set_dgm = []
    test_set_label = []
    training_set_label = []

    if params.TF_Learning:
        DgmsDF_test = args[0]

        # set a constant random_state number to be able to get same train-test for all classifiers
        skf = KFold(n_splits=params.k_fold_cv,
                    shuffle=True, random_state=params.seed)

        # generate list of indicies of training and test of training and test diagrams for transfer learning
        indices_train_train = []
        indices_train_test = []
        indices_test_train = []
        indices_test_test = []

        # find indices of training and test set for training set diagrams
        for train_ind, test_ind in skf.split(DgmsDF[dgm_col], DgmsDF[labels_col]):
            indices_train_train.append(train_ind)
            indices_train_test.append(test_ind)

        # find indices of training and test set for test set diagrams
        for train_ind, test_ind in skf.split(DgmsDF_test[dgm_col], DgmsDF_test[labels_col]):
            indices_test_train.append(train_ind)
            indices_test_test.append(test_ind)

        for k_fold in range(params.k_fold_cv):
            if type(dgm_col) == list:
                if params.feature_function != fF.tent and params.feature_function != fF.interp_polynomial:
                    raise Exception(
                        "The list format for dgm_col is only valid for template functions.")
                # user may enter a list of diagram label
                DgmsDF = DgmsDF.sort_index()
                D_train_train, D_train_test = DgmsDF.iloc[indices_train_train[k_fold]
                                                          ], DgmsDF.iloc[indices_train_test[k_fold]]
                L_train_train, L_train_test = D_train_train[labels_col], D_train_test[labels_col]

                DgmsDF_test = DgmsDF_test.sort_index()
                # training and test set for test set  diagrams/landscapes
                D_test_train, D_test_test = DgmsDF_test.iloc[indices_test_train[k_fold]
                                                             ], DgmsDF_test.iloc[indices_test_test[k_fold]]
                L_test_train, L_test_test = DgmsDF_test[labels_col][indices_test_train[k_fold]
                                                                    ], DgmsDF_test[labels_col][indices_test_test[k_fold]]

            else:
                DgmsDF = DgmsDF.sort_index()
                # training and test set for training set diagrams/landscapes/images
                D_train_train, D_train_test = DgmsDF[dgm_col][indices_train_train[k_fold]
                                                              ], DgmsDF[dgm_col][indices_train_test[k_fold]]
                L_train_train, L_train_test = DgmsDF[labels_col][indices_train_train[k_fold]
                                                                 ], DgmsDF[labels_col][indices_train_test[k_fold]]

                DgmsDF_test = DgmsDF_test.sort_index()
                # training and test set for test set  diagrams/landscapes/images
                D_test_train, D_test_test = DgmsDF_test[dgm_col][indices_test_train[k_fold]
                                                                 ], DgmsDF_test[dgm_col][indices_test_test[k_fold]]
                L_test_train, L_test_test = DgmsDF_test[labels_col][indices_test_train[k_fold]
                                                                    ], DgmsDF_test[labels_col][indices_test_test[k_fold]]

            training_set_dgm.append(D_train_train)
            test_set_dgm.append(D_test_train)
            training_set_label.append(L_train_train)
            test_set_label.append(L_test_train)

    else:
        # first step is to obtain training and test sets of the
        # set a constant random_state number to be able to get same train-test for all classifiers
        skf = KFold(n_splits=params.k_fold_cv,
                    shuffle=True, random_state=params.seed)

        # generate list of indicies of training and test of training and test diagrams for transfer learning
        indices_train = []
        indices_test = []

        # find indices of training and test set
        for train_ind, test_ind in skf.split(DgmsDF[dgm_col], DgmsDF[labels_col]):
            indices_train.append(train_ind)
            indices_test.append(test_ind)

        for k_fold in range(params.k_fold_cv):
            if type(dgm_col) == list:
                if params.feature_function != fF.tent and params.feature_function != fF.interp_polynomial:
                    raise Exception(
                        "The list format for dgm_col is only valid for template functions.")
                # user may enter a list of diagram label
                DgmsDF = DgmsDF.sort_index()
                D_train, D_test = DgmsDF.iloc[indices_train[k_fold]
                                              ], DgmsDF.iloc[indices_test[k_fold]]
                L_train, L_test = D_train[labels_col], D_test[labels_col]
            else:
                DgmsDF = DgmsDF.sort_index()
                # training and test set for training set diagrams/landscapes/images
                D_train, D_test = DgmsDF[dgm_col][indices_train[k_fold]
                                                  ], DgmsDF[dgm_col][indices_test[k_fold]]
                L_train, L_test = DgmsDF[labels_col][indices_train[k_fold]
                                                     ], DgmsDF[labels_col][indices_test[k_fold]]

            training_set_dgm.append(D_train)
            test_set_dgm.append(D_test)
            training_set_label.append(L_train)
            test_set_label.append(L_test)

    return training_set_dgm, test_set_dgm, training_set_label, test_set_label


def train_test_split_sklearn(DgmsFD, labels_col, train_size=.5, seed=12):
    """
    This function splits given dataframe of diagrams with labels into a training and testing set using the sklearn train_test_split function

    Parameters
    ----------
    DgmsDF : dataframe
        Dataframe that includes diagrams and labels 
    labels_col : str
        Name of the column that stores the labels of the diagrams
    train_size : float
        Integer representing number of samples of float between 0 and 1 representing a sampling percentage
    seed : int
        Seed to replicate sampling

    Returns
    -------
    training_dgms : pd DataFrame
        The dataframe includes training set diagrams for each fold
    testing_dgms : pd DataFrame
        The dataframe includes training set diagrams for each fold

    """
    from sklearn.model_selection import train_test_split
    labels = DgmsFD[labels_col]
    training_dgms, testing_dgms = train_test_split(
        DgmsFD, train_size=train_size, random_state=seed, stratify=labels)
    return training_dgms.reset_index(), testing_dgms.reset_index()


def getPercentScore(DgmsDF,
                    labels_col='trainingLabel',
                    dgm_col='Dgm1',
                    params=Base.ParameterBucket(),
                    precomputed=False,
                    saving=False,
                    saving_path=None,
                    **kwargs):
    """


    Parameters
    ----------
    DgmsDF : dataframe
        Data frame that includes persistence diagrams and labels. If user choose to performs transfer learning, DgmsDF_test should be given to algorithm. When transfer learning is performed, first diagram input is assumed as training set.

    labels_col : str
        Name of the column that stores the labels for persistence diagrams in a Pandas dataframe. The default is 'trainingLabel'.
    dgm_col : str
        Name of the column that stores the persistence diagrams in a Pandas dataframe. The default is 'Dgm1'.
    params : parameterbucket object
        Parameter bucket object. The default is Base.ParameterBucket().
    precomputed : boolean, optional
        If user already computed the persitence landscapes, this should be set to True, otherwise algorithm will 
        spend time on computing these. This option is only valid when persistence landscapes are used as featurization
        methods. If this parameter is True, algorithm treat 'DgmsDF' as persistence landscapes. The default is False.
    saving : boolean, optional
        If user wants to save classification results, this should be set to True and saving_path needs to be provided. The default is False.
    saving_path : str, optional
        The path where user wants to save the results. This should be provided when saving is True. The default is None.
    **kwargs : 
        Additional parameters. When user wants to apply transfer learning, the second set of persistence diagrams and their labels should be passed in a
        dataframe format. 

    Returns
    -------
    c_report_train : dict
        Classification report for training set results.
    c_report_test : dict
        Classification report for test set results.

    """

    # Note: DgmsDF_test and DgmsDF can represent persistence diagrams, persistence images
    # or persistence landscapes depending on the user input for precomputed. If precomputed is
    # set to True, algorithm will not spend time for computing landscapes or images. Instead, it will
    # expect user to pass them in the same data format as diagrams.

    # if verbose:
    #     print('---')
    #     print('Beginning experiment.')
    #     print(params)

    # generate lists for storing accuracy and reports of classification for training and test sets
    accuracy_train = []
    accuracy_test = []
    c_report_train = []
    c_report_test = []

    # obtain training and test sets for diagrams using the stratified k-fold
    if params.TF_Learning:
        # assign the test test set diagrams from additional variables
        try:
            DgmsDF_test = kwargs['DgmsDF_test']
        except:
            raise Exception("Please provide the second set of persistence diagrams that represent the test set for transfer learning."
                            "If you do not want to proceed with transfer learning, please set TF_Learning flag to False.")

        training_set_dgm, test_set_dgm, training_set_label, test_set_label = train_test_split(
            DgmsDF, labels_col, dgm_col, params, DgmsDF_test)
    else:
        training_set_dgm, test_set_dgm, training_set_label, test_set_label = train_test_split(
            DgmsDF, labels_col, dgm_col, params)

    # check to see if only one column label was passed. If so, turn it into a list.
    if type(dgm_col) == str and (params.feature_function == fF.tent or params.feature_function == fF.interp_polynomial):
        dgm_col = [dgm_col]

    print('Beginning experiments\n')
    for k_fold in range(params.k_fold_cv):

        # assign training and test  diagrams/landscapes/images  and labels
        D_train = training_set_dgm[k_fold]
        D_test = test_set_dgm[k_fold]
        L_train = training_set_label[k_fold]
        L_test = test_set_label[k_fold]

        ########-------TEMPLATE FUNCTION PART---------------###############
        # Note: Template function featurization part is seperated because
        #       its functions' structure is different

        if params.feature_function == fF.tent or params.feature_function == fF.interp_polynomial:

            D_train = pd.DataFrame(D_train)
            D_train[labels_col] = L_train
            D_test = pd.DataFrame(D_test)
            D_test[labels_col] = L_test

            # Get the portions of the test data frame with diagrams and concatenate into giant series:
            allDgms = pd.concat((D_train[label] for label in dgm_col))

            if params.useAdaptivePart == True:
                # Hand the series to the makeAdaptivePartition function
                params.makeAdaptivePartition(allDgms, meshingScheme='DV')
            else:
                # TODO this should work for both interp and tents but doesn't yet
                params.makeAdaptivePartition(allDgms, meshingScheme=None)

            #--------Training------------#
            # if verbose:
            #     print('Using ' + str(len(L_train)) + '/' +
            #           str(len(D_train)) + ' to train...')

            clf = Base.ML_via_featurization(
                D_train, labels_col=labels_col, dgm_col=dgm_col, params=params, normalize=False, verbose=False)
            listOfG_train = []

            for dgmColLabel in dgm_col:
                G_train = Base.build_G(D_train[dgmColLabel], params)
                listOfG_train.append(G_train)

            # feature matrix for training set
            G_train = np.concatenate(listOfG_train, axis=1)

            #--------Testing-------------#
            # if verbose:
            #     print('Using ' + str(len(L_test)) + '/' +
            #           str(len(D_test)) + ' to test...')
            listOfG_test = []
            for dgmColLabel in dgm_col:
                G_test = Base.build_G(D_test[dgmColLabel], params)
                listOfG_test.append(G_test)

            # feature matrix for test set
            G_test = np.concatenate(listOfG_test, axis=1)

            X_train, X_test = G_train, G_test

        # convert diagrams into object array
        D_train = D_train.sort_index().values
        D_test = D_test.sort_index().values
        L_train = L_train.sort_index().values
        L_test = L_test.sort_index().values

        ########------- LANDSCAPES---------------#########
        if params.feature_function == fF.F_Landscape:

            # user defines the landscapes number which are used in feature extraction
            if params.PL_Number is None:
                raise Exception(
                    "Please provide the landscape number that you want to extract features from.")
            LN = params.PL_Number

            # in case landscapes are computed in advance
            if precomputed:
                if k_fold == 0:
                    print(
                        'User provided precomputed landscapes, we are working on generating feature matrices...\n')
                # feature_matrix for training set
                X_train, mesh = fF.F_Landscape(D_train, params)
                PL_test = D_test

                # feature matrix for test set

                # This matrix should be computed based on the mesh obtained from training set
                interp_y = []
                N = len(PL_test)
                X_test = np.zeros((N, 1))
                for j in range(0, len(LN)):
                    xvals = mesh[j]
                    y_interp = np.zeros((len(xvals), N))
                    # loop iterates for all test set landscapes
                    for i in range(0, N):
                        # if current landscape number is exist in the current lansdcape set
                        if len(PL_test[i]) >= LN[j]:
                            L = PL_test[i]['Points'].sort_index().values
                            # x values of nth landscape for current persistence diagram
                            x = L[LN[j]-1][:, 0]
                            # y values of nth landscape
                            y = L[LN[j]-1][:, 1]
                            # piecewise linear interpolation
                            y_interp[:, i] = np.interp(xvals, x, y)
                    interp_y.append((y_interp[0:len(xvals), 0:N]).transpose())
                for j in range(0, len(LN)):
                    ftr = interp_y[j]
                    X_test = np.concatenate((X_test, ftr), axis=1)
                X_test = X_test[:, 1:]

            else:
                # in this case user does not provide landscapes, so algorithm will compute them itself
                PL_train = np.ndarray(shape=(len(D_train)), dtype=object)
                PL_test = np.ndarray(shape=(len(D_test)), dtype=object)

                # compute persistence landscape for training set
                for i in range(len(D_train)):
                    PL = fF.PLandscape(D_train[i])
                    PL_train[i] = PL.AllPL

                # compute persistence landscape for test set
                for i in range(len(D_test)):
                    PL = fF.PLandscape(D_test[i])
                    PL_test[i] = PL.AllPL

                X_train, mesh = fF.F_Landscape(PL_train, params)

                # feature matrix for test set

                # This matrix should be computed based on the mesh obtained from training set
                interp_y = []
                N = len(PL_test)
                X_test = np.zeros((N, 1))
                for j in range(len(LN)):
                    xvals = mesh[j]
                    y_interp = np.zeros((len(xvals), N))
                    # loop iterates for all test set landscapes
                    for i in range(0, N):
                        # if current landscape number is exist in the current lansdcape set
                        if len(PL_test[i]) >= LN[j]:
                            L = PL_test[i]['Points'].sort_index().values
                            # x values of nth landscape for current persistence diagram
                            x = L[LN[j]-1][:, 0]
                            # y values of nth landscape
                            y = L[LN[j]-1][:, 1]
                            # piecewise linear interpolation
                            y_interp[:, i] = np.interp(xvals, x, y)
                    interp_y.append((y_interp[0:len(xvals), 0:N]).transpose())
                for j in range(len(LN)):
                    ftr = interp_y[j]
                    X_test = np.concatenate((X_test, ftr), axis=1)
                X_test = X_test[:, 1:]

        ########------- PERSISTENCE IMAGES---------------#########
        elif params.feature_function == fF.F_Image:

            # persistence images algorithm has built-in transfer learning option
            output = fF.F_Image(D_train, params.pixel_size, params.var, False, [
            ], pers_imager=None, training=True)
            X_train = output['F_Matrix']
            pers_imager = output['pers_imager']
            # use these bounds to compute persistence images of test set
            output = fF.F_Image(D_test, params.pixel_size, params.var, False, [
            ], pers_imager=pers_imager, training=False)
            X_test = output['F_Matrix']

        ########------- CARLSSON COORDINATES---------------#########
        elif params.feature_function == fF.F_CCoordinates:
            if params.FN > 5:
                raise Exception(
                    "There are five coordinates available currenly. Please enter a number between 1 and 5.")

            X_train, TotalNumComb, CombList = fF.F_CCoordinates(
                D_train, params.FN)
            X_test, TotalNumComb, CombList = fF.F_CCoordinates(
                D_test, params.FN)

            # feature matrix for different combinations of Carlsson Coordinates are generated
            # To avoid complexity, we only consider the last feature matrix that includes the all
            # coordinates
            X_train = X_train[-1]
            X_test = X_test[-1]

        ########------- PATH SIGNATURES ---------------#########
        elif params.feature_function == fF.F_PSignature:
            # check if user provided the landscape number which will be used in feature extraction
            if params.L_number == None:
                raise Exception(
                    "Please pass the landscape number that will be used in feature extraction")

            if precomputed:
                X_train = fF.F_PSignature(D_train, params.L_number)
                X_test = fF.F_PSignature(D_test, params.L_number)
            else:
                # in this case user does not provide landscapes, so algorithm will compute them itself
                PL_train = np.ndarray(shape=(len(D_train)), dtype=object)
                PL_test = np.ndarray(shape=(len(D_test)), dtype=object)

                # compute persistence landscape for training set
                for i in range(len(D_train)):
                    PL = fF.PLandscape(D_train[i])
                    PL_train[i] = PL.AllPL

                # compute persistence landscape for test set
                for i in range(len(D_test)):
                    PL = fF.PLandscape(D_test[i])
                    PL_test[i] = PL.AllPL

                X_train = fF.F_PSignature(PL_train, params.L_number)
                X_test = fF.F_PSignature(PL_test, params.L_number)

        ########------- KERNEL METHODS---------------#########
        elif params.feature_function == fF.KernelMethod:
            if params.TF_Learning:
                raise Exception(
                    "Transfer learning for Kernel Method is not available. Please try another featurization approach.")
            else:
                if params.sigma is None:
                    raise Exception(
                        "Please provide the sigma variable for kernel computation.")

                if params.param_tuning:
                    raise Exception(
                        "Parameter tuning is not available for kernel methods.")

                N1 = len(D_train)
                N2 = len(D_test)

                # find the combinations so that we can only compute the upper diagonal and diagonal elements of pairwise kernel matrix
                poss_comb = np.array(list(combinations(range(N1), 2)))
                # create training kernel matrix
                KernelTrain = np.zeros((len(poss_comb)))
                # loop computes kernels for training set (except diagonal ones)
                for i in range(len(poss_comb)):
                    perDgm1 = D_train[(poss_comb[i, 0])]
                    perDgm2 = D_train[(poss_comb[i, 1])]
                    KernelTrain[i] = fF.KernelMethod(
                        perDgm2, perDgm1, params.sigma)
                KernelTrain = np.ravel(KernelTrain)
                KernelTrain = squareform(KernelTrain)

                # compute diagonal kernels separately and add them to kernel matrix
                for i in range(0, N1):
                    perdgm1 = D_train[i]
                    KernelTrain[i, i] = fF.KernelMethod(
                        perdgm1, perdgm1, params.sigma)

                # classifier
                clf = SVC(kernel='precomputed')
                # train the classifier
                clf.fit(KernelTrain, L_train)

                # test set kernel matrix
                KernelTest = np.zeros((N2, N1))
                for i in range(0, N2):
                    perDgm1 = D_test[i]
                    for k in range(0, N1):
                        perDgm2 = D_train[k]
                        KernelTest[i, k] = fF.KernelMethod(
                            perDgm1, perDgm2, params.sigma)

                # predicted labels
                predicted_labels_test = clf.predict(KernelTest)
                predicted_labels_train = clf.predict(KernelTrain)

                # classification report
                cr_test = classification_report(
                    L_test, predicted_labels_test, output_dict=True)
                cr_train = classification_report(
                    L_train, predicted_labels_train, output_dict=True)

                accuracy_test.append(cr_test['accuracy'])
                accuracy_train.append(cr_train['accuracy'])
                c_report_train.append(cr_train)
                c_report_test.append(cr_test)

                print('Run Number: {}'.format(k_fold+1))
                print('Test set acc.: {:.3f} \nTraining set acc.: {:.3f}'.format(
                    cr_test['accuracy'], cr_train['accuracy']))
                print('------------------------------')

        if params.feature_function != fF.KernelMethod:

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = params.clf_model()
            # if user wants to tune classifier parameters, we provide Grid Search approach
            # the list of parameters with their range needs to be provided otherwise
            # user will have an error.
            if params.param_tuning:
                if params.parToTune == None:
                    raise Exception(
                        "Please pass the parameters to tune using ParameterBucket object.")

                # number of features
                n_feat = len(X_train[0, :])
                # concantenate the labels and the train set
                X = np.concatenate((X_train, np.reshape(
                    L_train, (len(L_train), 1))), axis=1)
                # split training set into two
                splits = np.array_split(X, 2)
                T1, T2 = splits[0], splits[1]
                T2_train = T2[:, 0:n_feat]
                T2_label = T2[:, n_feat]
                # paramter tuning
                param_tune = GridSearchCV(model, params.parToTune)
                # use one half of the training set to tune the parameters
                param_tune.fit(T2_train, T2_label)
                best_params = param_tune.best_params_
                for key in sorted(best_params.keys()):
                    setattr(model, key, best_params[key])

            # retrain/train the model
            model.fit(X_train, L_train)

            # predicted labels
            predicted_labels_test = model.predict(X_test)
            predicted_labels_train = model.predict(X_train)

            # classification report
            cr_test = classification_report(
                L_test, predicted_labels_test, output_dict=True)
            cr_train = classification_report(
                L_train, predicted_labels_train, output_dict=True)

            accuracy_test.append(cr_test['accuracy'])
            accuracy_train.append(cr_train['accuracy'])
            c_report_train.append(cr_train)
            c_report_test.append(cr_test)

            print('Run Number: {}'.format(k_fold+1))
            print('Test set acc.: {:.3f} \nTraining set acc.: {:.3f}'.format(
                cr_test['accuracy'], cr_train['accuracy']))
            print('------------------------------')

    print("\nFinished with training/testing experiments")
    print("\nTest Set \n---------")
    print("Average accuracy: {:.3f}\nStandard deviation: {:.3f}".format(
        np.average(accuracy_test), np.std(accuracy_test)))
    print("\nTraining Set \n---------")
    print("Average accuracy: {:.3f}\nStandard deviation: {:.3f}".format(
        np.average(accuracy_train), np.std(accuracy_train)))
    print("\nFor more metrics, see the outputs.")
    # after all runs are completed, save results depending on user choice
    if saving:
        if saving_path is None:
            raise Exception(
                'Please provide saving path to save classification reports.')

        save_name = saving_path+'\\test_set_classification_report_run_number.pkl'
        c_report_train = np.asarray(c_report_train)
        f = open(save_name, "wb")
        pickle.dump(c_report_train, f)
        f.close()

        save_name = saving_path+'\\training_set_classification_report_run_number.pkl'
        c_report_train = np.asarray(c_report_test)
        f = open(save_name, "wb")
        pickle.dump(c_report_test, f)
        f.close()

    return c_report_train, c_report_test
