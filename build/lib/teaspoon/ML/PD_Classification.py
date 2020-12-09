"""
.. module:: PD_Classification
"""

import time
import numpy as np
import os,sys
import pandas as pd
from sklearn.svm import LinearSVC,NuSVC,SVC
from sklearn.linear_model import RidgeClassifierCV,RidgeClassifier
from sklearn.model_selection import train_test_split
from scipy.special import comb
from itertools import combinations
from scipy.spatial.distance import squareform


sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..','teaspoon','ML'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'teaspoon','ML'))

import teaspoon.ML.feature_functions as fF
import teaspoon.ML.Base as Base


def CL_PL(PL,params):
    """

    This function perform classification using persistence landscapes. There are two inputs and these are persistence landscape set and parameter bucket object.

    :param ndarray (PL):
        Object array that includes all landscape functions for each persistence diagram

    :param object (params):
        Parameterbucket object for landscapes. Please see :ref:`PB_Landscape` for more details.

    :Returns:

        **results:** 1x5 matrix that includes the classification results and total elapsed time. First and second columns are for test set score and deviation, while third and fourth column are for training set score and deviation. The fifth represents total elapsed time for classification.

    """
    start =time.time()
    # extract the parameter from parameter bucket
    clf = params.clf_model()
    test_size = params.test_size
    f_function = params.feature_function
    LN = params.PL_Number
    labels = params.Labels
    run_number = 10

    # generate a matrix which includes test and train set results and classification time
    results = np.zeros((1,5))

    #the loop that will
    accuracy=np.zeros((run_number,2))
    for k in range (0,run_number):
        #Training set and Test Set

        PL_train,PL_test,Label_train,Label_test= train_test_split(PL,labels, test_size=test_size)

        # feature_matrix for training set
        feature_train,mesh=fF.F_Landscape(PL_train,params)

        # feature matrix for test set
        # This matrix should be computed based on the mesh obtained from training set

        interp_y=[]
        N=len(PL_test)
        feature_test=np.zeros((N,1))
        for j in range(0,len(LN)):
            xvals = mesh[j]
            y_interp=np.zeros((len(xvals),N))
            for i in range(0,N):                                        # loop iterates for all test set landscapes
                if len(PL_test[i])>=LN[j]:                              # if current landscape number is exist in the current lansdcape set
                    L=PL_test[i].iloc[:, 1].values
                    x=L[LN[j]-1][:,0]                                   # x values of nth landscape for current persistence diagram
                    y=L[LN[j]-1][:,1]                                   # y values of nth landscape
                    y_interp[:,i]=np.interp(xvals,x,y)                  # piecewise linear interpolation
            interp_y.append((y_interp[0:len(xvals),0:N]).transpose())
        for j in range(0,len(LN)):
            ftr=interp_y[j]
            feature_test=np.concatenate((feature_test,ftr),axis=1)
        feature_test=feature_test[:,1:]


        #Classification
        Label_train=np.ravel(Label_train)
        Label_test=np.ravel(Label_test)
        clf.fit(feature_train,Label_train)
        accuracy[k,0]=clf.score(feature_test,Label_test)     # training set score
        accuracy[k,1]=clf.score(feature_train,Label_train)   # test set score


    results[0,0]=np.mean(accuracy[:,0]) # average of test set score
    results[0,1]=np.std(accuracy[:,0])  # deviation of test set score
    results[0,2]=np.mean(accuracy[:,1]) # average of training set score
    results[0,3]=np.std(accuracy[:,1])  # deviation of training set score

    end = time.time()
    results[0,4]=end-start               # time duration for classification

    print ('Landscapes used in feature matrix generation: {}'.format(LN))
    print ('Test set score: {}'.format(results[0,0]))
    print ('Test set deviation: {}'.format(results[0,1]))
    print ('Training set score: {}'.format(results[0,2]))
    print ('Training set deviation: {}'.format(results[0,3]))
    print ('Total elapsed time: {}'.format(results[0,4]))

    return results

def CL_PI(F_PImage1,params,*args):
    """

    This function takes parameter object and feature matrix/matrices which is/are generated using persistence images and returns classification results in a matrix.
    Function is capable of performing transfer learning if user specifies it in parameter bucket object and provides two feature matrices.

    :param ndarray (F_PImage):
        Feature matrix generated with persistence images. If user choose to perform transfer learning, algorithm will assume this parameter as the feature matrix of training set.

    :param object (params):
        Parameterbucket object for classification. Please see :ref:`CL_PB` for more details.

    :param (\*args): Optional paramters. For transfer learning, algorithm needs second feature matrix for test set.

    :Returns:

        **results:** 1x5 matrix that includes the classification results and total elapsed time. First and second columns are for test set score and deviation, while third and fourth column are for training set score and deviation. The fifth represents total elapsed time for classification.


    """
    start = time.time()
    run_number = 10

    clf = params.clf_model()
    test_size = params.test_size

    if params.TF_Learning:
        #transfer learning
        labels_train = params.training_labels   #labels for training data set
        labels_test = params.test_labels        #labels for test data set
    else:
        labels = params.Labels


    results = np.zeros((1,5))
    accuracy=np.zeros((run_number,2))
    for k in range (0,run_number):

        # if user choose to perform transfer learning, second persistence diagram set should be provided
        if params.TF_Learning:
            #check if user provided the second persistence diagram
            F_PImage2 = args[0]

            #Training set and Test Set
            F_Training_Train,F_Training_Test,Label_Training_Train,Label_Training_Test= train_test_split(F_PImage1,labels_train, test_size=0.33)
            F_Test_Train,F_Test_Test,Label_Test_Train,Label_Test_Test= train_test_split(F_PImage2,labels_test, test_size=0.70)

            F_PI_train =  F_Training_Train
            F_PI_test =  F_Test_Test

            Label_train=np.ravel(Label_Training_Train)
            Label_test=np.ravel(Label_Test_Test)

        else:

            F_PI_train,F_PI_test,Label_train,Label_test= train_test_split(F_PImage1,labels, test_size=test_size)

            Label_train=np.ravel(Label_train)
            Label_test=np.ravel(Label_test)


        #Classification

        clf.fit(F_PI_train,Label_train)
        accuracy[k,0]=clf.score(F_PI_test,Label_test)     # training set score
        accuracy[k,1]=clf.score(F_PI_train,Label_train)   # test set score

    results[0,0]=np.mean(accuracy[:,0]) # average of test set score
    results[0,1]=np.std(accuracy[:,0])  # deviation of test set score
    results[0,2]=np.mean(accuracy[:,1]) # average of training set score
    results[0,3]=np.std(accuracy[:,1])  # deviation of training set score

    end = time.time()
    results[0,4]=end-start               # time duration for classification

    print ('Test set score: {}'.format(results[0,0]))
    print ('Test set deviation: {}'.format(results[0,1]))
    print ('Training set score: {}'.format(results[0,2]))
    print ('Training set deviation: {}'.format(results[0,3]))
    print ('Total elapsed time: {}'.format(results[0,4]))

    return results



def CL_CC(PD1,params,*args):
    """

    This function takes persistence diagrams and parameter bucket object and returns classification results in a matrix.
    Function is capable of performing transfer learning if user specifies it in parameter bucket object and provides two persistence diagrams object array.

    :param ndarray (PD1):
        Object array that includes the persistence diagrams. If user wants to perform transfer learning, PD1 will be assumed as training set persistence diagrams, while PD2 is set for test set diagrams.

    :param object (params):
        Parameterbucket object for classification. Please see :ref:`CL_PB` for more details.

    :param ndarray (\*args):
        If user choose the transfer learning option, test set persistence diagrams should be given to algorithm as an input.

    :Returns:

        **results:** Kx4 matrix that includes the classification results. First and second column are for test set score and deviation, while third and fourth column are for training set score and deviation.

    """

    start = time.time()
    run_number=10

    clf = params.clf_model()
    test_size = params.test_size
    FN = params.FN                              #feature number

    if params.TF_Learning:
        #transfer learning
        labels_train = params.training_labels   #labels for training data set
        labels_test = params.test_labels        #labels for test data set
    else:
        labels = params.Labels


    Combinations = []                       #define a list that includes whole combinations for feature number.

    NComb = sum([comb(FN,i, exact=True) for i in np.arange(1,FN+1,1)])

    scores=np.zeros((NComb,2,10))
    results = np.zeros((NComb,4))

    for i in range (0,run_number):
        # if user choose to perform transfer learning, second persistence diagram set should be provided
        if params.TF_Learning:
            PD2 =args[0]
            #Training set and Test Set
            PD_Training_Train,PD_Training_Test,Label_Training_Train,Label_Training_Test= train_test_split(PD1,labels_train, test_size=0.33)
            PD_Test_Train,PD_Test_Test,Label_Test_Train,Label_Test_Test= train_test_split(PD2,labels_test, test_size=0.70)

            #feature matrix for training set
            feature_train,NumberofComb_train,ListofComb_train=fF.F_CCoordinates(PD_Training_Train,FN)

            #feature matrix for test set
            feature_test,NumberofComb_test,ListofComb_train=fF.F_CCoordinates(PD_Test_Test,FN)

            Label_train=np.ravel(Label_Training_Train)
            Label_test=np.ravel(Label_Test_Test)
        else :
            #Training set and Test Set
            PD_train,PD_test,Label_train,Label_test= train_test_split(PD1,labels, test_size=test_size)

            #feature matrix for training set
            feature_train,NumberofComb_train,ListofComb_test=fF.F_CCoordinates(PD_train,FN)

            #feature matrix for test set
            feature_test,NumberofComb_test,ListofComb_train=fF.F_CCoordinates(PD_test,FN)

            Label_train=np.ravel(Label_train)
            Label_test=np.ravel(Label_test)

        for m in range (0,NComb):
            clf.fit(feature_train[m],Label_train)
            scores[m,0,i]=clf.score(feature_test[m],Label_test)
            scores[m,1,i]=clf.score(feature_train[m],Label_train)

    for j in range(NComb):
        results[j,0] = np.mean(scores[j,0,:])
        results[j,1] = np.std(scores[j,0,:])
        results[j,2] = np.mean(scores[j,1,:])
        results[j,3] = np.std(scores[j,1,:])

    test_max = max(results[:,0])
    ind = np.where(results[:,0]==test_max)[0][0]

    end = time.time()
    duration = end-start

    print ('Number of combinations: {}'.format(NumberofComb_train))
    print ('Highest accuracy among all combinations:')
    print ('Test set score: {}'.format(results[ind,0]))
    print ('Test set deviation: {}'.format(results[ind,1]))
    print ('Training set score: {}'.format(results[ind,2]))
    print ('Training set deviation: {}'.format(results[ind,3]))
    print ('Total elapsed time: {}'.format(duration))


    return results

def CL_PS(F_PSignature,params,*args):
    """

    This function takes parameter object and feature matrix which is generated using signatures of paths and returns classification results in a matrix.
    Function is capable of performing transfer learning if user specifies it in parameter bucket object and provides two feature matrices.

    :param ndarray (F_PSignature):
        Feature matrix generated with path signatures. If user performs transfer learning, this parameter will be assumed as training set feature matrix.

    :param object (params):
        Parameterbucket object for classification. Please see :ref:`CL_PB` for more details.

    :param (\*args): Optional parameters. If TF_Learning in parameter bucket is true, algorithms will need test set feature matrix.

    :Returns:

        **results:** 1x5 matrix that includes the classification results and total elapsed time. First and second column are for test set score and deviation, while third and fourth column are for training set score and deviation. The fifth includes total elapsed time for classification.


    """
    start = time.time()
    run_number = 10

    #classification parameters
    clf = params.clf_model()
    test_size = params.test_size

    if params.TF_Learning:
        #transfer learning
        labels_train = params.training_labels   #labels for training data set
        labels_test = params.test_labels        #labels for test data set
    else:
        labels = params.Labels

    results = np.zeros((1,5))
    accuracy=np.zeros((run_number,2))

    for k in range (0,run_number):
        # if user choose to perform transfer learning, second persistence diagram set should be provided
        if params.TF_Learning:
            F_PSignature_test =args[0]
            #Training set and Test Set
            F_Training_Train,F_Training_Test,Label_Training_Train,Label_Training_Test= train_test_split(F_PSignature,labels_train, test_size=0.33)
            F_Test_Train,F_Test_Test,Label_Test_Train,Label_Test_Test= train_test_split(F_PSignature_test,labels_test, test_size=0.70)

            Label_train=np.ravel(Label_Training_Train)
            Label_test=np.ravel(Label_Test_Test)

            F_PS_test = F_Test_Test
            F_PS_train = F_Training_Train

        else:
            F_PS_train,F_PS_test,Label_train,Label_test= train_test_split(F_PSignature,labels, test_size=test_size)

            Label_train=np.ravel(Label_train)
            Label_test=np.ravel(Label_test)

        #Classification
        clf.fit(F_PS_train,Label_train)
        accuracy[k,0]=clf.score(F_PS_test,Label_test)     # training set score
        accuracy[k,1]=clf.score(F_PS_train,Label_train)   # test set score

    results[0,0]=np.mean(accuracy[:,0]) # average of test set score
    results[0,1]=np.std(accuracy[:,0])  # deviation of test set score
    results[0,2]=np.mean(accuracy[:,1]) # average of training set score
    results[0,3]=np.std(accuracy[:,1])  # deviation of training set score

    end = time.time()
    results[0,4]=end-start               # time duration for classification

    print ('Test set score: {}'.format(results[0,0]))
    print ('Test set deviation: {}'.format(results[0,1]))
    print ('Training set score: {}'.format(results[0,2]))
    print ('Training set deviation: {}'.format(results[0,3]))
    print ('Total elapsed time: {}'.format(results[0,4]))

    return results


def CL_KM(PD,params):
    """

    This function takes parameter object and persistence diagrams and computes pairwise kernels for training set and kernel matrix between test set and training set separately.
    The main difference of this function from others is that it uses  `LIBSVM <https://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_ (see `here <https://www.csie.ntu.edu.tw/~cjlin/libsvm/COPYRIGHT>`_ for LIBSVM copyright statement) for classification.
    Function will ask user to input the path to the folder where LIBSVM/python file is available, then it will import related functions.
    Computed kernels are used in SVM algorithm to perform classification.

    :param ndarray (PD):
        Object array that includes the persistence diagrams

    :param object (params):
        Parameterbucket object for classification. Please see :ref:`CL_PB` for more details.

    :Returns:

        **results:** 1x3 matrix that includes the classification results and total elapsed time. First and second column is for test set score and deviation, while third column is the elapsed time.

    """

    user_input = input("Enter the path of the LIBSVM python folder: ")

    assert os.path.exists(user_input), "Specified file does not exist at, "+str(user_input)
    
    sys.path.insert(0, os.path.join(user_input))
    
    from svmutil import svm_train, svm_predict
    
    
    sigma =params.sigma
    test_size=params.test_size
    Label = params.Labels
    start = time.time()
    run_number = 3


    # accuracy and mean squared error matrices
    accuracy_kernel_test = np.zeros((run_number))
    mse_kernel = np.zeros((run_number))

    for rep in range(0,run_number):

        #split data into test set and training set
        PD_train,PD_test,Label_train,Label_test= train_test_split(PD,Label, test_size=0.33)

        N1=len(PD_train)

        # find the combinations so that we can only compute the upper diagonal and diagonal elements of pairwise kernel matrix
        poss_comb = np.array(list(combinations(range(1,N1+1), 2)))

        # create training kernel matrix
        KernelTrain = np.zeros((len(poss_comb),1))
        #loop computes kernels for training set (except diagonal ones)
        for i in range (0,len(poss_comb)):
            perDgm1=PD_train[(poss_comb[i,0])-1]
            perDgm2=PD_train[(poss_comb[i,1])-1]
            KernelTrain[i,0] = fF.KernelMethod(perDgm2,perDgm1,sigma)

        KernelTrain=np.ravel(KernelTrain)
        KernelTrain=squareform(KernelTrain)

        #compute diagonal kernels separately and add them to kernel matrix
        for i in range(0,N1):
            perdgm1=PD_train[i]
            KernelTrain[i,i] = fF.KernelMethod(perdgm1,perdgm1,sigma)
        
        # # classifier        
        # clf = SVC(kernel='precomputed')
        # # train the classifier
        # clf.fit(KernelTrain, PD_test)
        
        Row1=np.zeros((N1,1))

        #concatane row matrix and kernel matrix
        KernelTrain=np.concatenate((Row1, KernelTrain),axis=1)
        KernelTrain[:,:1]=np.arange(N1)[:,np.newaxis]+1

        #Training
        m = svm_train(Label_train, [list(row) for row in KernelTrain], '-c 4 -t 4')

        N2=len(PD_test)
        #test set kernel matrix
        KernelTest = np.zeros((N2,N1))
        for i in range (0,N2):
            perDgm1=PD_test[i]
            for k in range (0,N1):
                perDgm2=PD_train[k]
                KernelTest[i,k] = fF.KernelMethod(perDgm1,perDgm2,sigma)

        # Testing

        p_label, p_acc, p_val = svm_predict(Label_test,[list(row) for row in KernelTest], m)
        accuracy_kernel_test[rep] = p_acc[0]
        mse_kernel[rep] = p_acc[1]
        # accuracy_kernel_train[rep]=clf.score(KernelTrain,Label_train)
        # accuracy_kernel_test[rep]=clf.score(KernelTest,Label_test)
    
    results = np.zeros((1,5))
    results[0,0] = np.mean(accuracy_kernel_test)
    results[0,1] = np.std(accuracy_kernel_test)
    results[0,2] = np.mean(accuracy_kernel_test)
    results[0,3] = np.std(accuracy_kernel_test)
    
    end = time.time()
    results[0,4] = end-start

    print ('Test set score: {}'.format(results[0,0]))
    print ('Test set deviation: {}'.format(results[0,1]))
    print ('Total elapsed time: {}'.format(results[0,2]))

    return results




def getPercentScore(DgmsDF, 
                    labels_col = 'Label', 
                    dgm_col = 'Dgm1', 
                    params = Base.CL_ParameterBucket(), 
                    normalize = False, 
                    verbose = True, 
                    **kwargs):

    """
    :param str DataFrame (DgmsDF):
        Data frame that includes persistence diagrams and labels. If user choose to performs transfer learning, DgmsDF_test should be given into algorithm. In case of transfer learning, first diagram input is assumed as training set.

    :param str (labels_col):
        Name of the labels' column in the data frame

    :param str (dgm_col):
        Name of the diagrams' columnn in the data frame

    :param (params): Parameter bucket object

    :Returns:

        :output:
            (dict) Classification results for training and test set

    """
    if verbose:
        print('---')
        print('Beginning experiment.')
        print(params)

	#check to see if only one column label was passed. If so, turn it into a list.
    if type(dgm_col) == str:
        dgm_col = [dgm_col]

    if params.TF_Learning:
        DgmsDF_test = kwargs['DgmsDF_test']
        # Run actual train/test experiment using sklearn. This part using %70 of each data set to generate random splits for different iterations
        D_train_train, D_train_test, L_train_train,L_train_test = train_test_split(DgmsDF,DgmsDF[labels_col],test_size=0.30,random_state = params.seed)
        D_test_train, D_test_test, L_test_train,L_test_test = train_test_split(DgmsDF_test,DgmsDF_test[labels_col],test_size=0.70,random_state = params.seed)

        D_train = D_train_train
        D_test = D_test_test
        L_train = L_train_train
        L_test = L_test_test
    else:
        D_train, D_test, L_train,L_test = train_test_split(DgmsDF,
                                                           DgmsDF[labels_col],
                                                           test_size=params.test_size,
                                                           random_state = params.seed)

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

    clf = Base.ML_via_featurization(D_train,labels_col = labels_col,dgm_col = dgm_col,params = params,normalize = normalize, verbose = verbose)
    listOfG_train = []

    for dgmColLabel in dgm_col:
        G_train = Base.build_G(D_train[dgmColLabel],params)
        listOfG_train.append(G_train)

    G_train = np.concatenate(listOfG_train,axis = 1)

	# Normalize G
    if normalize:
        G_train = scale(G_train)


	#--------Testing-------------#
    if verbose:
        print('Using ' + str(len(L_test)) + '/' + str(len(DgmsDF_test)) + ' to test...')
    listOfG_test = []
    for dgmColLabel in dgm_col:
        G_test = Base.build_G(D_test[dgmColLabel],params)
        listOfG_test.append(G_test)

    G_test = np.concatenate(listOfG_test,axis = 1)

	# Normalize G
    if normalize:
        G_test = scale(G_test)


	# Compute predictions
    L_predict = pd.Series(clf.predict(G_test),index = L_test.index)
    L_predict_train = pd.Series(clf.predict(G_train),index = L_train.index)

	# Compute scores
    score = clf.score(G_test,list(L_test))
    score_train = clf.score(G_train,list(L_train))
    if verbose:
        print('Score on testing set: ' + str(score) +"...\n")

        print('Finished with train/test experiment.')

    output = {}
    output['score'] = score
    output['score_training'] =score_train
    output['DgmsDF'] = L_predict
    output['clf'] = clf

    return output
