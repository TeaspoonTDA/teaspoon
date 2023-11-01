from teaspoon.ML import load_datasets
import time
import numpy as np
from sklearn.model_selection import train_test_split
import math
from numpy.linalg import norm as lnorm
from math import pi
from sklearn.metrics import DistanceMetric
from numpy.linalg import norm as lnorm
import numpy as np
from persim import heat

mnist = load_datasets.mnist()

def train_test_split_sklearn(DgmsFD, labels_col, train_size=.5, seed=12):
 
    labels = DgmsFD[labels_col]
    training_dgms, testing_dgms = train_test_split(DgmsFD, train_size=train_size, random_state=seed, stratify=labels)
    return training_dgms.reset_index(), testing_dgms.reset_index()

dgms_train, dgms_test = train_test_split_sklearn(mnist, 'labels', train_size = .95)
xdgm0_train = dgms_train['zero_dim_rtl']
xdgm0_test = dgms_test['zero_dim_rtl']
xdgm1_train = dgms_train['one_dim_rtl']
xdgm1_test = dgms_test['one_dim_rtl']
labels_train = dgms_train['labels']
labels_test = dgms_test['labels']

def KernelMethod(perDgm1, perDgm2, sigma):
    """

    This function computes the kernel for given two persistence diagram based on the formula provided in Ref. :cite:`5 <Reininghaus2015>`.
    There are three inputs and these are two persistence diagrams and the kernel scale sigma.    

    Parameters
    ----------
    perDgm1 : ndarray
        Object array that includes first persistence diagram set.
    perDgm2 : ndarray
        Object array that includes second persistence diagram set.
    sigma : float
        Kernel scale.

    Returns
    -------
    Kernel : float
        The kernel value for given two persistence diagrams.

    """
    
    L1 = len(perDgm1)
    L2 = len(perDgm2)
    kernel = np.zeros((L2, L1))

    Kernel = 0

    for i in range(0, L1):
        p = perDgm1[i]
        p = np.reshape(p, (2, 1))
        for j in range(0, L2):
            q = perDgm2[j]
            q = np.reshape(q, (2, 1))
            q_bar = np.zeros((2, 1))
            q_bar[0] = q[1]
            q_bar[1] = q[0]
            dist1 = lnorm(p-q)
            dist2 = lnorm(p-q_bar)
            kernel[j, i] = np.exp(-(math.pow(dist1, 2))/(8*sigma)) - \
                np.exp(-(math.pow(dist2, 2))/(8*sigma))
            Kernel = Kernel+kernel[j, i]
    Kernel = Kernel*(1/(8*pi*sigma))

    return Kernel

def heat_kernel_distance(dgm0, dgm1, sigma=.4):
    return np.sqrt(KernelMethod(dgm0, dgm0, sigma) + KernelMethod(dgm1, dgm1, sigma) - 2*KernelMethod(dgm0, dgm1, sigma))

def kernel_features(train, s):
    n_train = len(train)
    X_train_features = np.zeros((n_train, n_train))
    
    start = time.time()
    for i in range(0,n_train):
        for j in range(0,i):
            dgm0 = train[i]
            dgm1 = train[j]
            hka = heat_kernel_distance(dgm0, dgm1, sigma = s) 
            X_train_features[i,j] = hka
            X_train_features[j,i] = hka

    timing = time.time()-start
    return timing, X_train_features

def fast_hk(dgm0,dgm1,sigma=.4):
    
    dist = DistanceMetric.get_metric('euclidean')
    dist1 = (dist.pairwise(dgm0,dgm1))**2
    Qc = dgm1[:,1::-1]
    dist2 = (dist.pairwise(dgm0,Qc))**2
    exp_dist1 = sum(sum(np.exp(-dist1/(8*sigma))))
    exp_dist2 = sum(sum(np.exp(-dist2/(8*sigma))))
    hk = (exp_dist1-exp_dist2)/(8*np.pi*sigma)
    return(hk)

def heat_kernel_approx(dgm0, dgm1, sigma=.4):
    return np.sqrt(fast_hk(dgm0, dgm0, sigma) + fast_hk(dgm1, dgm1, sigma) - 2*fast_hk(dgm0, dgm1, sigma))

def fast_kernel_features(train, s):
    n_train = len(train)
    X_train_features = np.zeros((n_train, n_train))
    
    start = time.time()
    for i in range(0,n_train):
        for j in range(0,i):
            dgm0 = train[i]
            dgm1 = train[j]
            hka = heat_kernel_approx(dgm0, dgm1, sigma = s) 
            X_train_features[i,j] = hka
            X_train_features[j,i] = hka

    timing = time.time()-start
    return timing, X_train_features

def persim_kernel_features(train, s):
    n_train = len(train)
    X_train_features = np.zeros((n_train, n_train))
    
    start = time.time()
    for i in range(0,n_train):
        for j in range(0,i):
            dgm0 = train[i]
            dgm1 = train[j]
            hka = heat(dgm0, dgm1, sigma = s) 
            X_train_features[i,j] = hka
            X_train_features[j,i] = hka

    timing = time.time()-start
    return timing, X_train_features

#train_test = xdgm0_train[0:50000]

kernel_features(xdgm0_train, s = .3)