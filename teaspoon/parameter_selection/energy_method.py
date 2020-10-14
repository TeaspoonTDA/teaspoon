def Persistence0D(sample_data, min_or_max = 0, edges = False): 
    #Input: Sample_data is the data set and min_or_max is wether you want 
    #peaks (1) or valleys (0)
    #Output: Indices of features, and persistence points (birth/death)
    
    
    import operator
    import pandas as pd
    from scipy.signal import argrelmax
    from scipy.signal import argrelmin
    import numpy as np

    # replace min_or_max with an integer to facilitate later processing
    if min_or_max == 'localMax': 
        min_or_max = 1
    else:
        min_or_max = 0
        
    
    from itertools import groupby
    sample_data = [k for k,g in groupby(sample_data) if k!=0]
    NegEnd = -100*np.max(np.abs(sample_data))
    if edges == False:
        #force local minima at edges for clipping
        sample_data = np.insert(sample_data, 0, NegEnd, axis=0)
        sample_data = np.insert(sample_data, len(sample_data), NegEnd, axis=0)
        sample_data = np.insert(sample_data, len(sample_data), NegEnd/2, axis=0)
        # find local minima and local maxima with ends clipped
        maxloc = np.array(argrelmax(sample_data, mode = 'clip'))
        minloc = np.array(argrelmin(sample_data, mode = 'clip'))
        temp = np.array(argrelmax(sample_data, mode = 'clip'))
        
    else:  # find local minima and local maxima with ends
        maxloc = np.array(argrelmax(sample_data, mode = 'wrap'))
        minloc = np.array(argrelmin(sample_data, mode = 'wrap'))
        temp = np.array(argrelmax(sample_data, mode =  'wrap'))

    max_vals = sample_data[maxloc]
    min_vals = sample_data[minloc]
    # create a matrix that will be used in the following part of the code
    minmax_mat = np.concatenate((min_vals, max_vals, minloc, maxloc), axis=0)
    
    
    i = 1
    L = len(maxloc[0])
    # prellocate the persistence diagram
    persistenceDgm = np.zeros((L,2))
    # preallocate the vector that will hold the time a desired feature was born
    feature_ind_1 = np.zeros((L,1))
    feature_ind_2 = np.zeros((L,1))
    
    # find the pairwise diferences
    # get the min-max pairs
    
    while (minmax_mat).shape[1] > 0.5: #while there is still values left in the matrix
        #stack the first two columns of minmax matrix (min values and max values)
        if maxloc[0][0] < minloc[0][0]: #if first value is a peak
            y = np.vstack((minmax_mat[1],minmax_mat[0])).T 
        else:                           #if first value is a valley
            y = np.vstack((minmax_mat[0],minmax_mat[1])).T
            
        y = y.reshape(2*len(minmax_mat[0]),)
        
        pairwiseDiff = abs(np.diff(y))
        
        # find index of the smallest difference
        differences = pairwiseDiff.reshape(len(pairwiseDiff),)
        smallestDiff_ind = min(enumerate(differences), key=operator.itemgetter(1))[0]
        
        if maxloc[0][0] < minloc[0][0]: #if first value is a peak
            ind1 = (int((smallestDiff_ind+1)/2))
            ind2 = (int((smallestDiff_ind)/2))
        else:                           #if first value is a valley
            ind1 = (int((smallestDiff_ind)/2))
            ind2 = (int((smallestDiff_ind+1)/2))

        
        # find the index to the peak that is on one end of the 
        # smallest difference
        peak_val = minmax_mat[1][ind1]
        # find the index to the peak val
        peak_ind = minmax_mat[3][ind1] 

        
        # remove the peak for next iteration
        minmax_mat[1][ind1] = np.NaN
        minmax_mat[3][ind1] = np.NaN
        
        # find the closest local minima to peak_ind
        valley_val = minmax_mat[0][ind2]
        # find the index to the valley val
        valley_ind = minmax_mat[2][ind2]
        # remove the valley for next iteration
        minmax_mat[0][ind2] = np.NaN
        minmax_mat[2][ind2] = np.NaN

        

        # record the 'time' the desired feature was born
        if valley_val > NegEnd:
            feature_ind_1[i-1] = (1-min_or_max)*valley_ind + min_or_max *peak_ind
            feature_ind_2[i-1] = (min_or_max)*valley_ind + (1-min_or_max)*peak_ind
            persDgmPnt = [valley_val, peak_val]
            persistenceDgm[i-1] = persDgmPnt
        
        # remove the NaN (delete the peaks and valleys that were just used)
        for j in range(0,4):
            temp = np.append([0],minmax_mat[j][~pd.isnull(minmax_mat[j])])
            minmax_mat[j] = temp
        minmax_mat=np.delete(minmax_mat,0,axis = 1)
        i=i+1
        
    if edges == False:
        #remove artifact last persistence feature and indices
        feature_ind_1 = feature_ind_1[:-1]
        feature_ind_2 = feature_ind_2[:-1]
        persistenceDgm = persistenceDgm[:-1]
    
    return feature_ind_1, feature_ind_2, persistenceDgm



def takens_embedding(ts, dim, delay):
    S = np.arange(0, dim*delay, delay)
    A = []
    for i in range(len(S)):
        a = ts[S[i]:len(ts)-S[len(S)-1-i]]
        A.append(a)
    E = np.array(A).T
    return E


def monotonic_energy_for_delay(ts, dim = 2, plotting = False):
    delays = np.arange(1,50,1)
    L_sum = []
    L_max = []
    for d in delays:
        Emb = takens_embedding(ts, dim, d)
        En = np.zeros((len(Emb.T[0],)))
        for n in range(dim):
            En = Emb.T[n]**2 + En
        
        feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(En, 'localMin', edges = False)
        B = np.flip(persistenceDgm.T[0], axis = 0)
        D = np.flip(persistenceDgm.T[1], axis = 0)
        L = D-B
        L_sum.append(np.sum(L))
        L_max.append(np.max(L))
    plt.plot(delays, L_sum)
    plt.show()
    plt.plot(delays, L_max)
    plt.show()
    return tau


# In[ ]:


if __name__ == '__main__':
    
    import numpy as np
    
    fs = 10
    t = np.linspace(0, 100, fs*100) 
    ts = np.sin(t)# + np.sin(1.143*t)+ np.sin(2.143*t)+ np.random.normal(0,0.25, len(t))
    plt.plot(t,ts)
    plt.show()
    monotonic_energy_for_delay(ts)














