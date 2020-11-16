def Persistence0D(ts): 
    """This function calculates the zero-dimensional sublevel set persistence over a closed time domain.
    
    Args:
        ts (1-D array): time series.
        
    Returns:
        [px2 array]: peristence diagram with p persistence pairs.
    """
    
    #import needed packages
    import pandas as pd
    import numpy as np
    from scipy.signal import find_peaks
    sample_data = np.array(ts).astype(float)
    
    
    #assumes trend at edges continues to infinity
    slope = np.diff(sample_data)
    slope_o, slope_f = -slope[0], slope[-1]
    NegEnd, PosEnd = -float('inf'), float('inf')
    if slope_o < 0: sample_data = np.insert(sample_data, 0, NegEnd, axis=0)
    else: sample_data = np.insert(sample_data, 0, PosEnd, axis=0)    
    if slope_f < 0: sample_data = np.insert(sample_data, len(sample_data), NegEnd, axis=0)
    else: sample_data = np.insert(sample_data, len(sample_data), PosEnd, axis=0)
    
    
    #get extrema locations
    maxloc, _ = find_peaks(sample_data)
    minloc, _ = find_peaks(-sample_data)
    temp = np.argmax(sample_data)

    
    # add outside borders as infinity extrema
    if slope_o < 0: minloc = np.insert(minloc, 0, 0, axis=0)
    else: maxloc = np.insert(maxloc, 0, 0, axis=0)
    if slope_f < 0: minloc = np.insert(minloc, len(minloc), -1, axis=0)
    else: maxloc = np.insert(maxloc, len(maxloc), -1, axis=0) 
    if len(maxloc) != len(minloc): 
        if len(maxloc) > len(minloc): #if both outside borders are infinity
            maxloc = maxloc[:-1] #removes one of the positive infinity values
        else: #if both outside borders are negative infinity
            minloc = minloc[:-1] #removes one of the negative infinity values


    #gets minmax matrix
    maxloc, minloc = [maxloc], [minloc]
    max_vals = [sample_data[maxloc[0]]]
    min_vals = [sample_data[minloc[0]]]
    # create a matrix that will be used in the following part of the code
    M = np.concatenate((min_vals, max_vals, minloc, maxloc), axis=0)
    

    #initialize result data structures
    i, L = 1, len(maxloc[0])
    # prellocate the persistence diagram
    persistenceDgm = np.zeros((L,2))
    # preallocate the vector that will hold the time a desired feature was born
    birth_ind = np.zeros((L,1))
    death_ind = np.zeros((L,1))
    
    while (M).shape[1] > 0.5: #while there is still values left in the matrix
        #stack the first two columns of minmax matrix (peaks and valleys)
        if maxloc[0][0] < minloc[0][0]: #if first value is a peak
            peaks_valleys = np.vstack((M[1],M[0])).T 
        else:                           #if first value is a valley
            peaks_valleys = np.vstack((M[0],M[1])).T
        #flattens peaks_valleys to 1-D and orders chronologically for pairwise distance comparison
        time_sorted_peaks_valls = peaks_valleys.reshape(2*len(M[0]),) 
        pairwiseDiff = abs(np.diff(time_sorted_peaks_valls)) #gets all pairwise distances
        smallestDiff_ind = np.argmin(pairwiseDiff)
        #gets indices of smallest pairwise distance peak-valley pair
        if maxloc[0][0] < minloc[0][0]: #if first value is a peak
            ind1, ind2 = (int((smallestDiff_ind+1)/2)), (int((smallestDiff_ind)/2))
        else:                           #if first value is a valley
            ind1, ind2 = (int((smallestDiff_ind)/2)), (int((smallestDiff_ind+1)/2))
        
        
        # find the peak value from min pairwise distance
        peak_val = M[1][ind1]
        # find the time series index to the peak value
        peak_ind = M[3][ind1] 
        # remove the peak for next iteration
        M[1][ind1], M[3][ind1] = np.NaN, np.NaN
        
    
        # find the valley value from min pairwise distance
        valley_val = M[0][ind2]
        # find the time series index to the valley val
        valley_ind = M[2][ind2]
        # remove the valley for next iteration
        M[0][ind2], M[2][ind2] = np.NaN, np.NaN
        
        
        # record time series indices and birth and deaths and store persistence diagram point
        birth_ind[i-1] =  valley_ind
        death_ind[i-1] =  peak_ind
        persDgmPnt = [valley_val, peak_val]
        persistenceDgm[i-1] = persDgmPnt
        
        
        # remove the NaN (delete the peaks and valleys that were just used)
        for j in range(0,4): #go through each row in minmax matrix (4x4 matrix)
            temp = np.append([0],M[j][~pd.isnull(M[j])])
            M[j] = temp
        M=np.delete(M,0,axis = 1)
        
        i=i+1 #incremenet i
    return birth_ind, death_ind, persistenceDgm




# In[ ]: 
if __name__ == "__main__": #___________________example_________________________
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    
    
    fs, T = 100, 122234
    t = np.linspace(-0.2,T,fs*T+1)
    A = 20
    ts = A*np.sin(np.pi*t) + A*np.sin(1*t)
    
    feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(ts)
    D = persistenceDgm
    print(' Persistence Diagram Pairs: ', D)
    
    gs = gridspec.GridSpec(1,2) 
    plt.figure(figsize=(11,5))
        
    ax = plt.subplot(gs[0, 0])
    plt.title('Time Series')
    plt.xlabel('$t$')
    plt.ylabel('$x(t)$')
    plt.plot(t, ts)
    
    ax = plt.subplot(gs[0, 1])
    plt.title('Persistence Diagram')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.plot(D.T[0], D.T[1], 'ro')
    plt.plot([min(ts), max(ts)], [min(ts), max(ts)], 'k--')
    
    plt.show()
    
    
    
    
    