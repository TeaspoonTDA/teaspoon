def Persistence0D(ts, edge_values = 'automatic'): 
    """This function calculates the zero-dimensional sublevel set persistence over a closed time domain.
    
    Args:
        ts (1-D array): time series.
        
    Other Parameters:
        edge_values (Optional[string]): either 'automatic', 'NN', 'PP', 'PN', or 'NP'. Default is 'automatic'. This argument treats the end points of the time series as an infinity with automatic calulating the slope at each end and assigning the appropriate infinity value or the user may assign this to one of the 4 possible end point values of 'NN' as negative negative, 'PP' as positive positive, etc.
        
    Returns:
        [px2 array]: peristence diagram with p persistence pairs.
    """
    
    #import needed packages
    import operator
    import pandas as pd
    import numpy as np
    from scipy.signal import find_peaks
    
    sample_data = np.array(ts)
    min_or_max = 0 # set to find maxima on positive sample_data compared to minima
    
    slope = np.diff(sample_data)
    if edge_values == 'automatic':
        slope_o, slope_f = -slope[0], slope[-1]
    else:
        if edge_values == 'PN': #assumes trend at edges continues to infinity
            slope_o, slope_f = 1,-1
        if edge_values == 'NN': #assumes trend at edges continues to infinity
            slope_o, slope_f = -1,-1
        if edge_values == 'PP': #assumes trend at edges continues to infinity
            slope_o, slope_f = 1,1
        if edge_values == 'NP': #assumes trend at edges continues to infinity
            slope_o, slope_f = -1,1
    NegEnd, PosEnd = -float('inf'), float('inf')
    
    #assumes trend at edges continues to infinity
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

    maxloc, minloc = [maxloc], [minloc]
    max_vals = [sample_data[maxloc[0]]]
    min_vals = [sample_data[minloc[0]]]
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
            
        y = y.reshape(2*len(minmax_mat[0]),) #flattens y to 1 dimension for pairwise distance comparison
        pairwiseDiff = abs(np.diff(y)) #gets all pairwise distances
        
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
        minmax_mat[1][ind1], minmax_mat[3][ind1] = np.NaN, np.NaN
        
        
        # find the closest local minima to peak_ind
        valley_val = minmax_mat[0][ind2]
        
        # find the index to the valley val
        valley_ind = minmax_mat[2][ind2]
        
        # remove the valley for next iteration
        minmax_mat[0][ind2], minmax_mat[2][ind2] = np.NaN, np.NaN

        # record the 'time' the desired feature was born
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
        
    if edge_values == 'ignore':
        feature_ind_1, feature_ind_2, persistenceDgm = feature_ind_1[:-1], feature_ind_2[:-1], persistenceDgm[:-1] 
    
    return feature_ind_1, feature_ind_2, persistenceDgm




# In[ ]: 
if __name__ == "__main__": #___________________example_________________________
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    
    
    fs, T = 100, 7
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
    
    
    
    
    
    