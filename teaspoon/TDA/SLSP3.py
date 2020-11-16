def initialize_M(sample_data, ends):
    #import packages
    import numpy as np
    from scipy.signal import find_peaks
    
    slope = np.diff(sample_data)
    slope_o, slope_f = -slope[0], slope[-1]
        
    #assumes trend at edges continues to infinity
    if ends == False:
        NegEnd, PosEnd = -float('inf'), float('inf')
        if slope_o < 0: sample_data = np.insert(sample_data, 0, NegEnd, axis=0)
        else: sample_data = np.insert(sample_data, 0, PosEnd, axis=0)    
        if slope_f < 0: sample_data = np.insert(sample_data, len(sample_data), NegEnd, axis=0)
        else: sample_data = np.insert(sample_data, len(sample_data), PosEnd, axis=0)
    else:
        NegEnd, PosEnd = -float('inf'), float('inf')
        sample_data = np.insert(sample_data, 0, NegEnd, axis=0)
        sample_data = np.insert(sample_data, len(sample_data), NegEnd, axis=0)
    #get extrema locations
    maxloc, _ = find_peaks(sample_data)
    minloc, _ = find_peaks(-sample_data)
    
    # add outside borders as infinity extrema
    if ends == False:
        if slope_o < 0: minloc = np.insert(minloc, 0, 0, axis=0)
        else: maxloc = np.insert(maxloc, 0, 0, axis=0)
        if slope_f < 0: minloc = np.insert(minloc, len(minloc), -1, axis=0)
        else: maxloc = np.insert(maxloc, len(maxloc), -1, axis=0)
    else:
        minloc = np.insert(minloc, 0, 0, axis=0)
        minloc = np.insert(minloc, len(minloc), -1, axis=0)
    
    max_vals, min_vals = sample_data[maxloc], sample_data[minloc]
    
    # create a matrix that will be used in the following part of the code
    M = [min_vals.tolist(), max_vals.tolist(), minloc.tolist(), maxloc.tolist()]
    
    return M 

def initialize_Q(M):
    
    def interleave(list1, list2):
        newlist = []
        a1 = len(list1)
        a2 = len(list2)
        for i in range(max(a1, a2)):
            if i < a1:
                newlist.append(list1[i])
            if i < a2:
                newlist.append(list2[i])
        return newlist
    
    min_vals, max_vals = M[0], M[1]

    #flattens maxima and minima into chronologically sorted PV array
    if M[3][0] < M[2][0] > 0: #if peak first
        PV = interleave(max_vals, min_vals)
    else:
        PV = interleave(min_vals, max_vals)
        
    #get priority array
    v = abs(np.diff(PV)) #gets all pairwise distances
    #get pointer array
    ptr = np.arange(len(v))
    #generate priority matrix Q
    Q = np.array([ptr, v]).T
    I_sort = np.argsort(v)
    Q = Q[I_sort]
    return Q
    
def update_M(M, I1, I2):
    #remove values from M where where persistence pair was.
    M[0].pop(I2)
    M[1].pop(I1)
    M[2].pop(I2)
    M[3].pop(I1)
    return M
    
def update_Q(m, Q):
    #import packages
    import numpy as np

    #get new priority value for new ptr pair
    if m != 0 and m != len(Q)-1:
        indices = np.arange(len(Q))
        ind_m_prime = indices[Q.T[0] == m - 1][0]
        ind_m = indices[Q.T[0] == m][0]
        ind_m_next = indices[Q.T[0] == m + 1][0]
        v_new = Q[ind_m_prime][1] + Q[ind_m_next][1] - Q[ind_m][1]
        #get new row for Q
        q = np.array([m-1, v_new])
        #remove old rows of Q that were combined into 1
        Q = np.delete(Q, [ind_m_prime, ind_m, ind_m_next], axis= 0)
        #add new [ptr, v] to Q 
        insert_index = np.searchsorted(Q.T[1], v_new)
        Q = np.insert(Q, insert_index, [q], axis = 0)
        #decrease index of those greater than points removed by 2 since two were removed
        Q.T[0][Q.T[0] > m] = Q.T[0][Q.T[0] > m] - 2
    else: # else if it on an edge or the last few left
        if len(Q) == 1: #if last entry of Q
            #remove old rows of Q that were combined into 1
            Q = np.delete(Q, m, axis= 0)
        else:
            if (m == len(Q)-1): # if on right edge
                #remove old rows of Q that were combined into 1
                Q = np.delete(Q, [m, m-1], axis= 0)
                #decrease index of those greater than points removed by 2 since two were removed
                Q.T[0][Q.T[0] > m] = Q.T[0][Q.T[0] > m] - 2
            if (m == 0): # if on left edge
                #remove old rows of Q that were combined into 1
                Q = np.delete(Q, [m, m+1], axis= 0)
                #decrease index of those greater than points removed by 2 since two were removed
                Q.T[0][Q.T[0] > m] = Q.T[0][Q.T[0] > m] - 2
    return Q
        
def get_persistence_pair(m, M):
    minloc, maxloc = M[2], M[3]
    #gets indices of smallest pairwise distance peak-valley pair
    if maxloc[0] < minloc[0]: #if first, non-edge extrema is a peak
        I1, I2 = (int((m+1)/2)), (int((m)/2))
    else:                           #if first, non-edge extrema is a valley
        I1, I2 = (int((m)/2)), (int((m+1)/2))
    
    # find the peak value and time series index from M
    peak_value, peak_index = M[1][I1], M[3][I1] 
    # find the valley value and time series index from M
    valley_value, valley_index = M[0][I2], M[2][I2]
    
    return peak_value, peak_index, valley_value, valley_index, I1, I2

def Persistence0D(ts, ends = False): 
    """This function calculates the zero-dimensional sublevel set persistence over a closed time domain.
    
    Args:
        ts (1-D array): time series.
        
    Returns:
        [px2 array]: peristence diagram with p persistence pairs.
    """
    
    #import needed packages
    import numpy as np
    sample_data = np.array(ts).astype(float)
    
    M = initialize_M(sample_data, ends)
    Q = initialize_Q(M)
    #Initialize data for results
    birth_indices, death_indices, persistenceDgm = [], [], []
    while len(Q) > 0: #while there is still values left in the matrix
        
        #get persistence pair
        m = int(Q.T[0][np.argmin(Q.T[1])])
        peak_val, peak_ind, vall_val, vall_ind, I1, I2 = get_persistence_pair(m, M)

        # update M and Q
        M = update_M(M, I1, I2)
        Q = update_Q(m, Q)

        # record time series indices and birth and deaths and store persistence diagram point
        birth_indices.append(vall_ind)
        death_indices.append(peak_ind)
        persistenceDgm.append([vall_val, peak_val])
        
    if ends == True: #removes artifact persistence pair from -infinity at each end.
        birth_indices, death_indices, persistenceDgm = birth_indices[:-1], death_indices[:-1], persistenceDgm[:-1]
    return np.array(birth_indices), np.array(death_indices), np.array(persistenceDgm)




# In[ ]: 
if __name__ == "__main__": #___________________example_________________________
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    
    
    
    import time
    start = time.time()
    fs, T = 100, 20000
    t = np.linspace(-0.2,T,fs*T+1)
    A = 20
    ts = A*np.sin(np.pi*t) + A*np.sin(1*t)
    
    feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(ts, ends = False)
    D = persistenceDgm
    end = time.time()
    print('time elapsed: ', end - start)
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
    
    
    
    
    