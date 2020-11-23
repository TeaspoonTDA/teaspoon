def initialize_M(sample_data):
    #import packages
    import numpy as np
    from scipy.signal import find_peaks
    
    slope = np.diff(sample_data)
    slope_o, slope_f = -slope[0], slope[-1]
        
    #assumes trend at edges continues to infinity
    NegEnd, PosEnd = -float('inf'), float('inf')
    if slope_o < 0: sample_data = np.insert(sample_data, 0, NegEnd, axis=0)
    else: sample_data = np.insert(sample_data, 0, PosEnd, axis=0)    
    if slope_f < 0: sample_data = np.insert(sample_data, len(sample_data), NegEnd, axis=0)
    else: sample_data = np.insert(sample_data, len(sample_data), PosEnd, axis=0)
    #get extrema locations
    maxloc, _ = find_peaks(sample_data)
    minloc, _ = find_peaks(-sample_data)
    
    # add outside borders as infinity extrema
    if slope_o < 0: minloc = np.insert(minloc, 0, 0, axis=0)
    else: maxloc = np.insert(maxloc, 0, 0, axis=0)
    if slope_f < 0: minloc = np.insert(minloc, len(minloc), -1, axis=0)
    else: maxloc = np.insert(maxloc, len(maxloc), -1, axis=0)
    
    max_vals, min_vals = sample_data[maxloc], sample_data[minloc]
    
    # create a matrix that will be used in the following part of the code
    M = [min_vals.tolist(), max_vals.tolist(), minloc.tolist(), maxloc.tolist()]
    
    return M 
        
def initialize_Q(M):
    from sortedcontainers import SortedList
    import numpy as np
    def interleave(list1, list2):
        newlist = []
        a1, a2 = len(list1), len(list2)
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
    I_ref = np.repeat(np.arange(len(PV)), 2)[1:-1]
    I_ref = np.reshape(I_ref, (int(len(I_ref)/2),2)).T
    #get priority array
    v = abs(np.diff(PV)) #gets all pairwise distances
    ptr = np.arange(len(v))
    #generate priority matrix Q
    Q = (np.array([v, ptr, I_ref[0], I_ref[1]]).T).tolist()
    dictionary = Q
    Q = SortedList(Q)
    I = np.arange(len(Q))
    
    return Q, dictionary, I



def update_Q(m, Q, D, I):
    # import packages
    import numpy as np
    
    # get new priority value for new ptr pair
    I_m = np.argwhere(I == m)[0][0] #this is O(n)
    I_prev, I_next = I_m-1, I_m+1
    
    #get indices for peak/valley pairs in M
    i = np.array([I[I_m], I[I_next]])
    
    #get new difference value after min peak/vall diff removed
    v_new = Q[Q.index(D[I[I_next]])][0] + Q[Q.index(D[I[I_prev]])][0] - Q[Q.index(D[I[I_m]])][0]
    
    # define new element of Q
    q = [v_new, I[I_prev], I_prev, I_next+1]
    
    # remove old rows of Q that were combined into 1
    Q.remove(D[I[I_prev]])
    Q.remove(D[I[I_m]])
    Q.remove(D[I[I_next]])
    
    # add new [v, ptr, indices] to Q 
    Q.add(q)
    
    #update dictionary
    D[I[I_prev]] = q
    
    #update index array
    I = np.delete(I, [I_m, I_next])

    return Q, D, I, (i/2).astype(int)


def get_persistence_pair(m, M, i, p_min):
    minloc, maxloc = M[2], M[3]
    #gets indices of smallest pairwise distance peak-valley pair
    if maxloc[0] < minloc[0]: #if first, non-edge extrema is a peak
        I1, I2 = i[0], i[1]
    else:  #if first, non-edge extrema is a valley
        I1, I2 = i[1], i[0]
    
    #with not updating M, it is possible to need to flip indices
    comp_val = 2*np.abs(M[1][I2] - M[0][I1] - p_min)/np.abs(M[1][I2] - M[0][I1] + p_min)
    if comp_val > 0.01 and comp_val != float('inf'):
        I1, I2 = I2, I1
    
    # find the valley value and time series index from M
    valley_value, valley_index = M[0][I1], M[2][I1]
    # find the peak value and time series index from M
    peak_value, peak_index = M[1][I2], M[3][I2] 
    
    
    
    return peak_value, peak_index, valley_value, valley_index

        



def Persistence0D(ts): 
    """This function calculates the zero-dimensional sublevel set persistence over a closed time domain.
    
    Args:
        ts (1-D array): time series.
        
    Returns:
        [px2 array]: peristence diagram with p persistence pairs.
    """
    
    #import needed packages
    import numpy as np
    
    #change data to array
    sample_data = np.array(ts).astype(float)
    
    #initialize minmax matrix M and priority Q as a sorted list
    M = initialize_M(sample_data)
    Q, D, I = initialize_Q(M) #auxillary data dictionary (D) and removal indices I
    
    #Initialize data for results
    birth_indices, death_indices, persistenceDgm = [], [], []
    
    while len(Q) >= 3: #while there is still values left in the matrix
        #get persistence pair
        m = int(Q[0][1])
        p_min = Q[0][0] #minimum priority valeu
        # update Q with auxilary data D (dictionary), I (indices from M), i (indices for peak/valley)
        Q, D, I, i = update_Q(m, Q, D, I)
        peak_val, peak_ind, vall_val, vall_ind = get_persistence_pair(m, M, i, p_min)
        # record time series indices and birth and deaths and store persistence diagram point
        birth_indices.append(vall_ind)
        death_indices.append(peak_ind)
        persistenceDgm.append([vall_val, peak_val])
        
    return np.array(birth_indices), np.array(death_indices), np.array(persistenceDgm)




# In[ ]: 
if __name__ == "__main__": #___________________example_________________________
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    
    
    
    import time
    start = time.time()
    fs, T = 100, 1000
    t = np.linspace(-0.2,T,fs*T+1)
    A = 20
    ts = A*np.sin(np.pi*t) + A*np.sin(1*t)
    
    feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(ts)
    D = persistenceDgm
    end = time.time()
    print('time elapsed: ', end - start)
    #print(' Persistence Diagram Pairs: ', D)
    

    
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
    
    
    
    
    