# ~O(n) function that interleaves (alternates values from) two lists/arrays
def interleave(list1, list2):
    newlist = []
    a1, a2 = len(list1), len(list2)
    for i in range(max(a1, a2)):
        if i < a1:
            newlist.append(list1[i])
        if i < a2:
            newlist.append(list2[i])
    return newlist


def initialize_Q_M(sample_data):
    #import packages
    import numpy as np
    from scipy.signal import find_peaks
    from sortedcontainers import SortedList

    slope_o, slope_f = sample_data[0] - \
        sample_data[1], sample_data[-1]-sample_data[-2]

    # assumes trend at edges continues to infinity
    NegEnd, PosEnd = -float('inf'), float('inf')
    if slope_o < 0:
        sample_data[0] = NegEnd
    else:
        sample_data[0] = PosEnd
    if slope_f < 0:
        sample_data[-1] = NegEnd
    else:
        sample_data[-1] = PosEnd

    # get extrema locations
    max_locs, _ = find_peaks(sample_data)
    min_locs, _ = find_peaks(-sample_data)

    # add outside borders as infinity extrema
    if slope_o < 0:
        min_locs = np.insert(min_locs, 0, 0, axis=0)
    else:
        max_locs = np.insert(max_locs, 0, 0, axis=0)
    if slope_f < 0:
        min_locs = np.insert(min_locs, len(min_locs), -1, axis=0)
    else:
        max_locs = np.insert(max_locs, len(max_locs), -1, axis=0)

    # get extrema values
    max_vals, min_vals = sample_data[max_locs], sample_data[min_locs]

    if max_locs[0] < min_locs[0]:  # if peak first
        # chronologically ordered peaks and  valleys
        PV = interleave(max_vals, min_vals)
        # indices of ordered peaks and valleys
        I_PV = interleave(max_locs, min_locs)
    else:  # if valley first
        # chronologically ordered peaks and  valleys
        PV = interleave(min_vals, max_vals)
        # indices of ordered peaks and valleys
        I_PV = interleave(min_locs, max_locs)

    # get priority value array as chronological pairwise difference
    v = abs(np.diff(PV))

    # get pointer to M array before sorting
    ptr = np.arange(len(v))

    # generate priority matrix Q by stacking together v, ptr, PV and I_PV
    Q = (np.array([v, ptr, PV[:-1], PV[1:], I_PV[:-1], I_PV[1:]]).T).tolist()

    # generate dictionary for quick looking up indices in Q
    M = Q
    # M is defined as the unordered Q because it allows for pointers back to Q by a value lookup O(log(n)),
    # which is a lot faster than trying to update all the pointers in M to Q for each deleted pair.

    # created sorted list from Q
    Q = SortedList(Q)

    # get indice array for knowing which element of Q have been deleted.
    I = np.arange(len(Q)).tolist()
    I = SortedList(I)

    return Q, M, I


def update_Q_M(Q, M, I):

    # get indices to delete from priority value by searching indices left in I
    # index of I where correct index is found using binary search ~ O(log(n))
    i_q = I.bisect_left(int(Q[0][1]))
    i_prev, i_next = i_q-1, i_q+1  # get previous and next index ~ O(1)

    # get related indices in Q ~ O(log(n))
    I_next, I_q, I_prev = Q.index(M[I[i_next]]), 0, Q.index(M[I[i_prev]])

    # get new difference value after peak/vall diff removed ~ O(1)
    v_new = Q[I_next][0] + Q[I_prev][0] - Q[I_q][0]

    # define new element of Q ~ O(1)
    v1, v2, I_v1, I_v2 = Q[I_prev][2], Q[I_next][3], Q[I_prev][4], Q[I_next][5]
    q_new = [v_new, I[i_prev], v1, v2, I_v1, I_v2]

    # get current minimum difference array from priority queue ~ O(1)
    q = Q[I_q]

    # remove old rows of Q that were combined into one using dictionary values
    Q.pop(0)  # ~ O(1)
    Q.remove(M[I[i_prev]])  # ~ O(log(n))
    Q.remove(M[I[i_next]])  # ~ O(log(n))

    # add new element to Q from combining removed elements
    Q.add(q_new)  # ~ O(log(n))

    # update dictionary (don't need to remove dictionary values of other used elements). ~ O(1)
    M[I[i_prev]] = q_new

    # update index array by removing elements indices where elements of Q were removed. ~ O(2log(n))
    I.pop(i_next)
    I.pop(i_q)
    # Why use I? :
    # I is a list that keeps track of which element that is "left" in M.
    # The reason we are not directly deleting elements from M is that we chose to use M = Q.unsorted(),
    # which allows for quick lookup by index. Otherwise we would need to update multiple elements of M
    # for each iteration, slowing everything down.

    return Q, M, I, q


def get_persistence_pair(q):
    # get persistence values from priority element
    b, d, I_b, I_d = q[2], q[3], q[4], q[5]

    # sort persistence pair values so birth < death value
    if b > d:
        b, d, I_b, I_d = d, b, I_d, I_b  # ~ O(1)

    return b, d, I_b, I_d


def Persistence0D(ts):
    """This function calculates the zero-dimensional sublevel set persistence over a closed time domain 
       using a recursively updated priority Q as a SortedList data structure. The algorithm is approximately O(log(n)).

    Args:
        ts (1-D array): time series.

    Returns:
        [px2 array]: peristence diagram with p persistence pairs.
    """

    # import needed packages
    import numpy as np

    # change data to array
    sample_data = np.array(ts).astype(float)

    # initialize minmax matrix M and priority Q as a sorted list
    # auxillary data dictionary (D) and removal indices I
    Q, M, I = initialize_Q_M(sample_data)

    # Initialize data for results
    birth_indices, death_indices, persistenceDgm = [], [], []
    while len(Q) >= 3:  # while there is still values left in the matrix

        # update Q with auxilary data D (dictionary), I (indices from M), i (indices for peak/valley)
        Q, M, I, q = update_Q_M(Q, M, I)

        # get persistence pair from priority element
        b, d, I_b, I_d = get_persistence_pair(q)

        # record time series indices and birth and deaths and store persistence diagram point
        birth_indices.append(I_b)
        death_indices.append(I_d)
        persistenceDgm.append([b, d])

    return np.array(birth_indices), np.array(death_indices), np.array(persistenceDgm)


# In[ ]:
if __name__ == "__main__":  # ___________________example_________________________

    from teaspoon.TDA.SLSP import Persistence0D
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    fs, T = 100, 10
    t = np.linspace(-0.2, T, fs*T+1)
    A = 20
    ts = A*np.sin(np.pi*t) + A*np.sin(1*t)

    feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(ts)
    D = persistenceDgm
    print(' Persistence Diagram Pairs: ', D)

    gs = gridspec.GridSpec(1, 2)
    plt.figure(figsize=(11, 5))

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
