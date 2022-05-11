"""
Permutation Auto Mutual Information (PAMI) for time delay (tau).
=======================================================================

This function implements the mulutal information of permutations to find the delay (tau) 
that causes the first minima in the mutual information between permutations.
"""


def mutualPerm(ts, delay, n):
    """This function calculates the mutual information between permutations with tau = 1 and tau = delay  

    Args:
       ts (array):  Time series (1d).
       delay (int): Permutation delay
       n (int): Permutation dimension

    Returns:
       (float): I, Mutual permutation information.

    """
    import numpy as np
    import math
    x = ts
    y = ts
    Bins_x, PA_x = (stats_permutation_entropy(x, n, 1))
    Bins_y, PA_y = (stats_permutation_entropy(y, n, delay))
    Bins_x, Bins_y = np.array(Bins_x), np.array(Bins_y)
    PA_y = np.array(PA_y)
    PA_x = np.array(PA_x)[0:len(PA_y)]
    types = np.stack((np.tile(np.linspace(1, math.factorial(n), math.factorial(n)),  math.factorial(n)),
                      np.repeat(np.linspace(1, math.factorial(n), math.factorial(n)), math.factorial(n)))).T
    PAs = np.stack((PA_x, PA_y)).T
    Bins_xy = np.zeros((math.factorial(n), math.factorial(n)))
    count = 0
    for i in range(len(PA_x)):
        for j in range(len(types.T[0])):
            if PAs[i][0] == types[j][0] and PAs[i][1] == types[j][1]:
                Bins_xy[PAs[i][0]-1][PAs[i][1]-1] += 1
                count = count+1

    P_xy = Bins_xy/count
    P_x = Bins_x/sum(Bins_x)
    P_y = Bins_y/sum(Bins_y)
    I = 0
    for i in range(0, math.factorial(n)):
        for j in range(0, math.factorial(n)):
            if (P_x[i] != 0 and P_y[j] != 0 and P_xy[i][j] != 0):
                I_xy = P_xy[i][j]*np.log2(P_xy[i][j]/(P_x[i]*P_y[j]))
                I = I + I_xy
    return I


def stats_permutation_entropy(time_series, m, delay):

    def util_hash_term(perm):
        deg = len(perm)
        return sum([perm[k]*deg**k for k in range(deg)])

    def util_granulate_time_series(time_series, scale):
        n = len(time_series)
        b = int(np.fix(n / scale))
        cts = [0] * b
        for i in range(b):
            cts[i] = np.mean(time_series[i * scale: (i + 1) * scale])
        return cts

    import itertools
    import numpy as np
    L = len(time_series)
    perm_order = []
    permutations = np.array(list(itertools.permutations(range(m))))
    hashlist = [util_hash_term(perm) for perm in permutations]
    c = [0] * len(permutations)

    for i in range(L - delay * (m - 1)):
        # sorted_time_series =    np.sort(time_series[i:i+delay*m:delay], kind='quicksort')
        sorted_index_array = np.array(np.argsort(
            time_series[i:i + delay * m:delay], kind='quicksort'))
        hashvalue = util_hash_term(sorted_index_array)
        c[np.argwhere(hashlist == hashvalue)[0][0]] += 1
        perm_order = np.append(
            perm_order, np.argwhere(hashlist == hashvalue)[0][0])
    c = [element for element in c]  # if element != 0
    Bins = c
    perm_seq = perm_order.astype(int)+1
    return Bins, perm_seq


def PAMI_for_delay(ts, n=5, plotting=False):
    """This function calculates the mutual information between permutations with tau = 1 and tau = delay  

    Args:
       ts (array):  Time series (1d).

    Kwargs:
       plotting (bool): Plotting for user interpretation. defaut is False.

       n (int): dimension for calculating delay. delault is 5 as explain in On the Automatic Parameter Selection for Permutation Entropy

    Returns:
       (int): tau, The embedding delay for permutation formation.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks

    cutoff = 0.01
    max_delay = 100
    m = 2
    MP = []
    tau_a = []
    window_a = []
    flag = False
    delay = 0
    while flag == False:
        delay = delay+1
        tau_a.append(delay)
        window_a.append(delay*(n-1))
        MI_Perm = mutualPerm(ts, delay, m)
        MP.append(MI_Perm)  # calculates mutual information
        peaks, _ = find_peaks(-np.array(MP), height=-cutoff)
        if MI_Perm < cutoff and len(peaks) > 0:
            flag = True
        if delay > max_delay:
            delay = 0
            cutoff = cutoff*10
            MP = []
            tau_a = []
            window_a = []
    delay_2 = delay
    delay_n = int(delay_2/(n-1))
    if plotting == True:
        TextSize = 12
        plt.figure(1)
        plt.plot(tau_a, MP, label='n = ' + str(m), linewidth=2)
        plt.xlabel(r'$\tau(n-1)$', size=TextSize)
        plt.ylabel(r'$I_p(\tau,n)$', size=TextSize)
        plt.xticks(size=TextSize)
        plt.yticks(size=TextSize)
        plt.legend(loc='upper right', fontsize=TextSize)
        plt.ylim(0)
        plt.show()

    return delay_n

# In[ ]: running functions on time series


# _______________________________________EXAMPLE_________________________________________
if __name__ == "__main__":

    import numpy as np

    fs = 10
    t = np.linspace(0, 100, fs*100)
    ts = np.sin(t)
    tau = PAMI_for_delay(ts, n=5, plotting=True)
    print('Delay from PAMI: ', tau)
