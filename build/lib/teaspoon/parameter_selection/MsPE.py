"""
MsPE for time delay (tau) and dimension (n).
=================================================

This function implements Multi-scale Permutation Entropy (MsPE) for the selection of n and tau for permutation entropy. 
Additionally, it only requires a single time series, is robust to additive noise, and has a fast computation time.
"""


def MsPE_tau(time_series, delay_end = 200, plotting = False):
    """This function takes a time series and uses Multi-scale Permutation Entropy (MsPE) to find the optimum
    delay based on the first maxima in the MsPE plot
    
    Args:
       ts (array):  Time series (1d).
       delay_end (int): maximum delay in search. default is 200.
       
    Kwargs:
       plotting (bool): Plotting for user interpretation. defaut is False.

    Returns:
       (int): tau, The embedding delay for permutation formation.

    """
    trip = 0.9
    from pyentrp import entropy as ent
    import math
    import numpy as np
    
    MSPE = []
    delays = []
    ds, de = 0, delay_end
    m = 3
    start = False
    end = False
    delay = ds
    NPE_previous = 0
    while end == False:
        delay = delay+1
        ME = np.log2(math.factorial(m))
        PE = ent.permutation_entropy(time_series, m, delay)
        NPE = PE/ME
        if NPE < trip:
            start = True
        if NPE > trip and start == True and end == False:
            if NPE < NPE_previous:
                delay_peak = delay-1
                end = True
            NPE_previous = NPE
        MSPE = np.append(MSPE, NPE)
        delays.append(delay)
        
        if delay > de:
            delay = 1
            trip = trip-0.05
            if trip < 0.5:
                delay = 1
                end = True
    
    if plotting == True:
        import matplotlib.pyplot as plt
        plt.figure(2) 
        TextSize = 17
        plt.figure(figsize=(8,3))
        plt.plot(delays, MSPE, marker = '.')
        plt.xticks(size = TextSize)
        plt.yticks(size = TextSize)
        plt.ylabel(r'$h(3)$', size = TextSize)
        plt.xlabel(r'$\tau$', size = TextSize)
        plt.show()
    return delay_peak

    
def MsPE_n(time_series, delay, m_start = 3, m_end = 7, plotting = False):
    
    """This function returns a suitable embedding dimension, given a time series and embedding delay, based on the 
    dimenion normalized MsPE at the optimum delay for a range of dimensions n.

    Args:
       ts (array):  Time series (1d).
       delay (int):  Optimum delay from MsPE.
       m_start (int):  minimum dimension in dimension search. Default is 3.
       m_end (int):  maximum dimension in dimension search. Default is 8.
       
    Kwargs:
       plotting (bool): Plotting for user interpretation. defaut is False.

    Returns:
       (int): n, The embedding dimension for permutation formation.

    """
    from pyentrp import entropy as ent
    import numpy as np
    MnPE = []
    for m in range(m_start,m_end+1):
        PE = ent.permutation_entropy(time_series,m,delay)/(np.log(2))
        NPE = PE/(m-1)
        MnPE = np.append(MnPE, NPE)
    dim = np.argmax(MnPE)
    
    if plotting == True:
        D = delay
        MSPE = []
        MdoPE = []
        for delay in range(1, int(1.5*D)):
            for m in range(m_start,m_end+1):
                PE = ent.permutation_entropy(time_series,m,delay)
                NPE = PE/(m-1)
                MSPE = np.append(MSPE, NPE)
           
            MdoPE = np.concatenate((MdoPE, MSPE), axis = 0)
            MSPE = []
        MdoPE = MdoPE.reshape(( int(1.5*D)-1, 1 + m_end - m_start)).T
        import matplotlib.pyplot as plt
        plt.figure(1) 
        TextSize = 17
        plt.figure(figsize=(8,3))
        for m in range(0, m_end-m_start + 1):
            plt.plot(np.linspace(1,  int(1.5*D)-1, len(MdoPE[m])), MdoPE[m], label = 'n = '+str(m+m_start), linewidth = 2)
        plt.plot([D, D], [0,1.3], 'r--')
        plt.xticks(size = TextSize)
        plt.yticks(size = TextSize)
        plt.ylabel(r'$H/(n-1)$', size = TextSize)
        plt.xlabel(r'$\tau$', size = TextSize)
        plt.legend( loc='lower right', borderaxespad=0.)
        plt.show()
    
    return dim+m_start
        
# In[ ]: Start of Example where time series is defined

# _______________________________________EXAMPLE_________________________________________
if __name__ == "__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    fs = 10
    t = np.linspace(0, 100, fs*100) 
    ts = np.sin(t)
    
    m_s, m_e, d_s, d_e = 3, 7, 1, 200
    #m_s and m_e are the starting and ending dimensions n to search through
    #d_e = max delay tau to search through
    
    #plotting option will show you how delay tau or dimension n were selected
    tau = int(MsPE_tau(ts, d_e, plotting = True)) 
    n = MsPE_n(ts, tau, m_s, m_e, plotting = True)
    
    print('Embedding Delay:       '+str(tau))
    print('Embedding Dimension:   '+str(n))
    

        
        