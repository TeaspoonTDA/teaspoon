def MsPE(ts, n = 5, delay_end = 200, plotting = False, normalize = False):
    """This function takes a time series and calculates Multi-scale Permutation Entropy (MsPE) 
        over multiple time scales
    
    Args:
       ts (array):  Time series (1d).
       n (int): Permutation dimension. Default is 5.
       delay_end (int): maximum delay in search. default is 200.
       
    Kwargs:
       plotting (bool): Plotting for user interpretation. defaut is False.
       normalize (bool): Normalizes the permutation entropy on scale from 0 to 1. defaut is False.

    Returns:
       (array): MsPE, the permutation entropy over multiple time scales.

    """
    from pyentrp import entropy as ent
    import math
    import numpy as np
    
    MSPE = []
    delays = []
    m = n
    for delay in np.arange(1,delay_end):
        PE = ent.permutation_entropy(ts, m, delay)
        ME = 1
        if normalize == True: ME = np.log2(math.factorial(n))
        PE = PE/ME
        MSPE = np.append(MSPE, PE)
        delays.append(delay)
    
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
    return delays, MSPE


def PE(ts, n = 5, tau = 10, normalize = False):
    """This function takes a time series and calculates Permutation Entropy (PE).
    
    Args:
       ts (array):  Time series (1d).
       n (int): Permutation dimension. Default is 5.
       tau (int): Permutation delay. Default is 10.
       
    Kwargs:
       normalize (bool): Normalizes the permutation entropy on scale from 0 to 1. defaut is False.

    Returns:
       (float): PE, the permutation entropy.

    """
    from pyentrp import entropy as ent
    import math
    import numpy as np
    
    PE = ent.permutation_entropy(ts, n, tau)
    ME = 1
    if normalize == True: ME = np.log2(math.factorial(n))
    PE = PE/ME

    return PE




def PersistentEntropy(lifetimes, normalize = False):
    """This function takes a time series and calculates Permutation Entropy (PE).
    
    Args:
       lifetimes (array):  Lifetimes from persistence diagram (1d).
       
    Kwargs:
       normalize (bool): Normalizes the entropy on scale from 0 to 1. defaut is False.

    Returns:
       (float): PerEn, the persistence diagram entropy.

    """
    import numpy as np
    lt = lifetimes
    if len(lt) > 1:
        L = sum(lt)
        p = lt/L
        E = sum(-p*np.log2(p))
    if len(lt) == 1:
        E = 0
    if len(lt) == 0:
        E = np.nan
    Emax = 1
    if normalize == True: Emax = np.log2(sum(lt))
    PerEn = E/Emax

    return PerEn
