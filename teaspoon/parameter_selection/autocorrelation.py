"""
Auto-correlation for time delay (tau).
=======================================================================

This function implements Auto-Correlation (AC) for the selection of the delay tau for permutation entropy. 
Additionally, it only requires a single time series and has a fast computation time. 
However, this method is only designed for linear system.
"""


def autoCorrelation_tau(ts, cutoff=0.36788, AC_method='spearman', plotting=False):
    """This function takes a time series and uses AC to find the optimum
    delay based on the correlation being less than a specified cutoff (default  is 1/e, which is approximately 0.36788).

    Args:
       ts (array):  Time series (1d).
       cutoff (float): value for which correlation is considered insignificant (default is 1/e).
       method (string): either 'spearman' or 'pearson'. default is 'spearman'.

    Kwargs:
       plotting (bool): Plotting for user interpretation. defaut is False.

    Returns:
       (int): tau, The embedding delay for permutation formation.

    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    tau = 0
    tau_a = []
    R = 1
    R_a = []
    while R > cutoff:
        tau = tau+1
        ts_d = ts[tau:len(ts)]
        ts_o = ts[:len(ts)-tau]
        ts_o = pd.Series(ts_o)
        ts_d = pd.Series(ts_d)

        if AC_method == 'pearson':
            R = ts_o.corr(ts_d, method='pearson')
        if AC_method == 'spearman':
            R, p_value = spearmanr(ts_o, ts_d)
        if plotting == True:
            tau_a.append(tau)
            R_a.append(R)
    if plotting == True:
        TextSize = 18
        plt.xticks(size=TextSize)
        plt.yticks(size=TextSize)
        plt.xlabel(r'Delay $\tau$', size=TextSize)
        plt.ylabel('Correlation Value', size=TextSize)
        plt.plot(tau_a, R_a, 'k')
        plt.plot(tau_a, R_a, 'r.')
        plt.plot([0, tau+5], [cutoff, cutoff], 'b--', label='Cutoff')
        plt.legend(loc='upper right', fontsize=TextSize)
        plt.show()
    return tau - 1


# In[ ]:


if __name__ == '__main__':

    from teaspoon.parameter_selection.autocorrelation import autoCorrelation_tau
    import numpy as np

    fs = 10
    t = np.linspace(0, 100, fs*100)
    ts = np.sin(t) + np.sin((1/np.pi)*t)

    tau = autoCorrelation_tau(ts, cutoff=1/np.exp(1),
                              AC_method='pearson', plotting=True)
    print('Delay from AC: ', tau)
