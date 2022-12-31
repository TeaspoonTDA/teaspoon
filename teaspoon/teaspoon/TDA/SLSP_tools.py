
def cutoff(L, n, alpha=0.001, distribution='Gaussian', signal_compensation=True, noise_param=None):
    """This function calculates the zero-dimensional sublevel set persistence over a closed time domain.

    Args:
        L (1-D array): Zero-dimensional sublevel set persistence lifetimes.
        n (int): Length of time series.

    Other Parameters:
        alpha (Optional[float]): Confidence level. Default is 0.1% = 0.001.
        distribution (optional[string]): Assumed distribution of additve noise. 'Gaussian' is default. Current options are 'Gaussian', , 'Uniform', 'Rayleigh', and 'Exponential'
        signal_compensation (optional[boolean]): Signal compensation option for cutoff adjustment.
        noise_param (optional[float]): If the additive noise distribution parameter is known or estimated, it can be provided directly.

    Returns:
        [float]: Cutoff for the peristence diagram.
        [float]: Estimated noise distribution parameter.
    """

    import numpy as np
    import scipy
    # ------------------Error Checks----------------
    distributions = ['Gaussian', 'Uniform', 'Rayleigh', 'Exponential']
    if distribution not in distributions:
        print('Error: provided distribution not in available distributions.')
        print('available distributions: ', distributions)
        print('Defaulting to Gaussian distribution.')
        distribution = 'Gaussian'

    M_L = np.median(L)
    # ------------------Initial cutoff calculation based on distribution----------------

    if distribution == 'Gaussian':
        D_mean = 1.692
        r = 1.15
        if noise_param == None:
            sigma_est = r*M_L/D_mean
        else:
            sigma_est = noise_param
        cutoff = (2**1.5)*sigma_est * \
            scipy.special.erfinv(2*(1-np.sqrt(alpha))**(1/n) - 1)
        param_est = sigma_est

    if distribution == 'Uniform':
        D_mean = 1/2
        r = 1.00
        if noise_param == None:
            delta_est = 1.0*r*M_L/D_mean
        else:
            delta_est = noise_param
        cutoff = delta_est*(-1+2*((1-np.sqrt(alpha))**(1/n)))
        param_est = delta_est

    if distribution == 'Rayleigh':
        D_mean = 1.10126
        r = 1.13
        if noise_param == None:
            sigma_est = 1.01*r*M_L/D_mean
        else:
            sigma_est = noise_param
        cutoff = sigma_est*(np.sqrt(-2*np.log((1-np.sqrt(alpha))**(1/n))) -
                            np.sqrt(-2*np.log(1 - (1-np.sqrt(alpha))**(1/n))))
        param_est = sigma_est

    if distribution == 'Exponential':
        D_mean = 3/2
        r = 1.25
        if noise_param == None:
            beta_est = r*M_L/D_mean
        else:
            beta_est = noise_param
        cutoff = -beta_est*np.log((1-np.sqrt(alpha))
                                  ** (1/n) - (1-np.sqrt(alpha))**(2/n))
        param_est = beta_est

    # -------------------Compensation for median reduction from signal---------------

    Delta = None
    if noise_param == None:
        Delta = 2*(np.sum(L[L > cutoff])/n)

        if signal_compensation == True:
            if distribution == 'None':
                L_R = Delta
                R = (M_L + L_R)/M_L
            else:
                if distribution == 'Gaussian':
                    c1, c2 = 0.8436165995, 0.8089660065
                if distribution == 'Uniform':
                    c1, c2 = 0.8804987172, 0.638907256
                if distribution == 'Rayleigh':
                    c1, c2 = 0.725841855, 0.6046880833
                if distribution == 'Exponential':
                    c1, c2 = 0.4362525576, 0.3930373465
                R = 1/np.exp(-c1*(Delta/(M_L+Delta))**c2)

            cutoff = cutoff*R
            param_est = param_est*R

    else:
        param_est = noise_param

    return cutoff, param_est


def signficant_extrema(ts, height=None):
    import numpy as np
    from teaspoon.TDA.SLSP import Persistence0D
    ts_n = np.array(ts.flatten())
    ts_n = ts_n + np.random.normal(0, 10**-5*(np.max(ts)-np.min(ts)), len(ts))
    feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(
        ts_n, 'localMin', edges=False)
    B = persistenceDgm.T[0]
    D = persistenceDgm.T[1]
    L = D-B

    I_B = np.array(feature_ind_1.astype(int)).T[0]-1
    I_D = np.array(feature_ind_2.astype(int)).T[0]-1
    if height == None:
        C, noise_parameter_estimate = cutoff(
            L, n=len(ts), signal_compensation=False)
    else:
        C = height
    I_vall = np.append(I_B[L > C], np.argmin(ts))
    I_peak = np.append(I_D[L > C], np.argmax(ts))
    return I_vall, I_peak


# In[ ]:
if __name__ == "__main__":  # ___________________example_________________________

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec
    from teaspoon.TDA.SLSP import Persistence0D
    from teaspoon.TDA.SLSP_tools import cutoff

    # ----------------Assume the additive noise distribution---------------
    dist = 'Gaussian'

    # ---------------Generate a time series with additive noise------------
    fs, T = 40, 10
    t = np.linspace(0, T, fs*T)
    A, sd = 20, 1  # signal amplitude and standard deviation
    ts_0 = A*np.sin(np.pi*t) + A*np.sin(t)
    # gaussian distribution additive noise
    noise = np.random.normal(0, sd, len(ts_0))
    ts = ts_0 + noise  # add noise to time series

    # --------------Run sublevel set persistence--------------------------
    feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(ts)
    B = np.flip(persistenceDgm.T[0], axis=0)  # get birth times
    D = np.flip(persistenceDgm.T[1], axis=0)  # get death times
    L = D-B  # get lifetimes as difference between birth and death

    I_B = np.array(feature_ind_1.astype(int)).T  # indices of birth times
    T_B = np.flip(t[I_B], axis=0)  # time values at birth times

    # -------------get cutoff for persistence diagram---------------------
    C, param = cutoff(L, alpha=0.01, n=len(
        ts), distribution=dist)  # get cutoff
    print('Distribution parameter estimate: ', param)
    print('C: ', C)

    # -------------------------PLOTTING THE RESULTS-----------------------
    gs = gridspec.GridSpec(2, 3)
    plt.figure(figsize=(17, 5))
    TextSize = 15

    ax = plt.subplot(gs[0, 0:2])
    plt.plot(t, ts, 'k')
    plt.ylabel('$x(t)$', size=TextSize)
    plt.xticks(size=TextSize)
    plt.yticks(size=TextSize)
    plt.xlim(min(t), max(t))

    ax = plt.subplot(gs[1, 0:2])
    plt.ylabel('$L$', size=TextSize)
    plt.xlabel('$t$', size=TextSize)
    plt.xticks(size=TextSize)
    plt.yticks(size=TextSize)
    plt.plot(T_B[L > C], L[L > C], 'bd', label=r'$L$ (signal)')
    plt.plot(T_B[L < C], L[L < C], 'ko', alpha=0.7, label=r'$L$ (noise)')
    plt.plot([np.min(t), np.max(t)], [C, C], 'k--', label=r'$C^*_\alpha$')
    ax.fill_between([min(t), max(t)], [C, C], color='red', alpha=0.15)
    plt.ylim(0,)
    plt.xlim(min(t), max(t))
    plt.legend(loc='right', fontsize=TextSize-3, ncol=2)

    ax = plt.subplot(gs[0:2, 2])
    plt.ylabel('$D$', size=TextSize)
    plt.xlabel('$B$', size=TextSize)
    plt.xticks(size=TextSize)
    plt.yticks(size=TextSize)
    plt.plot(B[L > C], D[L > C], 'bd', label=r'signal')
    plt.plot(B[L < C], D[L < C], 'ro', alpha=0.7, label=r'noise')
    plt.plot([min(B), max(D)], [min(B), max(D)], 'k')
    plt.plot([min(B), max(D)], [min(B)+C, max(D)+C],
             'k--', label=r'$C_\alpha$')
    ax.fill_between(x=[min(B), max(D)], y1=[min(B)+C, max(D)+C], y2=[min(B), max(D)],
                    color='red', alpha=0.15)
    plt.legend(loc='lower right', fontsize=TextSize -
               3, bbox_to_anchor=(1.02, -0.02))

    plt.subplots_adjust(wspace=0.3)
    plt.show()
