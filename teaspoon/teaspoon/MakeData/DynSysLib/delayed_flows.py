import numpy as np

def delayed_flows(system, dynamic_state=None, L=None, fs=None,
                  SampleSize=None, parameters=None, InitialConditions=None):

    if system == 'mackey_glass':
        t, ts = mackey_glass()

    return t, ts

def mackey_glass(parameters=None, fs=5, SampleSize=1000, L=400, dynamic_state='periodic'):
    """
    The Mackey-Glass Delayed Differential Equation is

    .. math::
        \dot{x}(t) = -\\gamma*x(t) + \\beta*\\frac{x(t - \\tau)}{1 + x(t - \\tau)^n}
    
    where we chose the parameters :math:`\\tau = 2, \\beta = 2, \\gamma = 1,` and :math:`n = 9.65`. We solve this system for 400 seconds with a sampling rate of 50 Hz. The solution was then downsampled to 5 Hz and the last 200 seconds were used for the figure.

    .. figure:: ../../../figures/Delayed_Flows/Mackey_Glass_Delayed_Differential_Equation.png
    
    Parameters:
        parameters (Optional[floats]): Array of four floats [gamma, tau, beta, n]
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' is not supplying parameters.
        
    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    """

    # This requires installation of ddeint (pip install ddeint)
    from ddeint import ddeint

    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    if len(parameters) != 4:
        print(
            'Warning: needed 4 parameters. Defaulting to periodic solution parameters.')
        parameters = None
    else:
        gamma, τ, B, n = parameters[0], parameters[1], parameters[2], parameters[3]

    if parameters == None:
        if dynamic_state == 'periodic':
            n = 7.75
        if dynamic_state == 'chaotic':
            n = 9.65
        τ = 2.0
        B = 2.0
        gamma = 1.0001

    def mackey_glass(X, t, d):
        x = X(t)
        xd = X(t-d)
        return B*(xd/(1+xd**n)) - gamma*x

    fsolve = 50
    tt = np.linspace(0, L, int(L*fsolve))
    def g(t): return np.array([1, 1])
    d = τ
    states = ddeint(mackey_glass, g, tt, fargs=(d,)).T

    ts = [((states[0])[::int(fsolve/fs)])[-SampleSize:]]
    t = tt[::int(fsolve/fs)][-SampleSize:]

    return t, ts