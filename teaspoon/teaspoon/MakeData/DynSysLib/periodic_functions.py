import numpy as np


def periodic_functions(system, dynamic_state=None, L=None, fs=None,
                       SampleSize=None, parameters=None, InitialConditions=None):
    '''
    TODO Add docstring. Do we even want to keep this function? 
    '''
    if system == 'sine':

        # setting simulation time series parameters
        if fs == None:
            fs = 50
        if SampleSize == None:
            SampleSize = 2000
        if L == None:
            L = 40
        t = np.linspace(0, L, int(L*fs))

        # setting system parameters
        if parameters != None:
            if len(parameters) != 1:
                print(
                    'Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                parameters = None
            else:
                omega = parameters[0]
        if parameters == None:
            omega = 2*np.pi

        t, ts = sine(omega, L, fs, SampleSize, parameters)

    if system == 'incommensurate_sine':
        # setting simulation time series parameters
        if fs == None:
            fs = 50
        if SampleSize == None:
            SampleSize = 5000
        if L == None:
            L = 100
        t = np.linspace(0, L, int(L*fs))

        # setting system parameters
        if parameters != None:
            if len(parameters) != 2:
                print(
                    'Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                parameters = None
            else:
                omega1, omega2 = parameters[0], parameters[1]
        if parameters == None:
            omega1 = np.pi
            omega2 = 1

        t, ts = incommensurate_sine(omega1, omega2, L, fs, SampleSize)

    return t, ts


def sine(omega=2*np.pi,
         L=40, fs=50, SampleSize=2000):
    """
    The sinusoidal function is defined as

    .. math::
        x(t) = \sin(2\pi t) 

    This was solved for 40 seconds with a sampling rate of 50 Hz.

    TODO: Add fig from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        omega (Optional[float]): frequency of the sine wave.
        L (Optional[int]): amount of time to solve simulation for.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    """
    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    ts = [(np.sin(omega*t))[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def incommensurate_sine(omega1=np.pi, omega2=1,
                        L=100, fs=50, SampleSize=5000):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        omega1 (Optional[float]): frequency of the first sine wave.
        omega2 (Optional[float]): frequency of the second sine wave.
        L (Optional[int]): amount of time to solve simulation for.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    """

    t = np.linspace(0, L, int(L*fs))

    ts = [(np.sin(omega1*t) + np.sin(omega2*t))[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts
