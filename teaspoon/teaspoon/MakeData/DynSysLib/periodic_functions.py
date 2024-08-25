import numpy as np


def periodic_functions(system, dynamic_state=None, L=None, fs=None,
                       SampleSize=None, parameters=None, InitialConditions=None):
    if system == 'sine':
        t, ts = sine()

    if system == 'incommensurate_sine':
        t, ts = incommensurate_sine()

    return t, ts


def sine(omega=2*np.pi,
         L=40, fs=50, SampleSize=2000):
    """
    The sinusoidal function is defined as

    .. math::
        x(t) = \sin(2\pi t) 

    This was solved for 40 seconds with a sampling rate of 50 Hz.

    .. figure:: ../../../figures/Periodic_Quasiperiodic_Functions/Periodic_Sinosoidal_Function.png

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
    This function is generated using two incommensurate periodic functions as

    .. math::
        x(t) = \sin(\\omega_1 t) + \sin(\\omega_2 t)

    This was sampled such that :math:`t \in [0, 100]` at a rate of 50 Hz.

    .. figure:: ../../../figures/Periodic_Quasiperiodic_Functions/Quasiperiodic_Function.png

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
