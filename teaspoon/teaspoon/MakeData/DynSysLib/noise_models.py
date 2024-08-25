import numpy as np


def noise_models(system, dynamic_state=None, L=None, fs=None,
                 SampleSize=None, parameters=None, InitialConditions=None):
    if system == 'gaussian_noise':
        t, ts = gaussian_noise()

    if system == 'uniform_noise':
        t, ts = uniform_noise()

    if system == 'rayleigh_noise':
        t, ts = rayleigh_noise()

    if system == 'exponential_noise':
        t, ts = exponential_noise()

    return t, ts


def gaussian_noise(sigma=1.0, mu=0.0,
                   L=1000, fs=1, SampleSize=1000):
    """
    Generate a noise signal sampled from a Gaussian distribution. 

    .. figure:: ../../../figures/Noise_Models/gaussian_noise.png

    Parameters:
        sigma (Optional[float]): Standard deviation of the normal distribution.
        mu (Optional[float]): Mean of the normal distribution. 
        L (Optional[int]): amount of time to solve simulation for.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))
    ts = [(np.random.normal(mu, sigma, len(t)))[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def uniform_noise(a=-1.0, b=1.0,
                  L=1000, fs=1, SampleSize=1000):
    """
    Generate a noise signal sampled from a uniform distribution.

    .. figure:: ../../../figures/Noise_Models/uniform_noise.png

    Parameters:
        a (Optional[float]): Uniform distribution lower bound.
        b (Optional[float]): Uniform distribution upper bound. 
        L (Optional[int]): amount of time to solve simulation for.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))
    ts = [(np.random.uniform(a, b, size=len(t)))[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def rayleigh_noise(sigma=1.0,
                   L=1000, fs=1, SampleSize=1000):
    """
    Generate a noise signal sampled from a Rayleigh distribution. 

    .. figure:: ../../../figures/Noise_Models/rayleigh_noise.png

    Parameters:
        sigma (Optional[float]): Rayleigh distribution mode.
        L (Optional[int]): amount of time to solve simulation for.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))
    ts = [(np.random.rayleigh(sigma, size=len(t)))[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def exponential_noise(sigma=1.0,
                      L=1000, fs=1, SampleSize=1000):
    """
    Generate a noise signal sampled from an exponential distribution.

    .. figure:: ../../../figures/Noise_Models/exponential_noise.png

    Parameters:
        sigma (Optional[float]): Exponential distribution scale.
        L (Optional[int]): amount of time to solve simulation for.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))
    ts = [(np.random.exponential(sigma, len(t)))[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts
