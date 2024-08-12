import numpy as np

def noise_models(system, dynamic_state=None, L=None, fs=None,
                 SampleSize=None, parameters=None, InitialConditions=None):
    import numpy as np
    run = True
    if run == True:
        if system == 'gaussian_noise':
            # setting simulation time series parameters
            if fs == None:
                fs = 1
            if SampleSize == None:
                SampleSize = 1000
            if L == None:
                L = 1000
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print(
                        'Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    sigma, mu = parameters[0], parameters[1]
            if parameters == None:
                sigma, mu = 1, 0

            ts = [(np.random.normal(mu, sigma, len(t)))[-SampleSize:]]
            t = t[-SampleSize:]


        if system == 'uniform_noise':
            # setting simulation time series parameters
            if fs == None:
                fs = 1
            if SampleSize == None:
                SampleSize = 1000
            if L == None:
                L = 1000
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print(
                        'Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b = parameters[0], parameters[1]
            if parameters == None:
                a, b = -1, 1

            ts = [(np.random.uniform(a, b, size=len(t)))[-SampleSize:]]
            t = t[-SampleSize:]


        if system == 'rayleigh_noise':
            # setting simulation time series parameters
            if fs == None:
                fs = 1
            if SampleSize == None:
                SampleSize = 1000
            if L == None:
                L = 1000
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print(
                        'Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    sigma = parameters[0]
            if parameters == None:
                sigma = 1

            ts = [(np.random.rayleigh(sigma, size=len(t)))[-SampleSize:]]
            t = t[-SampleSize:]


        if system == 'exponential_noise':
            # setting simulation time series parameters
            if fs == None:
                fs = 1
            if SampleSize == None:
                SampleSize = 1000
            if L == None:
                L = 1000
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print(
                        'Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    sigma = parameters[0]
            if parameters == None:
                sigma = 1

            ts = [(np.random.exponential(sigma, len(t)))[-SampleSize:]]
            t = t[-SampleSize:]
    return t, ts



def gaussian_noise(sigma=1.0, mu=0.0, 
         L=1000, fs=1, SampleSize=1000):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

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
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

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
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

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
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

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