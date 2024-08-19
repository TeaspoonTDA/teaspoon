import numpy as np

def maps(system, dynamic_state=None, L=None, fs=None,
         SampleSize=None, parameters=None, InitialConditions=None):

    # https://www.mathworks.com/matlabcentral/fileexchange/34820-gingerbread_man-iterated-chaotic-map-with-parameters-attractor-explorer

    if system == 'gingerbread_man_map':
        t, ts = gingerbread_man_map()

    if system == 'sine_map':
        t, ts = sine_map()

    if system == 'tent_map':
        t, ts = tent_map()

    if system == 'linear_congruential_generator_map':
        t, ts = linear_congruential_generator_map()

    if system == 'rickers_population_map':
        t, ts = rickers_population_map()

    if system == 'gauss_map':
        t, ts = gauss_map()

    if system == 'cusp_map':
        t, ts = cusp_map()

    if system == 'pinchers_map':
        t, ts = pinchers_map()

    if system == 'sine_circle_map':
        t, ts = sine_circle_map()

    if system == 'logistic_map':
        t, ts = logistic_map()

    if system == 'henon_map':
        t, ts = henon_map()
        
    if system == 'lozi_map':
        t, ts = lozi_map()

    if system == 'delayed_logstic_map':
        t, ts = delayed_logstic_map()

    if system == 'tinkerbell_map':
        t, ts = tinkerbell_map()

    if system == 'burgers_map':
        t, ts = burgers_map()

    if system == 'holmes_cubic_map':
        t, ts = holmes_cubic_map()

    if system == 'kaplan_yorke_map':
        t, ts = kaplan_yorke_map()

    return t, ts


def logistic_map(r=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    The logistic map [1]_ was generated as

    .. math::
        x_{n+1} = rx_n(1-x_n)
    
    where we chose the parameters :math:`x_0 = 0.5` and :math:`r = 3.6` for a chaotic state. You can set :math:`r = 3.5` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients. A figure of the resulting time series can be found `here <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_.



    Parameters:
        r (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    References
    ----------
    .. [1] May, Robert. "Simple mathematical models with very complicated dynamics". Nature, 1976.
    """

    t = np.linspace(0, L, int(L*fs))

    if r == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 1 parameter. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            r = 3.5
        if dynamic_state == 'chaotic':
            r = 3.6

    if InitialConditions == None:
        InitialConditions = [0.5]
    xn = InitialConditions[0]

    t, ts = [], []
    for n in range(0, int(L)):
        xn = r*xn*(1-xn)
        ts = np.append(ts, xn)
        t = np.append(t, n)

    ts = [ts[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def henon_map(a=None, b=None, c=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    The Hénon map [2]_ was solved as

    .. math::

        x_{n+1} &= 1 -ax_n^2+y_n,

        y_{n+1}&=bx_n

    where we chose the parameters :math:`a = 1.20`, :math:`b = 0.30`, and :math:`c = 1.00` for a chaotic state with initial conditions :math:`x_0 = 0.1` and :math:`y_0 = 0.3`. You can set :math:`a = 1.25` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients. A figure of the resulting time series can be found `here <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_.

    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter.
        c (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    References
    ----------
    .. [2] Hénon Michel. "A two-dimensional mapping with a strange attractor". Communications in Mathematical Physics, 1976.
    """

    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    if a == None or b == None or c == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            a = 1.25
        if dynamic_state == 'chaotic':
            a = 1.20
        b = 0.3
        c = 1.0

    # defining simulation functions
    def henon(a, b, c, x, y):
        return y + c - a*x*x, b*x

    if InitialConditions == None:
        InitialConditions = [0.1, 0.3]

    xtemp = InitialConditions[0]
    ytemp = InitialConditions[1]
    x, y = [], []
    for n in range(0, int(L)):
        xtemp, ytemp = henon(a, b, c, xtemp, ytemp)
        x.append(xtemp)
        y.append(ytemp)

    ts = [x[-SampleSize:], y[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def sine_map(A=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):

    """
    The Sine map is defined as

    .. math::
        x_{n+1} = A\sin{(\pi x_n)}

    where we chose the parameter :math:`A = 1.0` for a chaotic state with initial condition :math:`x_0 = 0.1`. You can also change :math:`A = 0.8` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients. A figure of the resulting time series can be found `here <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_.

    Parameters:
        A (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    if A == None:
        if dynamic_state == None:
            print('A parameter not specified. Defaulting to periodic response (A=0.8).')
            dynamic_state = 'periodic'
        if dynamic_state == 'periodic':
            A = 0.8
        if dynamic_state == 'chaotic':
            A = 1.0

    if InitialConditions == None:
        InitialConditions = [0.1]
    xn = InitialConditions[0]

    t, ts = [], []
    for n in range(0, int(L)):
        xn = A*np.sin(np.pi*xn)
        ts = np.append(ts, xn)
        t = np.append(t, n)

    ts = [ts[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts





















def gingerbread_man_map(a=1.0, b=1.0, dynamic_state=None, InitialConditions=None,  
         L=2000, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_
    https://www.mathworks.com/matlabcentral/fileexchange/34820-gingerbread_man-iterated-chaotic-map-with-parameters-attractor-explorer

    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter. 
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    # defining simulation functions

    def gingerbread_man(a, b, x, y):
        return 1 - a*y + b*np.abs(x), x

    if InitialConditions == None:
        if dynamic_state == None:
            print('Either dynamic_state or InitialConditions need to be specified. Defaulting to periodic response.')
            dynamic_state='periodic'
        if dynamic_state == 'periodic':
            InitialConditions = [0.5, 1.5]
        if dynamic_state == 'chaotic':
            InitialConditions = [0.5, 1.8]

    xtemp = InitialConditions[0]
    ytemp = InitialConditions[1]
    x, y = [], []
    for n in range(0, int(L)):
        xtemp, ytemp = gingerbread_man(a, b, xtemp, ytemp)
        x.append(xtemp)
        y.append(ytemp)

    ts = [x[-SampleSize:], y[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts





def tent_map(A=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        A (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    if A == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
            
        if dynamic_state == 'periodic':
            A = 1.05
        if dynamic_state == 'chaotic':
            A = 1.5

    if InitialConditions == None:
        InitialConditions = [1/np.sqrt(2)]
    xn = InitialConditions[0]

    t, ts = [], []
    for n in range(0, int(L)):
        xn = A*np.min([xn, 1-xn])
        ts = np.append(ts, xn)
        t = np.append(t, n)

    ts = [ts[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def linear_congruential_generator_map(a=None, b=None, c=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter.
        c (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """
    t = np.linspace(0, L, int(L*fs))

    if a == None or b == None or c == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
            
        if dynamic_state == 'periodic':
            a = 0.9
        if dynamic_state == 'chaotic':
            a = 1.1
        b, c = 54773, 259200

    if InitialConditions == None:
        InitialConditions = [0.1]
    xn = InitialConditions[0]

    t, ts = [], []
    for n in range(0, int(L)):
        xn = (a*xn + b) % c
        ts = np.append(ts, xn)
        t = np.append(t, n)

    ts = [ts[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def rickers_population_map(a=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        a (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    if a == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 1 parameter. Defaulting to periodic solution parameters.')
            
        if dynamic_state == 'periodic':
            a = 13
        if dynamic_state == 'chaotic':
            a = 20

    if InitialConditions == None:
        InitialConditions = [0.1]
    xn = InitialConditions[0]

    t, ts = [], []
    for n in range(0, int(L)):
        xn = a*xn*np.exp(-xn)
        ts = np.append(ts, xn)
        t = np.append(t, n)

    ts = [ts[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def gauss_map(alpha=None, beta=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    taken from https://en.wikipedia.org/wiki/Gauss_iterated_map

    Parameters:
        alpha (Optional[float]): System parameter.
        beta (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    if alpha == None or beta == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
            
        if dynamic_state == 'periodic':
            beta = -0.20
        if dynamic_state == 'chaotic':
            beta = -0.35
        alpha = 6.20

    if InitialConditions == None:
        InitialConditions = [0.1]
    xn = InitialConditions[0]

    t, ts = [], []
    for n in range(0, int(L)):
        xn = np.exp(-alpha*xn**2) + beta
        ts = np.append(ts, xn)
        t = np.append(t, n)

    ts = [ts[-SampleSize:]]
    t = t[-SampleSize:]
    
    return t, ts


def cusp_map(a=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        a (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    if a == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 1 parameter. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            a = 1.1
        if dynamic_state == 'chaotic':
            a = 1.2

    if InitialConditions == None:
        InitialConditions = [0.5]
    xn = InitialConditions[0]

    t, ts = [], []
    for n in range(0, int(L)):
        xn = 1-a*np.sqrt(np.abs(xn))
        ts = np.append(ts, xn)
        t = np.append(t, n)

    ts = [ts[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def pinchers_map(s=None, c=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        s (Optional[float]): System parameter.
        c (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    if s == None or c == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            s = 1.3
        if dynamic_state == 'chaotic':
            s = 1.6
        c = 0.5

    if InitialConditions == None:
        InitialConditions = [0.0]
    xn = InitialConditions[0]

    t, ts = [], []
    for n in range(0, int(L)):
        xn = np.abs(np.tanh(s*(xn-c)))
        ts = np.append(ts, xn)
        t = np.append(t, n)

    ts = [ts[-SampleSize:]]
    t = t[-SampleSize:]
    
    return t, ts


def sine_circle_map(omega=None, k=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        omega (Optional[float]): System parameter.
        k (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    if omega == None or k == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            k = 1.5
        if dynamic_state == 'chaotic':
            k = 2.0
        omega = 0.5

    if InitialConditions == None:
        InitialConditions = [0.0]
    xn = InitialConditions[0]

    t, ts = [], []
    for n in range(0, int(L)):
        xn = xn + omega - (k/(2*np.pi))*np.sin(2*np.pi*xn) % 1
        ts = np.append(ts, xn)
        t = np.append(t, n)

    ts = [ts[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts



def lozi_map(a=None, b=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """
    
    t = np.linspace(0, L, int(L*fs))

    
    if a == None or b == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            a = 1.5
        if dynamic_state == 'chaotic':
            a = 1.7
        b = 0.5

    # defining simulation functions
    def lozi(a, b, x, y):
        return 1-a*np.abs(x) + b*y, x

    if InitialConditions == None:
        InitialConditions = [-0.1, 0.1]

    xtemp = InitialConditions[0]
    ytemp = InitialConditions[1]
    x, y = [], []
    for n in range(0, int(L)):
        xtemp, ytemp = lozi(a, b, xtemp, ytemp)
        x.append(xtemp)
        y.append(ytemp)

    ts = [x[-SampleSize:], y[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def delayed_logstic_map(a=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        a (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    if a == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 1 parameter. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            a = 2.20
        if dynamic_state == 'chaotic':
            a = 2.27

    # defining simulation functions
    def delay_log(a, x, y):
        return a*x*(1-y), x

    if InitialConditions == None:
        InitialConditions = [0.001, 0.001]

    xtemp = InitialConditions[0]
    ytemp = InitialConditions[1]
    x, y = [], []
    for n in range(0, int(L)):
        xtemp, ytemp = delay_log(a, xtemp, ytemp)
        x.append(xtemp)
        y.append(ytemp)

    ts = [x[-SampleSize:], y[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def tinkerbell_map(a=None, b=None, c=None, d=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter.
        c (Optional[float]): System parameter.
        d (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    if a == None or b == None or c == None or d == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 4 parameters. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            a = 0.7
        if dynamic_state == 'chaotic':
            a = 0.9
        b, c, d = -0.6, 2, 0.5

    # defining simulation functions
    def tinkerbell(a, b, c, d, x, y):
        return x**2 - y**2 + a*x + b*y, 2*x*y + c*x + d*y

    if InitialConditions == None:
        InitialConditions = [0.0, 0.5]

    xtemp = InitialConditions[0]
    ytemp = InitialConditions[1]
    x, y = [], []
    for n in range(0, int(L)):
        xtemp, ytemp = tinkerbell(a, b, c, d, xtemp, ytemp)
        x.append(xtemp)
        y.append(ytemp)

    ts = [x[-SampleSize:], y[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def burgers_map(a=None, b=None, dynamic_state=None, InitialConditions=None,  
    L=3000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    if a == None or b == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            b = 1.6
        if dynamic_state == 'chaotic':
            b = 1.75
        a = 0.75

    # defining simulation functions

    def burgers(a, b, x, y):
        return a*x - y**2, b*y + x*y

    if InitialConditions == None:
        InitialConditions = [-0.1, 0.1]

    xtemp = InitialConditions[0]
    ytemp = InitialConditions[1]
    x, y = [], []
    for n in range(0, int(L)):
        xtemp, ytemp = burgers(a, b, xtemp, ytemp)
        x.append(xtemp)
        y.append(ytemp)

    ts = [x[-SampleSize:], y[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def holmes_cubic_map(b=None, d=None, dynamic_state=None, InitialConditions=None,  
    L=3000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        b (Optional[float]): System parameter.
        d (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    if b == None or d == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            b = 0.27
        if dynamic_state == 'chaotic':
            b = 0.20
        d = 2.77

    # defining simulation functions

    def holmes(b, d, x, y):
        return y, -b*x + +d*y - y**3

    if InitialConditions == None:
        InitialConditions = [1.6, 0.0]

    xtemp = InitialConditions[0]
    ytemp = InitialConditions[1]
    x, y = [], []
    for n in range(0, int(L)):
        xtemp, ytemp = holmes(b, d, xtemp, ytemp)
        x.append(xtemp)
        y.append(ytemp)

    ts = [x[-SampleSize:], y[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def kaplan_yorke_map(a=None, b=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    Add description from `Audun's pdf <https://teaspoontda.github.io/teaspoon/_downloads/8d622bebe5abdc608bbc9616ffa444d9/dynamic_systems_library.pdf>`_

    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter.
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    """

    t = np.linspace(0, L, int(L*fs))

    if a == None or b == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            a = -1.0
        if dynamic_state == 'chaotic':
            a = -2.0
        b = 0.2

    # defining simulation functions

    def kaplan_yorke(a, b, x, y):
        return (a*x) % 0.99995, (b*y + np.cos(4*np.pi*x))

    if InitialConditions == None:
        InitialConditions = [1/np.sqrt(2), -0.4]

    xtemp = InitialConditions[0]
    ytemp = InitialConditions[1]
    x, y = [], []
    for n in range(0, int(L)):
        xtemp, ytemp = kaplan_yorke(a, b, xtemp, ytemp)
        x.append(xtemp)
        y.append(ytemp)

    ts = [x[-SampleSize:], y[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts