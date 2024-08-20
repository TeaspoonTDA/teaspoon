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
    
    where we chose the parameters :math:`x_0 = 0.5` and :math:`r = 3.6` for a chaotic state. You can set :math:`r = 3.5` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Logistic_map.png
    
    Parameters:
        r (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
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

    where we chose the parameters :math:`a = 1.20`, :math:`b = 0.30`, and :math:`c = 1.00` for a chaotic state with initial conditions :math:`x_0 = 0.1` and :math:`y_0 = 0.3`. You can set :math:`a = 1.25` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Henon_map.png
    
    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter.
        c (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    References
    ----------
    .. [2] Hénon, Michel. "A two-dimensional mapping with a strange attractor". Communications in Mathematical Physics, 1976.
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

    where we chose the parameter :math:`A = 1.0` for a chaotic state with initial condition :math:`x_0 = 0.1`. You can also change :math:`A = 0.8` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients. 

    .. figure:: ../../../figures/Maps/Sine_map.png

    Parameters:
        A (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
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


def tent_map(A=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=1, SampleSize=500):
    """
    The Tent map [3]_ was solved as

    .. math::
        x_{n+1} = A\min{([x_n, 1-x_n])}
    
    where we chose the parameter :math:`A = 1.50` for a chaotic state with initial condition :math:`x_0 = 1/\sqrt{2}`. You can also change :math:`A = 1.05` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Tent_map.png

    Parameters:
        A (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    References
    ----------
    .. [3] Crampin, Michael. "On the chaotic behaviour of the tent map". Teaching Mathematics and its Applications, 1994.
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
    The Linear Congruential Generator map is defined as

    .. math::
        x_{n+1}=(ax_n+b)\mod c
    
    where we chose the parameter :math:`a = 1.1` for a chaotic state with initial condition :math:`x_0 = 0.1`. You can also change :math:`a = 0.9` for a periodic response. :math:`b` and :math:`c` are set to 54,773 and 259,200 respectively for both dynamic states. We solve this system for 1000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Linear_Congruential_Generator_map.png
    
    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter.
        c (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
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
    The Ricker's Population map is defined [4]_ as

    .. math::
        x_{n+1}=ax_n e^{-x_n}
    
    where we chose the parameter :math:`a = 20` for a chaotic state with initial condition :math:`x_0 = 0.1`. You can set :math:`a = 13` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Rickers_Popoulation_map.png
    
    Parameters:
        a (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    References
    ----------
    .. [4] Ricker, William Edwin. "Stock and recruitment". Journal of the Fisheries Research Board of Canada, 1954.
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
    The Gauss map is defined [5]_ as

    .. math::
        x_{n+1}=e^{-\\alpha x_n^2}+\\beta
    
    where we chose the parameters :math:`\\alpha = 6.20` and :math:`\\beta = -0.35` for a chaotic state with initial condition :math:`x_0 = 0.1`. You can set :math:`\\beta = -0.20` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients. Taken from https://en.wikipedia.org/wiki/Gauss_iterated_map

    .. figure:: ../../../figures/Maps/Gauss_map.png
    
    Parameters:
        alpha (Optional[float]): System parameter.
        beta (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    References
    ----------
    .. [5] Hilborn, Robert C. "Chaos and nonlinear dynamics: an introduction for scientists and engineers". Oxford, Univ. Press, 2004.
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
    The Cusp map is defined [6]_ as

    .. math::
        x_{n+1} = 1 - a\sqrt{|x_n|}

    where we chose the parameter :math:`a = 1.2` for a chaotic state with initial condition :math:`x_0 = 0.5`. You can set :math:`a = 1.1` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Cusp_map.png

    Parameters:
        a (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    
    References
    ----------
    .. [6] Beck, Christian. "Thermodynamics of Chaotic Systems". Cambridge University Press, 2009.
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
    The Pincher's map is defined [7]_ as

    .. math::
        x_{n+1} = |\\tanh{(s(x_n-c))}|

    where we chose the parameters :math:`s = 1.6` and :math:`c = 0.5` for a chaotic state with initial condition :math:`x_0 = 0.0`. You can set :math:`s = 1.3` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Pinchers_map.png

    Parameters:
        s (Optional[float]): System parameter.
        c (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [7] Akgul, Akif. "Text Encryption by Using One-Dimensional Chaos Generators and Nonlinear Equations". International Conference on Electrical and Electronics Engineering, ELECO, 2013.
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
    The Sine Circle map is defined [8]_ as

    .. math::
        x_{n+1} = x_n + \\omega -\\left[\\frac{k}{2\\pi}\sin{(2\\pi x_n)}\\right]

    where we chose the parameters :math:`\\omega = 0.5` and :math:`k = 2.0` for a chaotic state with initial condition :math:`x_0 = 0.0`. You can set :math:`k = 1.5` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Sine_Circle_map.png

    Parameters:
        omega (Optional[float]): System parameter.
        k (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [8] Arnold, V.I. "Small denominators. i. mapping the circle onto itself". Izv. Akad. Nauk SSSR Ser. Mat., 1961.
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
    The Lozi map is defined [9]_ as

    .. math::
        x_{n+1} &= 1 - a|x_n| +by_n,

        y_{n+1} &= x_n

    where we chose the parameters :math:`a = 1.7` and :math:`b = 0.5` for a chaotic state with initial conditions :math:`x_0 = -0.1` and :math:`y_0 = 0.1`. You can set :math:`a = 1.5` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Lozi_map.png

    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    References
    ----------
    .. [9] Peitgen, Heinz-Otto. "Chaos and Fractals New Frontiers of Science". Springer-Verlag, 1992.
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
    The Delayed Logistic map is defined [10]_ as

    .. math::
        x_{n+1} &= ax_n(1-y_n)

        y_{n+1} &= x_n

    where we chose the parameter :math:`a = 2.27` for a chaotic state with initial conditions :math:`x_0 = 0.001` and :math:`y_0 = 0.001`. You can set :math:`a = 2.20` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Delayed_Logistic_map.png
    
    Parameters:
        a (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    References
    ----------
    .. [10] Ismail, Samar M. "Generalized Delayed Logistic Map Suitable For Pseudo-random Number Generation". International Conference on Science and Technology, 2015.
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
    The Tinkerbell map is defined [11]_ as

    .. math::
        x_{n+1} &= x_n^2 - y_n^2 + ax_n + by_n,

        y_{n+1} &= 2x_ny_n + cx_n + dy_n

    where we chose the parameters :math:`a = 0.9`, :math:`b = -0.6`, :math:`c = 2.0`, and :math:`d = 0.5` for a chaotic state with initial conditions :math:`x_0 = 0.0` and :math:`y_0 = 0.5`. You can set :math:`a = 0.7` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Tinkerbell_map.png
    
    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter.
        c (Optional[float]): System parameter.
        d (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [11] Goldsztejn, Alexandre M. "Tinkerbell is chaotic". SIAM Journal on Applied Dynamical Systems, 2011.
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
    The Burger's map is defined [12]_ as

    .. math::
        x_{n+1} &= ax_n - y_n^2,

        y_{n+1} &= by_n + x_ny_n

    where we chose the parameters :math:`a = 0.75` and :math:`b = 1.75` for a chaotic state with initial conditions :math:`x_0 = -0.1` and :math:`y_0 = 0.5`. You can set :math:`b = 1.60` for a periodic response. We solve this system for 3000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Burgers_map.png

    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    References
    ----------
    .. [12] Burgers, J. M. "Mathematical examples illustrating relations occurring in the theory of turbulent fluid motion". Trans. Roy. Neth. Acad. Sci. Amsterdam., 1995.
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
    The Holme's Cubic map is defined [13]_ as

    .. math::
        x_{n+1} &= y_n,

        y_{n+1} &= -bx_n + dy_n - y_n^3

    where we chose the parameters :math:`b = 0.20` and :math:`d = 2.77` for a chaotic state with initial conditions :math:`x_0 = -0.1` and :math:`y_0 = 0.5`. You can set :math:`b = 0.27` for a periodic response. We solve this system for 3000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Holmes_Cubic_map.png

    Parameters:
        b (Optional[float]): System parameter.
        d (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    References
    ----------
    .. [13] Chavoya-Aceves, O. "Symbolic dynamics of the cubic map". Physica D: Nonlinear Phenomena., 1985.
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
    The Kaplan Yorke map is defined [14]_ as

    .. math:: 
        x_{n+1} &= [ax_n](\mod 1),

        y_{n+1} &= by_n + \cos{(4\\pi x_n)}

    where we chose the parameters :math:`a = -2.0` and :math:`b = 0.2` for a chaotic state with initial conditions :math:`x_0 = -0.1` and :math:`y_0 = 0.5`. You can set :math:`a = -1.0` for a periodic response. We solve this system for 1000 data points and keep the second 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Kaplan_Yorke_map.png
        
    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    References
    ----------
    .. [14] Peitgen, Heinz-Otto. "Functional Differential Equations and Approximation of Fixed Points". Springer., 1979.
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


def gingerbread_man_map(a=1.0, b=1.0, dynamic_state=None, InitialConditions=None,  
         L=2000, fs=1, SampleSize=500):
    """
    The Gingerbread Man Map is defined [15]_ [16]_ as

    .. math::
        x_{n+1} &= 1 - ay_n + n|x_n|,

        y_{n+1} &= x_n

    where we chose the parameters :math:`a = 1.0` and :math:`b = 1.0`. For a chaotic state, initial conditions :math:`x_0 = 0.5` and :math:`y_0 = 1.8`, and for a periodic response :math:`x_0 = 0.5` and :math:`y_0 = 1.5`. We solve this system for 2000 data points and keep the last 500 to avoid transients.

    .. figure:: ../../../figures/Maps/Gingerbread_Man_map.png
    
    Parameters:
        a (Optional[float]): System parameter.
        b (Optional[float]): System parameter. 
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    References
    ----------
    .. [15] Devaney, Robert. "The gingerbreadman". Algorithms, 1992.
    .. [16] Devaney, Robert. "A piecewise linear model for the zones of instability of an area-preserving map". Physica D: Nonlinear Phenomena, 1984.
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




























