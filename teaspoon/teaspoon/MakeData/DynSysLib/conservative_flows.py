from scipy.integrate import odeint
import numpy as np


def conservative_flows(system, dynamic_state=None, L=None, fs=None,
                       SampleSize=None, parameters=None,
                       InitialConditions=None):

    if system == 'simplest_driven_chaotic_flow':
        t, ts = simplest_driven_chaotic_flow()

    if system == 'nose_hoover_oscillator':
        t, ts = nose_hoover_oscillator()

    if system == 'labyrinth_chaos':
        t, ts = labyrinth_chaos()

    if system == 'henon_heiles_system':
        t, ts = henon_heiles_system()

    return t, ts


def simplest_driven_chaotic_flow(fs=50, SampleSize=5000, L=300.0, parameters=[1], InitialConditions=[0, 0], dynamic_state=None):
    """
    The simplest driven chaotic flow can be reproduced with the following equations

    .. math::
        \dot{x} = y,

        \dot{y} = -x^3 + \sin{\\omega t}

    where we chose the parameters :math:`(x_0. y_0) = (0.0, 0.0)` for initial conditions, and :math:`\\omega = 1.0` (periodic) and :math:`\\omega = 1.88` (chaotic). 

    .. figure:: ../../../figures/Conservative_Flows/simplest_driven_chaotic_flow.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): array [:math:`\\omega`] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    """

    t = np.linspace(0, L, int(L*fs))

    num_param = 1

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            omega = 1
        elif dynamic_state == 'chaotic':
            omega = 1.88
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        omega = parameters[0]

    # defining simulation functions
    def simplest_driven_chaotic_flow(state, t):
        x, y = state  # unpack the state vector
        return y, -x**3 + np.sin(omega*t)

    states = odeint(simplest_driven_chaotic_flow, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def nose_hoover_oscillator(fs=20, SampleSize=5000, L=500.0, parameters=[6], InitialConditions=[0, 5, 0], dynamic_state=None):
    """
    The Nose Hoover Oscillator is represented by the following equations

    .. math::
        \dot{x} = y,

        \dot{y} = -x + yz,

        \dot{z} = a - y^2

    where we chose the parameters :math:`(x_0, y_0, z_0) = (0.0, 5.0, 0.0)` for initial conditions, and :math:`a = 6.0` (periodic) and :math:`a = 1` (chaotic). 

    .. figure:: ../../../figures/Conservative_Flows/nose_hoover_oscillator.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): list of values for [:math:`a`] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]    
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    """

    t = np.linspace(0, L, int(L*fs))

    num_param = 1

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 6
        elif dynamic_state == 'chaotic':
            a = 1
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        a = parameters[0]

    # defining simulation functions
    def nose_hoover_oscillator(state, t):
        x, y, z = state  # unpack the state vector
        return y, -x+y*z, a-y**2

    states = odeint(nose_hoover_oscillator, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
          [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def labyrinth_chaos(fs=10, SampleSize=5000, L=2000.0, parameters=[1, 1, 1], InitialConditions=[0.1, 0, 0], dynamic_state=None):
    """
    The Labyrinth Chaos Oscillator is represented by the following equations

    .. math::
        \dot{x} = y,

        \dot{y} = -x + yz,

        \dot{z} = a - y^2

    where we chose the parameters :math:`(x_0, y_0, z_0) = (0.1, 0.0, 0.0)` for initial conditions, and :math:`a = 1.0` (chaotic) with :math:`b = 1` and :math:`c = 1`. We could not find a periodic response. Any contributions would be appreciated!

    .. figure:: ../../../figures/Conservative_Flows/labyrinth_chaos.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): list of values for [:math:`a`, :math:`b`, :math:`c`] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    """

    t = np.linspace(0, L, int(L*fs))

    num_param = 3

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            print('We could not find a periodic response. Using the chaotic parameters.')
            a, b, c = 1, 1, 1
        elif dynamic_state == 'chaotic':
            a, b, c = 1, 1, 1
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        a, b, c = parameters[0], parameters[1], parameters[2]

    # defining simulation functions
    def labyrinth_chaos(state, t):
        x, y, z = state  # unpack the state vector
        return a*np.sin(y), b*np.sin(z), c*np.sin(x)

    states = odeint(labyrinth_chaos, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
          [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def henon_heiles_system(fs=20, SampleSize=5000, L=10000.0, parameters=[1], InitialConditions=[0.499, 0, 0, 0.03], dynamic_state=None):
    """
    The Henon Heiles System is represented by the following equations

    .. math::
        \dot{x} = px,

        \dot{px} = -x - 2axy,

        \dot{y} = py,

        \dot{py} = -y - a(x^2 - y^2)

    where we chose the parameters :math:`(x_0, px_0, y_0, py_0) = (0.499, 0, 0, 0.03)` for initial conditions, and :math:`a = 1.0` (chaotic). We could not find a periodic response. Any contributions would be appreciated!

    .. figure:: ../../../figures/Conservative_Flows/henon_heiles_system.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): list of values for [:math:`a`] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`px_0`, :math:`y_0`, :math:`py_0`]    
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    """

    t = np.linspace(0, L, int(L*fs))

    num_param = 1

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            print('We could not find a periodic response. Using the chaotic parameters.')
            a = 1
        elif dynamic_state == 'chaotic':
            a = 1
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        a = parameters[0]

    # defining simulation functions

    def henon_heiles_system(state, t):
        x, px, y, py = state  # unpack the state vector
        return px, -x - 2*a*x*y, py, -y - a*(x**2 - y**2)

    states = odeint(henon_heiles_system, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:],
          (states[:, 2])[-SampleSize:], (states[:, 3])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts
