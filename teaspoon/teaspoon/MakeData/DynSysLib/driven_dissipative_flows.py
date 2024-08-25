from scipy.integrate import odeint
import numpy as np


def driven_dissipative_flows(system, dynamic_state=None, L=None, fs=None,
                             SampleSize=None, parameters=None,
                             InitialConditions=None):

    if system == 'base_excited_magnetic_pendulum':
        t, ts = base_excited_magnetic_pendulum()

    if system == 'driven_pendulum':
        t, ts = driven_pendulum()

    if system == 'driven_van_der_pol_oscillator':
        t, ts = driven_van_der_pol_oscillator()

    if system == 'shaw_van_der_pol_oscillator':
        t, ts = shaw_van_der_pol_oscillator()

    if system == 'forced_brusselator':
        t, ts = forced_brusselator()

    if system == 'ueda_oscillator':
        t, ts = ueda_oscillator()

    if system == 'duffings_two_well_oscillator':
        t, ts = duffings_two_well_oscillator()

    if system == 'duffing_van_der_pol_oscillator':
        t, ts = duffing_van_der_pol_oscillator()

    if system == 'rayleigh_duffing_oscillator':
        t, ts = rayleigh_duffing_oscillator()

    return t, ts


def base_excited_magnetic_pendulum(parameters=[0.1038, 0.208, 9.81, 0.18775, 0.00001919, 0.022, 3*np.pi, 0.003, 1.2, 0.032, 1.257E-6], fs=200, SampleSize=5000, L=100.0, dynamic_state=None, InitialConditions=[0.0, 0.0]):
    """
    This is a simple pendulum with a magnet at its base. See Myers & Khasawneh [1]_. The system was simulated for 100 seconds at a rate of 200 Hz and the last 25 seconds were used for the chaotic response as shown in the figure below.

    .. figure:: ../../../figures/Driven_Dissipative_Flows/Base_Excited_Magnetic_Pendulum_Setup.png
    .. figure:: ../../../figures/Driven_Dissipative_Flows/Base_Excited_Magnetic_Pendulum.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): list of values for [m, l, g, r_cm, I_o, A, w, c, q, d, mu] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

        Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [1] Audun Myers and Firas A. Khasawneh. Dynamic State Analysis of a Driven Magnetic Pendulum using Ordinal Partition Networks and Topological Data Analysis. ASME 32nd Conference on Mechanical Vibration and Noise (VIB), November 3, 2020.

    """

    import sympy as sp
    # NEW EQUATIONS NEEDED (CORRECT)
    m, l, g, r_cm, I_o, A, w, c, q, d, mu = sp.symbols(
        "m \ell g r_{cm} I_o A \omega c q d \mu")  # declare constants
    t = sp.symbols("t")  # declare time variable
    # declare time dependent variables
    th = sp.Function(r'\theta')(t)

    r = sp.sqrt((l)**2 + (d+l)**2 - 2*l*(l+d)*sp.cos(th))
    a = 2*np.pi - np.pi/2
    b = (np.pi/2) - th
    phi = np.pi/2 - sp.asin((l/r)*sp.sin(th))
    Fr = (3*mu*q**2/(4*np.pi*r**4))*(2*sp.cos(phi-a) *
                                     sp.cos(phi-b) - sp.sin(phi-a)*sp.sin(phi-b))
    Fphi = (3*mu*q**2/(4*np.pi*r**4))*(sp.sin(2*phi-a-b))

    tau_m = l*Fr*sp.cos(phi-th) - l*Fphi*sp.sin(phi-th)
    tau_damping = c*th.diff(t)

    V = -m*g*r_cm*sp.cos(th)
    vx = r_cm*th.diff(t)*sp.cos(th) + A*w*sp.cos(w*t)
    vy = r_cm*th.diff(t)*sp.sin(th)
    T = (1/2)*I_o*th.diff(t)**2 + (1/2)*m*(vx**2 + vy**2)
    R = tau_damping + tau_m

    L_eq = T - V

    R_symb = sp.symbols("R_s")
    EOM = (L_eq.diff(th.diff(t))).diff(t) - L_eq.diff(th) + \
        R_symb  # lagranges equation applied

    # first solve both EOM for th1dd and th2dd
    thdd = sp.solve(EOM, th.diff(t).diff(t))[0]

    # we first need to change to th_1 and om_1 symbols and not functions to apply lambdify.
    ph, phi_dot = sp.symbols(r"\phi \dot{\phi}")
    phidd = thdd.subs([(R_symb, R)])
    phidd = phidd.subs([(th.diff(t), phi_dot)])
    phidd = phidd.subs([(th, ph)])

    # lambdified functions
    F_phidd = sp.lambdify(
        [(t, ph, phi_dot), (m, l, g, r_cm, I_o, A, w, c, q, d, mu)], phidd)

    t = np.linspace(0, L, int(L*fs))

    # setting system parameters

    num_param = 11

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            m, l, g, r_cm, I_o, A, w, c, q, d, mu = 0.1038, 0.208, 9.81, 0.18775, 0.00001919, 0.022, 3 * \
                np.pi, 0.003, 1.2, 0.032, 1.257E-6
        elif dynamic_state == 'chaotic':
            m, l, g, r_cm, I_o, A, w, c, q, d, mu = 0.1038, 0.208, 9.81, 0.18775, 0.00001919, 0.021, 3 * \
                np.pi, 0.003, 1.2, 0.032, 1.257E-6
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        m, l, g, r_cm, I_o = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
        A, w, c, q, d, mu = parameters[5], parameters[6], parameters[
            7], parameters[8], parameters[9], parameters[10]

    def vectorfield(state, t):
        ph, phi_dot = state
        return phi_dot, F_phidd((t, ph, phi_dot), (m, l, g, r_cm, I_o, A, w, c, q, d, mu))

    states = odeint(vectorfield, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def driven_pendulum(fs=50, SampleSize=5000, L=300, parameters=[1, 9.81, 1, 0.1, 5, 1], InitialConditions=[0, 0], dynamic_state=None):
    """
    The point mass, driven simple pendulum with viscous damping is described as

    .. math::
        \dot{\\theta} = \\omega,

        \dot{\\omega} = -\\frac{g}{l}\sin{\\theta} + \\frac{A}{ml^2}\sin{\\omega_m t} - c\\omega

    where we chose the parameters :math:`g = 9.81` for gravitational constant, :math:`l = 1` for the length of pendulum arm, :math:`m = 1` for the point mass, :math:`A = 5` for the amplitude of force, and :math:`\\omega = 1` for driving force (periodic) and :math:`\\omega = 2` for a chaotic state. The system was simulated for 300 seconds at a rate of 50 Hz and the last 100 seconds were used for the chaotic response as shown in the figure below.

    .. figure:: ../../../figures/Driven_Dissipative_Flows/Driven_Simple_Pendulum.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): list of values for [m, g, l, c, A, w] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`\\theta_0`, :math:`\\omega_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    """

    t = np.linspace(0, L, int(L*fs))

    num_param = 6

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            m, g, l, c, A, w = 1, 9.81, 1, 0.1, 5, 1
        elif dynamic_state == 'chaotic':
            m, g, l, c, A, w = 1, 9.81, 1, 0.1, 5, 2
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        m, g, l, c, A, w = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]

    # defining simulation functions
    def driven_pendulum(state, t):
        th, om = state  # unpack the state vector
        return om, (-g/l)*np.sin(th) + (A/(m*l**2))*np.sin(w*t) - c*om

    states = odeint(driven_pendulum, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def driven_van_der_pol_oscillator(fs=40, SampleSize=5000, L=300, parameters=[2.9, 5, 1.788], InitialConditions=[-1.9, 0], dynamic_state=None):
    """
    The Driven Van der Pol Oscillator is defined as

    .. math::
        \dot{x} = y,

        \dot{y} = -x + b(1 - x^2)y + A\sin{\\omega t}

    where we chose the parameters :math:`(x_0. y_0) = (-1.0, 0.0)` for initial conditions, :math:`A = 5`, :math:`\\omega = 1.788`, :math:`b = 2.9` (periodic) and :math:`b = 3.0` (chaotic). 

    .. figure:: ../../../figures/Driven_Dissipative_Flows/Driven_VanderPol_Oscillator.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): list of values for [b, A, :math:`\\omega`] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`]
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
            b, A, omega = 2.9, 5, 1.788
        elif dynamic_state == 'chaotic':
            b, A, omega = 3.0, 5, 1.788
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        b, A, omega = parameters[0], parameters[1], parameters[2]

    # defining simulation functions

    def driven_van_der_pol(state, t):
        x, y = state  # unpack the state vector
        return y, -x + b*(1-x**2)*y + A*np.sin(omega*t)

    states = odeint(driven_van_der_pol, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def shaw_van_der_pol_oscillator(fs=25, SampleSize=5000, L=500.0, parameters=[1, 5, 1.4], InitialConditions=[1.3, 0], dynamic_state=None):
    """
    The Shaw Van der Pol Oscillator is defined as

    .. math::
        \dot{x} = y + \sin{\\omega t},

        \dot{y} = -x + b(1 - x^2)y

    where we chose the parameters :math:`(x_0. y_0) = (1.3, 0.0)` for initial conditions, :math:`A = 5`, :math:`b = 1`, :math:`\\omega = 1.4` (periodic) and :math:`\\omega = 1.8` (chaotic). 

    .. figure:: ../../../figures/Driven_Dissipative_Flows/Shaw_VanderPol_Oscillator.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): list of values for [b, A, :math:`\\omega`] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`]    
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
            b, A, omega = 1, 5, 1.4
        elif dynamic_state == 'chaotic':
            b, A, omega = 1, 5, 1.8
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        b, A, omega = parameters[0], parameters[1], parameters[2]

    # defining simulation functions
    def shaw_van_der_pol_oscillator(state, t):
        x, y = state  # unpack the state vector
        return y + np.sin(omega*t), -x + b*(1-x**2)*y

    states = odeint(shaw_van_der_pol_oscillator, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def forced_brusselator(fs=20, SampleSize=5000, L=500.0, parameters=[0.4, 1.2, 0.05, 1.1], InitialConditions=[0.3, 2], dynamic_state=None):
    """
    The Forced Brusselator is defined as

    .. math::
        \dot{x} = x^2y - (b+1)x + a + A\sin{\\omega t},

        \dot{y} = -x^2y + bx

    where we chose the parameters :math:`(x_0. y_0) = (0.3, 2.0)` for initial conditions, :math:`a = 0.4`, :math:`b = 1.2`, :math:`A = 0.05`, :math:`\\omega = 1.1` (periodic) and :math:`\\omega = 1.0` (chaotic). 

    .. figure:: ../../../figures/Driven_Dissipative_Flows/Forced_Brusselator.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): list of values for [a, b, A, :math:`\\omega`] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`]    
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    """

    t = np.linspace(0, L, int(L*fs))

    num_param = 4

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a, b, A, omega = 0.4, 1.2, 0.05, 1.1
        elif dynamic_state == 'chaotic':
            a, b, A, omega = 0.4, 1.2, 0.05, 1.0
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        a, b, A, omega = parameters[0], parameters[1], parameters[2], parameters[3]

    # defining simulation functions
    def forced_brusselator(state, t):
        x, y = state  # unpack the state vector
        return (x**2)*y - (b+1)*x + a + A*np.sin(omega*t), -(x**2)*y + b*x

    states = odeint(forced_brusselator, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def ueda_oscillator(fs=50, SampleSize=5000, L=500.0, parameters=[0.05, 7.5, 1.2], InitialConditions=[2.5, 0.0], dynamic_state=None):
    """
    The Ueda Oscillator is defined as

    .. math::
        \dot{x} = y,

        \dot{y} = -x^3 - by + A\sin{\\omega t}

    where we chose the parameters :math:`(x_0. y_0) = (2.5, 0.0)` for initial conditions, :math:`b = 0.05`, :math:`A = 7.5`, :math:`\\omega = 1.2` (periodic) and :math:`\\omega = 1.0` (chaotic). 

    .. figure:: ../../../figures/Driven_Dissipative_Flows/Ueda_Oscillator.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): list of values for [b, A, :math:`\\omega`] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`]    
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
            b, A, omega = 0.05, 7.5, 1.2
        elif dynamic_state == 'chaotic':
            b, A, omega = 0.05, 7.5, 1.0
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        b, A, omega = parameters[0], parameters[1], parameters[2]

    # defining simulation functions
    def ueda_oscillator(state, t):
        x, y = state  # unpack the state vector
        return y, -x**3 - b*y + A*np.sin(omega*t)

    states = odeint(ueda_oscillator, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def duffings_two_well_oscillator(fs=20, SampleSize=5000, L=500.0, parameters=[0.25, 0.4, 1.1], InitialConditions=[0.2, 0.0], dynamic_state=None):
    """
    The Duffings Two-Well Oscillator is defined as

    .. math::
        \dot{x} = y,

        \dot{y} = -x^3 + x - by + A\sin{\\omega t}

    where we chose the parameters :math:`(x_0. y_0) = (2.5, 0.0)` for initial conditions, :math:`b = 0.25`, :math:`A = 0.4`, :math:`\\omega = 1.1` (periodic) and :math:`\\omega = 1.0` (chaotic). 

    .. figure:: ../../../figures/Driven_Dissipative_Flows/Duffings_TwoWell_Oscillator.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): list of values for [b, A, :math:`\\omega`] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`]  
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
            b, A, omega = 0.25, 0.4, 1.1
        elif dynamic_state == 'chaotic':
            b, A, omega = 0.25, 0.4, 1.0
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        b, A, omega = parameters[0], parameters[1], parameters[2]

    # defining simulation functions
    def duffings_two_well_oscillator(state, t):
        x, y = state  # unpack the state vector
        return y, -x**3 + x - b*y + A*np.sin(omega*t)

    states = odeint(duffings_two_well_oscillator, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def duffing_van_der_pol_oscillator(fs=20, SampleSize=5000, L=500.0, parameters=[0.2, 8, 0.35, 1.3], InitialConditions=[0.2, -0.2], dynamic_state=None):
    """
    The Duffings Two-Well Oscillator is defined as

    .. math::
        \dot{x} = y,

        \dot{y} = \\mu (1 - \\gamma x^2)y - x^3 + A\sin{\\omega t}

    where we chose the parameters :math:`(x_0. y_0) = (0.2, 0.0)` for initial conditions, :math:`\\mu = 0.2`, :math:`\\gamma = 8`, :math:`A = 0.35`, :math:`\\omega = 1.3` (periodic) and :math:`\\omega = 1.2` (chaotic). 

    .. figure:: ../../../figures/Driven_Dissipative_Flows/Duffing_VanderPol_Oscillator.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): list of values for [:math:`\\mu`, :math:`\\gamma` A, :math:`\\omega`] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    """

    t = np.linspace(0, L, int(L*fs))

    num_param = 4

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            mu, gamma, A, omega = 0.2, 8, 0.35, 1.3
        elif dynamic_state == 'chaotic':
            mu, gamma, A, omega = 0.2, 8, 0.35, 1.2
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        mu, gamma, A, omega = parameters[0], parameters[1], parameters[2], parameters[3]

    # defining simulation functions
    def duffing_van_der_pol_oscillator(state, t):
        x, y = state  # unpack the state vector
        return y, mu*(1-gamma*x**2)*y - x**3 + A*np.sin(omega*t)

    states = odeint(duffing_van_der_pol_oscillator,
                    InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def rayleigh_duffing_oscillator(fs=20, SampleSize=5000, L=500.0, parameters=[0.2, 4, 0.3, 1.4], InitialConditions=[0.3, 0.0], dynamic_state=None):
    """
    The Rayleigh Duffing Oscillator is defined as

    .. math::
        \dot{x} = y,

        \dot{y} = \\mu (1 - \\gamma y^2)y - x^3 + A\sin{\\omega t}

    where we chose the parameters :math:`(x_0. y_0) = (0.3, 0.0)` for initial conditions, :math:`\\mu = 0.2`, :math:`\\gamma = 4`, :math:`A = 0.3`, :math:`\\omega = 1.4` (periodic) and :math:`\\omega = 1.2` (chaotic). 

    .. figure:: ../../../figures/Driven_Dissipative_Flows/Rayleigh_Duffing_Oscillator.png

    Parameters:
        L (Optional[int]): Number of iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series
        parameters (Optional[floats]): list of values for [:math:`\\mu`, :math:`\\gamma` A, :math:`\\omega`] or None if using the dynamic_state variable
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    """

    t = np.linspace(0, L, int(L*fs))

    num_param = 4

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            mu, gamma, A, omega = 0.2, 4, 0.3, 1.4
        elif dynamic_state == 'chaotic':
            mu, gamma, A, omega = 0.2, 4, 0.3, 1.2
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        mu, gamma, A, omega = parameters[0], parameters[1], parameters[2], parameters[3]

    # defining simulation functions
    def rayleigh_duffing_oscillator(state, t):
        x, y = state  # unpack the state vector
        return y, mu*(1-gamma*y**2)*y - x**3 + A*np.sin(omega*t)

    states = odeint(rayleigh_duffing_oscillator, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts
