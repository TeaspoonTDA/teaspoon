from scipy.integrate import odeint
import scipy.integrate as integrate
import numpy as np

def autonomous_dissipative_flows(system, dynamic_state=None, L=None, fs=None,
                                 SampleSize=None, parameters=None,
                                 InitialConditions=None):
    

    if system == 'lorenz':
        t, ts = lorenz()

    if system == 'rossler':
        t, ts = rossler()

    if system == 'chua':
        t, ts = chua()


    if system == 'double_pendulum':
        t, ts = double_pendulum()


    if system == 'coupled_lorenz_rossler':
        t, ts = coupled_lorenz_rossler()


    if system == 'coupled_rossler_rossler':
        t, ts = coupled_rossler_rossler()


    if system == 'diffusionless_lorenz_attractor':
        t, ts = diffusionless_lorenz_attractor()


    if system == 'complex_butterfly':
        t, ts = complex_butterfly()


    if system == 'chens_system':
        t, ts = chens_system()


# Sunia

    if system == 'hadley_circulation':
        # setting simulation time series parameters
        if fs == None:
            fs = 50
        if SampleSize == None:
            SampleSize = 4000
        if L == None:
            L = 500.0
        t = np.linspace(0, L, int(L*fs))

        # setting system parameters
        if parameters != None:
            if len(parameters) != 3:
                print(
                    'Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                parameters = None
            else:
                a, b, F, G = parameters[0], parameters[1], parameters[2], parameters[3]
        if parameters == None:
            if dynamic_state == 'periodic':
                a = 0.30
            if dynamic_state == 'chaotic':
                a = 0.25
            b, F, G = 4, 8, 1

        # defining simulation functions
        def hadley_circulation(state, t):
            x, y, z = state  # unpack the state vector
            return -y**2 - z**2 - a*x + a*F, x*y - b*x*z - y + G, b*x*y + x*z - z

        if InitialConditions == None:
            InitialConditions = [-10, 0, 37]

        states = odeint(hadley_circulation, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
                [-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]


    if system == 'ACT_attractor':
        # setting simulation time series parameters
        if fs == None:
            fs = 50
        if SampleSize == None:
            SampleSize = 4000
        if L == None:
            L = 500.0
        t = np.linspace(0, L, int(L*fs))

        # setting system parameters
        if parameters != None:
            if len(parameters) != 4:
                print(
                    'Warning: needed 4 parameters. Defaulting to periodic solution parameters.')
                parameters = None
            else:
                alpha, mu, delta, beta = parameters[0], parameters[1], parameters[2], parameters[3]
        if parameters == None:
            if dynamic_state == 'periodic':
                alpha = 2.5
            if dynamic_state == 'chaotic':
                alpha = 2.0
            mu, delta, beta = 0.02, 1.5, -0.07

        # defining simulation functions
        def ACT_attractor(state, t):
            x, y, z = state  # unpack the state vector
            return alpha*(x-y), -4*alpha*y + x*z + mu*x**3, -delta*alpha*z + x*y + beta*z**2

        if InitialConditions == None:
            InitialConditions = [0.5, 0, 0]

        states = odeint(ACT_attractor, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
                [-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]


    if system == 'rabinovich_frabrikant_attractor':
        # setting simulation time series parameters
        if fs == None:
            fs = 30
        if SampleSize == None:
            SampleSize = 3000
        if L == None:
            L = 500.0
        t = np.linspace(0, L, int(L*fs))

        # setting system parameters
        if parameters != None:
            if len(parameters) != 2:
                print(
                    'Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                parameters = None
            else:
                alpha, gamma = parameters[0], parameters[1]
        if parameters == None:
            if dynamic_state == 'periodic':
                alpha = 1.16
            if dynamic_state == 'chaotic':
                alpha = 1.13
            gamma = 0.87

        # defining simulation functions
        def rabinovich_frabrikant_attractor(state, t):
            x, y, z = state  # unpack the state vector
            return y*(z-1+x**2)+gamma*x, x*(3*z + 1 - x**2) + gamma*y, -2*z*(alpha + x*y)

        if InitialConditions == None:
            InitialConditions = [-1, 0, 0.5]

        states = odeint(rabinovich_frabrikant_attractor,
                        InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
                [-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]


    if system == 'linear_feedback_rigid_body_motion_system':
        # system from https://ir.nctu.edu.tw/bitstream/11536/26522/1/000220413000019.pdf
        # setting simulation time series parameters
        if fs == None:
            fs = 100
        if SampleSize == None:
            SampleSize = 3000
        if L == None:
            L = 500.0
        t = np.linspace(0, L, int(L*fs))

        # setting system parameters
        if parameters != None:
            if len(parameters) != 3:
                print(
                    'Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                parameters = None
            else:
                a, b, c = parameters[0], parameters[1], parameters[2]
        if parameters == None:
            if dynamic_state == 'periodic':
                a = 5.3
            if dynamic_state == 'chaotic':
                a = 5.0
            b, c = -10, -3.8
        # defining simulation functions

        def linear_feedback_rigid_body_motion_system(state, t):
            x, y, z = state  # unpack the state vector
            return -y*z + a*x, x*z + b*y, (1/3)*x*y + c*z

        if InitialConditions == None:
            InitialConditions = [0.2, 0.2, 0.2]

        states = odeint(
            linear_feedback_rigid_body_motion_system, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
                [-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]


    if system == 'moore_spiegel_oscillator':
        # setting simulation time series parameters
        if fs == None:
            fs = 100
        if SampleSize == None:
            SampleSize = 5000
        if L == None:
            L = 500.0
        t = np.linspace(0, L, int(L*fs))

        # setting system parameters
        if parameters != None:
            if len(parameters) != 2:
                print(
                    'Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                parameters = None
            else:
                T, R = parameters[0], parameters[1]
        if parameters == None:
            if dynamic_state == 'periodic':
                T = 7.8
            if dynamic_state == 'chaotic':
                T = 7.0
            R = 20
        # defining simulation functions

        def moore_spiegel_oscillator(state, t):
            x, y, z = state  # unpack the state vector
            return y, z, -z - (T-R + R*x**2)*y - T*x

        if InitialConditions == None:
            InitialConditions = [0.2, 0.2, 0.2]

        states = odeint(moore_spiegel_oscillator, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
                [-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]


    if system == 'thomas_cyclically_symmetric_attractor':
        # setting simulation time series parameters
        if fs == None:
            fs = 10
        if SampleSize == None:
            SampleSize = 5000
        if L == None:
            L = 1000.0
        t = np.linspace(0, L, int(L*fs))

        # setting system parameters
        if parameters != None:
            if len(parameters) != 1:
                print(
                    'Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                parameters = None
            else:
                b = parameters[0]
        if parameters == None:
            if dynamic_state == 'periodic':
                b = 0.17
            if dynamic_state == 'chaotic':
                b = 0.18

        # defining simulation functions
        def thomas_cyclically_symmetric_attractor(state, t):
            x, y, z = state  # unpack the state vector
            return -b*x + np.sin(y), -b*y + np.sin(z), -b*z + np.sin(x)

        if InitialConditions == None:
            InitialConditions = [0.1, 0, 0]

        states = odeint(
            thomas_cyclically_symmetric_attractor, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
                [-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]


    if system == 'halvorsens_cyclically_symmetric_attractor':
        # setting simulation time series parameters
        if fs == None:
            fs = 200
        if SampleSize == None:
            SampleSize = 5000
        if L == None:
            L = 200.0
        t = np.linspace(0, L, int(L*fs))

        # setting system parameters
        if parameters != None:
            if len(parameters) != 3:
                print(
                    'Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                parameters = None
            else:
                a, b, c = parameters[0], parameters[1], parameters[2]
        if parameters == None:
            if dynamic_state == 'periodic':
                a = 1.85
            if dynamic_state == 'chaotic':
                a = 1.45
            b, c = 4, 4

        # defining simulation functions
        def halvorsens_cyclically_symmetric_attractor(state, t):
            x, y, z = state  # unpack the state vector
            return -a*x - b*y - c*z - y**2, -a*y - b*z - c*x - z**2, -a*z - b*x - c*y - x**2

        if InitialConditions == None:
            InitialConditions = [-5, 0, 0]

        states = odeint(
            halvorsens_cyclically_symmetric_attractor, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
                [-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]


# Max

    if system == 'burke_shaw_attractor':
        t, ts = burke_shaw_attractor()


    if system == 'rucklidge_attractor':
        t, ts = rucklidge_attractor()


    if system == 'WINDMI':
        # setting simulation time series parameters
        if fs == None:
            fs = 20
        if SampleSize == None:
            SampleSize = 5000
        if L == None:
            L = 1000.0
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
            if dynamic_state == 'periodic':
                a = 0.9
            if dynamic_state == 'chaotic':
                a = 0.8
            b = 2.5

        # defining simulation functions
        def WINDMI(state, t):
            x, y, z = state  # unpack the state vector
            return y, z, -a*z - y + b - np.exp(x)

        if InitialConditions == None:
            InitialConditions = [1, 0, 4.5]

        states = odeint(WINDMI, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
                [-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]


    if system == 'simplest_quadratic_chaotic_flow':
        # setting simulation time series parameters
        if fs == None:
            fs = 20
        if SampleSize == None:
            SampleSize = 5000
        if L == None:
            L = 1000.0
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
            if dynamic_state == 'periodic':
                print('We could not find a periodic response near $a = 2.017$.')
                print('Any contributions would be appreciated!')
                print('Defaulting to chaotic state.')
                a = 2.017
            if dynamic_state == 'chaotic':
                a = 2.017
            b = 1
        # defining simulation functions

        def simplest_quadratic_chaotic_flow(state, t):
            x, y, z = state  # unpack the state vector
            return y, z, -a*z + b*y**2 - x

        if InitialConditions == None:
            InitialConditions = [-0.9, 0, 0.5]

        states = odeint(simplest_quadratic_chaotic_flow,
                        InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
                [-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]


    if system == 'simplest_cubic_chaotic_flow':
        # setting simulation time series parameters
        if fs == None:
            fs = 20
        if SampleSize == None:
            SampleSize = 5000
        if L == None:
            L = 1000.0
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
            if dynamic_state == 'periodic':
                a = 2.11
            if dynamic_state == 'chaotic':
                a = 2.05
            b = 2.5

        # defining simulation functions
        def simplest_cubic_chaotic_flow(state, t):
            x, y, z = state  # unpack the state vector
            return y, z, -a*z + x*y**2 - x

        if InitialConditions == None:
            InitialConditions = [0, 0.96, 0]

        states = odeint(simplest_cubic_chaotic_flow, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
                [-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]


    if system == 'simplest_piecewise_linear_chaotic_flow':
        # setting simulation time series parameters
        if fs == None:
            fs = 40
        if SampleSize == None:
            SampleSize = 5000
        if L == None:
            L = 1000.0
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
            if dynamic_state == 'periodic':
                a = 0.7
            if dynamic_state == 'chaotic':
                a = 0.6

        # defining simulation functions
        def simplest_piecewise_linear_chaotic_flow(state, t):
            x, y, z = state  # unpack the state vector
            return y, z, -a*z - y + np.abs(x) - 1

        if InitialConditions == None:
            InitialConditions = [0, -0.7, 0]

        states = odeint(
            simplest_piecewise_linear_chaotic_flow, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
                [-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]


    if system == 'double_scroll':
        # setting simulation time series parameters
        if fs == None:
            fs = 20
        if SampleSize == None:
            SampleSize = 5000
        if L == None:
            L = 1000.0
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
            if dynamic_state == 'periodic':
                a = 1.0
            if dynamic_state == 'chaotic':
                a = 0.8

        # defining simulation functions
        def double_scroll(state, t):
            x, y, z = state  # unpack the state vector
            return y, z, -a*(z + y + x - np.sign(x))

        if InitialConditions == None:
            InitialConditions = [0.01, 0.01, 0]

        states = odeint(double_scroll, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
                [-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]

    return t, ts

















def lorenz(parameters=[100, 10, 8.0/3.0], dynamic_state=None, InitialConditions=[10.0**-10.0, 0.0, 1.0], L=100.0, fs=100, SampleSize=2000):
    """
    The Lorenz system used is defined as

    .. math::
        \dot{x} &= \sigma (y - x),

        \dot{y} &= x (\\rho - z) - y,

        \dot{z} &= x y - \\beta z
    
    The Lorenz system was solved with a sampling rate of 100 Hz for 100 seconds with only the last 20 seconds used to avoid transients. For a chaotic response, parameters of :math:`\\sigma = 10.0`, :math:`\\beta = 8.0/3.0`, and :math:`\\rho = 105` and initial conditions :math:`[x_0,y_0,z_0] = [10^{-10},0,1]` are used. For a periodic response set :math:`\\rho = 100`.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Lorenz_System.png

    Parameters:
        parameters (Optional[floats]): Array of three floats [:math:`\\rho`, :math:`\\sigma`, :math:`\\beta`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_{0}`, :math:`y_{0}`, :math:`z_{0}`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    """

    t = np.linspace(0, L, int(L*fs))

    num_param = 3

    if len(parameters) != num_param:
        raise ValueError(f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            rho = 100.0
            sigma = 10.0
            beta = 8.0 / 3.0
        elif dynamic_state == 'chaotic':
            rho = 105.0
            sigma = 10.0
            beta = 8.0 / 3.0
        else:
            raise ValueError(f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        rho, sigma, beta = parameters[0], parameters[1], parameters[2]

    # defining simulation functions

    def lorenz_sys(state, t):
        x, y, z = state  # unpack the state vector
        # derivatives
        return sigma*(y - x), x*(rho - z) - y, x*y - beta*z

    states = odeint(lorenz_sys, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
            [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def rossler(parameters=[0.1, 0.2, 14], dynamic_state=None, InitialConditions=[-0.4, 0.6, 1], L=1000.0, fs=15, SampleSize=2500):
    """
    The Rössler system used was defined as

    .. math::
        \dot{x} &= -y-z,

        \dot{y} &= x + ay,

        \dot{z} &= b + z(x-c)
    
    The Rössler system was solved with a sampling rate of 15 Hz for 1000 seconds with only the last 166 seconds used to avoid transients. For a chaotic response, parameters of :math:`a = 0.15`, :math:`b = 0.2`, and :math:`c = 14` and initial conditions :math:`[x_0,y_0,z_0] = [-0.4,0.6,1.0]` are used. For a periodic response set :math:`a = 0.10`.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Rossler_System.png

    Parameters:
        parameters (Optional[floats]): Array of three floats [a, b, c] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_{0}`, :math:`y_{0}`, :math:`z_{0}`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    """
    t = np.linspace(0, L, int(L*fs))

    num_param = 3

    if len(parameters) != num_param:
        raise ValueError(f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 0.10
            b = 0.20
            c = 14
        elif dynamic_state == 'chaotic':
            a = 0.15
            b = 0.20
            c = 14
        else:
            raise ValueError(f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        a, b, c = parameters[0], parameters[1], parameters[2]

    # defining simulation functions

    def rossler_sys(state, t):
        x, y, z = state  # unpack the state vector
        return -y - z, x + a*y, b + z*(x-c)

    states = odeint(rossler_sys, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
            [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def coupled_lorenz_rossler(parameters=[0.25, 8/3, 0.2, 5.7, 0.1, 0.1, 0.1, 28, 10], dynamic_state=None, InitialConditions=[0.1, 0.1, 0.1, 0, 0, 0], L=500.0, fs=50, SampleSize=15000):
    """
    The coupled Lorenz-Rössler system is defined as

    .. math::
        \dot{x}_1 &= -y_1-z_1+k_1(x_2-x_1),

        \dot{y}_1 &= x_1 + ay_1+k_2(y_2-y_1),

        \dot{z}_1 &= b_2 + z_1(x_1-c_2) + k_3(z_2-z_1),

        \dot{x}_2 &= \\sigma (y_2-x_2),

        \dot{y}_2 &= \\lambda x_2 - y_2 - x_2z_2,

        \dot{z}_2 &= x_2y_2 - b_1z_2
    
    where :math:`b_1 =8/3`, :math:`b_2 =0.2`, :math:`c_2 =5.7`, :math:`k_1 =0.1`, :math:`k_2 =0.1`, :math:`k_3 =0.1`, :math:`\\lambda =28`, :math:`\\sigma =10`,and :math:`a=0.25` for a periodic response and :math:`a = 0.51` for a chaotic response. This system was simulated at a frequency of 50 Hz for 500 seconds with the last 300 seconds used. The default initial condition is :math:`[x_1, y_1, z_1, x_2, y_2, z_2]=[0.1,0.1,0.1,0,0,0]`.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Coupled_Rossler_Lorenz_System.png

    Parameters:
        parameters (Optional[floats]): Array of three floats [:math:`a`, :math:`b_1`, :math:`b_2`, :math:`c_2`, :math:`k_1`, :math:`k_2`, :math:`k_3`, :math:`\\lambda`, :math:`\\sigma`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_{1, 0}`, :math:`y_{1, 0}`, :math:`z_{1, 0}`, :math:`x_{2, 0}`, :math:`y_{2, 0}`, :math:`z_{2, 0}`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    """
    t = np.linspace(0, L, int(L*fs))

    num_param = 9

    if len(parameters) != num_param:
        raise ValueError(f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a, b1, b2, c2, k1, k2, k3, lam, sigma = 0.25, 8/3, 0.2, 5.7, 0.1, 0.1, 0.1, 28, 10
        elif dynamic_state == 'chaotic':
            a, b1, b2, c2, k1, k2, k3, lam, sigma = 0.51, 8/3, 0.2, 5.7, 0.1, 0.1, 0.1, 28, 10
        else:
            raise ValueError(f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        a, b1, b2, c2, k1, k2, k3, lam, sigma = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8]

    # defining simulation functions

    def coupled_lorenz_rossler_sys(state, t):
        x1, y1, z1, x2, y2, z2 = state  # unpack the state vector
        D = [-y1 - z1 + k1*(x2-x1),
                x1 + a*y1 + k2*(y2-y1),
                b2 + z1*(x1-c2) + k3*(z2-z1),
                sigma*(y2-x2),
                lam*x2 - y2 - x2*z2,
                x2*y2 - b1*z2]
        return D[0], D[1], D[2], D[3], D[4], D[5]

    states = odeint(coupled_lorenz_rossler_sys, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:],
            (states[:, 1])[-SampleSize:],
            (states[:, 2])[-SampleSize:],
            (states[:, 3])[-SampleSize:],
            (states[:, 4])[-SampleSize:],
            (states[:, 5])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def coupled_rossler_rossler(parameters=[0.25, 0.99, 0.95], dynamic_state=None, InitialConditions=[-0.4, 0.6, 5.8, 0.8, -2, -4], L=1000.0, fs=10, SampleSize=1500):
    """
    The coupled Lorenz-Rössler system is defined as

    .. math::
        \dot{x}_1 &= -w_1y_1 - z_1 +k(x_2-x_1),

        \dot{y}_1 &= w_1x_1 + 0.165y_1,

        \dot{z}_1 &= 0.2 + z_1(x_1-10),

        \dot{x}_2 &= -w_2y_2-z_2 + k(x_1-x_2),

        \dot{y}_2 &= w_2x_2 + 0.165y_2,

        \dot{z}_2 &= 0.2 + z_2(x_2-10)
    
    with :math:`w_1 = 0.99`, :math:`w_2 = 0.95`, and :math:`k = 0.05`. This was solved for 1000 seconds with a sampling rate of 10 Hz. Only the last 150 seconds of the solution are used and the default initial condition is :math:`[x_1, y_1, z_1, x_2, y_2, z_2]=[-0.4,0.6,5.8,0.8,-2,-4]`.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/BiDirectional_Coupled_Rossler_System.png

    Parameters:
        parameters (Optional[floats]): Array of three floats [:math:`k`, :math:`w_1`, :math:`w_2`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_{1, 0}`, :math:`y_{1, 0}`, :math:`z_{1, 0}`, :math:`x_{2, 0}`, :math:`y_{2, 0}`, :math:`z_{2, 0}`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    """
    t = np.linspace(0, L, int(L*fs))

    num_param = 3

    if len(parameters) != num_param:
        raise ValueError(f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            k, w1, w2 = 0.25, 0.99, 0.95
        elif dynamic_state == 'chaotic':
            k, w1, w2 = 0.30, 0.99, 0.95
        else:
            raise ValueError(f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        k, w1, w2 = parameters[0], parameters[1], parameters[2]

    # defining simulation functions

    def coupled_rossler_rossler_sys(state, t):
        x1, y1, z1, x2, y2, z2 = state  # unpack the state vector
        D = [-w1*y1 - z1 + k*(x2-x1),
                w1*x1 + 0.165*y1,
                0.2 + z1*(x1-10),
                -w2*y2 - z2 + k*(x1-x2),
                w2*x2 + 0.165*y2,
                0.2 + z2*(x2-10)]
        return D[0], D[1], D[2], D[3], D[4], D[5]

    states = odeint(coupled_rossler_rossler_sys, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:],
            (states[:, 1])[-SampleSize:],
            (states[:, 2])[-SampleSize:],
            (states[:, 3])[-SampleSize:],
            (states[:, 4])[-SampleSize:],
            (states[:, 5])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def chua(parameters=[10.8, 27, 1.0, -3/7, 3/7], dynamic_state=None, InitialConditions=[1.0, 0.0, 0.0],  
    L=200.0, fs=50, SampleSize=4000):
    """
    Chua's circuit is based on a non-linear circuit and is described as

    .. math::
        \dot{x} &= \\alpha (y-f(x)),

        \dot{y} &= \\gamma (x-y+z),

        \dot{z} &= -\\beta y,
    
    where :math:`f(x)` is based on a non-linear resistor model defined as

    .. math::
        f(x) = m_1x + \\frac{1}{2}(m_0+m_1)[|x+1| - |x-1|]
    
    The system parameters are set to :math:`\\beta=27`, :math:`\\gamma=1`, :math:`m_0 =-3/7`, :math:`m_1 =3/7`, and :math:`\\alpha=10.8` for a periodic response and :math:`\\alpha = 12.8` for a chaotic response. The system was simulated for 200 seconds at a rate of 50 Hz and the last 80 seconds are used.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Chua_Circuit.png

    Parameters:
        parameters (Optional[floats]): Array of three floats [:math:`a`, :math:`B`, :math:`g`, :math:`m_0`, :math:`m_1`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    """
    t = np.linspace(0, L, int(L*fs))

    num_param = 5

    if len(parameters) != num_param:
        raise ValueError(f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a, B, g, m0, m1 = 10.8, 27, 1.0, -3/7, 3/7
        elif dynamic_state == 'chaotic':
            a, B, g, m0, m1 = 12.8, 27, 1.0, -3/7, 3/7
        else:
            raise ValueError(f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        a, B, g, m0, m1 = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
    
    # defining simulation functions

    def f(x):
        f = m1*x+(m0-m1)/2.0*(abs(x+1.0)-abs(x-1.0))
        return f

    def chua_sys(H, t=0):
        return np.array([a*(H[1]-f(H[0])),
                            g*(H[0]-H[1]+H[2]),
                            -B*H[1]])


    states, infodict = integrate.odeint(
        chua_sys, InitialConditions, t, full_output=True)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
            [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def double_pendulum(parameters=[1, 1, 1, 1, 9.81], dynamic_state=None, InitialConditions=[0.4, 0.6, 1, 1], L=100.0, fs=100, SampleSize=5000):
    """
    The double pendulum is a staple bench top experiment for investigated chaos in a mechanical system. A point-mass double pendulum's equations of motion are defined as 

    .. math::
        \dot{\\theta}_1 &= \\omega_1,

        \dot{\\theta}_2 &= \\omega_2,

        \dot{\\omega}_1 &= \\frac{-g(2m_1+m_2)\sin(\\theta_1) - m_2g\sin(\\theta_1-2\\theta_2) - 2\sin(\\theta_1-\\theta2)m_2(\\omega_2^2 l_2 + \\omega_1^2 l_1\cos(\\theta_1-\\theta_2))}{l_1(2m_1+m_2-m_2\cos(2\\theta_1-2\\theta_2))},

        \dot{\\omega}_2 &= \\frac{2\sin(\\theta_1-\\theta_2)(\\omega_1^2 l_1(m_1+m_2)+g(m_1+m_2)\cos(\\theta_1)+\\omega_2^2 l_2m_2\cos(\\theta_1-\\theta_2))}{l_2(2m_1+m_2-m_2\cos(2\\theta_1-2\\theta_2))}
    
    where the system parameters are :math:`g=9.81 m/s^2`, :math:`m_1 =1 kg`, :math:`m_2 =1 kg`, :math:`l_1 = 1 m`, and :math:`l_2 =1 m`. The system was solved for 200 seconds at a rate of 100 Hz and only the last 30 seconds were used as shown in the figure below for the chaotic response with initial conditions :math:`[\\theta_1, \\theta_2, \\omega_1, \\omega_2] = [0, 3 rad, 0, 0]`. This system will have different dynamic states based on the initial conditions, which can vary from periodic, quasiperiodic, and chaotic.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Double_Pendulum.png

    Parameters:
        parameters (Optional[floats]): Array of three floats [:math:`m_1`, :math:`m_2`, :math:`l_1`, :math:`l_2`, :math:`g`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`\\theta_{1, 0}`, :math:`\\theta_{2, 0}`, :math:`\\omega_{1, 0}`, :math:`\\omega_{2, 0}`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    """
    t = np.linspace(0, L, int(L*fs))

    num_param = 5

    if len(parameters) != num_param:
        raise ValueError(f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            m1, m2, l1, l2, g = 1, 1, 1, 1, 9.81
            InitialConditions = [0.4, 0.6, 1, 1]
        elif dynamic_state == 'chaotic':
            m1, m2, l1, l2, g = 1, 1, 1, 1, 9.81
            InitialConditions = [0.0, 3, 0, 0]
        elif dynamic_state == 'quasiperiodic':
            m1, m2, l1, l2, g = 1, 1, 1, 1, 9.81
            InitialConditions = [1, 0, 0, 0]            
        else:
            raise ValueError(f'dynamic_state needs to be either "periodic", "quasiperiodic" or "chaotic", or provide an array of length {num_param} in parameters.')
    else:
        m1, m2, l1, l2, g = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]

    # defining simulation functions
    def double_pendulum_sys(state, t):
        th1, th2, om1, om2 = state  # unpack the state vector
        numerator1 = -g*(2*m1+m2)*np.sin(th1) - m2*g*np.sin(th1-2*th2) - \
            2*np.sin(th1-th2)*m2*(om2**2 * l2 +
                                    om1**2 * l1*np.cos(th1-th2))
        numerator2 = 2*np.sin(th1-th2)*(om1**2 * l1*(m1+m2) +
                                        g*(m1+m2)*np.cos(th1)+om2**2 * l2*m2*np.cos(th1-th2))
        denomenator1 = l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))
        denomenator2 = l2*(2*m1+m2-m2*np.cos(2*th1-2*th2))
        D = [om1,
                om2,
                numerator1/denomenator1,
                numerator2/denomenator2]
        return D[0], D[1], D[2], D[3]


    states = odeint(double_pendulum_sys, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:],
            (states[:, 2])[-SampleSize:], (states[:, 3])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def diffusionless_lorenz_attractor(parameters=[0.25], dynamic_state=None, InitialConditions=[1, -1, 0.01], L=1000.0, fs=40, SampleSize=10000):
    """
    The Diffusionless Lorenz attractor is defined as

    .. math::
        \dot{x} &= -y-x,

        \dot{y} &= -xz,

        \dot{z} &= xy + R
    
    The system parameter is set to :math:`R = 0.40` for a chaotic response and :math:`R = 0.25` for a periodic response. The initial conditions were set to :math:`[x, y, z] = [1.0, -1.0, 0.01]`. The system was simulated for 1000 seconds at a rate of 40 Hz and the last 250 seconds were used for the chaotic response.
    
    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Diffusionless_Lorenz.png

    Parameters:
        parameters (Optional[floats]): Array of one float [:math:`R`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    """

    t = np.linspace(0, L, int(L*fs))
    
    num_param = 1

    if len(parameters) != num_param:
        raise ValueError(f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            R = 0.25
        elif dynamic_state == 'chaotic':
            R = 0.40
        else:
            raise ValueError(f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        R = parameters[0]

    # defining simulation functions

    def diffusionless_lorenz_attractor(state, t):
        x, y, z = state  # unpack the state vector
        return -y - x, -x*z, x*y + R

    if InitialConditions == None:
        InitialConditions = [1, -1, 0.01]

    states = odeint(diffusionless_lorenz_attractor,
                    InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
            [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def complex_butterfly(parameters=[0.15], dynamic_state=None, InitialConditions=[0.2,0.0,0.0], L=1000.0, fs=10, SampleSize=5000):
    """
    The Complex Butterfly attractor [1]_ is defined as

    .. math::
        \dot{x} &= a(y-x),

        \dot{y} &= z~\\text{sgn}(x),

        \dot{z} &= |x|-1
    
    The system parameter is set to :math:`a = 0.55` for a chaotic response and :math:`a = 0.15` for a periodic response. The initial conditions were set to :math:`[x, y, z] = [0.2, 0.0, 0.0]`. The system was simulated for 1000 seconds at a rate of 10 Hz and the last 500 seconds were used for the chaotic response.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Complex_Butterfly.png

    Parameters:
        parameters (Optional[floats]): Array of one float [:math:`a`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [1] Ahmed, Elwakil. "Creation of a complex butterfly attractor using a novel Lorenz-Type system". IEEE Transactions on Circuits and Systems I: Fundamental Theory and Applications, 2002.

    """
    # setting simulation time series parameters
    
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 1

    if len(parameters) != num_param:
        raise ValueError(f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 0.15
        elif dynamic_state == 'chaotic':
            a = 0.55
        else:
            raise ValueError(f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        a = parameters[0]

    # defining simulation functions

    def complex_butterfly(state, t):
        x, y, z = state  # unpack the state vector
        return a*(y-x), -z*np.sign(x), np.abs(x) - 1

    if InitialConditions == None:
        InitialConditions = [0.2, 0.0, 0.0]

    states = odeint(complex_butterfly, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
            [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def chens_system(parameters=[30.0,3.0,28.0], dynamic_state=None, InitialConditions=[-10, 0, 37], L=500.0, fs=200, SampleSize=3000):
    """
    Chen's System is defined [2]_ as

    .. math::
        \dot{x} &= a(y-x),

        \dot{y} &= (c-a)x-xz+cy,

        \dot{z} &= xy-bz
    
    The system parameters are set to :math:`a = 35`, :math:`b = 3`, and :math:`c = 28` for a chaotic response and :math:`a = 30` for a periodic response. The initial conditions were set to :math:`[x, y, z] = [-10, 0, 37]`. The system was simulated for 500 seconds at a rate of 200 Hz and the last 15 seconds were used for the chaotic response.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Chens_System.png

    Parameters:
        parameters (Optional[floats]): Array of one float [:math:`a`, :math:`b`, :math:`c`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [2] Liang, Xiyin. "Mechanical analysis of Chen chaotic system". Chaos, Solitons & Fractals, 2017.

    """

    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 3

    if len(parameters) != num_param:
        raise ValueError(f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 30.0
        elif dynamic_state == 'chaotic':
            a = 35.0
        else:
            raise ValueError(f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
        b, c = 3, 28
    else:
        a, b, c = parameters[0], parameters[1], parameters[2]

    # defining simulation functions
    def chens_system(state, t):
        x, y, z = state  # unpack the state vector
        return a*(y-x), (c-a)*x - x*z + c*y, x*y - b*z

    if InitialConditions == None:
        InitialConditions = [-10, 0, 37]

    states = odeint(chens_system, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
            [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def burke_shaw_attractor(parameters=[12.0,4.0], dynamic_state=None, InitialConditions=[0.6,0.0,0.0], L=500.0, fs=200, SampleSize=5000):
    """
    The Burke-Shaw Attractor is defined [8]_ as 

    .. math::
        \dot{x} &= s(x+y),

        \dot{y} &= -y - sxz,

        \dot{z} &= sxy + V

    The system parameters are set to :math:`s = 12`, :math:`V = 4`, and :math:`c = 28` for a periodic response and :math:`s = 10` for a chaotic response. The initial conditions were set to :math:`[x, y, z] = [0.6,0.0,0.0]`. The system was simulated for 500 seconds at a rate of 200 Hz and the last 25 seconds were used for the chaotic response.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Burke_Shaw_Attractor.png

    Parameters:
        parameters (Optional[floats]): Array of one float [:math:`s`, :math:`V`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [8] Shaw, Robert. "Strange attractors, chaotic behavior, and information flow". Zeitschrift für Naturforschung, 1981.
    """
    # system from http://www.atomosyd.net/spip.php?article33

    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 2

    if len(parameters) != num_param:
        raise ValueError(f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            s = 12.0
        elif dynamic_state == 'chaotic':
            s = 10.0
        else:
            raise ValueError(f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
        V = 4.0
    else:
        s, V = parameters[0], parameters[1]

    # defining simulation functions
    def burke_shaw_attractor(state, t):
        x, y, z = state  # unpack the state vector
        return -s*(x+y), -y - s*x*z, s*x*y + V

    if InitialConditions == None:
        InitialConditions = [0.6, 0, 0]

    states = odeint(burke_shaw_attractor, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
            [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def rucklidge_attractor(parameters=[1.1, 6.7], dynamic_state=None, InitialConditions=[1.0,0.0,4.5], L=1000.0, fs=50, SampleSize=5000):
    """
    The Rucklidge Attractor is defined [9]_ as

    .. math::
        \dot{x} &= -kx + \\lambda y - yz,

        \dot{y} &= x,

        \dot{z} &= -z + y^2

    The system parameters are set to :math:`k = 1.1`, :math:`\\lambda = 6.7` for a periodic response and :math:`k = 1.6` for a chaotic response. The initial conditions were set to :math:`[x, y, z] = [1.0,0.0,4.5]`. The system was simulated for 1000 seconds at a rate of 50 Hz and the last 100 seconds were used for the chaotic response.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Rucklidge_Attractor.png

    Parameters:
        parameters (Optional[floats]): Array of one float [:math:`k`, :math:`\\lambda`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [9] Chandrasekaran, Ramanathan. "A new chaotic attractor from Rucklidge system and its application in secured communication using OFDM". 11th International Conference on Intelligent Systems and Control (ISCO), 2017.
    """
    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 2

    if len(parameters) != num_param:
        raise ValueError(f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            k = 1.1
        elif dynamic_state == 'chaotic':
            k = 1.6
        else:
            raise ValueError(f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
        lamb = 6.7
    else:
        k, lamb = parameters[0], parameters[1]

    # defining simulation functions
    def rucklidge_attractor(state, t):
        x, y, z = state  # unpack the state vector
        return -k*x + lamb*y - y*z, x, -z + y**2

    if InitialConditions == None:
        InitialConditions = [1, 0, 4.5]

    states = odeint(rucklidge_attractor, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
            [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts
