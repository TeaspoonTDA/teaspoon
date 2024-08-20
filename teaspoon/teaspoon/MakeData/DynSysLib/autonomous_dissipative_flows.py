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
        # setting simulation time series parameters
        if fs == None:
            fs = 40
        if SampleSize == None:
            SampleSize = 10000
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
                R = parameters[0]
        if parameters == None:
            if dynamic_state == 'periodic':
                R = 0.25
            if dynamic_state == 'chaotic':
                R = 0.40

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


    if system == 'complex_butterfly':
        # system from https://pdfs.semanticscholar.org/3794/50ca6b8799d0b3c2f35bbe6df47676c69642.pdf?_ga=2.68291732.442509117.1595011450-840911007.1542643809
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
                a = parameters[0]
        if parameters == None:
            if dynamic_state == 'periodic':
                a = 0.15
            if dynamic_state == 'chaotic':
                a = 0.55

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


    if system == 'chens_system':
        # setting simulation time series parameters
        if fs == None:
            fs = 200
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
                a = 30
            if dynamic_state == 'chaotic':
                a = 35
            b, c = 3, 28

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


    if system == 'burke_shaw_attractor':
        # system from http://www.atomosyd.net/spip.php?article33
        # setting simulation time series parameters
        if fs == None:
            fs = 200
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
                s, V = parameters[0], parameters[1]
        if parameters == None:
            if dynamic_state == 'periodic':
                s = 12
            if dynamic_state == 'chaotic':
                s = 10
            V = 4

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


    if system == 'rucklidge_attractor':
        # setting simulation time series parameters
        if fs == None:
            fs = 50
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
                k, lamb = parameters[0], parameters[1]
        if parameters == None:
            if dynamic_state == 'periodic':
                k = 1.1
            if dynamic_state == 'chaotic':
                k = 1.6
            lamb = 6.7

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


def lorenz(rho=None, sigma=None, beta=None, dynamic_state=None, InitialConditions=None,  
    L=100.0, fs=100, SampleSize=2000):
    """
    The Lorenz system used is defined as

    .. math::
        \dot{x} &= \sigma (y - x),

        \dot{y} &= x (\\rho - z) - y,

        \dot{z} &= x y - \\beta z
    
    The Lorenz system was solved with a sampling rate of 100 Hz for 100 seconds with only the last 20 seconds used to avoid transients. For a chaotic response, parameters of :math:`\\sigma = 10.0`, :math:`\\beta = 8.0/3.0`, and :math:`\\rho = 105` and initial conditions :math:`[x_0,y_0,z_0] = [10^{-10},0,1]` are used. For a periodic response set :math:`\\rho = 100`.


    Parameters:
        rho (Optional[float]): System parameter.
        sigma (Optional[float]): System parameter.
        beta (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    """

    t = np.linspace(0, L, int(L*fs))


    if rho == None or sigma == None or beta == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            rho = 100.0
        if dynamic_state == 'chaotic':
            rho = 105.0
        sigma = 10.0
        beta = 8.0 / 3.0

    # defining simulation functions

    def lorenz_sys(state, t):
        x, y, z = state  # unpack the state vector
        # derivatives
        return sigma*(y - x), x*(rho - z) - y, x*y - beta*z

    if InitialConditions == None:
        InitialConditions = [10.0**-10.0, 0.0, 1.0]

    states = odeint(lorenz_sys, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
            [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def rossler(a=None, b=None, c=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=15, SampleSize=2500):
    """
    The Rössler system used was defined as

    .. math::
        \dot{x} &= -y-z,

        \dot{y} &= x + ay,

        \dot{z} &= b + z(x-c)
    
    The Rössler system was solved with a sampling rate of 15 Hz for 1000 seconds with only the last 166 seconds used to avoid transients. For a chaotic response, parameters of :math:`a = 0.15`, :math:`b = 0.2`, and :math:`c = 14` and initial conditions :math:`[x_0,y_0,z_0] = [-0.4,0.6,1.0]` are used. For a periodic response set :math:`a = 0.10`.

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
            a = 0.10
        if dynamic_state == 'chaotic':
            a = 0.15
        b = 0.20
        c = 14

    # defining simulation functions

    def rossler_sys(state, t):
        x, y, z = state  # unpack the state vector
        return -y - z, x + a*y, b + z*(x-c)

    if InitialConditions == None:
        InitialConditions = [-0.4, 0.6, 1]

    states = odeint(rossler_sys, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
            [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts

def coupled_lorenz_rossler(a=None, b1=None, b2=None, c2=None, k1=None, k2=None, k3=None, lam=None, sigma=None, dynamic_state=None, InitialConditions=None,  
    L=500.0, fs=50, SampleSize=15000):
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

    Parameters:
        a (Optional[float]): System parameter.
        b1 (Optional[float]): System parameter.
        b2 (Optional[float]): System parameter.
        c2 (Optional[float]): System parameter.
        k1 (Optional[float]): System parameter.
        k2 (Optional[float]): System parameter.
        k3 (Optional[float]): System parameter.
        lam (Optional[float]): System parameter.
        sigma (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    """

    t = np.linspace(0, L, int(L*fs))

    if a==None or b1==None or b2==None or c2==None or k1==None or k2==None or k3==None or lam==None or sigma==None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                    'Warning: needed 9 parameters. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            a = 0.25
        if dynamic_state == 'chaotic':
            a = 0.51
        b1, b2 = 8/3, 0.2
        c2 = 5.7
        k1, k2, k3 = 0.1, 0.1, 0.1
        lam, sigma = 28, 10

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

    if InitialConditions == None:
        InitialConditions = [0.1, 0.1, 0.1,
                                0, 0, 0]  # inital conditions

    states = odeint(coupled_lorenz_rossler_sys, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:],
            (states[:, 1])[-SampleSize:],
            (states[:, 2])[-SampleSize:],
            (states[:, 3])[-SampleSize:],
            (states[:, 4])[-SampleSize:],
            (states[:, 5])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts

def coupled_rossler_rossler(k=None, w1=None, w2=None, dynamic_state=None, InitialConditions=None,  
    L=1000.0, fs=10, SampleSize=1500):
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

    Parameters:
        k (Optional[float]): System parameter.
        w1 (Optional[float]): System parameter.
        w2 (Optional[float]): System parameter.
        dynamic_state (Optional[string]): Dynamic state ('periodic' or 'chaotic')
        L (Optional[int]): Number of map iterations.
        fs (Optional[int]): sampling rate for simulation.
        SampleSize (Optional[int]): length of sample at end of entire time series

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`
    
    """

    t = np.linspace(0, L, int(L*fs))

    if k == None or w1 == None or w2 == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            k = 0.25
        if dynamic_state == 'chaotic':
            k = 0.30
        w1 = 0.99
        w2 = 0.95

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

    if InitialConditions == None:
        InitialConditions = [-0.4, 0.6, 5.8,
                                0.8, -2, -4]  # inital conditions

    states = odeint(coupled_rossler_rossler_sys, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:],
            (states[:, 1])[-SampleSize:],
            (states[:, 2])[-SampleSize:],
            (states[:, 3])[-SampleSize:],
            (states[:, 4])[-SampleSize:],
            (states[:, 5])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts

def chua(a=None, g=None, B=None, m0=None, m1=None, dynamic_state=None, InitialConditions=None,  
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

    """
    t = np.linspace(0, L, int(L*fs))

    
    if a==None or g==None or B==None or m0==None or m1==None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print(
                'Warning: needed 5 parameters. Defaulting to periodic solution parameters.')
        if dynamic_state == 'periodic':
            a = 10.8  # alpha
        if dynamic_state == 'chaotic':
            a = 12.8
        B = 27  # betta
        g = 1.0  # gamma
        m0 = -3/7
        m1 = 3/7

    # defining simulation functions

    def f(x):
        f = m1*x+(m0-m1)/2.0*(abs(x+1.0)-abs(x-1.0))
        return f

    def chua_sys(H, t=0):
        return np.array([a*(H[1]-f(H[0])),
                            g*(H[0]-H[1]+H[2]),
                            -B*H[1]])

    if InitialConditions == None:
        InitialConditions = [1.0, 0.0, 0.0]

    states, infodict = integrate.odeint(
        chua_sys, InitialConditions, t, full_output=True)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
            [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts

def double_pendulum(m1=None, m2=None, l1=None, l2=None, g=None, dynamic_state=None, InitialConditions=None,  
    L=100.0, fs=100, SampleSize=5000):
    """
    The double pendulum is a staple bench top experiment for investigated chaos in a mechanical system. A point-mass double pendulum's equations of motion are defined as 

    .. math::
        \dot{\\theta}_1 &= \\omega_1,

        \dot{\\theta}_2 &= \\omega_2,

        \dot{\\omega}_1 &= \\frac{-g(2m_1+m_2)\sin(\\theta_1) - m_2g\sin(\\theta_1-2\\theta_2) - 2\sin(\\theta_1-\\theta2)m_2(\\omega_2^2 l_2 + \\omega_1^2 l_1\cos(\\theta_1-\\theta_2))}{l_1(2m_1+m_2-m_2\cos(2\\theta_1-2\\theta_2))},

        \dot{\\omega}_2 &= \\frac{2\sin(\\theta_1-\\theta_2)(\\omega_1^2 l_1(m_1+m_2)+g(m_1+m_2)\cos(\\theta_1)+\\omega_2^2 l_2m_2\cos(\\theta_1-\\theta_2))}{l_2(2m_1+m_2-m_2\cos(2\\theta_1-2\\theta_2))}
    
    where the system parameters are :math:`g=9.81 m/s^2`, :math:`m_1 =1 kg`, :math:`m_2 =1 kg`, :math:`l_1 = 1 m`, and :math:`l_2 =1 m`. The system was solved for 200 seconds at a rate of 100 Hz and only the last 30 seconds were used as shown in the figure below for the chaotic response with initial conditions :math:`[\\theta_1, \\theta_2, \\omega_1, \\omega_2] = [0, 3 rad, 0, 0]`. This system will have different dynamic states based on the initial conditions, which can vary from periodic, quasiperiodic, and chaotic.

    """

    t = np.linspace(0, L, int(L*fs))

    if m1==None or m2==None or l1==None or l2==None or g==None:
        print('Warning: needed 5 parameters. Using default parameters.')
        print('Parameters needed are [m1, m2, l1 ,l2, g].')
        m1, m2, l1, l2, g = 1, 1, 1, 1, 9.81

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

    if InitialConditions == None:
        if dynamic_state == None:
            dynamic_state = 'periodic'
            print('Warning: Dynamic state not specified. Using periodic initial condition.')
        if dynamic_state == 'periodic':
            InitialConditions = [0.4, 0.6, 1, 1]
        if dynamic_state == 'quasiperiodic':
            InitialConditions = [1, 0, 0, 0]
        if dynamic_state == 'chaotic':
            InitialConditions = [0.0, 3, 0, 0]

    states = odeint(double_pendulum_sys, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:],
            (states[:, 2])[-SampleSize:], (states[:, 3])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts
