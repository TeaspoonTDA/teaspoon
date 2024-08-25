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

    if system == 'hadley_circulation':
        t, ts = hadley_circulation()

    if system == 'ACT_attractor':
        t, ts = ACT_attractor()

    if system == 'rabinovich_frabrikant_attractor':
        t, ts = rabinovich_frabrikant_attractor()

    if system == 'linear_feedback_rigid_body_motion_system':
        t, ts = linear_feedback_rigid_body_motion_system()

    if system == 'moore_spiegel_oscillator':
        t, ts = moore_spiegel_oscillator()

    if system == 'thomas_cyclically_symmetric_attractor':
        t, ts = thomas_cyclically_symmetric_attractor()

    if system == 'halvorsens_cyclically_symmetric_attractor':
        t, ts = halvorsens_cyclically_symmetric_attractor()

    if system == 'burke_shaw_attractor':
        t, ts = burke_shaw_attractor()

    if system == 'rucklidge_attractor':
        t, ts = rucklidge_attractor()

    if system == 'WINDMI':
        t, ts = WINDMI()

    if system == 'simplest_quadratic_chaotic_flow':
        t, ts = simplest_quadratic_chaotic_flow()

    if system == 'simplest_cubic_chaotic_flow':
        t, ts = simplest_cubic_chaotic_flow()

    if system == 'simplest_piecewise_linear_chaotic_flow':
        t, ts = simplest_piecewise_linear_chaotic_flow()

    if system == 'double_scroll':
        t, ts = double_scroll()

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
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
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
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
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
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
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
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
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
        parameters (Optional[floats]): Array of nine floats [:math:`a`, :math:`b_1`, :math:`b_2`, :math:`c_2`, :math:`k_1`, :math:`k_2`, :math:`k_3`, :math:`\\lambda`, :math:`\\sigma`] or None if using the dynamic_state variable
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
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a, b1, b2, c2, k1, k2, k3, lam, sigma = 0.25, 8/3, 0.2, 5.7, 0.1, 0.1, 0.1, 28, 10
        elif dynamic_state == 'chaotic':
            a, b1, b2, c2, k1, k2, k3, lam, sigma = 0.51, 8/3, 0.2, 5.7, 0.1, 0.1, 0.1, 28, 10
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        a, b1, b2, c2, k1, k2, k3, lam, sigma = parameters[0], parameters[1], parameters[
            2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8]

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
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            k, w1, w2 = 0.25, 0.99, 0.95
        elif dynamic_state == 'chaotic':
            k, w1, w2 = 0.30, 0.99, 0.95
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
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
        parameters (Optional[floats]): Array of five floats [:math:`a`, :math:`B`, :math:`g`, :math:`m_0`, :math:`m_1`] or None if using the dynamic_state variable
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
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a, B, g, m0, m1 = 10.8, 27, 1.0, -3/7, 3/7
        elif dynamic_state == 'chaotic':
            a, B, g, m0, m1 = 12.8, 27, 1.0, -3/7, 3/7
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
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
        parameters (Optional[floats]): Array of five floats [:math:`m_1`, :math:`m_2`, :math:`l_1`, :math:`l_2`, :math:`g`] or None if using the dynamic_state variable
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
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
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
            raise ValueError(
                f'dynamic_state needs to be either "periodic", "quasiperiodic" or "chaotic", or provide an array of length {num_param} in parameters.')
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
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            R = 0.25
        elif dynamic_state == 'chaotic':
            R = 0.40
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
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


def complex_butterfly(parameters=[0.15], dynamic_state=None, InitialConditions=[0.2, 0.0, 0.0], L=1000.0, fs=10, SampleSize=5000):
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
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 0.15
        elif dynamic_state == 'chaotic':
            a = 0.55
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
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


def chens_system(parameters=[30.0, 3.0, 28.0], dynamic_state=None, InitialConditions=[-10, 0, 37], L=500.0, fs=200, SampleSize=3000):
    """
    Chen's System is defined [2]_ as

    .. math::
        \dot{x} &= a(y-x),

        \dot{y} &= (c-a)x-xz+cy,

        \dot{z} &= xy-bz

    The system parameters are set to :math:`a = 35`, :math:`b = 3`, and :math:`c = 28` for a chaotic response and :math:`a = 30` for a periodic response. The initial conditions were set to :math:`[x, y, z] = [-10, 0, 37]`. The system was simulated for 500 seconds at a rate of 200 Hz and the last 15 seconds were used for the chaotic response.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Chens_System.png

    Parameters:
        parameters (Optional[floats]): Array of three floats [:math:`a`, :math:`b`, :math:`c`] or None if using the dynamic_state variable
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
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 30.0
        elif dynamic_state == 'chaotic':
            a = 35.0
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
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


def hadley_circulation(parameters=[0.25, 4, 8, 1], dynamic_state=None, InitialConditions=[-10, 0, 37], L=500.0, fs=50, SampleSize=4000):
    """
    Hadley Circulation System is defined as

    .. math::
        \dot{x} &= -y^2 - z^2 - ax + aF,

        \dot{y} &= xy - bxz - y + G,

        \dot{z} &= bxy + xz - z

    The system parameters are set to :math:`a = 0.25`, :math:`b = 4`, :math:`F = 8`and :math:`G = 1` for a periodic response and :math:`a = 0.3` for a chaotic response. The initial conditions were set to :math:`[x, y, z] = [-10, 0, 37]`.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/hadley_circulation.png

    Parameters:
        parameters (Optional[floats]): Array of four floats [:math:`a`, :math:`b`, :math:`F``, :math:`G`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`


    """

    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 4

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 0.30
        elif dynamic_state == 'chaotic':
            a = 0.25
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
        b, F, G = 4, 8, 1
    else:
        a, b, F, G = parameters[0], parameters[1], parameters[2], parameters[3]

    # defining simulation functions
    def hadley_circulation(state, t):
        x, y, z = state  # unpack the state vector
        return -y**2 - z**2 - a*x + a*F, x*y - b*x*z - y + G, b*x*y + x*z - z

    states = odeint(hadley_circulation, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
          [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def ACT_attractor(parameters=[2.5, 0.02, 1.5, -0.07], dynamic_state=None, InitialConditions=[0.5, 0, 0], L=500.0, fs=50, SampleSize=4000):
    """
    ACT Attractor is defined [3]_ as

    .. math::
        \dot{x} &= \\alpha(x-y),

        \dot{y} &= -4\\alpha y + xz + \\mu x^3,

        \dot{z} &= -\\delta \\alpha z + xy + \\beta z^2

    The system parameters are set to :math:`\\alpha = 2.5`, :math:`\\mu = 0.02`, :math:`\\delta = 1.5`and :math:`\\beta = -0.07` for a periodic response and :math:`a = 2.0` for a chaotic response. The initial conditions were set to :math:`[x, y, z] = [0.5, 0, 0]`.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/ACT_attractor.png

    Parameters:
        parameters (Optional[floats]): Array of four floats [:math:`\\alpha`, :math:`\\mu`, :math:`\\delta``, :math:`\\beta`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [3] A. Arneodo, P. Coullet, and C. Tresser. Possible new strange attractors with spiral structure. Communications in Mathematical Physics, 79(4):573-579, dec 1981.

    """

    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 4

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            alpha = 2.5
        elif dynamic_state == 'chaotic':
            alpha = 2.0
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
        mu, delta, beta = 0.02, 1.5, -0.07
    else:
        alpha, mu, delta, beta = parameters[0], parameters[1], parameters[2], parameters[3]

    # defining simulation functions
    def ACT_attractor(state, t):
        x, y, z = state  # unpack the state vector
        return alpha*(x-y), -4*alpha*y + x*z + mu*x**3, -delta*alpha*z + x*y + beta*z**2

    states = odeint(ACT_attractor, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
          [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def rabinovich_frabrikant_attractor(parameters=[1.16, 0.87], dynamic_state=None, InitialConditions=[-1, 0, 0.5], L=500.0, fs=30, SampleSize=3000):
    """
    Rabinovich-Frabrikant Attractor is defined [4]_ as

    .. math::
        \dot{x} &= y(z - 1 + x^2) + \\alpha x,

        \dot{y} &= x(3z + 1 - x^2) + \\alpha y,

        \dot{z} &= -2z(\\gamma + xy)

    The system parameters are set to :math:`\\alpha = 1.16` and :math:`\\gamma = 0.87` for a periodic response and :math:`\\alpha = 1.13` for a chaotic response. The initial conditions were set to :math:`[x, y, z] = [-1, 0, 0.5]`.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/rabinovich_frabrikant_attractor.png

    Parameters:
        parameters (Optional[floats]): Array of two floats [:math:`\\alpha`, :math:`\\gamma`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [4]  Marius-F. Danca. Hidden transient chaotic attractors of rabinovich-fabrikant system. Nonlinear Dynamics, 86(2):1263-1270, jul 2016.

    """

    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 2

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            alpha = 1.16
        elif dynamic_state == 'chaotic':
            alpha = 1.13
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
        gamma = 0.87
    else:
        alpha, gamma = parameters[0], parameters[1]

    # defining simulation functions
    def rabinovich_frabrikant_attractor(state, t):
        x, y, z = state  # unpack the state vector
        return y*(z-1+x**2)+gamma*x, x*(3*z + 1 - x**2) + gamma*y, -2*z*(alpha + x*y)

    states = odeint(rabinovich_frabrikant_attractor,
                    InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
          [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def linear_feedback_rigid_body_motion_system(parameters=[5.3, -10, -3.8], dynamic_state=None, InitialConditions=[0.2, 0.2, 0.2], L=500.0, fs=100, SampleSize=3000):
    """
    Linear Feedback Rigid Body Motion System is defined [5]_ as

    .. math::
        \dot{x} &= yx, 

        \dot{y} &= xz + by,

        \dot{z} &= \\frac{1}{3}xy + cz

    The system parameters are set to :math:`a = 5.3`, :math:`b = -10`, :math:`c = -3.8` for a periodic response and :math:`a = 5` for a chaotic response. The initial conditions were set to :math:`[x, y, z] = [0.2, 0.2, 0.2]`.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/linear_feedback_rigid_body_motion_system.png

    Parameters:
        parameters (Optional[floats]): Array of three floats [:math:`a`, :math:`b`, :math:`c`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [5]  Hsien-Keng Chen and Ching-I Lee. Anti-control of chaos in rigid body motion. Chaos, Solitons \& Fractals, 21(4):957-965, aug 2004.

    """

    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 3

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 5.3
        elif dynamic_state == 'chaotic':
            a = 5.0
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
        b, c = -10, -3.8
    else:
        a, b, c = parameters[0], parameters[1], parameters[2]

    def linear_feedback_rigid_body_motion_system(state, t):
        x, y, z = state  # unpack the state vector
        return -y*z + a*x, x*z + b*y, (1/3)*x*y + c*z

    states = odeint(
        linear_feedback_rigid_body_motion_system, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
          [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def moore_spiegel_oscillator(parameters=[7.8, 20], dynamic_state=None, InitialConditions=[0.2, 0.2, 0.2], L=500.0, fs=100, SampleSize=5000):
    """
    The Moore-Spiegel Oscillator is defined [6]_ as

    .. math::
        \dot{x} &= y, 

        \dot{y} &= z,

        \dot{z} &= -z - (T - R + Rx^2)y - Tx

    The system parameters are set to :math:`T = 7.8`, :math:`R = 20` for a periodic response and :math:`T = 7` for a chaotic response. The initial conditions were set to :math:`[x, y, z] = [0.2, 0.2, 0.2]`.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/moore_spiegel_oscillator.png

    Parameters:
        parameters (Optional[floats]): Array of two floats [:math:`T`, :math:`R`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [6]  N. J. Balmforth and R. V. Craster. Synchronizing moore and spiegel. Chaos: An Interdisciplinary Journal of Nonlinear Science, 7(4):738-752, dec 1997.

    """

    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 2

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            T = 7.8
        elif dynamic_state == 'chaotic':
            T = 7.0
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
        R = 20
    else:
        T, R = parameters[0], parameters[1]

    def moore_spiegel_oscillator(state, t):
        x, y, z = state  # unpack the state vector
        return y, z, -z - (T-R + R*x**2)*y - T*x

    if InitialConditions == None:
        InitialConditions = [0.2, 0.2, 0.2]

    states = odeint(moore_spiegel_oscillator, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
          [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def thomas_cyclically_symmetric_attractor(parameters=[0.17], dynamic_state=None, InitialConditions=[0.1, 0, 0], L=1000.0, fs=10, SampleSize=5000):
    """
    The Thomas Cyclically Symmetric Attractor is defined [7]_ as

    .. math::
        \dot{x} &= -bx + \sin{y}, 

        \dot{y} &= -by + \sin{z},

        \dot{z} &= -bz + \sin{x}

    The system parameters are set to :math:`b = 0.17` for a periodic response and :math:`b = 0.18` for a chaotic response. The initial conditions were set to :math:`[x, y, z] = [0.1, 0, 0]`.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/thomas_cyclically_symmetric_attractor.png

    Parameters:
        parameters (Optional[floats]): Array of one float [:math:`b`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [7]  R. Thomas. Deterministic chaos seen in terms of feedback circuits: Analysis, synthesis, ”labyrinth chaos”. International Journal of Bifurcation and Chaos, 9:1889-1905, 1999.


    """

    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 1

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            b = 0.17
        elif dynamic_state == 'chaotic':
            b = 0.18
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        b = parameters[0]

    # defining simulation functions
    def thomas_cyclically_symmetric_attractor(state, t):
        x, y, z = state  # unpack the state vector
        return -b*x + np.sin(y), -b*y + np.sin(z), -b*z + np.sin(x)

    states = odeint(
        thomas_cyclically_symmetric_attractor, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
          [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def halvorsens_cyclically_symmetric_attractor(parameters=[1.85, 4, 4], dynamic_state=None, InitialConditions=[-5, 0, 0], L=200.0, fs=200, SampleSize=5000):
    """
    The Halvorsens Cyclically Symmetric Attractor is defined as

    .. math::
        \dot{x} &= -ax - by - cz - y^2, 

        \dot{y} &= -ay - bz - cz - z^2,

        \dot{z} &= -az - bx - cy - x^2

    The system parameters are set to :math:`a = 1.85`, :math:`b = 4`, :math:`c = 4` for a periodic response and :math:`a = 1.45` for a chaotic response. The initial conditions were set to :math:`[x, y, z] = [-5, 0, 0]`.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/halvorsens_cyclically_symmetric_attractor.png

    Parameters:
        parameters (Optional[floats]): Array of three floats [:math:`a`, :math:`b`, :math:`c`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`


    """

    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 3

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 1.85
        elif dynamic_state == 'chaotic':
            a = 1.45
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
        b, c = 4, 4
    else:
        a, b, c = parameters[0], parameters[1], parameters[2]

    # defining simulation functions
    def halvorsens_cyclically_symmetric_attractor(state, t):
        x, y, z = state  # unpack the state vector
        return -a*x - b*y - c*z - y**2, -a*y - b*z - c*x - z**2, -a*z - b*x - c*y - x**2

    states = odeint(
        halvorsens_cyclically_symmetric_attractor, InitialConditions, t)
    ts = [(states[:, 0])[-SampleSize:], (states[:, 1])
          [-SampleSize:], (states[:, 2])[-SampleSize:]]
    t = t[-SampleSize:]

    return t, ts


def burke_shaw_attractor(parameters=[12.0, 4.0], dynamic_state=None, InitialConditions=[0.6, 0.0, 0.0], L=500.0, fs=200, SampleSize=5000):
    """
    The Burke-Shaw Attractor is defined [8]_ as 

    .. math::
        \dot{x} &= s(x+y),

        \dot{y} &= -y - sxz,

        \dot{z} &= sxy + V

    The system parameters are set to :math:`s = 12`, :math:`V = 4`, and :math:`c = 28` for a periodic response and :math:`s = 10` for a chaotic response. The initial conditions were set to :math:`[x, y, z] = [0.6,0.0,0.0]`. The system was simulated for 500 seconds at a rate of 200 Hz and the last 25 seconds were used for the chaotic response.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Burke_Shaw_Attractor.png

    Parameters:
        parameters (Optional[floats]): Array of two floats [:math:`s`, :math:`V`] or None if using the dynamic_state variable
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
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            s = 12.0
        elif dynamic_state == 'chaotic':
            s = 10.0
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
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


def rucklidge_attractor(parameters=[1.1, 6.7], dynamic_state=None, InitialConditions=[1.0, 0.0, 4.5], L=1000.0, fs=50, SampleSize=5000):
    """
    The Rucklidge Attractor is defined [9]_ as

    .. math::
        \dot{x} &= -kx + \\lambda y - yz,

        \dot{y} &= x,

        \dot{z} &= -z + y^2

    The system parameters are set to :math:`k = 1.1`, :math:`\\lambda = 6.7` for a periodic response and :math:`k = 1.6` for a chaotic response. The initial conditions were set to :math:`[x, y, z] = [1.0,0.0,4.5]`. The system was simulated for 1000 seconds at a rate of 50 Hz and the last 100 seconds were used for the chaotic response.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Rucklidge_Attractor.png

    Parameters:
        parameters (Optional[floats]): Array of two floats [:math:`k`, :math:`\\lambda`] or None if using the dynamic_state variable
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
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            k = 1.1
        elif dynamic_state == 'chaotic':
            k = 1.6
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
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


def WINDMI(parameters=[0.9, 2.5], dynamic_state=None, InitialConditions=[1.0, 0.0, 4.5], L=1000.0, fs=20, SampleSize=5000):
    """
    The WINDMI Attractor is defined [10]_ as

    .. math::
        \dot{x} &= y,

        \dot{y} &= z,

        \dot{z} &= -az - y + b - e^x

    The system parameters are set to :math:`a = 0.9`, :math:`b = 2.5` for a periodic response and :math:`a = 0.8` for a chaotic response. The initial conditions were set to :math:`[x, y, z] = [1.0,0.0,4.5]`. The system was simulated for 1000 seconds at a rate of 20 Hz and the last 250 seconds were used for the chaotic response.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/WINDMI_Attractor.png

    Parameters:
        parameters (Optional[floats]): Array of two floats [:math:`a`, :math:`b`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [10] Vaidyanathan, S. "Adaptive backstepping controller design for the anti - synchronization of identical WINDMI chaotic systems with unknown parameters and its SPICE implementation". Journal of Engineering Science and Technology Review, 2015.
    """
    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 2

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 0.9
        elif dynamic_state == 'chaotic':
            a = 0.8
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
        b = 2.5
    else:
        a, b = parameters[0], parameters[1]

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

    return t, ts


def simplest_quadratic_chaotic_flow(parameters=[2.017, 1.0], dynamic_state=None, InitialConditions=[-0.9, 0.0, 0.5], L=1000.0, fs=20, SampleSize=5000):
    """
    The Simplest Quadratic Chaotic Flow is defined [11]_ as

    .. math::
        \dot{x} &= y,

        \dot{y} &= z,

        \dot{z} &= -az - by^2 - x

    The system parameters are set to :math:`a = 2.017`, :math:`b = 1.0` for a chaotic response we could not find any periodic response near this value. The initial conditions were set to :math:`[x, y, z] = [-0.9,0.0,0.5]`. The system was simulated for 1000 seconds at a rate of 20 Hz and the last 250 seconds were used for the chaotic response.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Simplest_Quadratic_Chaotic_Flow.png

    Parameters:
        parameters (Optional[floats]): Array of two floats [:math:`a`, :math:`b`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [11] Sprott, J.C. "Simplest dissipative chaotic flow". Physics Letters A, 1997.
    """

    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 2

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            print('We could not find a periodic response near $a = 2.017$.')
            print('Any contributions would be appreciated!')
            print('Defaulting to chaotic state.')
            a = 2.017
        elif dynamic_state == 'chaotic':
            a = 2.017
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
        b = 1.0
    else:
        a, b = parameters[0], parameters[1]

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

    return t, ts


def simplest_cubic_chaotic_flow(parameters=[2.11, 2.5], dynamic_state=None, InitialConditions=[0.0, 0.96, 0.0], L=1000.0, fs=20, SampleSize=5000):
    """
    The Simplest Cubic Chaotic Flow is defined [12]_ as

    .. math::
        \dot{x} &= y,

        \dot{y} &= z,

        \dot{z} &= -az - xy^2 - x

    The system parameters are set to :math:`a = 2.11`, :math:`b = 2.5` for a periodic response and :math:`a = 2.05` for chaotic. The initial conditions were set to :math:`[x, y, z] = [0.0,0.96,0.0]`. The system was simulated for 1000 seconds at a rate of 20 Hz and the last 250 seconds were used for the chaotic response.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Simplest_Cubic_Chaotic_Flow.png

    Parameters:
        parameters (Optional[floats]): Array of two floats [:math:`a`, :math:`b`] or None if using the dynamic_state variable
        fs (Optional[float]): Sampling rate for simulation
        SampleSize (Optional[int]): length of sample at end of entire time series
        L (Optional[int]): Number of iterations
        InitialConditions (Optional[floats]): list of values for [:math:`x_0`, :math:`y_0`, :math:`z_0`]
        dynamic_state (Optional[str]): Set dynamic state as either 'periodic' or 'chaotic' if not supplying parameters.

    Returns:
        array: Array of the time indices as `t` and the simulation time series `ts`

    References
    ----------
    .. [12] Malasomam, J.-M. "What is the simplest dissipative chaotic jerk equation which is parity invariant?" Physics Letters A, 2000.
    """
    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 2

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 2.11
        elif dynamic_state == 'chaotic':
            a = 2.05
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
        b = 2.5
    else:
        a, b = parameters[0], parameters[1]

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

    return t, ts


def simplest_piecewise_linear_chaotic_flow(parameters=[0.7], dynamic_state=None, InitialConditions=[0.0, -0.7, 0.0], L=1000.0, fs=40, SampleSize=5000):
    """
    The Simplest Piecewise-Linear Chaotic Flow is defined [13]_ as

    .. math::
        \dot{x} &= y,

        \dot{y} &= z,

        \dot{z} &= -az - y + |x| - 1

    The system parameter is set to :math:`a = 0.7` for a periodic response and :math:`a = 0.6` for chaotic. The initial conditions were set to :math:`[x, y, z] = [0.0,-0.7,0.0]`. The system was simulated for 1000 seconds at a rate of 20 Hz and the last 250 seconds were used for the chaotic response.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Simplest_Piecewise_Linear_Chaotic_Flow.png

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
    .. [13] TAO YANG. "PIECEWISE-LINEAR CHAOTIC SYSTEMS WITH a SINGLE EQUILIBRIUM POINT" International Journal of Bifurcation and Chaos, 2000.
    """

    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 1

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 0.7
        elif dynamic_state == 'chaotic':
            a = 0.6
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        a = parameters[0]

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

    return t, ts


def double_scroll(parameters=[1.0], dynamic_state=None, InitialConditions=[0.01, 0.01, 0.0], L=1000.0, fs=20, SampleSize=5000):
    """
    The Double Scroll Attractor is defined [14]_ as

    .. math::
        \dot{x} &= y,

        \dot{y} &= z,

        \dot{z} &= -a(x + y + z - \\text{sgn}(x))

    The system parameter is set to :math:`a = 1.0` for a periodic response and :math:`a = 0.8` for chaotic. The initial conditions were set to :math:`[x, y, z] = [0.01,0.01,0.0]`. The system was simulated for 1000 seconds at a rate of 20 Hz and the last 250 seconds were used for the chaotic response.

    .. figure:: ../../../figures/Autonomous_Dissipative_Flows/Double_Scroll_Attractor.png

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
    .. [14] A.S Elwakil "Chua's circuit decomposition: a systematic design approach for chaotic oscillators" Journal of the Franklin Institute, 2000.
    """
    # setting simulation time series parameters
    t = np.linspace(0, L, int(L*fs))

    # setting system parameters
    num_param = 1

    if len(parameters) != num_param:
        raise ValueError(
            f'Need {num_param} parameters as specified in documentation.')
    elif dynamic_state != None:
        if dynamic_state == 'periodic':
            a = 1.0
        elif dynamic_state == 'chaotic':
            a = 0.8
        else:
            raise ValueError(
                f'dynamic_state needs to be either "periodic" or "chaotic" or provide an array of length {num_param} in parameters.')
    else:
        a = parameters[0]

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
