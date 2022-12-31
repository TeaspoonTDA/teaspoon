

def driven_dissipative_flows(system, dynamic_state=None, L=None, fs=None,
                             SampleSize=None, parameters=None,
                             InitialConditions=None):
    from scipy.integrate import odeint
    import numpy as np

    run = True
    if run == True:

        # In[ ]: Complete
        if system == 'base_excited_magnetic_pendulum':

            get_EOM = True
            if get_EOM == True:
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

            if fs == None:
                fs = 200
            if SampleSize == None:
                SampleSize = 5000
            if L == None:
                L = 100.0
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 13:
                    print(
                        'Warning: needed 13 parameters. Defaulting to periodic solution parameters.')
                    print(
                        'Parameters needed are [m, l, g, r_cm, I_o, A, w, c, q, d, mu].')
                    parameters = None
                else:
                    m, l, g, r_cm, I_o = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
                    A, w, c, q, d, mu = parameters[5], parameters[6], parameters[
                        7], parameters[8], parameters[9], parameters[10]
            if parameters == None:
                if dynamic_state == 'periodic':
                    A = 0.022
                if dynamic_state == 'chaotic':
                    A = 0.021
                w = 3*np.pi
                q, mu, g = 1.2, 1.257E-6, 9.81
                m, l, d, r_cm, I_o, c = 0.1038, 0.208, 0.032, 0.18775, 0.00001919, 0.003

                # Parameter values
                # m (mass)                    = estimated
                # l (length)                  = 0.208 meters pm 5%
                # g (gravity)                 = 9.81 m/s^2 pm 0.05%
                # r_cm (distance to cm)       = estimated
                # I_o (inertia about o)       = 0.00002389 kg/m^2
                # w (base frequency)          = 3pi rad/s
                # A (base amplitude)          = varies (0.01 through 0.5)?
                # c (viscous damping)         = estimated
                # mu (Univ. Mag. Const.)      = 1.257E-6
                # d (distance)                = 0.036 meters pm 5%

            if InitialConditions == None:
                InitialConditions = [0.0, 0.0]

            def vectorfield(state, t):
                ph, phi_dot = state
                return phi_dot, F_phidd((t, ph, phi_dot), (m, l, g, r_cm, I_o, A, w, c, q, d, mu))

            states = odeint(vectorfield, InitialConditions, t)
            ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
            t = t[-SampleSize:]

    # In[ ]: Complete

        if system == 'driven_pendulum':
            # setting simulation time series parameters
            if fs == None:
                fs = 50
            if SampleSize == None:
                SampleSize = 5000
            if L == None:
                L = 300.0
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 6:
                    print(
                        'Warning: needed 6 parameters. Defaulting to periodic solution parameters.')
                    print('Parameters needed are [m, g, l, c, A, w].')
                    parameters = None
                else:
                    m, g, l, A, w = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
            if parameters == None:
                if dynamic_state == 'periodic':
                    w = 1
                if dynamic_state == 'chaotic':
                    w = 2
                m, g, l, c, A = 1, 9.81, 1, 0.1, 5

            # defining simulation functions
            def driven_pendulum(state, t):
                th, om = state  # unpack the state vector
                return om, (-g/l)*np.sin(th) + (A/(m*l**2))*np.sin(w*t) - c*om

            if InitialConditions == None:
                InitialConditions = [0, 0]

            states = odeint(driven_pendulum, InitialConditions, t)
            ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
            t = t[-SampleSize:]

    # In[ ]:

        if system == 'driven_van_der_pol_oscillator':
            # setting simulation time series parameters
            if fs == None:
                fs = 40
            if SampleSize == None:
                SampleSize = 5000
            if L == None:
                L = 300.0
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print(
                        'Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [b, A, $\Omega$].')
                    parameters = None
                else:
                    b, A, omega = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic':
                    b = 2.9
                if dynamic_state == 'chaotic':
                    b = 3.0
                A, omega = 5, 1.788

            # defining simulation functions
            def driven_van_der_pol(state, t):
                x, y = state  # unpack the state vector
                return y, -x + b*(1-x**2)*y + A*np.sin(omega*t)

            if InitialConditions == None:
                InitialConditions = [-1.9, 0]

            states = odeint(driven_van_der_pol, InitialConditions, t)
            ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
            t = t[-SampleSize:]

    # In[ ]:

        if system == 'shaw_van_der_pol_oscillator':
            # setting simulation time series parameters
            if fs == None:
                fs = 25
            if SampleSize == None:
                SampleSize = 5000
            if L == None:
                L = 500.0
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print(
                        'Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [b, A, $\Omega$].')
                    parameters = None
                else:
                    b, A, omega = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic':
                    omega = 1.4
                if dynamic_state == 'chaotic':
                    omega = 1.8
                A, b = 5, 1

            # defining simulation functions
            def shaw_van_der_pol_oscillator(state, t):
                x, y = state  # unpack the state vector
                return y + np.sin(omega*t), -x + b*(1-x**2)*y

            if InitialConditions == None:
                InitialConditions = [1.3, 0]

            states = odeint(shaw_van_der_pol_oscillator, InitialConditions, t)
            ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
            t = t[-SampleSize:]

    # In[ ]:

        if system == 'forced_brusselator':
            # setting simulation time series parameters
            if fs == None:
                fs = 20
            if SampleSize == None:
                SampleSize = 5000
            if L == None:
                L = 500.0
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 4:
                    print(
                        'Warning: needed 4 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [a, b, A, $\Omega$].')
                    parameters = None
                else:
                    a, b, A, omega = parameters[0], parameters[1], parameters[2], parameters[3]
            if parameters == None:
                if dynamic_state == 'periodic':
                    omega = 1.1
                if dynamic_state == 'chaotic':
                    omega = 1.0
                a, b, A = 0.4, 1.2, 0.05

            # defining simulation functions
            def forced_brusselator(state, t):
                x, y = state  # unpack the state vector
                return (x**2)*y - (b+1)*x + a + A*np.sin(omega*t), -(x**2)*y + b*x

            if InitialConditions == None:
                InitialConditions = [0.3, 2]

            states = odeint(forced_brusselator, InitialConditions, t)
            ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
            t = t[-SampleSize:]

    # In[ ]:

        if system == 'ueda_oscillator':
            # setting simulation time series parameters
            if fs == None:
                fs = 50
            if SampleSize == None:
                SampleSize = 5000
            if L == None:
                L = 500.0
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print(
                        'Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [b, A, $\Omega$].')
                    parameters = None
                else:
                    b, A, omega = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic':
                    omega = 1.2
                if dynamic_state == 'chaotic':
                    omega = 1.0
                b, A = 0.05, 7.5

            # defining simulation functions
            def ueda_oscillator(state, t):
                x, y = state  # unpack the state vector
                return y, -x**3 - b*y + A*np.sin(omega*t)

            if InitialConditions == None:
                InitialConditions = [2.5, 0.0]

            states = odeint(ueda_oscillator, InitialConditions, t)
            ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
            t = t[-SampleSize:]

    # In[ ]:

        if system == 'duffings_two_well_oscillator':
            # setting simulation time series parameters
            if fs == None:
                fs = 20
            if SampleSize == None:
                SampleSize = 5000
            if L == None:
                L = 500.0
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print(
                        'Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [b, A, $\Omega$].')
                    parameters = None
                else:
                    b, A, omega = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic':
                    omega = 1.1
                if dynamic_state == 'chaotic':
                    omega = 1.0
                b, A = 0.25, 0.4

            # defining simulation functions
            def duffings_two_well_oscillator(state, t):
                x, y = state  # unpack the state vector
                return y, -x**3 + x - b*y + A*np.sin(omega*t)

            if InitialConditions == None:
                InitialConditions = [0.2, 0.0]

            states = odeint(duffings_two_well_oscillator, InitialConditions, t)
            ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
            t = t[-SampleSize:]

    # In[ ]:

        if system == 'duffing_van_der_pol_oscillator':
            # setting simulation time series parameters
            if fs == None:
                fs = 20
            if SampleSize == None:
                SampleSize = 5000
            if L == None:
                L = 500.0
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 4:
                    print(
                        'Warning: needed 4 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [mu, gamma, A, omega].')
                    parameters = None
                else:
                    mu, gamma, A, omega = parameters[0], parameters[1], parameters[2], parameters[3]
            if parameters == None:
                if dynamic_state == 'periodic':
                    omega = 1.3
                if dynamic_state == 'chaotic':
                    omega = 1.2
                mu, gamma, A = 0.2, 8, 0.35

            # defining simulation functions
            def duffing_van_der_pol_oscillator(state, t):
                x, y = state  # unpack the state vector
                return y, mu*(1-gamma*x**2)*y - x**3 + A*np.sin(omega*t)

            if InitialConditions == None:
                InitialConditions = [0.2, -0.2]

            states = odeint(duffing_van_der_pol_oscillator,
                            InitialConditions, t)
            ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
            t = t[-SampleSize:]

    # In[ ]:

        if system == 'rayleigh_duffing_oscillator':
            # setting simulation time series parameters
            if fs == None:
                fs = 20
            if SampleSize == None:
                SampleSize = 5000
            if L == None:
                L = 500.0
            t = np.linspace(0, L, int(L*fs))

            # setting system parameters
            if parameters != None:
                if len(parameters) != 4:
                    print(
                        'Warning: needed 4 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [mu, gamma, A, omega].')
                    parameters = None
                else:
                    mu, gamma, A, omega = parameters[0], parameters[1], parameters[2], parameters[3]
            if parameters == None:
                if dynamic_state == 'periodic':
                    omega = 1.4
                if dynamic_state == 'chaotic':
                    omega = 1.2
                mu, gamma, A = 0.2, 4, 0.3

            # defining simulation functions
            def rayleigh_duffing_oscillator(state, t):
                x, y = state  # unpack the state vector
                return y, mu*(1-gamma*y**2)*y - x**3 + A*np.sin(omega*t)

            if InitialConditions == None:
                InitialConditions = [0.3, 0.0]

            states = odeint(rayleigh_duffing_oscillator, InitialConditions, t)
            ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
            t = t[-SampleSize:]

    # In[ ]:

    return t, ts
