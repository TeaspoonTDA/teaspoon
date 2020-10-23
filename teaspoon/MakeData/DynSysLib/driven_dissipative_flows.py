def driven_dissipative_flows(system, dynamic_state = None, L = None, fs = None, 
                                 SampleSize = None, parameters = None, 
                                 InitialConditions = None):
    from scipy.integrate import odeint
    import numpy as np
    
    run = True
    if run == True:
        
                    
    # In[ ]: Complete
        if system == 'base_excited_magnetic_pendulum': 
            print(system,'Model or data is not currently available.')
            t = np.nan
            ts = [np.nan]   
            
            
            
    # In[ ]: Complete
            
        if system == 'driven_pendulum': 
            #setting simulation time series parameters
            if fs == None: fs = 50
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 300.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 6:
                    print('Warning: needed 6 parameters. Defaulting to periodic solution parameters.')
                    print('Parameters needed are [m, g, l, c, A, w].')
                    parameters = None
                else:
                    m, g, l, A, w = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
            if parameters == None:
                if dynamic_state == 'periodic': w = 1
                if dynamic_state == 'chaotic': w = 2
                m, g, l, c, A = 1, 9.81, 1, 0.1, 5
                    
            #defining simulation functions
            def driven_pendulum(state, t):
                th, om = state  # unpack the state vector
                return om, (-g/l)*np.sin(th) + (A/(m*l**2))*np.sin(w*t) - c*om
                    
            
            if InitialConditions == None:
                InitialConditions = [0, 0]
            
            states = odeint(driven_pendulum, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:]]
            t = t[-SampleSize:]
            
            
                               
            
    # In[ ]: 
            
        if system == 'driven_can_der_pol_oscillator': 
            #setting simulation time series parameters
            if fs == None: fs = 40
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 300.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [b, A, $\Omega$].')
                    parameters = None
                else:
                    b, A, omega = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': b = 2.9
                if dynamic_state == 'chaotic': b = 3.0
                A, omega = 5, 1.788
                    
            #defining simulation functions
            def driven_van_der_pol(state, t):
                x, y = state  # unpack the state vector
                return y, -x + b*(1-x**2)*y + A*np.sin(omega*t)
                    
            
            if InitialConditions == None:
                InitialConditions = [-1.9, 0]
            
            states = odeint(driven_van_der_pol, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:]]
            t = t[-SampleSize:]
                  
            
                 
    # In[ ]: 
            
        if system == 'shaw_van_der_pol_oscillator': 
            #setting simulation time series parameters
            if fs == None: fs = 25
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [b, A, $\Omega$].')
                    parameters = None
                else:
                    b, A, omega = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': omega = 1.4
                if dynamic_state == 'chaotic': omega = 1.8
                A, b = 5, 1
                    
            #defining simulation functions
            def shaw_van_der_pol_oscillator(state, t):
                x, y = state  # unpack the state vector
                return y + np.sin(omega*t), -x + b*(1-x**2)*y
                    
            
            if InitialConditions == None:
                InitialConditions = [1.3, 0]
            
            states = odeint(shaw_van_der_pol_oscillator, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:]]
            t = t[-SampleSize:]
                                
                 
    # In[ ]: 
            
        if system == 'forced_brusselator': 
            #setting simulation time series parameters
            if fs == None: fs = 20
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 4:
                    print('Warning: needed 4 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [a, b, A, $\Omega$].')
                    parameters = None
                else:
                    a, b, A, omega = parameters[0], parameters[1], parameters[2], parameters[3]
            if parameters == None:
                if dynamic_state == 'periodic': omega = 1.1
                if dynamic_state == 'chaotic': omega = 1.0
                a, b, A = 0.4, 1.2, 0.05
                    
            #defining simulation functions
            def forced_brusselator(state, t):
                x, y = state  # unpack the state vector
                return (x**2)*y - (b+1)*x + a + A*np.sin(omega*t), -(x**2)*y + b*x
                    
            
            if InitialConditions == None:
                InitialConditions = [0.3, 2]
            
            states = odeint(forced_brusselator, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:]]
            t = t[-SampleSize:]
                                               
                 
    # In[ ]: 
            
        if system == 'ueda_oscillator': 
            #setting simulation time series parameters
            if fs == None: fs = 50
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [b, A, $\Omega$].')
                    parameters = None
                else:
                    b, A, omega = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': omega = 1.2
                if dynamic_state == 'chaotic': omega = 1.0
                b, A = 0.05, 7.5
                    
            #defining simulation functions
            def ueda_oscillator(state, t):
                x, y = state  # unpack the state vector
                return y, -x**3 - b*y + A*np.sin(omega*t)
                    
            
            if InitialConditions == None:
                InitialConditions = [2.5, 0.0]
            
            states = odeint(ueda_oscillator, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:]]
            t = t[-SampleSize:]     
            
                                                        
                 
    # In[ ]: 
            
        if system == 'duffings_two_well_oscillator': 
            #setting simulation time series parameters
            if fs == None: fs = 20
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [b, A, $\Omega$].')
                    parameters = None
                else:
                    b, A, omega = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': omega = 1.1
                if dynamic_state == 'chaotic': omega = 1.0
                b, A = 0.25, 0.4
                    
            #defining simulation functions
            def duffings_two_well_oscillator(state, t):
                x, y = state  # unpack the state vector
                return y, -x**3 + x - b*y + A*np.sin(omega*t)
                    
            
            if InitialConditions == None:
                InitialConditions = [0.2, 0.0]
            
            states = odeint(duffings_two_well_oscillator, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:]]
            t = t[-SampleSize:]     
            
                                                                   
                 
    # In[ ]: 
            
        if system == 'duffing_van_der_pol_oscillator': 
            #setting simulation time series parameters
            if fs == None: fs = 20
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 4:
                    print('Warning: needed 4 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [mu, gamma, A, omega].')
                    parameters = None
                else:
                    mu, gamma, A, omega = parameters[0], parameters[1], parameters[2], parameters[3]
            if parameters == None:
                if dynamic_state == 'periodic': omega = 1.3
                if dynamic_state == 'chaotic': omega = 1.2
                mu, gamma, A  = 0.2, 8, 0.35
                    
            #defining simulation functions
            def duffing_van_der_pol_oscillator(state, t):
                x, y = state  # unpack the state vector
                return y, mu*(1-gamma*x**2)*y - x**3 + A*np.sin(omega*t)
                    
            
            if InitialConditions == None:
                InitialConditions = [0.2, -0.2]
            
            states = odeint(duffing_van_der_pol_oscillator, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:]]
            t = t[-SampleSize:]  
                                                                            
                 
    # In[ ]: 
            
        if system == 'rayleigh_duffing_oscillator': 
            #setting simulation time series parameters
            if fs == None: fs = 20
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 4:
                    print('Warning: needed 4 parameters. Defaulting to periodic solution parameters.')
                    print(r'Parameters needed are [mu, gamma, A, omega].')
                    parameters = None
                else:
                    mu, gamma, A, omega = parameters[0], parameters[1], parameters[2], parameters[3]
            if parameters == None:
                if dynamic_state == 'periodic': omega = 1.4
                if dynamic_state == 'chaotic': omega = 1.2
                mu, gamma, A  = 0.2, 4, 0.3
                    
            #defining simulation functions
            def rayleigh_duffing_oscillator(state, t):
                x, y = state  # unpack the state vector
                return y, mu*(1-gamma*y**2)*y - x**3 + A*np.sin(omega*t)
                    
            
            if InitialConditions == None:
                InitialConditions = [0.3, 0.0]
            
            states = odeint(rayleigh_duffing_oscillator, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:]]
            t = t[-SampleSize:]  
            
            
    # In[ ]:
        
    return t, ts