def periodic_functions(system, dynamic_state = None, L = None, fs = None, 
         SampleSize = None, parameters = None, InitialConditions = None):
    import numpy as np
    run = True
    if run == True:
        if system == 'sine':
            
            #setting simulation time series parameters
            if fs == None: fs = 50
            if SampleSize == None: SampleSize = 2000
            if L == None: L = 40
            t = np.linspace(0, L, int(L*fs))
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    omega = parameters[0]
            if parameters == None:
                omega = 2*np.pi
            
            ts = [(np.sin(omega*t))[-SampleSize:]]
            t = t[-SampleSize:]
                
    # In[ ]: Complete
    
        if system == 'incommensurate_sine':
            #setting simulation time series parameters
            if fs == None: fs = 50
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 100
            t = np.linspace(0, L, int(L*fs))
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    omega1, omega2 = parameters[0], parameters[1]
            if parameters == None:
                omega1 = np.pi
                omega2 = 1
                
            ts = [(np.sin(omega1*t) + np.sin(omega2*t))[-SampleSize:]]
            t = t[-SampleSize:]
    return t, ts