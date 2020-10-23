def delayed_flows(system, dynamic_state = None, L = None, fs = None, 
         SampleSize = None, parameters = None, InitialConditions = None):
    import numpy as np
    run = True
    if run == True:
        if system == 'mackey_glass':  
            #This requires installation of ddeint (pip install ddeint)
            from ddeint import ddeint
            #setting simulation time series parameters
            if fs == None: fs = 5
            if SampleSize == None: SampleSize = 1000
            if L == None: L = 400
            t = np.linspace(0, L, int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 4:
                    print('Warning: needed 4 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    gamma, τ, B, n = parameters[0], parameters[1], parameters[2], parameters[3]
            if parameters == None:
                if dynamic_state == 'periodic': n = 7.75
                if dynamic_state == 'chaotic': n = 9.65
                τ = 2.0
                B = 2.0
                gamma = 1.0001
    
        
            def mackey_glass(X, t, d):
                x = X(t)
                xd = X(t-d)
                return B*(xd/(1+xd**n)) - gamma*x
            
            fsolve = 50
            tt = np.linspace(0, L, int(L*fsolve))
            g = lambda t: np.array([1, 1])
            d = τ
            states = ddeint(mackey_glass, g, tt, fargs=(d,)).T
            
            ts = [((states[0])[::int(fsolve/fs)])[-SampleSize:]]
            t = tt[::int(fsolve/fs)][-SampleSize:]
            
    return t, ts