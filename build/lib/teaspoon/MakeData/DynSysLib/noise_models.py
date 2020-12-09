def noise_models(system, dynamic_state = None, L = None, fs = None, 
         SampleSize = None, parameters = None, InitialConditions = None):
    import numpy as np
    run = True
    if run == True:
        if system == 'gaussian_noise':
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 1000
            if L == None: L = 1000
            t = np.linspace(0, L, int(L*fs))
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    sigma, mu = parameters[0], parameters[1]
            if parameters == None:
                sigma, mu = 1, 0
                
            ts = [(np.random.normal(mu, sigma, len(t)))[-SampleSize:]]
            t = t[-SampleSize:]
            
        
    # In[ ]: Complete
    
        if system == 'uniform_noise':
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 1000
            if L == None: L = 1000
            t = np.linspace(0, L, int(L*fs))
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b = parameters[0], parameters[1]
            if parameters == None:
                a, b = -1, 1
                
            ts = [(np.random.uniform(a, b, size = len(t)))[-SampleSize:]]
            t = t[-SampleSize:]
            
                
    # In[ ]: Complete
    
        if system == 'rayleigh_noise':
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 1000
            if L == None: L = 1000
            t = np.linspace(0, L, int(L*fs))
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    sigma = parameters[0]
            if parameters == None:
                sigma = 1
                
            ts = [(np.random.rayleigh(sigma, size = len(t)))[-SampleSize:]]
            t = t[-SampleSize:]
                
    # In[ ]: Complete
    
        if system == 'exponential_noise':
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 1000
            if L == None: L = 1000
            t = np.linspace(0, L, int(L*fs))
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    sigma = parameters[0]
            if parameters == None:
                sigma = 1
                
            ts = [(np.random.exponential(sigma, len(t)))[-SampleSize:]]
            t = t[-SampleSize:]
    return t, ts