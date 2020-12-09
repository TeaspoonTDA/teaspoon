def conservative_flows(system, dynamic_state = None, L = None, fs = None, 
                                 SampleSize = None, parameters = None, 
                                 InitialConditions = None):
    from scipy.integrate import odeint
    import numpy as np
    
    run = True
    if run == True:
                           
                    
    # In[ ]: Complete      
        
        if system == 'simplest_driven_chaotic_flow': 
            #setting simulation time series parameters
            if fs == None: fs = 50
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 300.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    print('Parameters needed are [omega].')
                    parameters = None
                else:
                    omega = parameters[0]
            if parameters == None:
                if dynamic_state == 'periodic': omega = 1
                if dynamic_state == 'chaotic': omega = 1.88
                    
            #defining simulation functions
            def simplest_driven_chaotic_flow(state, t):
                x, y = state  # unpack the state vector
                return y, -x**3 + np.sin(omega*t)
                    
            
            if InitialConditions == None:
                InitialConditions = [0, 0]
            
            states = odeint(simplest_driven_chaotic_flow, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:]]
            t = t[-SampleSize:]
             
                    
    # In[ ]: Complete
            
        if system == 'nose_hoover_oscillator': 
            #setting simulation time series parameters
            if fs == None: fs = 20
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    print('Parameters needed are [a].')
                    parameters = None
                else:
                    a = parameters[0]
            if parameters == None:
                if dynamic_state == 'periodic': a = 6
                if dynamic_state == 'chaotic': a = 1
                    
            #defining simulation functions
            def nose_hoover_oscillator(state, t):
                x, y, z = state  # unpack the state vector
                return y, -x+y*z, a-y**2
                    
            
            if InitialConditions == None:
                InitialConditions = [0, 5, 0]
            
            states = odeint(nose_hoover_oscillator, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
            
                                  
    # In[ ]: Complete
            
        if system == 'labyrinth_chaos': 
            #setting simulation time series parameters
            if fs == None: fs = 10
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 2000.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    print('Parameters needed are [a, b, c].')
                    parameters = None
                else:
                    a, b, c = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': 
                    print('We could not find a periodic response.')
                    print('Any contributions would be appreciated!')
                    print('Defaulting to chaotic state.')
                    a = 1
                if dynamic_state == 'chaotic': a = 1
                b, c = 1, 1
                    
            #defining simulation functions
            def labyrinth_chaos(state, t):
                x, y, z = state  # unpack the state vector
                return a*np.sin(y), b*np.sin(z), c*np.sin(x)
                    
            
            if InitialConditions == None:
                InitialConditions = [0.1, 0, 0]
            
            states = odeint(labyrinth_chaos, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
                       
                    
                                
            
                                  
    # In[ ]: Complete
            
        if system == 'henon_heiles_system': 
            #setting simulation time series parameters
            if fs == None: fs = 20
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 10000.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    print('Parameters needed are [a].')
                    parameters = None
                else:
                    a = parameters[0]
            if parameters == None:
                if dynamic_state == 'periodic': 
                    print('We could not find a periodic response.')
                    print('Any contributions would be appreciated!')
                    print('Defaulting to chaotic state.')
                    a = 1
                if dynamic_state == 'chaotic': a = 1.0
                    
            #defining simulation functions
            def henon_heiles_system(state, t):
                x, px, y, py = state  # unpack the state vector
                return px, -x - 2*a*x*y, py, -y - a*(x**2 - y**2)
                    
            
            if InitialConditions == None:
                InitialConditions = [0.499, 0, 0, 0.03]
            
            states = odeint(henon_heiles_system, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], 
                  (states[:,2])[-SampleSize:], (states[:,3])[-SampleSize:]]
            t = t[-SampleSize:]
            
            
    # In[ ]: Complete
    
    
    return t, ts