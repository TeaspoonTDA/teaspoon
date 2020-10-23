
def maps(system, dynamic_state = None, L = None, fs = None, 
         SampleSize = None, parameters = None, InitialConditions = None):
    import numpy as np
    run = True
    if run == True:
        
                                 
    # In[ ]: Complete
    
        if system == 'sine_map': 
            
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 500
            if L == None: L = 1000.0
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    A = parameters[0]
            if parameters == None:
                if dynamic_state == 'periodic': A = 0.8
                if dynamic_state == 'chaotic': A = 1.0
                
                    
            if InitialConditions == None:
                InitialConditions = [0.1]
            xn = InitialConditions[0]
                
                
            t, ts = [], []
            for n in range(0, int(L)): 
                xn = A*np.sin(np.pi*xn)
                ts = np.append(ts, xn)
                t = np.append(t, n)
                    
            ts = [ts[-SampleSize:]]
            t = t[-SampleSize:]    
                         
    # In[ ]: Complete
        
        if system == 'tent_map': 
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 500
            if L == None: L = 1000.0
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    A = parameters[0]
            if parameters == None:
                if dynamic_state == 'periodic': A = 1.05
                if dynamic_state == 'chaotic': A = 1.5
                
                    
            if InitialConditions == None:
                InitialConditions = [1/np.sqrt(2)]
            xn = InitialConditions[0]
                
                
            t, ts = [], []
            for n in range(0, int(L)): 
                xn = A*np.min([xn, 1-xn])
                ts = np.append(ts, xn)
                t = np.append(t, n)
                    
            ts = [ts[-SampleSize:]]
            t = t[-SampleSize:]     
                             
    # In[ ]: Complete
        
        if system == 'linear_congruential_generator_map': 
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 500
            if L == None: L = 1000.0
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b, c = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': a = 0.9
                if dynamic_state == 'chaotic': a = 1.1
                b, c = 54773, 259200
                
                    
            if InitialConditions == None:
                InitialConditions = [0.1]
            xn = InitialConditions[0]
                
                
            t, ts = [], []
            for n in range(0, int(L)): 
                xn = (a*xn + b)%c
                ts = np.append(ts, xn)
                t = np.append(t, n)
                    
            ts = [ts[-SampleSize:]]
            t = t[-SampleSize:]  
                             
    # In[ ]: Complete
        
        if system == 'rickers_population_map': 
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize =500
            if L == None: L = 1000.0
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a = parameters[0]
            if parameters == None:
                if dynamic_state == 'periodic': a = 13
                if dynamic_state == 'chaotic': a = 20
                
                    
            if InitialConditions == None:
                InitialConditions = [0.1]
            xn = InitialConditions[0]
                
                
            t, ts = [], []
            for n in range(0, int(L)): 
                xn = a*xn*np.exp(-xn)
                ts = np.append(ts, xn)
                t = np.append(t, n)
                    
            ts = [ts[-SampleSize:]]
            t = t[-SampleSize:]  
                        
    # In[ ]: Complete
    
        if system == 'gauss_map': 
            # taken from https://en.wikipedia.org/wiki/Gauss_iterated_map
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize =500
            if L == None: L = 1000.0
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    alpha, beta = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': beta = -0.20
                if dynamic_state == 'chaotic': beta = -0.35
                alpha = 6.20
                    
            if InitialConditions == None:
                InitialConditions = [0.1]
            xn = InitialConditions[0]
                
                
            t, ts = [], []
            for n in range(0, int(L)): 
                xn = np.exp(-alpha*xn**2) + beta
                ts = np.append(ts, xn)
                t = np.append(t, n)
                    
            ts = [ts[-SampleSize:]]
            t = t[-SampleSize:]     
                                    
    # In[ ]: Complete
    
        if system == 'cusp_map':
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize =500
            if L == None: L = 1000.0
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a = parameters[0]
            if parameters == None:
                if dynamic_state == 'periodic': a = 1.1
                if dynamic_state == 'chaotic': a = 1.2
                    
            if InitialConditions == None:
                InitialConditions = [0.5]
            xn = InitialConditions[0]
                
            t, ts = [], []
            for n in range(0, int(L)): 
                xn = 1-a*np.sqrt(np.abs(xn))
                ts = np.append(ts, xn)
                t = np.append(t, n)
                    
            ts = [ts[-SampleSize:]]
            t = t[-SampleSize:]   
                     
                                       
    # In[ ]: Complete
    
        if system == 'pinchers_map':  
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize =500
            if L == None: L = 1000.0
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    s, c = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': s = 1.3
                if dynamic_state == 'chaotic': s = 1.6
                c = 0.5
                    
            if InitialConditions == None:
                InitialConditions = [0.0]
            xn = InitialConditions[0]
            
                
            t, ts = [], []
            for n in range(0, int(L)): 
                xn = np.abs(np.tanh(s*(xn-c)))
                ts = np.append(ts, xn)
                t = np.append(t, n)
                    
            ts = [ts[-SampleSize:]]
            t = t[-SampleSize:]  
    
                                     
    # In[ ]: Complete
        
        if system == 'sine_circle_map': 
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize =500
            if L == None: L = 1000.0
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    omega, k = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': k = 1.5
                if dynamic_state == 'chaotic': k = 2.0
                omega = 0.5
                
            if InitialConditions == None:
                InitialConditions = [0.0]
            xn = InitialConditions[0]
                
                
            t, ts = [], []
            for n in range(0, int(L)): 
                xn = xn + omega - (k/(2*np.pi))*np.sin(2*np.pi*xn)%1
                ts = np.append(ts, xn)
                t = np.append(t, n)
                    
            ts = [ts[-SampleSize:]]
            t = t[-SampleSize:]  
            
    # In[ ]: Complete
        
        if system == 'logistic_map': 
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 500
            if L == None: L = 1000.0
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    r = parameters[0]
            if parameters == None:
                if dynamic_state == 'periodic': r = 3.5
                if dynamic_state == 'chaotic': r = 3.6 
                
                    
            if InitialConditions == None:
                InitialConditions = [0.5]
            xn = InitialConditions[0]
                
                
            t, ts = [], []
            for n in range(0, int(L)): 
                xn = r*xn*(1-xn)
                ts = np.append(ts, xn)
                t = np.append(t, n)
                    
            ts = [ts[-SampleSize:]]
            t = t[-SampleSize:]       
                    
    # In[ ]: Complete
               
        if system == 'henon_map': 
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 500
            if L == None: L = 1000
            t = np.linspace(0, L, int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b, c = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': a = 1.25
                if dynamic_state == 'chaotic': a = 1.20
                b = 0.3
                c = 1.0
                    
            #defining simulation functions
            def henon(a, b, c, x,y):
                return y + c - a*x*x, b*x
                    
            
            if InitialConditions == None:
                InitialConditions = [0.1, 0.3] 
            
            xtemp = InitialConditions[0]
            ytemp = InitialConditions[1]
            x, y = [], []
            for n in range(0,int(L)):
                xtemp, ytemp = henon(a,b,c,xtemp,ytemp)
                x.append(xtemp)
                y.append(ytemp)
                
            ts = [x[-SampleSize:], y[-SampleSize:]]
            t = t[-SampleSize:]
                    
                            
    # In[ ]: Complete
               
        if system == 'lozi_map':  
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 500
            if L == None: L = 1000.0
            t = np.linspace(0, L, int(L*fs))
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': a = 1.5
                if dynamic_state == 'chaotic': a = 1.7
                b = 0.5
                    
            #defining simulation functions
            def lozi(a, b, x,y):
                return 1-a*np.abs(x) + b*y, x
                
            if InitialConditions == None:
                InitialConditions = [-0.1, 0.1]
                
                
            xtemp = InitialConditions[0]
            ytemp = InitialConditions[1]
            x, y = [], []
            for n in range(0,int(L)):
                xtemp, ytemp = lozi(a,b,xtemp,ytemp)
                x.append(xtemp)
                y.append(ytemp)
                    
            ts = [x[-SampleSize:], y[-SampleSize:]]
            t = t[-SampleSize:]      
            
                                
    # In[ ]: Complete
               
        if system == 'delayed_logstic_map':
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 500
            if L == None: L = 1000.0
            t = np.linspace(0, L, int(L*fs))
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a = parameters[0]
            if parameters == None:
                if dynamic_state == 'periodic': a = 2.20
                if dynamic_state == 'chaotic': a = 2.27
                    
            #defining simulation functions
            def delay_log(a, x, y):
                return a*x*(1-y), x
                
            if InitialConditions == None:
                InitialConditions = [0.001, 0.001]
                
                
            xtemp = InitialConditions[0]
            ytemp = InitialConditions[1]
            x, y = [], []
            for n in range(0,int(L)):
                xtemp, ytemp = delay_log(a,xtemp,ytemp)
                x.append(xtemp)
                y.append(ytemp)
                    
            ts = [x[-SampleSize:], y[-SampleSize:]]
            t = t[-SampleSize:]      
            
                                        
    # In[ ]: Complete
               
        if system == 'tinkerbell_map':  
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 500
            if L == None: L = 1000.0
            t = np.linspace(0, L, int(L*fs))
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 4:
                    print('Warning: needed 4 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b, c, d = parameters[0], parameters[1], parameters[2], parameters[3]
            if parameters == None:
                if dynamic_state == 'periodic': a = 0.7
                if dynamic_state == 'chaotic': a = 0.9
                b, c, d = -0.6, 2, 0.5
                    
            #defining simulation functions
            def tinkerbell(a, b, c, d, x, y):
                return x**2 - y**2 + a*x + b*y, 2*x*y + c*x + d*y
                
            if InitialConditions == None:
                InitialConditions = [0.0, 0.5]
                
                
            xtemp = InitialConditions[0]
            ytemp = InitialConditions[1]
            x, y = [], []
            for n in range(0,int(L)):
                xtemp, ytemp = tinkerbell(a, b, c, d,xtemp,ytemp)
                x.append(xtemp)
                y.append(ytemp)
                    
            ts = [x[-SampleSize:], y[-SampleSize:]]
            t = t[-SampleSize:]      
            
                                                
    # In[ ]: Complete
               
        if system == 'burgers_map': 
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 500
            if L == None: L = 3000.0
            t = np.linspace(0, L, int(L*fs))
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': b = 1.6
                if dynamic_state == 'chaotic': b =  1.75
                a = 0.75
                
                    
            #defining simulation functions
            def burgers(a, b, x, y):
                return a*x - y**2, b*y + x*y
                
            if InitialConditions == None:
                InitialConditions = [-0.1, 0.1]
                
                
            xtemp = InitialConditions[0]
            ytemp = InitialConditions[1]
            x, y = [], []
            for n in range(0,int(L)):
                xtemp, ytemp = burgers(a, b,xtemp,ytemp)
                x.append(xtemp)
                y.append(ytemp)
                    
            ts = [x[-SampleSize:], y[-SampleSize:]]
            t = t[-SampleSize:]      
                                                      
    # In[ ]: Complete
               
        if system == 'holmes_cubic_map':
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 500
            if L == None: L = 3000.0
            t = np.linspace(0, L, int(L*fs))
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    b, d = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': b = 0.27
                if dynamic_state == 'chaotic': b =  0.20
                d = 2.77
                
                    
            #defining simulation functions
            def holmes(b, d, x, y):
                return y, -b*x + +d*y - y**3
                
            if InitialConditions == None:
                InitialConditions = [1.6, 0.0]
                
                
            xtemp = InitialConditions[0]
            ytemp = InitialConditions[1]
            x, y = [], []
            for n in range(0,int(L)):
                xtemp, ytemp = holmes(b, d,xtemp,ytemp)
                x.append(xtemp)
                y.append(ytemp)
                    
            ts = [x[-SampleSize:], y[-SampleSize:]]
            t = t[-SampleSize:]      
                                                                
    # In[ ]: Complete
               
        if system == 'kaplan_yorke_map': 
            #setting simulation time series parameters
            if fs == None: fs = 1
            if SampleSize == None: SampleSize = 500
            if L == None: L = 1000.0
            t = np.linspace(0, L, int(L*fs))
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': a = -1.0
                if dynamic_state == 'chaotic': a= -2.0
                b = 0.2
                
                    
            #defining simulation functions
            def kaplan_yorke(a, b, x, y):
                return (a*x)%0.99995, (b*y + np.cos(4*np.pi*x))
                
            if InitialConditions == None:
                InitialConditions = [1/np.sqrt(2), -0.4]
                
                
            xtemp = InitialConditions[0]
            ytemp = InitialConditions[1]
            x, y = [], []
            for n in range(0,int(L)):
                xtemp, ytemp = kaplan_yorke(a,b,xtemp,ytemp)
                x.append(xtemp)
                y.append(ytemp)
                    
            ts = [x[-SampleSize:], y[-SampleSize:]]
            t = t[-SampleSize:] 
            
            
    return t, ts