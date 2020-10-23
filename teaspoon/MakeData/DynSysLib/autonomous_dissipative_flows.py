



def autonomous_dissipative_flows(system, dynamic_state = None, L = None, fs = None, 
                                 SampleSize = None, parameters = None, 
                                 InitialConditions = None):
    from scipy.integrate import odeint
    import scipy.integrate as integrate
    import numpy as np
# In[ ]:
    run = True
    if run == True:
              
        
# In[ ]:Complete
        
        if system == 'lorenz': 
            
            #setting simulation time series parameters
            if fs == None: fs = 100
            if SampleSize == None: SampleSize = 2000
            if L == None: L = 100.0
            t = np.linspace(0, L, int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    rho, sigma, beta = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': rho = 100
                if dynamic_state == 'chaotic': rho = 105
                sigma = 10.0
                beta = 8.0 / 3.0
                     
                    
            #defining simulation functions
            def lorenz(state, t):
               x, y, z = state  # unpack the state vector
               return sigma*(y - x), x*(rho - z) - y, x*y - beta*z  # derivatives
                    
           
            if InitialConditions == None:
                InitialConditions = [10.0**-10.0, 0.0, 1.0]
            
            states = odeint(lorenz, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:] 
                
    # In[ ]: Complete
    
        if system == 'rossler':
            #setting simulation time series parameters
            if fs == None: fs = 15
            if SampleSize == None: SampleSize = 2500
            if L == None: L = 1000.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b, c = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': a = 0.10
                if dynamic_state == 'chaotic': a = 0.15 
                b = 0.20
                c = 14
                
                    
            #defining simulation functions
            def rossler(state, t):
                x, y, z = state  # unpack the state vector
                return -y - z, x + a*y, b + z*(x-c)
                    
            
            if InitialConditions == None:
                InitialConditions = [-0.4, 0.6, 1] 
            
            
            states = odeint(rossler, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
    # In[ ]: Complete
    
        if system == 'chua':
            #setting simulation time series parameters
            if fs == None: fs = 50
            if SampleSize == None: SampleSize = 4000
            if L == None: L = 200
            t = np.linspace(0, L, int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 5:
                    print('Warning: needed 5 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, g, B, m0, m1 = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
            if parameters == None:
                if dynamic_state == 'periodic': a = 10.8 #alpha
                if dynamic_state == 'chaotic': a = 12.8
                B = 27 #betta
                g = 1.0 #gamma
                m0 = -3/7
                m1 = 3/7
                    
                    
            #defining simulation functions
            def f(x):
                f = m1*x+(m0-m1)/2.0*(abs(x+1.0)-abs(x-1.0))
                return f
            
            def chua(H, t=0):
                return np.array([a*(H[1]-f(H[0])),
                              g*(H[0]-H[1]+H[2]),
                              -B*H[1]])
            
    
            if InitialConditions == None:
                InitialConditions = [1.0, 0.0, 0.0]
            
            states, infodict = integrate.odeint(chua, InitialConditions, t, full_output=True)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
            
                    
    # In[ ]: Complete
    
    
        if system == 'double_pendulum': 
            #setting simulation time series parameters
            if fs == None: fs = 100
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 100.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 5:
                    print('Warning: needed 5 parameters. Defaulting to periodic solution parameters.')
                    print('Parameters needed are [m1, m2, l1 ,l2, g].')
                    parameters = None
                else:
                    m1, m2, l1 ,l2, g = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
            if parameters == None:
                m1, m2, l1 ,l2, g = 1, 1, 1, 1, 9.81
                    
            #defining simulation functions
            def double_pendulum(state, t):
                th1, th2, om1, om2 = state  # unpack the state vector
                numerator1 = -g*(2*m1+m2)*np.sin(th1) - m2*g*np.sin(th1-2*th2) - 2*np.sin(th1-th2)*m2*(om2**2 * l2 + om1**2 *l1*np.cos(th1-th2))
                numerator2 = 2*np.sin(th1-th2)*(om1**2 *l1*(m1+m2) + g*(m1+m2)*np.cos(th1)+om2**2 * l2*m2*np.cos(th1-th2))
                denomenator1 = l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))
                denomenator2 = l2*(2*m1+m2-m2*np.cos(2*th1-2*th2))
                D = [om1,
                     om2,
                     numerator1/denomenator1,
                     numerator2/denomenator2]
                return D[0], D[1], D[2], D[3]
                    
            
            if InitialConditions == None:
                if dynamic_state == 'periodic': InitialConditions = [0.4, 0.6, 1, 1] 
                if dynamic_state == 'quasiperiodic': InitialConditions = [1, 0, 0, 0] 
                if dynamic_state == 'chaotic': InitialConditions = [0.0, 3, 0, 0] 
                
            
            
            states = odeint(double_pendulum, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], 
                  (states[:,2])[-SampleSize:], (states[:,3])[-SampleSize:]]
            t = t[-SampleSize:]

            
    # In[ ]: Complete
            
        if system == 'coupled_lorenz_rossler': 
            #setting simulation time series parameters
            if fs == None: fs = 50
            if SampleSize == None: SampleSize = 1500
            if L == None: L = 500.0
            t = np.linspace(0, L, int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 9:
                    print('Warning: needed 9 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a2, b1, b2, c2 = parameters[0], parameters[1], parameters[2], parameters[3]
                    k1, k2, k3, lam, sigma = parameters[4], parameters[5], parameters[6], parameters[7], parameters[8]
            if parameters == None:
                if dynamic_state == 'periodic': a2 = 0.25
                if dynamic_state == 'chaotic': a2 = 0.51
                b1, b2 = 8/3, 0.2
                c2 = 5.7
                k1, k2, k3 = 0.1, 0.1, 0.1
                lam, sigma = 28, 10
                
                    
            #defining simulation functions
            def coupled_lorenz_rossler(state, t):
                x1, y1, z1, x2, y2, z2 = state  # unpack the state vector
                D = [-y1 - z1 + k1*(x2-x1),
                     x1 + a2*y1 + k2*(y2-y1),
                     b2 + z1*(x1-c2) + k3*(z2-z1),
                     sigma*(y2-x2),
                     lam*x2 -y2 - x2*z2,
                     x2*y2 - b1*z2]
                return D[0], D[1], D[2], D[3], D[4], D[5]
                    
            
            if InitialConditions == None:
                InitialConditions = [0.1, 0.1, 0.1, 0, 0, 0] #inital conditions
            
            
            states = odeint(coupled_lorenz_rossler, InitialConditions, t)
            ts = [(states[:,0])[::10][-SampleSize:], 
                  (states[:,1])[::10][-SampleSize:], 
                  (states[:,2])[::10][-SampleSize:],
                  (states[:,3])[::10][-SampleSize:], 
                  (states[:,4])[::10][-SampleSize:], 
                  (states[:,5])[::10][-SampleSize:]]
            t = t[::10][-SampleSize:]
            
            
    
    # In[ ]: Complete
            
        if system == 'coupled_rossler_rossler': 
            #setting simulation time series parameters
            if fs == None: fs = 10
            if SampleSize == None: SampleSize = 1500
            if L == None: L = 1000.0
            t = np.linspace(0, L, int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    k, w1, w2 = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': k = 0.25
                if dynamic_state == 'chaotic': k = 0.30
                w1 =0.99 
                w2 = 0.95 
                
                    
            #defining simulation functions
            def coupled_rossler_rossler(state, t):
                x1, y1, z1, x2, y2, z2 = state  # unpack the state vector
                D = [-w1*y1 - z1 + k*(x2-x1), 
                     w1*x1 + 0.165*y1, 
                     0.2 + z1*(x1-10), 
                     -w2*y2 - z2 + k*(x1-x2), 
                     w2*x2 + 0.165*y2, 
                     0.2 + z2*(x2-10)]
                return D[0], D[1], D[2], D[3], D[4], D[5]
                    
            
            if InitialConditions == None:
                InitialConditions = [-0.4, 0.6, 5.8, 0.8, -2, -4] #inital conditions
            
            
            states = odeint(coupled_rossler_rossler, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], 
                  (states[:,1])[-SampleSize:], 
                  (states[:,2])[-SampleSize:],
                  (states[:,3])[-SampleSize:], 
                  (states[:,4])[-SampleSize:], 
                  (states[:,5])[-SampleSize:]]
            t = t[-SampleSize:]
                        
    
    # In[ ]: Complete
    
        if system == 'diffusionless_lorenz_attractor':
            #setting simulation time series parameters
            if fs == None: fs = 40
            if SampleSize == None: SampleSize = 10000
            if L == None: L = 1000.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    R = parameters[0]
            if parameters == None:
                if dynamic_state == 'periodic': R = 0.25
                if dynamic_state == 'chaotic': R = 0.40
                
                    
            #defining simulation functions
            def diffusionless_lorenz_attractor(state, t):
                x, y, z = state  # unpack the state vector
                return -y - x, -x*z, x*y + R
                    
            
            if InitialConditions == None:
                InitialConditions = [1, -1, 0.01] 
            
            
            states = odeint(diffusionless_lorenz_attractor, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
                
    # In[ ]: Complete
    
        if system == 'complex_butterfly':
            #system from https://pdfs.semanticscholar.org/3794/50ca6b8799d0b3c2f35bbe6df47676c69642.pdf?_ga=2.68291732.442509117.1595011450-840911007.1542643809
            #setting simulation time series parameters
            if fs == None: fs = 10
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 1000.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a = parameters[0]
            if parameters == None:
                if dynamic_state == 'periodic': a = 0.15
                if dynamic_state == 'chaotic': a = 0.55
                
                    
            #defining simulation functions
            def complex_butterfly(state, t):
                x, y, z = state  # unpack the state vector
                return a*(y-x), -z*np.sign(x), np.abs(x) - 1
                    
            
            if InitialConditions == None:
                InitialConditions = [0.2, 0.0, 0.0] 
            
            
            states = odeint(complex_butterfly, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
                            
    # In[ ]: Complete
    
        if system == 'chens_system':
            #setting simulation time series parameters
            if fs == None: fs = 200
            if SampleSize == None: SampleSize = 3000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b, c = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': a = 30
                if dynamic_state == 'chaotic': a = 35
                b, c = 3, 28
                    
            #defining simulation functions
            def chens_system(state, t):
                x, y, z = state  # unpack the state vector
                return a*(y-x), (c-a)*x - x*z + c*y, x*y - b*z
                    
            
            if InitialConditions == None:
                InitialConditions = [-10, 0, 37] 
            
            
            states = odeint(chens_system, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
                                        
    # In[ ]: Complete
    
        if system == 'hadley_circulation':
            #setting simulation time series parameters
            if fs == None: fs = 50
            if SampleSize == None: SampleSize = 4000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b, F, G = parameters[0], parameters[1], parameters[2], parameters[3]
            if parameters == None:
                if dynamic_state == 'periodic': a = 0.30
                if dynamic_state == 'chaotic': a = 0.25
                b, F, G = 4, 8, 1
                    
            #defining simulation functions
            def hadley_circulation(state, t):
                x, y, z = state  # unpack the state vector
                return -y**2 - z**2 - a*x + a*F, x*y - b*x*z - y + G, b*x*y + x*z - z
                    
            
            if InitialConditions == None:
                InitialConditions = [-10, 0, 37] 
            
            
            states = odeint(hadley_circulation, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
            
                        
                                                    
    # In[ ]: Complete
    
        if system == 'ACT_attractor':
            #setting simulation time series parameters
            if fs == None: fs = 50
            if SampleSize == None: SampleSize = 4000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 4:
                    print('Warning: needed 4 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    alpha, mu, delta, beta = parameters[0], parameters[1], parameters[2], parameters[3]
            if parameters == None:
                if dynamic_state == 'periodic': alpha = 2.5
                if dynamic_state == 'chaotic': alpha = 2.0
                mu, delta, beta = 0.02, 1.5, -0.07
                    
            #defining simulation functions
            def ACT_attractor(state, t):
                x, y, z = state  # unpack the state vector
                return alpha*(x-y), -4*alpha*y + x*z + mu*x**3, -delta*alpha*z + x*y + beta*z**2
                    
            
            if InitialConditions == None:
                InitialConditions = [0.5, 0, 0] 
            
            
            states = odeint(ACT_attractor, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
            
                                                                
    # In[ ]: Complete
    
        if system == 'rabinovich_frabrikant_attractor':
            #setting simulation time series parameters
            if fs == None: fs = 30
            if SampleSize == None: SampleSize = 3000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    alpha, gamma = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': alpha = 1.16
                if dynamic_state == 'chaotic': alpha = 1.13
                gamma = 0.87
                    
            #defining simulation functions
            def rabinovich_frabrikant_attractor(state, t):
                x, y, z = state  # unpack the state vector
                return y*(z-1+x**2)+gamma*x, x*(3*z + 1 -x**2) + gamma*y, -2*z*(alpha + x*y)
                    
            
            if InitialConditions == None:
                InitialConditions = [-1, 0, 0.5] 
            
            
            states = odeint(rabinovich_frabrikant_attractor, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
                        
                                                                
    # In[ ]: Complete
    
        if system == 'linear_feedback_rigid_body_motion_system':
            #system from https://ir.nctu.edu.tw/bitstream/11536/26522/1/000220413000019.pdf
            #setting simulation time series parameters
            if fs == None: fs = 100
            if SampleSize == None: SampleSize = 3000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b, c = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': a = 5.3
                if dynamic_state == 'chaotic': a = 5.0
                b, c = -10, -3.8
            #defining simulation functions
            def linear_feedback_rigid_body_motion_system(state, t):
                x, y, z = state  # unpack the state vector
                return -y*z + a*x, x*z + b*y, (1/3)*x*y + c*z 
            
                    
            
            if InitialConditions == None:
                InitialConditions = [0.2, 0.2, 0.2] 
            
            
            states = odeint(linear_feedback_rigid_body_motion_system, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
                                                                     
    # In[ ]: Complete
    
        if system == 'moore_spiegel_oscillator':
            #setting simulation time series parameters
            if fs == None: fs = 100
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    T, R = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': T = 7.8
                if dynamic_state == 'chaotic': T = 7.0
                R = 20
            #defining simulation functions
            def moore_spiegel_oscillator(state, t):
                x, y, z = state  # unpack the state vector
                return y, z, -z - (T-R + R*x**2)*y - T*x
            
                    
            
            if InitialConditions == None:
                InitialConditions = [0.2, 0.2, 0.2] 
            
            
            states = odeint(moore_spiegel_oscillator, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
                   
            
                                                                                 
    # In[ ]: Complete
    
        if system == 'thomas_cyclically_symmetric_attractor':
            #setting simulation time series parameters
            if fs == None: fs = 10
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 1000.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 1:
                    print('Warning: needed 1 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    b = parameters[0]
            if parameters == None:
                if dynamic_state == 'periodic': b = 0.17
                if dynamic_state == 'chaotic': b = 0.18
                
            #defining simulation functions
            def thomas_cyclically_symmetric_attractor(state, t):
                x, y, z = state  # unpack the state vector
                return -b*x + np.sin(y), -b*y + np.sin(z), -b*z + np.sin(x)
            
            if InitialConditions == None:
                InitialConditions = [0.1, 0, 0] 
            
            states = odeint(thomas_cyclically_symmetric_attractor, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
            
                        
                                                                                 
    # In[ ]: Complete
    
        if system == 'halvorsens_cyclically_symmetric_attractor':
            #setting simulation time series parameters
            if fs == None: fs = 200
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 200.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 3:
                    print('Warning: needed 3 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b, c = parameters[0], parameters[1], parameters[2]
            if parameters == None:
                if dynamic_state == 'periodic': a = 1.85
                if dynamic_state == 'chaotic': a = 1.45
                b, c = 4, 4
                
            #defining simulation functions
            def halvorsens_cyclically_symmetric_attractor(state, t):
                x, y, z = state  # unpack the state vector
                return -a*x - b*y - c*z - y**2, -a*y - b*z - c*x - z**2, -a*z - b*x - c*y - x**2
            
            if InitialConditions == None:
                InitialConditions = [-5, 0, 0] 
            
            states = odeint(halvorsens_cyclically_symmetric_attractor, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
            
                                                                                             
    # In[ ]: Complete
    
        if system == 'burke_shaw_attractor':
            #system from http://www.atomosyd.net/spip.php?article33
            #setting simulation time series parameters
            if fs == None: fs = 200
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 500.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    s, V = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': s = 12
                if dynamic_state == 'chaotic': s = 10
                V = 4
                
            #defining simulation functions
            def burke_shaw_attractor(state, t):
                x, y, z = state  # unpack the state vector
                return -s*(x+y), -y - s*x*z, s*x*y + V
            
            if InitialConditions == None:
                InitialConditions = [0.6, 0, 0] 
            
            states = odeint(burke_shaw_attractor, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
                                                                                             
    # In[ ]: Complete
    
        if system == 'rucklidge_attractor':
            #setting simulation time series parameters
            if fs == None: fs = 50
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 1000.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    k, lamb = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': k = 1.1
                if dynamic_state == 'chaotic': k = 1.6
                lamb = 6.7
                
            #defining simulation functions
            def rucklidge_attractor(state, t):
                x, y, z = state  # unpack the state vector
                return -k*x + lamb*y - y*z, x, -z + y**2
            
            if InitialConditions == None:
                InitialConditions = [1, 0, 4.5] 
            
            states = odeint(rucklidge_attractor, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
                                                                                             
    # In[ ]: Complete
    
        if system == 'WINDMI':
            #setting simulation time series parameters
            if fs == None: fs = 20
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 1000.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': a = 0.9
                if dynamic_state == 'chaotic': a = 0.8
                b = 2.5
                
            #defining simulation functions
            def WINDMI(state, t):
                x, y, z = state  # unpack the state vector
                return y, z, -a*z - y + b - np.exp(x)
            
            if InitialConditions == None:
                InitialConditions = [1, 0, 4.5] 
            
            states = odeint(WINDMI, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
                        
                                                                                             
    # In[ ]: Complete
    
        if system == 'simplest_quadratic_chaotic_flow':
            #setting simulation time series parameters
            if fs == None: fs = 20
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 1000.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': 
                    print('We could not find a periodic response near $a = 2.017$.')
                    print('Any contributions would be appreciated!')
                    print('Defaulting to chaotic state.')
                    a = 2.017
                if dynamic_state == 'chaotic': a = 2.017
                b = 1
            #defining simulation functions
            def simplest_quadratic_chaotic_flow(state, t):
                x, y, z = state  # unpack the state vector
                return y, z, -a*z + b*y**2 -x
            
            if InitialConditions == None:
                InitialConditions = [-0.9, 0, 0.5] 
            
            states = odeint(simplest_quadratic_chaotic_flow, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
                        
                                                                                             
    # In[ ]: Complete
    
        if system == 'simplest_cubic_chaotic_flow':
            #setting simulation time series parameters
            if fs == None: fs = 20
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 1000.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': a = 2.11
                if dynamic_state == 'chaotic': a = 2.05
                b = 2.5
                
            #defining simulation functions
            def simplest_cubic_chaotic_flow(state, t):
                x, y, z = state  # unpack the state vector
                return y, z, -a*z + x*y**2 - x
            
            if InitialConditions == None:
                InitialConditions = [0, 0.96, 0] 
            
            states = odeint(simplest_cubic_chaotic_flow, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
                                  
                                                                                             
    # In[ ]: Complete
    
        if system == 'simplest_piecewise_linear_chaotic_flow':
            #setting simulation time series parameters
            if fs == None: fs = 40
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 1000.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': a = 0.7
                if dynamic_state == 'chaotic': a = 0.6
                b = 2.5
                
            #defining simulation functions
            def simplest_piecewise_linear_chaotic_flow(state, t):
                x, y, z = state  # unpack the state vector
                return y, z, -a*z -y  + np.abs(x) - 1
            
            if InitialConditions == None:
                InitialConditions = [0, -0.7, 0] 
            
            states = odeint(simplest_piecewise_linear_chaotic_flow, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
                                           
                                                                                             
    # In[ ]: Complete
    
        if system == 'double_scroll':
            #setting simulation time series parameters
            if fs == None: fs = 20
            if SampleSize == None: SampleSize = 5000
            if L == None: L = 1000.0
            t = np.linspace(0, L,int(L*fs))
            
            
            #setting system parameters
            if parameters != None:
                if len(parameters) != 2:
                    print('Warning: needed 2 parameters. Defaulting to periodic solution parameters.')
                    parameters = None
                else:
                    a, b = parameters[0], parameters[1]
            if parameters == None:
                if dynamic_state == 'periodic': a = 1.0
                if dynamic_state == 'chaotic': a = 0.8
                b = 2.5
                
            #defining simulation functions
            def double_scroll(state, t):
                x, y, z = state  # unpack the state vector
                return y, z, -a*(z + y + x - np.sign(x))
            
            if InitialConditions == None:
                InitialConditions = [0.01, 0.01, 0] 
            
            states = odeint(double_scroll, InitialConditions, t)
            ts = [(states[:,0])[-SampleSize:], (states[:,1])[-SampleSize:], (states[:,2])[-SampleSize:]]
            t = t[-SampleSize:]
            
    return t, ts
