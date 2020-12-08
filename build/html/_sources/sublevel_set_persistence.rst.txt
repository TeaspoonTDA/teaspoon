Zero-Dimensional Sublevel Set Persistence
==========================================

.. automodule:: teaspoon.TDA.SLSP
    :members: Persistence0D

The following is an example implementing the zero-dimensional sublevel set persistence algorithm::

    from teaspoon.TDA.SLSP import Persistence0D
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    
    
    fs, T = 100, 10
    t = np.linspace(-0.2,T,fs*T+1)
    A = 20
    ts = A*np.sin(np.pi*t) + A*np.sin(1*t) 
    
    feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(ts)
    D = persistenceDgm
    print(' Persistence Diagram Pairs: ', D)
    
    
    gs = gridspec.GridSpec(1,2)
    plt.figure(figsize=(11,5))
    
    ax = plt.subplot(gs[0, 0])
    plt.title('Time Series')
    plt.xlabel('$t$')
    plt.ylabel('$x(t)$')
    plt.plot(t, ts)
    
    ax = plt.subplot(gs[0, 1])
    plt.title('Persistence Diagram')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.plot(D.T[0], D.T[1], 'ro')
    plt.plot([min(ts), max(ts)], [min(ts), max(ts)], 'k--')
    
    plt.show()
    

Where the output for this example is::

    Persistence Diagram Pairs:  
    [[-27.87625241   0.4981549 ]
     [ -0.05341814  30.33373693]
     [-34.58506622  25.25230112]
     [        -inf  32.58419686]]

.. figure:: figures/SLSP_example.png
   :scale: 75 %










.. automodule:: teaspoon.TDA.SLSP_tools
    :members: cutoff

The following is an example implementing the zero-dimensional sublevel set persistence algorithm::

        import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec
    from teaspoon.TDA.SLSP import Persistence0D
    from teaspoon.TDA.SLSP_tools import cutoff
    
    #----------------Assume the additive noise distribution---------------
    dist = 'Gaussian'

    #---------------Generate a time series with additive noise------------
    fs, T = 40, 10
    t = np.linspace(0,T,fs*T)
    A, sd = 20, 1 #signal amplitude and standard deviation
    ts_0 = A*np.sin(np.pi*t) + A*np.sin(t) 
    noise = np.random.normal(0,sd,len(ts_0)) #gaussian distribution additive noise
    ts = ts_0 + noise #add noise to time series
    
    #--------------Run sublevel set persistence--------------------------
    feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(ts)
    B = np.flip(persistenceDgm.T[0], axis = 0) #get birth times
    D = np.flip(persistenceDgm.T[1], axis = 0) #get death times
    L = D-B #get lifetimes as difference between birth and death
    
    I_B = np.array(feature_ind_1.astype(int)).T #indices of birth times
    T_B = np.flip(t[I_B], axis = 0) #time values at birth times
    
    #-------------get cutoff for persistence diagram---------------------
    C, param = cutoff(L, alpha = 0.01, n = len(ts), distribution = dist) #get cutoff
    print('Distribution parameter estimate: ', param)
    print('C: ', C)
    
    
    #-------------------------PLOTTING THE RESULTS-----------------------
    gs = gridspec.GridSpec(2,3) 
    plt.figure(figsize=(17,5))
    TextSize = 15
            
    ax = plt.subplot(gs[0, 0:2])
    plt.plot(t,ts, 'k')
    plt.ylabel('$x(t)$', size = TextSize)
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.xlim(min(t),max(t))
    
    ax = plt.subplot(gs[1, 0:2])
    plt.ylabel('$L$', size = TextSize)
    plt.xlabel('$t$', size = TextSize)
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.plot(T_B[L>C], L[L>C], 'bd', label = r'$L$ (signal)')
    plt.plot(T_B[L<C], L[L<C], 'ko', alpha = 0.7, label = r'$L$ (noise)')
    plt.plot([np.min(t),np.max(t)],[C, C],'k--', label = r'$C^*_\alpha$')
    ax.fill_between([min(t), max(t)], [C, C], color = 'red', alpha = 0.15)
    plt.ylim(0,)
    plt.xlim(min(t),max(t))
    plt.legend(loc = 'right', fontsize = TextSize-3, ncol = 2)
                
    ax = plt.subplot(gs[0:2, 2])
    plt.ylabel('$D$', size = TextSize)
    plt.xlabel('$B$', size = TextSize)
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.plot(B[L>C], D[L>C], 'bd', label = r'signal')
    plt.plot(B[L<C], D[L<C], 'ro', alpha = 0.7, label = r'noise')
    plt.plot([min(B), max(D)], [min(B), max(D)],'k')
    plt.plot([min(B), max(D)], [min(B)+C, max(D)+C],'k--', label = r'$C_\alpha$')
    ax.fill_between(x = [min(B), max(D)], y1 = [min(B)+C, max(D)+C], y2 = [min(B), max(D)], 
                                     color = 'red', alpha = 0.15)
    plt.legend(loc = 'lower right', fontsize = TextSize-3, bbox_to_anchor = (1.02, -0.02))
                
    plt.subplots_adjust(wspace=0.3)
    plt.show()


Where the output for this example is::

    Distribution parameter estimate:  1.0665900603015368
    C:  7.408859563871353

.. figure:: figures/SLSP_tools_cutoff.png
   :scale: 70 %


    
