
.. automodule:: teaspoon.parameter_selection.autocorrelation
    :members:

The following is an example implementing autocorrelation for selecting tau::

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    from PE_parameter_functions import autocorrelation
    
    t = np.linspace(0, 100, 1000)
    ts = np.sin(t)

    Delay = autocorrelation.autoCorrelation_tau(ts, cutoff = 1/np.exp(1), AC_method = 'pearson', plotting = True) 
    #calculates delay from autocorrelation (Pearson's)
    
    print('Delay from AC: ', Delay)

Where the output for this example is::

    Delay from AC:   13

.. figure:: figures/AC_example.png
   :scale: 20 %


