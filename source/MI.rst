

.. automodule:: teaspoon.parameter_selection.MI_delay
    :members:

The following is an example implementing the MI method for selecting tau::

    import numpy as np
    
    fs = 10
    t = np.linspace(0, 100, fs*100) 
    ts = np.sin(t) + np.sin((1/np.pi)*t)
    
    tau = MI_for_delay(ts, plotting = True, method = 'basic', h_method = 'sturge', k = 2, ranking = True)
    print('Delay from MI: ',tau)

Where the output for this example is::

    Delay from MI:  23

.. figure:: figures/MI_fig.png
   :scale: 15 %