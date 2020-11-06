
.. automodule:: teaspoon.parameter_selection.FNN_n
    :members:


The following is an example implementing the False Nearest Neighbors (FNN) algorithm for the dimension::

    from teaspoon.parameter_selection.FNN_n import FNN_n
    import numpy as np
    
    fs = 10
    t = np.linspace(0, 100, fs*100)
    ts = np.sin(t)
    
    tau=15 #embedding delay
    
    perc_FNN, n = FNN_n(ts, tau, plotting = True)
    print('FNN embedding Dimension: ',n)

Where the output for this example is::

    FNN embedding Dimension:  2

.. figure:: figures/FNN_fig.png
   :scale: 16 %


