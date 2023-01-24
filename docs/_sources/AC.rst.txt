
.. automodule:: teaspoon.parameter_selection.autocorrelation
    :members:

The following is an example implementing autocorrelation for selecting tau::

    from teaspoon.parameter_selection.autocorrelation import autoCorrelation_tau
    import numpy as np
    
    fs = 10
    t = np.linspace(0, 100, fs*100)
    ts = np.sin(t) + np.sin((1/np.pi)*t)
    
    tau = autoCorrelation_tau(ts, cutoff = 1/np.exp(1), AC_method = 'pearson', plotting = False)
    print('Delay from AC: ', tau)

Where the output for this example is::

    Delay from AC:   13

.. figure:: figures/AC_example.png
   :scale: 20 %


