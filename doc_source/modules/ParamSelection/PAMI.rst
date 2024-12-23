====================================================================
Permutation Auto Mutual Information (PAMI) for time delay (tau)
====================================================================

.. automodule:: teaspoon.parameter_selection.PAMI_delay
    :members:


The following is an example implementing Permutation Auto-Mutual Information for the purpose of selecting tau::

    from teaspoon.parameter_selection.PAMI_delay import PAMI_for_delay 
    import numpy as np
    
    fs = 10
    t = np.linspace(0, 100, fs*100)
    ts = np.sin(t)
    tau = PAMI_for_delay(ts, n = 3, plotting = True)
    print('Delay from PAMI: ',tau)

Where the output for this example is::

    Permutation Embedding Delay: 8

.. figure:: ../../figures/PAMI.png
   :scale: 13 %


