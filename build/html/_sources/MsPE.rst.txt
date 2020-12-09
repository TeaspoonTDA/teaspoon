
.. automodule:: teaspoon.parameter_selection.MsPE
    :members:


The following is an example implementing the MsPE method for selecting both n and tau::

    import numpy as np
    from teaspoon.parameter_selection.MsPE import MsPE_n,  MsPE_tau
    
    t = np.linspace(0, 100, 1000)
    ts = np.sin(t)
    
    m_s, m_e, d_s, d_e = 3, 7, 1, 200
    #m_s and m_e are the starting and ending dimensions n to search through
    #d_e = max delay tau to search through
    
    #plotting option will show you how delay tau or dimension n were selected
    tau = int(MsPE_tau(ts, d_e, plotting = True))
    n = MsPE_n(ts, tau, m_s, m_e, plotting = True)
    
    print('Embedding Delay:       '+str(tau))
    print('Embedding Dimension:   '+str(n))
    
    print('Embedding Delay:       '+str(tau))
    print('Embedding Dimension:   '+str(n))

Where the output for this example is::

    Embedding Delay:       21
    Embedding Dimension:   3

.. figure:: figures/MsPE_tau_example.png
   :scale: 20 %

.. figure:: figures/MsPE_n_example.png
   :scale: 20 %


