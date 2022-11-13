Network Representation of Time Series
=======================================

This module provides algorithms for forming both ordinal partition networks (see example GIF below) and k-NN networks. The formation of the networks is described in detail in "`Persistent Homology of Complex Networks for Dynamic State Detection <https://arxiv.org/abs/1904.07403>`_."


.. image:: figures/ordinal_partition_network_video_complex.gif
   :alt: Left floating image
   :class: with-shadow float-left
   :scale: 27

Figure: Example formation of an ordinal partition network for 6 dimensional permutations. As the ordinal ranking of the 6 spaced points changes, so does the node in the network. This shows how the network captures the periodic structure of the time series through a resulting double loop topology.

.. rst-class::  clear-both




k Nearest Neighbors Graph
*****************************************************************************************************************************************************************


.. automodule:: teaspoon.SP.network
    :members: Adjacency_KNN
    :noindex:

.. automodule:: teaspoon.SP.network
    :members: knn_graph
    :noindex:

The network output has the data structure of an adjacency matrix (or networkx graph for displaying). An example is shown below::

    #import needed packages
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    from teaspoon.SP.network import knn_graph
    from teaspoon.SP.network_tools import make_network

    # Time series data
    t = np.linspace(0,30,200)
    ts = np.sin(t) + np.sin(2*t) #generate a simple time series
    
    A = knn_graph(ts) #adjacency matrix
    
    G, pos = make_network(A) #get networkx representation
    
    plt.figure(figsize = (8,8))
    plt.title('Network', size = 16)
    nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='blue',
            width=1, font_size = 10, node_size = 30)
    plt.show()


This code has the output as follows:

.. figure:: figures/example_knn_graph.png
   :scale: 60 %



Ordinal Partition Graph
*****************************************************************************************************************************************************************


.. automodule:: teaspoon.SP.network
    :members: Adjaceny_OP
    :noindex:

.. automodule:: teaspoon.SP.network
    :members: ordinal_partition_graph
    :noindex:

The network output has the data structure of an adjacency matrix (or networkx graph for displaying). An example is shown below::

    #import needed packages
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    #teaspoon functions
    from teaspoon.SP.network import ordinal_partition_graph
    from teaspoon.SP.network_tools import remove_zeros
    from teaspoon.SP.network_tools import make_network

    # Time series data
    t = np.linspace(0,30,200)
    ts = np.sin(t) + np.sin(2*t) #generate a simple time series

    A = ordinal_partition_graph(ts, n = 6) #adjacency matrix
    A = remove_zeros(A) #remove nodes of unused permutation

    G, pos = make_network(A) #get networkx representation

    plt.figure(figsize = (8,8))
    plt.title('Network', size = 20)
    nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='red',
            width=1, font_size = 10, node_size = 30)
    plt.show()


This code has the output as follows:

.. figure:: figures/example_opn_graph.png
   :scale: 18 %




Coarse Grained State Space (CGSS) Graph
*****************************************************************************************************************************************************************


.. automodule:: teaspoon.SP.network
    :members: Adjacency_CGSS
    :noindex:

.. automodule:: teaspoon.SP.network
    :members: cgss_graph
    :noindex:

The network output has the data structure of an adjacency matrix (or networkx graph for displaying). An example is shown below::

    #import needed packages
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    #teaspoon functions
    from teaspoon.SP.network import cgss_graph
    from teaspoon.SP.network_tools import remove_zeros
    from teaspoon.SP.network_tools import make_network
    from teaspoon.parameter_selection import MsPE

    # Time series data
    t = np.linspace(0,30,600)
    ts = np.sin(t) + np.sin(2*t) #generate a simple time series
    
    n = 3 #dimension of SSR
    b = 8 #number of states per dimension
    tau = MsPE.MsPE_tau(ts)
    
    B_array = tsa_tools.cgss_binning(ts, n, tau, b) #binning array
    SSR = tsa_tools.takens(ts, n, tau)  #state space reconstruction
    
    A = cgss_graph(ts, B_array, n, tau) #adjacency matrix
    A = remove_zeros(A) #remove nodes of unused permutation

    G, pos = make_network(A) #get networkx representation

    plt.figure(figsize = (8,8))
    plt.title('Network', size = 20)
    nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='green',
            width=1, font_size = 10, node_size = 30)
    plt.show()

This code has the output as follows:

.. figure:: figures/example_cgssn_graph.png
   :scale: 18 %





Network Tools
*****************************************************************************************************************************************************************

.. automodule:: teaspoon.SP.network_tools
    :members: 














