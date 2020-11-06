Network Representation of Time Series
=======================================

This module provides algorithms for forming both ordinal partition networks (see example GIF below) and k-NN networks. The formation of the networks is described in detail in "`Persistent Homology of Complex Networks for Dynamic State Detection <https://arxiv.org/abs/1904.07403>`_."


.. image:: figures/ordinal_partition_network_video_complex.gif
   :alt: Left floating image
   :class: with-shadow float-left
   :scale: 27

Figure: Example formation of an ordinal partition network for 6 dimensional permutations. As the ordinal ranking of the 6 spaced points changes, so does the node in the network. This shows how the network captures the periodic structure of the time series through a resulting double loop topology.

.. rst-class::  clear-both


.. automodule:: teaspoon.SP.network
    :members: 

.. automodule:: teaspoon.SP.network_tools
    :members: 

The network output has the data structure of an adjacency matrix. An example is shown below::

    #import needed packages
    import numpy as np
    from teaspoon.SP.network import knn_graph
    from teaspoon.SP.network import ordinal_partition_graph
    
    t = np.linspace(0,30,200)
    ts = np.sin(t) + np.sin(2*t) #generate a simple time series
    
    A_knn = knn_graph(ts) #ordinal partition network from time series
    
    A_op = ordinal_partition_graph(ts) #knn network from time series
    


Additionally, there is methods to change the data structure to anything available in networkx through changing the adjacency matrix to the networkx data structure (dictionary of dictionaries) with an example as follows::

    t = np.linspace(0,30,200)
    ts = np.sin(t) + np.sin(2*t) #generate a simple time series
    
    from teaspoon.SP.network import knn_graph
    A = knn_graph(ts)
    
    from teaspoon.SP.network_tools import make_network
    G, pos = make_network(A)
    
    
    import matplotlib.pyplot as plt
    import networkx as nx
    plt.figure(figsize = (8,8))
    plt.title('Network', size = 16)
    nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='blue',
            width=1, font_size = 10, node_size = 30)
    plt.show()

This coe has the output as follows:

.. figure:: figures/example_knn_graph.png
   :scale: 100 %










