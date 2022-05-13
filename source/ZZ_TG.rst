Bifurcations using ZigZag (BuZZ)
====================================

This module contains code to run zigzag persistence on temporal hypergraph data. This code is based on the work performed in "Temporal Network Analysis Using Zigzag Persistence." The general pipeline for this work is shown in the following figure.

  .. image:: figures/TG_ZZ_pipeline.png
    :width: 700 px

Specifically, a temporal graph is a graph whose edges are active during specified intervals. For example, conisder the edge between nodes a and b with a set of intervals associated to it. This set of intervals describes when this edge is active. An example of this is shown on the left side of the pipeline figure. Next, we generate graph snapshots from the temporal graph by using a sliding window technique. These graph snapshots are generated based on adding edges to the graph snapshot if there is an overlap between the edge's interval and the sliding window interval. Additionally. we include union windows between sliding windows for the zigzag persistence pipeline. These union windows are formed the same except with the sliding window of the union of two temporally adjacent windows. Next, we construct simplicial complex representations for each graph and lastly apply zigzag persistence to the simplicial complex sequence.

Functions
################

.. automodule:: teaspoon.TDA.TGZZ
   :members: 

Note
################
To run dionysus2 follow the instructions available on the dionysus2 website. If running with windows OS we advise using windows subsystem for linux (WSL) and following dionysus2 installation directions.

Example
################

To demonstrate the function of zigzag persistence on graphs we will leverage a simple example. Namely, we have a temporal graph constructed that has a loop and several components that break and then recombine. The goal of zigzag persistence in dimension 0 and 1 is to track the components and holes in the graph, respectively.::

    import numpy as np
    import networkx as nx
    import dionysus as dio
    import gudhi
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
        
    # list of edges and nodes
    nodes = [0, 1, 2, 3, 4]
    edges = [(0,1),                #edge 0
             (1,2),                #edge 1
             (2,3),                #edge 2
             (3,4),                #edge 3
             (4,0)]                #edge 4
    
    # I is a list of intervals for each edge.
    I = [[[0.1, 2.3], [3.6, 8.7]], # edge 0
         [[4.4, 9.9]],             # edge 1
         [[2.1, 3.9], [6.1, 9.1]], # edge 2
         [[1.3, 9.3]],             # edge 3
         [[3.2, 7.6]]]             # edge 4
    
    TG =nx.Graph() #temporal graph as attributed networkx representation
    #add attribute information to each node and edge and construct as networkx graph
    for i, node in enumerate(nodes): #add position information to nodes
        TG.add_node(node)
    for i, edge in enumerate(edges):
        TG.add_edge(edge[0],edge[1])
    
    #define sliding window intervals
    window_width, overlap_ratio = 1.0, 0.0
    time_range = [0,10]
    windows = sliding_windows(window_width, overlap_ratio, time_range)
    
    #get graph snapshots
    G_snapshots = graph_snapshots(edges, I, windows)
    
    #get simplices and associated times based on dionysus format needed.
    S, T = simplicial_complex_representation(TG, G_snapshots, windows, K = 1)
    
    #run dionysus2 zigzag persistence
    f = dio.Filtration(S)
    zz, dgms, cells = dio.zigzag_homology_persistence(f, T)
    
    #print and plot persistence
    for i,dgm in enumerate(dgms):
        print("Dimension:", i)
        for j, p in enumerate(dgm):
            print(p)
    diag = plot_persistence_diagram(dgms, windows, dimensions = [0, 1], FS = 22)



Where the output for this example is::
	
    Dimension: 0
    (1,3)
    (0.5,10)
    Dimension: 1
    (4,4.5)
    (6,8.5)

.. figure:: figures/example_PD_dimension.png
   :scale: 12 %

To better understand the resulting persistence diagram, we can visualize the changing structure of the network by looking at the intervals and corresponding graph snapshots. The following figure shows the graph snapshots and the corresponding unions. It is clear that there is a cycle born at graph 3,4 and dissapears at graph 4 correspodning to the persistence pair (4,4.5) and the cycle is born again at graph 5,6 and breaks at graph 8 corresponding to the persistence pair (6,8.5). For more details on this example we direct the user to our publication titled "Temporal Network Analysis Using Zigzag Persistence."

.. figure:: figures/example_graph_sequence_and_intervals.png
   :scale: 12 %
