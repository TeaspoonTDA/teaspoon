
def make_network(A, position_iterations = 1000, remove_deg_zero_nodes = False):
    
    """This function creates a networkx graph and position of the nodes using the adjacency matrix.
    
    Args:
        A (2-D array): 2-D square adjacency matrix
        
    Other Parameters:
        position_iterations (Optional[int]): Number of spring layout position iterations. Default is 1000.
    
    Returns:
        [dictionaries, list]: G (networkx graph representation), pos (position of nodes for networkx drawing)
    """
    
    import numpy as np
    A = A + A.T #make undirected adjacency matrix
    np.fill_diagonal(A, 0) #get rid of diagonal 
    A[A>0] = 1 #make unweighted
    
    
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(A[0])))
        
    edges1 = []
    edges2 = []
    for h in range(0,len(A[0])):
        edges1 = np.append(edges1, np.nonzero(A[h]))
        L = len(np.nonzero(A[h])[0])
        edges2 = np.append(edges2, np.zeros(L) + h)
    edges1 = edges1.astype(int)+1
    edges2 = edges2.astype(int)+1
    edges = zip(edges1,edges2)
    G.add_edges_from(edges)
        
        
    fixed_nodes = []
    for i in range(0,len(A[0])):
        if G.degree[i] == 0:
                fixed_nodes.append(i)
            
            
    pos = nx.spring_layout(G, iterations = 0)
    pos = nx.spring_layout(G, pos = pos, fixed = fixed_nodes, iterations = position_iterations)
    
    if remove_deg_zero_nodes == False:
        if 0 in G.nodes(): #removes this node because it shouldn't be in graph
            G.remove_node(0)
    else:
        for i in range(0,len(A[0])):
            if G.degree[i] == 0:
                G.remove_node(i)
        
    
        
    return G, pos



    
# In[ ]:
    
if __name__ == "__main__": #Only runs if running from this file (This will show basic example)
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..', '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))


    import numpy as np
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












