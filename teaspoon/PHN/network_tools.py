def MakeNetwork(A):
    import numpy as np
    #ordinal network stuff
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
    if 0 in G.nodes():
        G.remove_node(0)
        
    pos = nx.spring_layout(G, iterations = 10000)
    
    return G, pos

def PHN_statistics(diagram, A):
    import numpy as np
    def persistentEntropy(lt):
        if len(lt) > 1:
            L = sum(lt)
            p = lt/L
            E = sum(-p*np.log2(p))
            Emax = np.log2(sum(lt))
            E = E/Emax
        if len(lt) == 1:
            E = 0
        if len(lt) == 0:
            E = 1
        return E
        
    lt = np.array(diagram[1].T[1]-diagram[1].T[0])
    D1 = np.array([diagram[1].T[0], diagram[1].T[1]])
    D1 = D1
    lt = lt[lt<10**10]
    num_lifetimes1 = len(lt)
    num_unique = len(A[0])
    En = persistentEntropy(lt)
    
    H1 = diagram[1].T
    delta = 0.1
    if len(H1[H1<10**10])>0:
        R = (1-np.nanmax(H1)/np.floor((num_unique/3)-delta))
    else:
        R = np.nan
    M = num_lifetimes1/num_unique
    statistics = [R, En, M]
    return statistics










