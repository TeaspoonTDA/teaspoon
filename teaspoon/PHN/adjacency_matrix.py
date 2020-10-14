def Adjaceny_OP(perm_seq, n): #Gets Adjacency Matrix (weighted and direction) using permutation sequence
    import numpy as np
    N = np.math.factorial(n) #number of possible nodes
    A = np.zeros((N,N)) #prepares A
    for i in range(len(perm_seq)-1): #go through all permutation transitions (This could be faster wiuthout for loop)
        A[perm_seq[i]-1][perm_seq[i+1]-1] += 1 #for each transition between permutations increment A_ij
    return A #this A is directional and weighted

def Adjacency_KNN(indices):
    import numpy as np
    A = np.zeros((len(indices.T[0]),len(indices.T[0])))
    for h in range(len(indices.T[0])):
        KNN_i = indices[h][indices[h]!=h] #indices of k nearest neighbors
        A[h][KNN_i]+=1 #increment A_ij for kNN indices
        A.T[h][KNN_i]+=1
    A[A>0] = 1
    return A