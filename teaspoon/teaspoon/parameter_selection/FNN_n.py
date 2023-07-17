"""
False Nearest Neighbors (FNN) for dimension (n).
==================================================


"""

def ts_recon(ts, dim, tau):
    import numpy as np
    xlen = len(ts)-(dim-1)*tau
    a = np.linspace(0, xlen-1, xlen)
    a = np.reshape(a, (xlen, 1))
    delayVec = np.linspace(0, (dim-1), dim)*tau
    delayVec = np.reshape(delayVec, (1, dim))
    delayMat = np.tile(delayVec, (xlen, 1))
    vec = np.tile(a, (1, dim))
    indRecon = np.reshape(vec, (xlen, dim)) + delayMat
    indRecon = indRecon.astype(np.int64)
    tsrecon = ts[indRecon]
    tsrecon = tsrecon[:, :, 0]
    return tsrecon

def find_neighbors(dist_array, ind_array, wind):
    import numpy as np
    neighbors = []
    dist = []
    for i, pt in enumerate(ind_array):
        if len(pt[np.abs(pt[0]-pt) > wind]) > 0:
            j = np.min(pt[np.abs(pt[0]-pt) > wind])
        else:
            j = pt[1]

        neighbors.append([pt[0], j])
        dist.append([dist_array[0][0], dist_array[i][np.where(pt==j)[0][0]]])
    return np.array(dist), np.array(neighbors)


def FNN_n(ts, tau, maxDim=10, plotting=False, Rtol=15, Atol=2, Stol=0.9, threshold=10, strand=False):
    """This function implements the False Nearest Neighbors (FNN) algorithm described by Kennel et al.
    to select the minimum embedding dimension.

    Args:
       ts (array):  Time series (1d).
       tau (int):  Embedding delay.


    Kwargs:
       maxDim (int):  maximum dimension in dimension search. Default is 10.

       plotting (bool): Plotting for user interpretation. Default is False.

       Rtol (float): Ratio tolerance. Default is 15. (10 recommended for false nearest strands)

       Atol (float): A tolerance. Default is 2.

       Stol (float): S tolerance. Default is 0.9.

       threshold (float): Tolerance threshold for percent of nearest neighbors. Default is 10%.

       strand (bool): Use the composite false nearest strands algorithm (David Chelidze 2017)

    Returns:
       (int): n, The embedding dimension.

    """

    import numpy as np
    from scipy.spatial import KDTree
    if len(ts)-(maxDim-1)*tau < 20:
        maxDim = len(ts)-(maxDim-1)*tau-1
    ts = np.reshape(ts, (len(ts), 1))  # ts is a column vector
    st_dev = np.std(ts)  # standart deviation of the time series

    Xfnn = []
    dim_array = []


    # Set theiler window for false nearest strands. (4 times delay)
    if strand:
        strands = []
        w = 4 * tau
    else:
        w = 1
    
    flag = False
    i = 0
    while flag == False:
        i = i+1
        dim = i
        tsrecon = ts_recon(ts, dim, tau)  # delay reconstruction

        tree = KDTree(tsrecon)
        D, IDX = tree.query(tsrecon, k=w+1)

        if strand:
            D, IDX = find_neighbors(D, IDX, w)

        # Calculate the false nearest neighbor ratio for each dimension
        if i > 1:
            if strand:
                starting_index = 0
                epsilon_k = 0
                delta_k = 0
                num1 = []
                num2 = []
                for k in range(0, len(strand_ind)):
                    strand_norm = np.linalg.norm(s_dm[k])
                    if strand_norm > 0 and starting_index < len(tsrecon):
                        epsilon_k = np.linalg.norm(tsrecon[s_ind_m[k], :] - tsrecon[ind, :][starting_index:starting_index+len(s_ind_m[k])])/strand_norm
                        delta_k = np.sum(abs(tsrecon[s_ind_m[k], :][:,i-1] - tsrecon[ind, :][starting_index:starting_index+len(s_ind_m[k])][:,i-1]))/strand_norm
                    # Criteria 1
                    num1.append(np.heaviside(delta_k - epsilon_k*Rtol, 0.5))
                    # Criteria 2
                    num2.append(np.heaviside(delta_k - Stol*st_dev, 0.5))

                    starting_index += len(s_ind_m[k])

                den = len(strand_ind)
                num = sum(np.logical_or(num1, num2))

            else:
                D_mp1 = np.sqrt(
                    np.sum((np.square(tsrecon[ind_m, :]-tsrecon[ind, :])), axis=1))
                # Criteria 1 : increase in distance between neighbors is large
                num1 = np.heaviside(
                    np.divide(abs(tsrecon[ind_m, -1]-tsrecon[ind, -1]), Dm)-Rtol, 0.5)
                # Criteria 2 : nearest neighbor not necessarily close to y(n)
                num2 = np.heaviside(Atol-D_mp1/st_dev, 0.5)
                num = sum(np.multiply(num1, num2))
                den = sum(num2)

            Xfnn.append((num / den) * 100)
            dim_array.append(dim-1)

            if (num/den)*100 <= threshold or i == maxDim:
                flag = True

            print(dim-1, (num / den) * 100)

        # Save the index to D and k(n) in dimension m for comparison with the
        # same distance in m+1 dimension
        xlen2 = len(ts)-dim*tau
        Dm = D[0:xlen2, -1]
        ind_m = IDX[0:xlen2, -1]
        ind = ind_m <= xlen2-1
        ind_m = ind_m[ind]
        Dm = Dm[ind]

        # Get strands from index of nearest neighbors
        if strand:
            strand_ind = []
            s_ind = []
            s_ind_m = []
            s_dm = []
            #############################################################################
            # Loop through all points
            for rnum, row in enumerate(IDX[0:xlen2]):
                sort_status = 0
                # Loop through all strands
                for k in range(len(strand_ind)):
                    # Loop through all points on each strand
                    if row[1] in k + np.array(strand_ind[k])[:,1]:
                        strand_ind[k].append(row)
                        sort_status = 1
                        break
                # If item is not allocated to strand, make new strand
                if not sort_status:
                    strand_ind.append([])
                    strand_ind[-1].append(row)
            #############################################################################

            # Assign distances to strands
            for k in range(len(strand_ind)):
                s_ind.append(np.array(strand_ind[k])[:,-1]<=xlen2-1)
                s_ind_m.append(np.array(strand_ind[k])[:,-1][s_ind[k]])
                s_dm.append(D[s_ind_m[k]][:, -1])
        else:
            pass

        

    Xfnn = np.array(Xfnn)

    if plotting == True:
        import matplotlib.pyplot as plt
        TextSize = 14
        plt.figure(1)
        plt.plot(dim_array, Xfnn, marker='x', linestyle='-', color='blue')
        plt.xlabel(r'Dimension $n$', size=TextSize)
        plt.ylabel('Percent FNN', size=TextSize)
        plt.xticks(size=TextSize)
        plt.yticks(size=TextSize)
        plt.ylim([-1, 101])
        plt.show()

    return Xfnn, dim-1


# In[ ]:

if __name__ == '__main__':

    import numpy as np

    fs = 10
    t = np.linspace(0, 100, fs*100)
    ts = np.sin(t)

    tau = 15  # embedding delay

    perc_FNN, n = FNN_n(ts, tau, plotting=True)
    print('FNN embedding Dimension: ', n)
