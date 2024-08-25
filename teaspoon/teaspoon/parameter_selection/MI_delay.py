"""
Mutual Information (MI) for time delay (tau).
=======================================================================
uses mutual information to find a suitable delay via the location
of the first minima in the mutual information vs delay plot, which is calculated using multiple
x(t) vs x(t+tau) plots. These plots have their individual mutual information calculated. Various methods
for partitioning the x(t) vs x(t+tau) plots for calculating the mutual information.
"""

import scipy
import numpy as np


class Partitions:
    def __init__(self, data=None,
                 meshingScheme=None,
                 numParts=3,
                 alpha=0.05):
        import scipy

        if data is not None:
            # check that the data is in ordinal coordinates
            if not self.isOrdinal(data):
                print("Converting the data to ordinal...")
                # perform ordinal sampling (ranking) transformation
                xRanked = scipy.stats.rankdata(data[:, 0], method='ordinal')
                yRanked = scipy.stats.rankdata(data[:, 1], method='ordinal')

                xFloats = np.copy(data[:, 0])
                xFloats.sort()
                yFloats = np.copy(data[:, 1])
                yFloats.sort()

                self.xFloats = xFloats
                self.yFloats = yFloats

                data = np.column_stack((xRanked, yRanked))

                # and return an empty partition bucket

            # If there is data, set the bounding box to be the max and min in the data

            xmin = data[:, 0].min()
            xmax = data[:, 0].max()
            ymin = data[:, 1].min()
            ymax = data[:, 1].max()

            self.borders = {}
            self.borders['nodes'] = np.array([xmin, xmax, ymin, ymax])
            self.borders['npts'] = data.shape[0]
            self.numParts = numParts
            self.alpha = alpha

            # If there is data, use the chosen meshing scheme to build the partitions.
            if meshingScheme == 'DV' and self.isOrdinal(data):
                # Figure out
                self.partitionBucket = self.return_partition_DV(data=data,
                                                                borders=self.borders,
                                                                r=self.numParts,
                                                                alpha=self.alpha)
            else:  # meshingScheme == None
                # Note that right now, this will just do the dumb thing for every other input
                self.partitionBucket = [self.borders]
                #  set the partitions to just be the bounding box

        else:
            self.partitionBucket = []

    def convertOrdToFloat(self, partitionEntry):
        bdyList = partitionEntry['nodes'].copy()
        # Need to subtract one to deal with counting from
        # 0 vs counting from 1 problems
        xLow = int(bdyList[0])-1
        xHigh = int(bdyList[1])-1
        yLow = int(bdyList[2])-1
        yHigh = int(bdyList[3])-1

        if hasattr(self, 'xFloats'):
            xLowFloat = self.xFloats[xLow]
            xHighFloat = self.xFloats[xHigh]
            yLowFloat = self.yFloats[yLow]
            yHighFloat = self.yFloats[yHigh]
            convertedBdyList = [xLowFloat, xHighFloat, yLowFloat, yHighFloat]
            partitionEntry['nodes'] = convertedBdyList
            return partitionEntry
        else:
            print("You're trying to convert your ordinal data")
            print("back to floats, but you must have had ordinal")
            print("to begin with so I can't.  Exiting...")

    def __len__(self):
        return len(self.partitionBucket)

    def __getitem__(self, key):
        if hasattr(self, 'xFloats'):  # if the data wasn't ordinal
            entry = self.partitionBucket[key].copy()
            entry = self.convertOrdToFloat(entry)
            return entry
        else:
            return self.partitionBucket[key]

    def getOrdinal(self, key):
        # overrides the builtin magic method in the case where
        # you had non-ordinal data but still want the ordinal
        # stuff back.
        # If the data wasn't ordinal, this has the exact same
        # effect as self[key]
        return self.partitionBucket[key]

    def __iter__(self):
        # iterates over the converted entries in the
        # parameter bucket
        if hasattr(self, 'xFloats'):
            return map(self.convertOrdToFloat, deepcopy(self.partitionBucket))
        else:
            return iter(self.partitionBucket)

    def iterOrdinal(self):
        # functions just like iter magic method without
        # converting each entry back to its float
        return iter(self.partitionBucket)

    def __str__(self):
        """!
        @brief Nicely prints all currently set values in the bucket.
        """
        attrs = vars(self)
        output = ''
        output += 'Variables in partition bucket\n'
        output += '---\n'
        for key in attrs.keys():
            output += str(key) + ' : '
            output += str(attrs[key]) + '\n'
            output += '---\n'
        return output

    def plot(self):
        import matplotlib.pyplot as plt
        import matplotlib
        # plot the partitions
        fig1, ax1 = plt.subplots()
        for binNode in self:
            # print(binNode)
            # get the bottom left corner
            corner = (binNode['nodes'][0], binNode['nodes'][2])

            # get the width and height
            width = binNode['nodes'][1] - binNode['nodes'][0]
            height = binNode['nodes'][3] - binNode['nodes'][2]

            # add the corresponding rectangle
            ax1.add_patch(matplotlib.patches.Rectangle(
                corner, width, height, fill=False))

        # Doesn't show unless we do this
        plt.axis('tight')

    # helper function for error checking. Used to make sure input is in
    # ordinarl coordinates. It checks that when the two data columns are sorted
    # they are each equal to an ordered vector with the same number of rows.
    def isOrdinal(self, dd):
        return np.all(np.equal(np.sort(dd, axis=0),
                               np.reshape(np.repeat(np.arange(start=1, stop=dd.shape[0]+1),
                                                    2), (dd.shape[0], 2))))

    # data: is a manyx2 numpy array that contains all the original data
    # borders: a dictionary that contains 'nodes' with a numpyu array of Xmin, Xmax, Ymin, Ymax,
    # and 'npts' which contains the number of points in the bin
    # r: is the number of partitions
    # alpha: the significance level to test for independence

    def return_partition_DV(self, data, borders, r=2, alpha=0.05):
        import scipy
        # extract the bin boundaries
        Xmin = borders['nodes'][0]
        Xmax = borders['nodes'][1]
        Ymin = borders['nodes'][2]
        Ymax = borders['nodes'][3]

        # find the number of bins
    #    numBins = r ** 2
        idx = np.where((data[:, 0] >= Xmin)
                       & (data[:, 0] <= Xmax)
                       & (data[:, 1] >= Ymin)
                       & (data[:, 1] <= Ymax))

        # extract the points in the bin
        Xsub = data[idx, 0]
        Ysub = data[idx, 1]

    #    print(Xsub.shape, '\t', Ysub.shape)

        # find the indices of the points in the x- and y-patches
        idx_x = np.where((data[:, 0] >= Xmin) & (data[:, 0] <= Xmax))
        idx_y = np.where((data[:, 1] >= Ymin) & (data[:, 1] <= Ymax))

        # get the subpartitions
        ai = np.floor(np.percentile(
            data[idx_x, 0], 1/r * np.arange(1, r) * 100))
        bj = np.floor(np.percentile(
            data[idx_y, 1], 1/r * np.arange(1, r) * 100))

        # get the bin edges
        edges1 = np.concatenate(([Xmin], ai, [Xmax]))
        edges2 = np.concatenate(([Ymin], bj, [Ymax]))

        # first exit criteria: we cannot split inot unique boundaries any more
        # preallocate the partition list
        partitions = []
        if (len(np.unique(edges1, return_counts=True)[1]) < r + 1 or
                len(np.unique(edges2, return_counts=True)[1]) < r + 1):

            # reject futher partitions, and return original bin
            partitions.insert(0, {'nodes': np.array([Xmin, Xmax, Ymin, Ymax]),
                                  'npts': len(idx[0])})
            return partitions

        # figure out the shift in the edges so that boundaries do not overlap
        xShift = np.zeros((2 * r, 2 * r))
        yShift = xShift
        xShift[:, 1:-1] = np.tile(np.array([[-1, 0]]), (2 * r, r - 1))
        yShift = xShift.T

        # find the boundaries for each bin
        # duplicate inner nodes for x mesh
        dupMidNodesX = np.append(np.insert(np.repeat((edges1[1:-1]), 2, axis=0),
                                           0, edges1[0]), edges1[-1])

        # duplicate inner nodes for y mesh
        dupMidNodesY = np.append(np.insert(np.repeat((edges2[1:-1]), 2, axis=0),
                                           0, edges2[0]), edges2[-1])
        # reshape
        dupMidNodesY = np.reshape(dupMidNodesY, (-1, 1))

        # now find the nodes for each bin
        xBinBound = dupMidNodesX + xShift
        yBinBound = dupMidNodesY + yShift
        # find the number of points in each bin, and put this info into array
        binned_data = scipy.stats.binned_statistic_2d(Xsub.flatten(), Ysub.flatten(), None, 'count',
                                                      bins=[edges1, edges2])
        # get the counts. Flatten columnwise to match the bin definition in the
        # loop that creates the dictionaries below
        binCounts = binned_data.statistic.flatten('F')

        # define an empty list to hold the dictionaries of the fresh partitions
        bins = []
        # create dictionaries for each bin
        # start with the loop over y
        # note how the loop counts were obtained above to match the convention
        # here
        for yInd in np.arange(r):
            # this is the loop over x
            for xInd in np.arange(r):
                # get the bin number
                binNo = yInd * r + xInd
                xLow, xHigh = xBinBound[yInd, 2*xInd + np.arange(2)]
                yLow, yHigh = yBinBound[2*yInd + np.arange(2), xInd]
                bins.append({'nodes': np.array([xLow, xHigh, yLow, yHigh]),
                             'npts': binCounts[binNo]})

        # calculate the chi square statistic
        chi2 = scipy.stats.chisquare(binCounts)

        # check for independence and start recursion
        # if the chi2 test fails, do further partitioning:
        if (chi2.pvalue < alpha and Xmax != Xmin and Ymax != Ymin).all():
            for binInfo in bins:
                if binInfo['npts'] != 0:  # if the bin is not empty:
                    # append entries to the tuple
                    partitions.extend(self.return_partition_DV(data=data,
                                                               borders=binInfo,
                                                               r=r, alpha=alpha))

        # Second exit criteria:
        # if the partitions are independent, reject further partitioning and
        # save the orignal, unpartitioned bin
        elif len(idx[0]) != 0:
            partitions.insert(0, {'nodes': np.array([Xmin, Xmax, Ymin, Ymax]),
                                  'npts': len(idx[0])})

        return partitions


# In[ ]:


def MI_DV(x, y):
    """This function calculates the mutual information between the time series x(t) and its delayed version x(t+tau)
    using adaptive partitioning of the plots of the time series x(t) and its delayed version x(t+tau). 
    This method was developed by Georges Darbellay and Igor Vajda in 1999 and was published as
    Estimation of information by an adaptive partitioning of the observation.

    Args:
       x (array): time series x(t)
       y (array): delayed time series x(t + tau) 

    Returns:
       (float): I, mutual information between x(t) and x(t+tau).

    """
    # input: x: x cooridnate array (could be time array), y: y coorindate array (could be time series)
    import numpy as np
    from scipy.stats import rankdata

    # perform ordinal sampling (ranking) transformation
    xRanked = rankdata(x, method='ordinal')
    yRanked = rankdata(y, method='ordinal')

    # obtain the adaptive mesh
    numParts = 4
    # get the adaptive partition of the data
    partitionList = Partitions(np.column_stack(
        (xRanked, yRanked)), meshingScheme="DV", numParts=numParts)
    # partitions are sorted in class as:
    # 1. overall borders with number of points total: xmin, xmax, ymin, ymax
    # 2. numParts = max number of parts in a single bin/partition
    # 3. alpha
    # 4. partition bucket: this has all the partition's borders and number of points in each.

    # extract the bin counts
    binCounts = np.array([partitionList.partitionBucket[i].get(
        "npts") for i in range(len(partitionList))])

    # get the total number of points
    N = partitionList.borders.get("npts")

    # grab the probability information from the partition
    Pn_AB = binCounts / N

    # grab the probbility of the horizontal strips for each bin
    PC = partitionList  # partition cells

    Pn_AR = np.zeros(len(partitionList))
    Pn_RB = np.zeros(len(partitionList))

    for Bin in range(len(partitionList)):  # go through each bin
        Pn_AR[Bin] = len([xRanked for i in range(N)  # count number of point between x bounds of bin
                          if xRanked[i] >= PC[Bin].get('nodes')[0] and xRanked[i] <= PC[Bin].get('nodes')[1]])
        Pn_RB[Bin] = len([yRanked for i in range(N)  # count number of point between y bounds of bin
                          if yRanked[i] >= PC[Bin].get('nodes')[2] and yRanked[i] <= PC[Bin].get('nodes')[3]])
    Pn_AR = (Pn_AR)/N  # divide for probability
    Pn_RB = (Pn_RB)/N

    # find the approximation for the mutual information function
    Iapprox = np.dot(Pn_AB, np.log(Pn_AB/(Pn_AR*Pn_RB)))
    return Iapprox

# This function computes the mutual information function based on the
# algorithm described in:
# "Estimating Mutual Information," Alexander Kraskov, Harald Stoegbauer, Peter Grassberger, 2003.


def MI_kraskov(x, y, k=2, ranking=True):
    """This function estimates the mutual information between the time series x(t) and its delayed version x(t+tau) in two different ways. 
    This method was developed by Alexander Kraskov, Harald Stoegbauer, and Peter Grassberger in 2003 and published as 
    Estimating Mutual Information.

    Args:
       x (array): time series x(t)
       y (array): delayed time series x(t + tau) 

    Kwargs:
       k (int): number of nearest neighbors used in MI estimation. Default is k = 2.

       ranking (bool): whether the ranked or unranked x and y inputs will be used. Default is ranking = True.

    Returns:
       (float): I1, first mutual information estimation method between x(t) and x(t+tau).
       (float): I2, second mutual information estimation method between x(t) and x(t+tau).

    """
    if ranking == True:  # rank x and y ordinally
        from scipy.stats import rankdata
        x = rankdata(x, method='ordinal')
        y = rankdata(y, method='ordinal')

    import numpy as np
    # put into a column vector and perturb the data to improve uniqueness
    x = x  # + (10**-5)*np.random.random_sample((len(x),))
    y = y  # + (10**-5)*np.random.random_sample((len(y),))

    lenZ = len(x)
    # find nearest neighbors in s-q plane using the chebyshev distance
    from sklearn.neighbors import NearestNeighbors
    vec = np.stack((x, y)).T
    nbrs = NearestNeighbors(
        n_neighbors=k+1, algorithm='ball_tree', metric='chebyshev').fit(vec)
    distances, indZ = nbrs.kneighbors(vec)
    # from the subset of nearest neighbors, find the distance to the kth nearest neighbor after
    # projecting the points on each axis
    cDx = abs(
        x[indZ] - np.tile(np.reshape((x[indZ].T[0]), (len(x[indZ].T[0]), 1)), k+1))
    cDx = np.sort(cDx, axis=-1).T[-1]
    cDy = abs(
        y[indZ] - np.tile(np.reshape((y[indZ].T[0]), (len(y[indZ].T[0]), 1)), k+1))
    cDy = np.sort(cDy, axis=-1).T[-1]
    # finding closest nearest kth neighbor distance for each node
    cD = distances.T[-1]

    # METHOD 1
    # find number of points (s-q plane) that fall within strips with widths defined by cDx and cDy
    nxI = []
    nyI = []
    # array of distances in x from point i
    x_d = abs((np.tile(x.reshape((len(x), 1)), len(x)) - x))
    # array of distances in y from point i
    y_d = abs((np.tile(y.reshape((len(y), 1)), len(y)) - y))
    for i in range(len(cDx)):
        nxI.append(len(x_d[i][x_d[i] <= cD[i]]))
        nyI.append(len(y_d[i][y_d[i] <= cD[i]]))
    nxI = np.array(nxI) - 1
    nyI = np.array(nyI) - 1

    # METHOD 2
    # find number of points falling within epsilon/2 in x and y
    nxI2 = []
    nyI2 = []
    for i in range(len(cDx)):
        # append if distance is less than epsilon_x/2
        nxI2.append(len(x_d[i][x_d[i] <= (cDx[i])]))
        # append if distance is less than epsilon_y/2
        nyI2.append(len(y_d[i][y_d[i] <= (cDy[i])]))
    nxI2 = np.array(nxI2) - 1  # subtract one for self
    nyI2 = np.array(nyI2) - 1  # subtract one for self

    # compute the mutual information function using the digamma function
    from scipy.special import digamma
    # alogirthm 1 mutual information:
    I1 = digamma(k) - np.mean(digamma(nxI + 1) +
                              digamma(nyI+1)) + digamma(lenZ)
    # algorithm 2 mutual information
    I2 = digamma(k) - (1/k) - np.mean(digamma(nxI2) +
                                      digamma(nyI2)) + digamma(lenZ)

    return I1, I2


# This function computes the mutual information function based on
# equal sized bins for x and y arrays (seperately)

def MI_basic(x, y, h_method='sturge', ranking=True):
    """This function calculates the mutual information between the time series x(t) and its delayed version x(t+tau)
    using equi-spaced partitions. The size of the partition is based on the desired bin size method commonly selected for histograms.

    Args:
       x (array): time series x(t)
       y (array): delayed time series x(t + tau) 

    Kwargs:
       h_method (string): bin size selection method. Methods are struge, sqrt, or rice. Default is sturge.

       ranking (bool): whether the ranked or unranked x and y inputs will be used. Default is ranking = True.

    Returns:
       (float): I, mutual information between x(t) and x(t+tau).

    """
    # input: x: x array (could be time array), y: y array (could be time series)
    import numpy as np
    from scipy.stats import rankdata

    # number of points
    Nx = len(x)
    Ny = len(y)
    if Nx == Ny:
        N = Nx
    else:
        N = max([Nx, Ny])

    # perform ordinal sampling (ranking) transformation
    if ranking == True:
        x = rankdata(x, method='ordinal')
        y = rankdata(y, method='ordinal')

    # first find number of bins based on given method
    # Either Sturge's or Rice Rule provides the most acurate results so far
    # strurges works better for smaller data sets
    if h_method == 'sturge':
        Bx = np.log2(Nx) + 1
        By = np.log2(Ny) + 1
        B = [Bx, By]

    if h_method == 'sqrt':
        Bx = np.sqrt(Nx)
        By = np.sqrt(Ny)
        B = [Bx, By]

    if h_method == 'rice':
        Bx = 2*(Nx**(1/3))
        By = 2*(Ny**(1/3))
        B = [Bx, By]

    B = np.round(np.array(B)).astype(int)

    # calculates 1D histogram for both x and y seperately
    Hx = np.histogram(x, bins=int(B[0]))[0]
    Hy = np.histogram(y, bins=int(B[1]))[0]

    # calculates histogram for 2D data on s-q plot with given number of bins
    Hxy = np.histogram2d(x, y, bins=B)[0]

    Px = Hx/N
    Py = Hy/N
    Pxy = Hxy/N

    # Ignore divide by zero warnings (NaN set to zero in next line)
    with np.errstate(divide='ignore', invalid='ignore'):
        I_matrix = Pxy*np.log(Pxy/(Px*Py))
    where_are_NaNs = np.isnan(I_matrix)
    I_matrix[where_are_NaNs] = 0
    I = sum(sum(I_matrix))
    return (I)


def MI_for_delay(ts, plotting=False, method='basic', h_method='sturge', k=2, ranking=True):
    """This function calculates the mutual information until a first minima is reached, which is estimated as a sufficient embedding dimension for permutation entropy.

    Args:
       ts (array):  Time series (1d).

    Kwargs:
       plotting (bool): Plotting for user interpretation. defaut is False.

       method (string): Method for calculating MI. Options include basic, kraskov 1, kraskov 2, or adaptive partitions. Default is basic.

       h_method (string): Bin size selection method for basic method. Methods are struge, sqrt, or rice. Default is sturge.

       ranking (bool): Whether the ranked or unranked x and y inputs will be used for kraskov and basic methods. Default is ranking = True.

       k (int): Number of nearest neighbors used in MI estimation for Kraskov methods. Default is k = 2.

    Returns:
       (int): tau, The embedding delay for permutation formation based on first mutual information minima.

    """
    delayMax = 250
    min_flag = False
    I = []  # initializes information array
    tau = []
    delay = 0
    method_flag = False
    while delay < delayMax and min_flag == False:
        delay = delay + 1
        if method == 'adaptive partitions':
            method_flag = True
            # takes all terms from time series besides last (delay) terms
            x = ts[:-delay]
            # takes all terms from time series besides first (delay) terms
            y = ts[delay:]
            I.append(MI_DV(x, y))
            tau.append(delay)

        if method == 'kraskov 1':
            method_flag = True
            # takes all terms from time series besides last (delay) terms
            x = ts[:-delay]
            # takes all terms from time series besides first (delay) terms
            y = ts[delay:]
            MI_k = MI_kraskov(x, y)
            I.append(MI_k[0])
            tau.append(delay)

        if method == 'kraskov 2':
            method_flag = True
            # takes all terms from time series besides last (delay) terms
            x = ts[:-delay]
            # takes all terms from time series besides first (delay) terms
            y = ts[delay:]
            MI_k = MI_kraskov(x, y)
            I.append(MI_k[1])
            tau.append(delay)

        if method == 'basic':
            method_flag = True
            # takes all terms from time series besides last (delay) terms
            x = ts[:-delay]
            # takes all terms from time series besides first (delay) terms
            y = ts[delay:]
            I.append(MI_basic(x, y))
            tau.append(delay)

        if delay > 1 or delay == delayMax-1:
            if I[delay-2]-I[delay-1] < 0 or delay == delayMax-1:  # if increasing
                delay_at_min = delay
                min_flag = True
        if method_flag == False:
            print('Warning: invalid method entered.')
            break
    if plotting == True:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        TextSize = 18
        plt.plot(tau, I, label=method + ': ' + str(delay_at_min-1))
        plt.xlabel(r'$\tau$', size=TextSize)
        plt.ylabel('MI', size=TextSize)
        plt.xticks(size=TextSize)
        plt.yticks(size=TextSize)
        #plt.legend(loc = 'upper right', fontsize = TextSize)
        plt.ylim(0)
        plt.show()

    return int(delay_at_min - 1)


# In[ ]:
if __name__ == "__main__":

    from teaspoon.parameter_selection.MI_delay import MI_for_delay
    import numpy as np

    fs = 10
    t = np.linspace(0, 100, fs*100)
    ts = np.sin(t) + np.sin((1/np.pi)*t)

    tau = MI_for_delay(ts, plotting=True, method='basic',
                       h_method='sturge', k=2, ranking=True)
    print('Delay from MI: ', tau)
