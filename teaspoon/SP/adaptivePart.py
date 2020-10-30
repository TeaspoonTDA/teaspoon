# -*- coding: utf-8 -*-
"""
Methods of partitioning birth-lifetime plane for persistence diagrams. This is
used for the adaptive partitioning version of template function featurization.

"""

# Created on Tue Aug 14 09:35:30 2018
#
# @author: khasawn3

import numpy as np
from scipy.stats import binned_statistic_2d, chisquare, rankdata
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
from sklearn.cluster import KMeans, MiniBatchKMeans

class Partitions:
    '''
    A data structure for storing a partition coming from an adapative meshing scheme.

    Parameters:
        data (np.array):
            A numpy array of type many by 2

        convertToOrd (bool):
            Boolean variable to decide if you want to use ordinals for
            partitioning. Ordinals make things faster but not as nice partitions.

        meshingScheme (str):
            The type of meshing scheme you want to use. Options include:

                - 'DV' method is based on (mention paper here). For more details see function return_partition_DV.
                - 'clustering' uses a clustering algorithm to find clusters in the data, then takes the bounding box of all points assigned to each cluster. For more details see the function return_partition_clustering.

        partitionParams (dict):
            Dictionary of parameters needed for the particular meshing scheme selected.
            For the explanation of the parameters see the function for the specific meshingScheme
            (i.e. return_partition_DV or return_partition_clustering)

                - For 'DV' the adjustable parameters are 'alpha', 'c', 'nmin', 'numParts', 'split'.
                - For 'clustering' the adjustable parameters are 'numClusters', 'clusterAlg', 'weights', 'boxOption', 'boxWidth', 'split'.

        kwargs:
            Any leftover inputs are stored as attributes.

    '''


    def __init__(self, data = None,
                 convertToOrd = False,
                 meshingScheme = None,
                 partitionParams = {},
                 **kwargs):

        self.convertToOrd = convertToOrd
        self.meshingScheme = meshingScheme
        self.__dict__.update(kwargs)

        if data is not None:

            # # if using kmeans, we dont want to convert to ordinals
            if meshingScheme == 'DV':
                convertToOrd = True

            # check if we want to convert to ordinals
            # may not want to for certain partitioning schemes
            if convertToOrd:
                # check that the data is in ordinal coordinates
                # data converted to ordinal and stored locally if not already
                if not self.isOrdinal(data):
                    print("Converting the data to ordinal...")

                    # perform ordinal sampling (ranking) transformation
                    xRanked = rankdata(data[:,0], method='ordinal')
                    yRanked = rankdata(data[:,1], method='ordinal')

                    # copy original data and save
                    xFloats = np.copy(data[:,0])
                    xFloats.sort()
                    yFloats = np.copy(data[:,1])
                    yFloats.sort()

                    self.xFloats = xFloats
                    self.yFloats = yFloats


                    data = np.column_stack((xRanked,yRanked))

                # and return an empty partition bucket

            # If there is data, set the bounding box to be the max and min in the data
            xmin = data[:,0].min()
            xmax = data[:,0].max()
            ymin = data[:,1].min()
            ymax = data[:,1].max()

            # self.borders stores x and y min and max of overall bounding box in 'nodes' and the number of points in the bounding box in 'npts'
            self.borders = {}
            self.borders['nodes'] = np.array([xmin, xmax, ymin, ymax])
            self.borders['npts'] = data.shape[0]

            # set parameters for partitioning algorithm
            self.setParameters(partitionParams=partitionParams)

            # If there is data, use the chosen meshing scheme to build the partitions.
            if meshingScheme == 'DV':

                self.partitionBucket = self.return_partition_DV(data = data,
                                        borders = self.borders,
                                        r = self.numParts,
                                        alpha = self.alpha,
                                        c = self.c,
                                        nmin = self.nmin)

            elif meshingScheme == 'clustering':

                self.partitionBucket = self.return_partition_clustering(data = data,
                                        clusterAlg = self.clusterAlg,
                                        num_clusters = self.numClusters,
                                        weights= self.weights,
                                        boxOption = self.boxOption,
                                        boxSize = self.boxSize)

            else: # meshingScheme == None
            # Note that right now, this will just do the dumb thing for every other input
                self.partitionBucket = [self.borders]
                #  set the partitions to just be the bounding box

        else:
            self.partitionBucket = []

    def convertOrdToFloat(self,partitionEntry):
        '''
        Converts to nodes of a partition entry from ordinal back to floats.

        Parameters:
            partitionEntry (dict):
                The partition that you want to convert.

        Returns:
            Partition entry with converted nodes. Also sets dictionary element to the converted version.

        '''

        bdyList = partitionEntry['nodes'].copy()
        # Need to subtract one to deal with counting from
        # 0 vs counting from 1 problems
        xLow = int(bdyList[0])-1
        xHigh = int(bdyList[1])-1
        yLow = int(bdyList[2])-1
        yHigh = int(bdyList[3])-1


        if hasattr(self, 'xFloats'):
            xLowFloat = self.xFloats[xLow]
            xHighFloat= self.xFloats[xHigh]
            yLowFloat = self.yFloats[yLow]
            yHighFloat = self.yFloats[yHigh]
            convertedBdyList = [xLowFloat, xHighFloat, yLowFloat,yHighFloat]
            partitionEntry['nodes'] = convertedBdyList
            return partitionEntry
        else:
            print("You're trying to convert your ordinal data")
            print("back to floats, but you must have had ordinal")
            print("to begin with so I can't.  Exiting...")

    def __len__(self):
        return len(self.partitionBucket)

    def __getitem__(self,key):
        if hasattr(self,'xFloats'): #if the data wasn't ordinal
            entry = self.partitionBucket[key].copy()
            entry = self.convertOrdToFloat(entry)
            return entry
        else:
            return self.partitionBucket[key]

    def getOrdinal(self,key):
        '''
        Overrides the builtin magic method in the case where you had non-ordinal data but still want the ordinal stuff back.
        If the data wasn't ordinal, this has the exact same effect as self[key].

        '''
        # overrides the builtin magic method in the case where
        # you had non-ordinal data but still want the ordinal
        # stuff back.
        # If the data wasn't ordinal, this has the exact same
        # effect as self[key]
        return self.partitionBucket[key]

    def __iter__(self):
        # iterates over the converted entries in the
        # parameter bucket
        if hasattr(self,'xFloats'):
            return map( self.convertOrdToFloat, deepcopy(self.partitionBucket)  )
        else:
            return iter(self.partitionBucket)

    def iterOrdinal(self):
        '''
        Functions just like iter magic method without converting each entry back to its float

        '''
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
            output += str(attrs[key])+ '\n'
            output += '---\n'
        return output

    def plot(self):
        '''
        Plot the partitions. Can plot in ordinal or float, whichever is in the partition bucket when it's called.

        '''
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
            ax1.add_patch(patches.Rectangle(corner, width, height, fill=False))

        # Doesn't show unless we do this
        plt.axis('tight')

    # helper function for error checking. Used to make sure input is in
    # ordinarl coordinates. It checks that when the two data columns are sorted
    # they are each equal to an ordered vector with the same number of rows.
    def isOrdinal(self, dd):
        '''
        Helper function for error checking. Used to make sure input is in ordinal coordinates.
        It checks that when the two data columns are sorted they are each equal to an ordered vector with the same number of rows.

        :param dd:
            Data in a manyx2 numpy array

        '''
        return np.all(np.equal(np.sort(dd, axis=0),
                        np.reshape(np.repeat(np.arange(start=1,stop=dd.shape[0]+1),
                                             2), (dd.shape[0], 2))))




    # data: is a manyx2 numpy array that contains all the original data
    # borders: a dictionary that contains 'nodes' with a numpyu array of Xmin, Xmax, Ymin, Ymax,
    # and 'npts' which contains the number of points in the bin
    # r: is the number of partitions
    # alpha: the significance level to test for independence
    def return_partition_DV(self, data, borders, r=2, alpha=0.05, c=0, nmin=5):
        '''
        Recursive method that partitions the data based on the DV method.

        Parameters:
            data (np.array):
                A manyx2 numpy array that contains all the data in ordinal format.

            borders (dict):
                A dictionary that contains 'nodes' with a numpy array of Xmin, Xmax, Ymin, Ymax.

            r (int):
                The number of partitions to split in each direction
                (i.e. r=2 means each partition is recursively split into a
                2 by 2 grid of partitions)

            alpha (float):
                The required significance level for independence to stop partitioning

            c (int):
                Parameter for an exit criteria. Partitioning stops if min(width of
                partition, height of partition) < max(width of bounding box, height
                of bounding box)/c.

            nmin (int):
                Minimum average number of points in each partition to keep recursion going.
                The default is 5 because chisquare test breaks down with less than 5 points
                per partition, thus we recommend choosing nmin >= 5.

        Returns:
            List of dictionaries. Each dictionary corresponds to a partition and
            contains 'nodes', a numpy array of Xmin, Xmax, Ymin, Ymax of the partition,
            and 'npts', the number of points in the partition.

        '''
        # extract the bin boundaries
        Xmin = borders['nodes'][0]
        Xmax = borders['nodes'][1]
        Ymin = borders['nodes'][2]
        Ymax = borders['nodes'][3]

        # find the number of bins
    #    numBins = r ** 2
        idx = np.where((data[:, 0] >= Xmin)
                       & (data[:, 0] <= Xmax )
                       & (data[:, 1] >= Ymin)
                       & (data[:, 1] <= Ymax))

        partitions = []

        # Exit Criteria:
        # if either height or width is less than the max size, return
        width = self.xFloats[int(Xmax-1)] - self.xFloats[int(Xmin-1)]
        height = self.yFloats[int(Ymax-1)] - self.yFloats[int(Ymin-1)]
        if ( ( c != 0 ) and ( min(width,height) < c) ):
            # print('Box getting too small, min(width,height)<', c)
            # reject futher partitions, and return original bin
            partitions.insert(0, {'nodes': np.array([Xmin, Xmax, Ymin, Ymax]),
                      'npts': len(idx[0])})
            return partitions

        # extract the points in the bin
        Xsub = data[idx, 0]
        Ysub = data[idx, 1]

    #    print(Xsub.shape, '\t', Ysub.shape)

        # find the indices of the points in the x- and y-patches
        idx_x = np.where((data[:, 0] >= Xmin) & (data[:, 0] <= Xmax))
        idx_y = np.where((data[:, 1] >= Ymin) & (data[:, 1] <= Ymax))

        # get the subpartitions
        ai = np.floor(np.percentile(data[idx_x, 0], 1/r * np.arange(1, r) * 100))
        bj = np.floor(np.percentile(data[idx_y, 1], 1/r * np.arange(1, r) * 100))

        # get the bin edges
        edges1 = np.concatenate(([Xmin], ai, [Xmax]))
        edges2 = np.concatenate(([Ymin], bj, [Ymax]))

        # first exit criteria: we cannot split inot unique boundaries any more
        # preallocate the partition list
        if (len(np.unique(edges1, return_counts=True)[1]) < r + 1 or
             len(np.unique(edges2, return_counts=True)[1])< r + 1):

            # reject futher partitions, and return original bin
            partitions.insert(0, {'nodes': np.array([Xmin, Xmax, Ymin, Ymax]),
                      'npts': len(idx[0])})
            return partitions

        # figure out the shift in the edges so that boundaries do not overlap
        xShift = np.zeros( (2 * r, 2 * r))
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
        binned_data = binned_statistic_2d(Xsub.flatten(), Ysub.flatten(), None, 'count',
                                          bins=[edges1, edges2])
        # get the counts. Flatten columnwise to match the bin definition in the
        # loop that creates the dictionaries below
        binCounts = binned_data.statistic.flatten('F')

        # Exit Criteria:
        # check if sum of bin counts is less than threshold of nmin per bin
        # nmin is necessary because chisquare breaks down if you have less than
        # 5 points in each bin
        if nmin != 0:
            if np.sum(binCounts) < nmin * (r**2):
                partitions.insert(0, {'nodes': np.array([Xmin, Xmax, Ymin, Ymax]),'npts': len(idx[0])})
                return partitions

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
                binNo = yInd * r  + xInd
                xLow, xHigh = xBinBound[yInd, 2*xInd + np.arange(2)]
                yLow, yHigh = yBinBound[2*yInd + np.arange(2), xInd]
                bins.append({'nodes': np.array([xLow, xHigh, yLow, yHigh]),
                    'npts': binCounts[binNo] })

        # calculate the chi square statistic
        chi2 = chisquare(binCounts)

        # check for independence and start recursion
        # if the chi2 test fails, do further partitioning:
        if (chi2.pvalue < alpha and Xmax!=Xmin and Ymax!=Ymin).all():
            for binInfo in bins:
                if binInfo['npts'] !=0:  # if the bin is not empty:
                    # append entries to the tuple
                    partitions.extend(self.return_partition_DV(data=data,
                                                            borders=binInfo,
                                                            r=r, alpha=alpha))

        # Second exit criteria:
        # if the partitions are independent, reject further partitioning and
        # save the orignal, unpartitioned bin
        elif len(idx[0]) !=0:
            partitions.insert(0, {'nodes': np.array([Xmin, Xmax, Ymin, Ymax]),
                      'npts': len(idx[0])})

        return partitions


    def return_partition_clustering(self, data, clusterAlg = KMeans, num_clusters=5, weights = None, boxOption="boundPoints", boxSize=2):
        '''
        Partitioning method based on clustering algorithms. First cluster the data, then using the cluster centers and labels determine the partitions.

        Parameters:
            data (np.array):
                A manyx2 numpy array that contains all the original data (not ordinals).

            cluster_algorithm (function):
                Clustering algorithm you want to use. Only options right now are
                KMeans and MiniBatchKMeans from scikit learn.

            num_clusters (int):
                The number of clusters you want. This is the number of partitions
                you want to divide your space into.

            weights (np.array):
                An array of the same length as data containing weights of points to use weighted clustering

            boxOption (str):
                Specifies how to choose the boxes based on cluster centers. Only option right now is
                "boundPoints" which takes the bounding box of all data points assigned to that cluster center.
                Additional options may be added in the future.

            boxSize (int):
                This input is not used as of now.

        Returns:
            List of dictionaries. Each dictionary corresponds to a partition and contains 'nodes',
            a numpy array of Xmin, Xmax, Ymin, Ymax of the partition, and 'center', the center of the
            cluster for that partition.

        '''

        if weights == 0:
            sample_weights = data[:,0]
        elif weights == 1:
            sample_weights = data[:,1]
        else:
            sample_weights = None

        # Fit using whatever the chosen cluster algorithm is
        kmeans = clusterAlg(n_clusters=num_clusters).fit(data,sample_weight = sample_weights)

        # Get the centers of each cluster and the labels for the data points
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        bins = []

        # Using this optin, take the bounding box of points closest to each cluster center
        # These will be the partitions
        if boxOption == "boundPoints":

            for l in np.unique(labels):
                cluster = data[labels == l]

                xmin = min(cluster[:,0])
                xmax = max(cluster[:,0])
                ymin = min(cluster[:,1])
                ymax = max(cluster[:,1])

                # issues if bounding box touches x-axis so print a warning if it does
                # if ymin == 0:
                #     print("Uh oh can't have points with zero lifetime!")


                bins.insert(0,{'nodes': [xmin,xmax,ymin,ymax], 'center': centers[l]})

        # # Using this option, put the equal size box centered at each cluster center
        # # These are then the partitions and we ignore anything outside them
        # elif boxOption == "equalSize":
        #     print("STOP: the 'equalSize' option has not been debugged. It may be available later.")
        #     print("If you used this option, I'm just giving you back the bounding box.")
        #
        #     bins.insert(0, {'nodes':[ min(data[:,0]), max(data[:,0]), min(data[:,1]), max(data[:,1]) ]})
        #
        #     ######################################################################
        #     ### DON'T DELETE
        #     ### This is a starting point but commented out because it doesn't work
        #     ### properly. There is no error checking so boxes could cross x axis
        #     ### which we can't have so needs more before it is usable
        #     ######################################################################
        #     # if isinstance(boxSize, int):
        #     #     boxSize = list([boxSize,boxSize])
        #     #
        #     # for l in np.unique(labels):
        #     #     center = centers[l]
        #     #
        #     #     xmin = center[0] - boxSize[0]/2
        #     #     xmax = center[0] + boxSize[0]/2
        #     #     ymin = center[1] - boxSize[1]/2
        #     #     ymax = center[1] + boxSize[1]/2
        #     #
        #     #     bins.insert(0,{'nodes': [xmin,xmax,ymin,ymax], 'center': centers[l]})
        #     ######################################################################

        return bins

    def setParameters(self, partitionParams):
        '''
        Helper function to set the parameters depending on the meshing scheme.
        If any are not specified, it is set to a default value.

        Parameters:
            partitionParams:
                Dictionary containing parameters needed for the partitioning algorithm.

        '''

        if self.meshingScheme == 'DV':

            xmin, xmax, ymin, ymax = self.borders['nodes']

            # c det
            if 'c' in partitionParams:
                c = partitionParams['c']

                if c != 0:
                    # convert c from integer to the corresponding width/height
                    width = (self.xFloats[xmax-1]-self.xFloats[xmin-1]) / c
                    height = (self.yFloats[ymax-1]-self.yFloats[ymin-1]) / c
                    self.c = max( width, height )
                else:
                    # c=0 means we don't use this paramter for an exit criteria
                    self.c = 0
            else:
                c = 10
                # convert c from integer to the corresponding width/height
                width = (self.xFloats[xmax-1]-self.xFloats[xmin-1]) / c
                height = (self.yFloats[ymax-1]-self.yFloats[ymin-1]) / c
                self.c = max( width, height )

            if 'alpha' in partitionParams:
                self.alpha = partitionParams['alpha']
            else:
                self.alpha = 0.05

            if 'nmin' in partitionParams:
                self.nmin = partitionParams['nmin']
            else:
                self.nmin = 5

            if 'numParts' in partitionParams:
                self.numParts = partitionParams['numParts']
            else:
                self.numParts = 2

            # if self.convertToOrd == False:
            #     self.convertToOrd = True

        elif self.meshingScheme == 'clustering':
            if 'clusterAlg' in partitionParams:
                self.clusterAlg = partitionParams['clusterAlg']
            else:
                self.clusterAlg = KMeans

            if 'numClusters' in partitionParams:
                self.numClusters = partitionParams['numClusters']
            else:
                self.numClusters = 5

            if 'weights' in partitionParams:
                self.weights = partitionParams['weights']
            else:
                self.weights = None

            if 'pad' in partitionParams:
                self.pad = partitionParams['pad']
            else:
                self.pad = 0.1

            if 'boxOption' in partitionParams:
                self.boxOption = partitionParams['boxOption']
            else:
                self.boxOption = "boundPoints"

            if 'boxSize' in partitionParams:
                self.boxSize = partitionParams['boxSize']
            else:
                self.boxSize = 2

            # if self.convertToOrd == True:
            #     self.convertToOrd = False

        if 'split' in partitionParams:
            self.split = partitionParams['split']
        else:
            self.split = False

#--------------------------------------------------

# this part tests the adaptive meshing algorithm
if __name__ == "__main__":
    # generate a bivariate Gaussian

    # fix the seed For reproducibility
    np.random.seed(48824)

    # create a bivariate Gaussian
    mu = np.array([0, 0])  # the means
    cov = 0.7 # covariance
    sigma = np.array([[1, cov], [cov, 1]])  # covariance matrix

    # create the multivariate random variable
    nsamples = 100  # number of random samples
    x, y = np.random.multivariate_normal(mu, sigma, nsamples).T


    # # perform ordinal sampling (ranking) transformation
    # xRanked = rankdata(x, method='ordinal')
    # yRanked = rankdata(y, method='ordinal')

    # obtain the adaptive mesh
    numParts = 2

    # get the adaptive partition of the data
    partitionList = Partitions(np.column_stack((x, y)),
                                       meshingScheme = "DV",
                                       partitionParams={'numParts':numParts},
                                       convertToOrd = True)

    # plot the partitions
    partitionList.plot()

    # overlay the data
    plt.plot(x, y, 'r*')

    # add formatting
    plt.axis('tight')
    # show the figure
    plt.show()
