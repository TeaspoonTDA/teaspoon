
# '''
# Machine learning featurization method

# If you make use of this code, please cite the following paper:<br/>
# J.A. Perea, E. Munch, and F. Khasawneh.  "Approximating Continuous Functions On Persistence Diagrams." Preprint, 2017.
# '''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import PersistenceImages.persistence_images as pimg
import math
from math import pi
from numpy.linalg import norm as lnorm
from sympy.abc import t
from sympy import Piecewise
from sympy import diff,integrate
from termcolor import colored
from itertools import combinations

# -------------------------------------------- #
# -------------------------------------------- #
# ----------Featurization--------------------- #
# -------------------------------------------- #
# -------------------------------------------- #

"""
.. module::feature_functions
"""

def tent(Dgm, params, dgm_type='BirthDeath'):
    '''
    Applies the tent function to a diagram.

    Parameters:

    :Parameter Dgm:
     	A persistence diagram, given as a :math:`K \times 2` numpy array
    :Parameter params:
     	An tents.ParameterBucket object.  Really, we need d, delta, and epsilon from that.
    :Parameter type:
    	This code accepts diagrams either
            1. in (birth, death) coordinates, in which case `type = 'BirthDeath'`, or
            2. in (birth, lifetime) = (birth, death-birth) coordinates, in which case `type = 'BirthLifetime'`

    :returns:

    :math:`\sum_{x,y \in \text{Dgm}}g_{i,j}(x,y)` where

    .. math::

        \\bigg| 1- \\max\\left\\{ \\left|\\frac{x}{\\delta} - i\\right|, \\left|\\frac{y-x}{\\delta} - j\\right|\\right\\} \\bigg|_+]

    where :math:`| * |_+` is positive part; equivalently, min of * and 0.

.. note:: This code does not take care of the maxPower polynomial stuff.  The build_G() function does it after all the rows have been calculated.

    '''
    d = params.d

    delta = params.delta
    epsilon = params.epsilon
    # print(Dgm[:3])
    # Move to birth,lifetime plane
    if dgm_type == 'BirthDeath':
        T = np.array(((1, -1), (0, 1)))
        A = np.dot(Dgm, T)

    elif dgm_type == 'BirthLifetime':

        A = Dgm
    else:
        print('Your choices for type are "BirthDeath" or "BirthLifetime".')
        print('Exiting...')
        return

    I, J = np.meshgrid(range(d + 1), range(1, d + 1))

    Iflat = delta * I.reshape(np.prod(I.shape))
    Jflat = delta * J.reshape(np.prod(I.shape)) + epsilon

    Irepeated = Iflat.repeat(Dgm.shape[0])
    Jrepeated = Jflat.repeat(Dgm.shape[0])

    DgmRepeated = np.tile(A, (len(Iflat), 1))

    BigIJ = np.array((Irepeated, Jrepeated)).T

    B = DgmRepeated - BigIJ
    B = np.abs(B)
    B = np.max(B, axis=1)
    B = delta - B
    B = np.where(B >= 0, B, 0)
    B = B.reshape((Iflat.shape[0], Dgm.shape[0]))
    out = np.sum(B, axis=1)

    out = out.reshape((d, d + 1)).T.flatten()
    out = out / delta

    # TO BE REMOVED.... THIS HAS BEEN MOVED TO build_G()
    # if params.maxPower >1:

    # 	BigOuts = [out]
    # 	# Make 2 using np.triu_indices
    # 	indices = np.array(np.triu_indices(len(out)))
    # 	C = out[indices.T]
    # 	C = np.prod(C,1)
    # 	BigOuts.append(C)
    # 	# Make 3 or above using itertools
    # 	# NOTE: This is incredibly slow and should be improved.
    # 	for i in range(3,params.maxPower + 1):
    # 		C = [a for a in itertools.combinations_with_replacement(out,i)]
    # 		C = np.array(C)
    # 		C = np.prod(C,1)
    # 		BigOuts.append(C)
    # 	# turn all of them into one long vector
    # 	out = np.concatenate(BigOuts)

    return out

def sub2ind(array_shape, rows, cols):
    '''
    Converts subscripts to indices
    '''
    ind = rows * array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    return ind


# convert indices to subscripts
def ind2sub(array_shape, ind):
    '''
    Converts indices to subscripts
    '''
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return rows, cols


# find the quadrature/interpolation weights for common orthognal functions
# define the function blocks
# Chebyshev-Gauss points of the first kind
def quad_cheb1(npts=10):
    '''
    Find the quadrature/interpolation weights for common orthognal functions

    Define the function blocks

    Chebyshev-Gauss points of the first kind
    '''
    # get the Chebyshev nodes of the first kind
    x = np.cos(np.pi * np.arange(0, npts + 1) / npts)

    # get the corresponding quadrature weights
    w = np.pi / (1 + np.repeat(npts, npts + 1))

    return x, w


# Computes the Legendre-Gauss-Lobatto nodes, and their quadrate weights.
# The LGL nodes are the zeros of (1-x**2)*P'_N(x), wher P_N is the nth Legendre
# polynomial
def quad_legendre(npts=10):
    '''
    Computes the Legendre-Gauss-Lobatto nodes, and their quadrate weights.

    The LGL nodes are the zeros of (1-x**2)*P'_N(x), wher P_N is the nth Legendre polynomial
    '''
    # Truncation + 1
    nptsPlusOne = npts + 1
    eps = np.spacing(1)  # find epsilon, the machine precision

    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
    x = np.cos(np.pi * np.arange(0, npts + 1) / npts)

    # Get the Legendre Vandermonde Matrix
    P = np.zeros((nptsPlusOne, nptsPlusOne))

    # Compute P_(N) using the recursion relation
    # Compute its first and second derivatives and
    # update x using the Newton-Raphson method.

    xold = 2

    while np.max(np.abs(x - xold)) > eps:
        xold = x

        P[:, 0] = 1
        P[:, 1] = x

        for k in range(1, npts):
            P[:, k + 1] = ((2 * k + 1) * x * P[:, k] - k * P[:, k - 1]) / (k + 1)

        x = xold - (x * P[:, npts] - P[:, npts - 1]) \
            / (nptsPlusOne * P[:, npts])

    # get the corresponding quadrature weights
    w = 2 / (npts * nptsPlusOne * P[:, npts] ** 2)

    return x, w

'''
Map the inputs to the function block.
You can invoke the desired function block using:
quad_pts_and_weights['name_of_function'](npts)
'''
quad_pts_and_weights = {'cheb1': quad_cheb1,'legendre': quad_legendre}

# map the inputs to the function blocks
# you can invoke the desired function block using:
# quad_pts_and_weights['name_of_function'](npts)
quad_pts_and_weights = {'cheb1': quad_cheb1,
                        'legendre': quad_legendre
                        }


# find the barycentric interplation weights
def bary_weights(x):
    '''
    Find the barycentric interplation weights

    Calculate Barycentric weights for the base points x.

    .. note:: this algorithm may be numerically unstable for high degree

    Polynomials (e.g. 15+). If using linear or Chebyshev spacing of base
    points then explicit formulae should then be used. See Berrut and
    Trefethen, SIAM Review, 2004.
    '''

    # find the length of x
    n = x.size

    # Rescale for numerical stability
    eps = np.spacing(1)  # find epsilon, the machine precision
    x = x * (1 / np.prod(x[x > -1 + eps] + 1)) ** (1 / n)

    # find the weights
    w = np.zeros((1, n))
    for i in np.arange(n):
        w[0, i] = np.prod(x[i] - np.delete(x, i))

    return 1 / w


def bary_diff_matrix(xnew, xbase, w=None):
    '''
    Calculate both the derivative and plain Lagrange interpolation matrix
    using Barycentric formulae from Berrut and Trefethen, SIAM Review, 2004.

    Parameters:

    :Parameter xnew:
        Interpolation points
    :Parameter xbase:
        Base points for interpolation
    :Parameter w:
        Weights calculated for base points (optional)

    '''

    # if w is not set, set it

    if w is None:
        w = bary_weights(xbase)

    # get the length of the base points
    n = xbase.size

    # get the length of the requested points
    nn = xnew.size

    # replicate the weights vector into a matrix
    wex = np.tile(w, (nn, 1))

    # Barycentric Lagrange interpolation (from Berrut and Trefethen, SIAM Review, 2004)
    xdiff = np.tile(xnew[np.newaxis].T, (1, n)) - np.tile(xbase, (nn, 1))

    M = wex / xdiff

    divisor = np.tile(np.sum(M, axis=1)[np.newaxis].T, (1, n))
    divisor[np.isinf(divisor)] = np.inf

    M[np.isinf(M)] = np.inf

    M = M / divisor

    #	M[np.isnan(M)] = 0

    M[xdiff == 0] = 1

    #	# Construct the derivative (Section 9.3 of Berrut and Trefethen)
    #	xdiff2 = xdiff ** 2
    #
    #	frac1 = wex / xdiff
    #	frac1[np.isinf(frac1)] = float("inf")
    #
    #	frac2 = wex / xdiff2
    #
    #	DM = (M * np.tile(np.sum(frac2, axis=1)[np.newaxis].T, (1, n)) - frac2) / np.tile(np.sum(frac1, axis=1)[np.newaxis].T, (1, n))
    #	row, col = np.where(xdiff == 0)
    #
    #
    #	if np.all(row == 0):  # or, row.size == 0:
    #		DM[row, ] = (wex[row, ] / np.tile(w[col].T, (1, n))) / xdiff[row, ]
    #		idx = sub2ind(DM.shape, row, col)
    #		DM[idx] = 0
    #		DM[idx] = -np.sum(DM[row, ], axis=1)
    return M


def interp_polynomial(Dgm, params, dgm_type='BirthDeath'):
    '''
    Extracts the weights on the interpolation mesh using barycentric Lagrange interpolation.

    Parameters:

    :Parameter Dgm:
     	A persistence diagram, given as a :math:`K \times 2` numpy array
    :Parameter params:
     	An tents.ParameterBucket object.  Really, we need d, delta, and epsilon from that.
    :Parameter type:
    	This code accepts diagrams either
    	1. in (birth, death) coordinates, in which case `type = 'BirthDeath'`, or
    	2. in (birth, lifetime) = (birth, death-birth) coordinates, in which case `dgm_type = 'BirthLifetime'`

    :returns:

    interp_weight-
        a matrix with each entry representiting the weight of an interpolation
    	function on the base mesh. This matrix assumes that on a 2D mesh the functions are ordered row-wise.

    '''
    #	jacobi_func = params.jacobi_func

    # check if we asked for a squre mesh or not
    if isinstance(params.d, int):
        nx = params.d
        ny = params.d
    else:
        nx, ny = params.d

    # check if the Dgm is empty. If it is, pass back zeros
    if Dgm.size == 0:
        return  np.zeros((nx + 1) * (ny + 1))

    # Move to birth,lifetime plane
    if dgm_type == 'BirthDeath':
        T = np.array(((1, -1), (0, 1)))
        A = np.dot(Dgm, T)
    elif dgm_type == 'BirthLifetime':
        A = Dgm
    else:
        print('Your choices for type are "BirthDeath" or "BirthLifetime".')
        print('Exiting...')
        return

    # first, get the entries in Dgm that are within each partition
    for partition in params.partitions:
        query_Dgm_pts = getSubset(A, partition)

        # get the number of query points
        num_query_pts = len(query_Dgm_pts)

        # check if the intersection of the Dgm and the partition is empty.
        # If it is, pass back zeros
        if num_query_pts == 0:
            return np.zeros((nx + 1) * (ny + 1))

        # get the query points. xq are the brith times, yq are the death times.
        xq, yq = query_Dgm_pts[:, 0], query_Dgm_pts[:, 1]
        # xq, yq = np.sort(query_Dgm_pts[:, 0]), np.sort(query_Dgm_pts[:, 1])

        # 1) Get the base nodes:
        # get the 1D base nodes in x and y

        xmesh, w = quad_pts_and_weights[params.jacobi_poly](nx)
        ymesh, w = quad_pts_and_weights[params.jacobi_poly](ny)
        xmesh = np.sort(xmesh)
        ymesh = np.sort(ymesh)

        # shift the base mesh points to the interval of interpolation [ax, bx], and
        # [ay, by]
        ax, bx, ay, by = partition['nodes']
        # ax = 5
        # bx = 6
        xmesh = (bx - ax) / 2 * xmesh + (bx + ax) / 2

        # ay = 5
        # by = 6
        ymesh = (by - ay) / 2 * ymesh + (by + ay) / 2

        # define a mesh on the base points
        x_base, y_base = np.meshgrid(xmesh, ymesh, sparse=False, indexing='ij')

        # get the x and y interpolation matrices
        # get the 1D interpolation matrix for x
        x_interp_mat = bary_diff_matrix(xnew=xq, xbase=xmesh)
        x_interp_mat = x_interp_mat.T  # transpose the x-interplation matrix

        # get the 1D interpolation matrix for y
        y_interp_mat = bary_diff_matrix(xnew=yq, xbase=ymesh)

        # replicate each column in the x-interpolation matrix n times
        Gamma = np.repeat(x_interp_mat, ny + 1, axis=1)

        # unravel, then replicate each row in the y-interpolation matrix m times
        y_interp_mat.shape = (1, y_interp_mat.size)
        Phi = np.repeat(y_interp_mat, nx + 1, axis=0)

        # element-wise multiply Gamma and Phi
        Psi = Gamma * Phi

        # split column-wise, then concatenate row-wise
        #    if Psi.size > 0:  # check that Psi is not empty
        Psi = np.concatenate(np.split(Psi, num_query_pts, axis=1), axis=0)

        # now reshape Psi so that each row corresponds to the weights of one query pt
        Psi = np.reshape(Psi, (num_query_pts, -1))

        # get the weights for each interpolation function/base-point
        interp_weights = np.sum(np.abs(Psi), axis=0)

        #    print('I ran the feature function!')
        #    plt.figure(10)
        #    plt.plot(np.abs(interp_weights),'x')
        #    plt.show()

        return np.abs(interp_weights)


# this function returns the points from querSet that are within the baseRecatangle
def getSubset(querySet, baseRectangle):
    # get the rectangel corners
    xmin = baseRectangle['nodes'][0];
    xmax = baseRectangle['nodes'][1];
    ymin = baseRectangle['nodes'][2];
    ymax = baseRectangle['nodes'][3];

    # subset
    return querySet[(querySet[:, 0] >= xmin) &
                    (querySet[:, 0] <= xmax) &
                    (querySet[:, 1] >= ymin) &
                    (querySet[:, 1] <= ymax), :]


def PLandscapes(A,L_number=[]):
    """
    This function computes persistence landscapes for given persistence diagrams.
    Function has two inputs which are persistence diagrams and chosen landscapes function.
    Algorithm computes all persistence landscapes for the persistence diagram.
    The output is an dictionary that includes all landscape functions, total number of landscapes and desired landscapes (if user specify L_number). 
    
    """ 
     # :param ndarray (A): 
     #     Persistence diagram points--(Nx2) matrix.
     
     # :param list (L_number): 
     #     Landscape numbers user wants to compute
    
     # :Returns:

     #     :output:
     #         (dict) Dictionary that includes the outputs
    
     #         * all: (DataFrame) Includes all landscape functions of the persistece diagram
     #         * LN : (int) Total number of landscapes for persistence diagram A
     #         * DesPL: (ndarray) Includes only selected landscapes functions given in L_number

    # sort persistence diagrams with respect to increasing order of birth time
    A=np.array(sorted(A, key = lambda x: x[0]))

    # sort persistence diagrams that has same birth time with respect to descending order of death time
    # group persistence diagrams with respect to same birth times
    compressed_data = np.split(A, np.unique(A[:,0], return_index=1)[1][1:],axis=0)
    compressed_data = np.array(compressed_data)                                 #convert list object to array

    # sorting each birth time arrays with respect to descending order of death time
    N = np.arange(0,len(compressed_data),1)
    compressed_data = [np.array(sorted(compressed_data[i], key = lambda x: x[1],reverse=True)) for i in N]
    A = np.vstack(compressed_data)

    # create p as zero matrix
    p=np.zeros((1,2))
    infnt=float('inf') #define infinity value

    l_n=0
    pl=np.zeros((1,3))
    Land=[]
    A1=[0,0]
    while A1!=[]:
        # initialize L
        L=[]
        # the first term (b,d) form A
        b=A[0,0]
        d=A[0,1]
        # let next point to be p
        length=len(A)
        if length>1:
            p[0,0]=A[1,0]
            p[0,1]=A[1,1]
        else:
            p[0,0]=infnt
            p[0,1]=0
        # add (-inf,0),(b,0),((b+d)/2,(d-b)/2) to L
        L.append([-infnt,0])
        L.append([b,0])
        L.append([0.5*(b+d),0.5*(d-b)])
        # pop the first term (b,d) form A
        A=np.delete(A,(0),axis=0)

        while L[-1]!=[infnt,0]:
            # convert death_time into list to find dmax
            death_t=list(A[:,1])
            if death_t==[]:
                dmax=d
            else:
                dmax=np.amax(death_t)
            # if maximum death time of the remaning terms in A is bigger or equal to d, add those two terms into the L
            if d>=dmax:
                L.append([d,0])
                L.append([infnt,0])
            else:                                    # if maximum death time of the remaning terms in A is not bigger or equal to d
                for index, item in enumerate(death_t):
                    if item > d:                     # find the first item that is bigger than d
                        d_p=item                     # define new d_prime
                        b_p=A[index,0]               # define new b_prime
                        break
                if d_p!=A[-1,1]:                     # assign next elment as p if d_p is not last element of matrix A
                    p[0,0]=A[index+1,0]              # let next point to be p
                    p[0,1]=A[index+1,1]
                else:
                    p[0,0]=infnt
                    p[0,1]=0
                # delete (b_p,d_p) form A
                A=np.delete(A,(index),axis=0)

                if b_p>d:
                    L.append([d,0])                  # add (d,0) to L
                if b_p>=d:
                    L.append([b_p,0])                # add (b_p,0) to L
                else:
                    L.append([(b_p+d)/2,(d-b_p)/2])
                    # push (b_p,d) in A in order of p
                    A=np.insert(A,0, np.array((b_p,d)),axis=0)
                    # sort persistence diagrams with respect to increasing order of birth time
                    A=np.array(sorted(A, key = lambda x: x[0]))
                    # sort persistence diagrams that has same birth time with respect to descending order of death time
                    # group persistence diagrams with respect to same birth times
                    compressed_data=np.split(A, np.unique(A[:,0], return_index=1)[1][1:],axis=0)
                    compressed_data = np.array(compressed_data) #convert list object to array
                    N = np.arange(0,len(compressed_data),1)
                    compressed_data = [np.array(sorted(compressed_data[i], key = lambda x: x[1],reverse=True)) for i in N]
                    A = np.vstack(compressed_data)
                    #find index of point p - use min command if multiple of rows of A matrix has same value
                    index_p=min(((np.where(A[:,0]==b_p)))[0])
                    # assign next elment as p if b_p is not last element of matrix A
                    if (len(A)-1)!=index_p:
                        p[0,0]=A[index_p+1,0]
                        p[0,1]=A[index_p+1,1]
                    else:
                        p[0,0]=infnt
                        p[0,1]=0
                L.append([(b_p+d_p)/2,(d_p-b_p)/2])
                b=b_p                              # b_prime becomes b
                d=d_p                              # d_prime becomes d
        l_n=l_n+1                                  # counter for persistence landscape
        tag=np.full((len(L),1),l_n)                # add tag information to calculated landscape so that np.groupby can be used
        Land=np.append(L,tag,axis=1)
        Land=Land[1:-1]                            # exclude the infinity terms inside the landscape
        pl=np.concatenate((pl,Land),axis=0)        # add tagging information to list of landscapes point
        A1=list(A[:,0])                            # create list of last version of A to be able to check while condition above

    pl=np.delete(pl,0,axis=0)                                  # delete first zero row from pl
    pl=pd.DataFrame(pl)                                        # convert landscape matrix into dataframe
    pl = pl.rename(columns={0: "x",1:"y",2:"Landscape"})       # rename columns
    # make group of points with respect to their landscape number
    comp_pl=pl.groupby(['Landscape']).apply(lambda x: np.transpose(np.array([x.x, x.y])))
    comp_pl = comp_pl.reset_index()
    comp_pl.columns = ['Landscape Number', 'Points']
    PL=comp_pl.iloc[:, 1].values
    Landscape_number=l_n
    
    output={}
    output['all']=comp_pl
    output['LN']=Landscape_number
    # in the case of user wants specific landscape data points
    if len(L_number)!=0 and np.all(np.array(L_number)<Landscape_number):
        L_number[:] = [number -1 for number in L_number]
        output['DesPL']=PL[L_number]

    return output



class PLandscape():

    def __init__(self,PD,L_number=[]):              # if user wants to see the plot and the points of the specific landscapes, L_number variable should be given as input.
        """
        
        This class uses landscapes algorithm (:meth:`PD_Featurization.PLandscapes`) to compute persistence landscapes and plot them based on user preference.
        The algorithm computes the persistence landscapes is written based on Ref. :cite:`1 <Bubenik2017>`.

        :param ndarray (PD):
            Persistence diagram points--(Nx2) matrix.
            
        :param list (L_Number):
            Desired landscape numbers in a list form. If the list is empty, all landscapes will be plotted.
            
        :Returns:

            :PL_number:
                (int) Total number of landscapes for given persistence diagram 
        
            :DesPL:
                (ndarray) Includes only selected landscapes functions given in L_number
            
            :AllPL: 
                (ndarray) Includes all landscape functions of the persistece diagram    
            
        """

        # L_number is a Nx1 matrix that includes the desired numbers of landscapes
        out=PLandscapes(PD,L_number)

        self.PL_number=out['LN']
        if len(L_number)!=0:
        # if user gives landscape number which does not exist:
            for i in range(0,len(L_number)):
                if L_number[i]>out['LN']:
                    print (colored('Warning:', 'red')+' Landscape number {} does not exist. Number of Landscapes are {}. Please enter desired landscape values less than or equal to {}.'.format(L_number,out['LN'],out['LN']))
                break
        # Warning: If user enters a specific landscape number which does not exist for current persistence diagrams,
        # class returns a warning message to user.
            if np.all(np.array(L_number)<out['LN']):
                self.DesPL=out['DesPL']
        else:
            self.DesPL = 'Warning: Desired landscape numbers were not specified.'
        self.AllPL=out['all']


    def PLandscape_plot(self,PL,L_number=[]):
        """
        
        This function plots selected persistence landscapes or it plots all of them if user does not provide desired landscape functions.

        :param ndarray (PD):
            Persistence diagram points--(Nx2) matrix.
            
        :param list (L_Number):
            Desired landscape numbers in a list form. If the list is empty, all landscapes will be plotted.
            
        :Returns:

            :PL_plot:
                (fig) The figure that includes chosen or all landsape functions  
                     
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import rc
        k=len(PL)                                   # number of landscapes
        if L_number==[]:                            # if user does not enter any landscape function, all landscpaes will be plotted
            L_number=np.linspace(1,k,k)
        plt.figure()
        plt.ioff()
        for i in range(0,len(L_number)):            # plotting the landscapes
            index1=int(L_number[i])
            x=PL[index1-1][:,0]
            y=PL[index1-1][:,1]
            plt.plot(x,y,label=r'Landscape %s' % index1 )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig=plt.gcf()
        PL_plot=fig
        return PL_plot






def F_Landscape(PL,params,max_l_number=None):
    """
        
    This function computes the features for selected persistence landscape numbers.
    There are three inputs to the function. 
    These are all landscape functions for each persistence diagram, parameter bucket object and the maximum level of landscape function.
    If user does not specify the third input, algorithm will automatically compute it.
    The second parameter includes the parameters needed to compute features and perform classification.
    Please see :meth:`PD_ParameterBucket.LandscapesParameterBucket` for more details about parameters.
    
    :param ndarray (PL):
        Object array that includes all landscape functions for each persistence diagram
        
    :param object (params):
        Parameterbucket object. We need landscape numbers defined by user to generate feature matrix.
        
    :Returns:

        :feature:
            (Array) NxM matrix that includes the features for each persistence diagram, where N is the number of persistence diagrams and M is the numbers of features which is equal to length of sorted mesh of landscapes functions.
        
        :Sorted_mesh: 
            (list) It includes the sorted mesh for each landscape function chosen by user.
                     
    """
    a=PL
    N=len(a)
    
    # landscape number will be used to generate feature matrix
    PLN=params.PL_Number
    if PLN == None:
        PLN[0]=1
    
    # find mesh for given landscape number
    l_number=[]
    if max_l_number==None:                            # find maximum number of landscapes in whole sets
        [l_number.append(len(a[i])) for i in np.arange(0,N,1)]
        max_l_number=max(l_number)                    # maximum landscape number for given persistence diagram set
    

    Sorted_mesh=[]                                    # this list includes x points of specific landscapes for whole landscapes in increasing order and duplicates are removed
    interp_y=[]
    for j in range(len(PLN)):
        X=[]
        for i in range(0,N):
            k=a[i].iloc[:, 1].values                  # landscapes set
            if len(k)>=PLN[j]:
                m=k[PLN[j]-1][:,0]
                X.append(m)                           # combine whole x values of same landscapes into one vector
        X=np.hstack(X)
        # sorting x values in increasing order
        K=np.array(X)
        Sorted_X=np.array(sorted(K))
        Sorted_X_unique=np.unique(Sorted_X)           # removing duplicates
        Sorted_mesh.append(Sorted_X_unique)

        desPL=PLN[j]
        xvals = np.array(Sorted_mesh[j])              # taking mesh for nth landscapes
        y_interp=np.zeros((len(xvals),N))
        for i in range(0,N):                          # the loop which iterates for whole cases
            if len(a[i])>=PLN[j]:                     # check if the landscape function exist for the current set of landscapes 
                L=a[i].iloc[:, 1].values              # landscapes set
                x=L[desPL-1][:,0]                     # x values of nth landscape
                y=L[desPL-1][:,1]                     # y values of nth landscape
                y_interp[:,i]=np.interp(xvals,x,y)    # piecewise linear interpolation
        interp_y.append(y_interp[0:len(xvals),0:N])

    feature=np.zeros((N,1))
    for i in range(0, len(PLN)):
        Piecewise_Linear= np.array(interp_y[i])
        ftr=Piecewise_Linear.transpose()
        feature=np.concatenate((feature,ftr),axis=1)
    feature=feature[:,1:]
    
    return feature, Sorted_mesh

def F_Image(PD1,PS,var, plot, TF_Learning, D_Img=[], *args):
    """
    
    This function returns user feature matrix generated using persistence images.
    We use `PersistenceImages <https://gitlab.com/csu-tda/PersistenceImages>`_ package to compute persistence images.
    User can also specify a list that includes the persistence diagram number to plot images of them.
    Algorithms returns plot only if plotting option is True and transfer learning is false.
    
    :param ndarray (PD):
        Object array that includes all persistence diagrams
        
    :param float (PS):
        Pixel size. 
        
    :param float (var):
        Variance of the Gaussian distribution. 
    
    :param list (D_Img):
        The number of persistence diagrams in a list. If this parameter is provided, algorithm will only plot the persistence images of these persistence diagrams. 
    
    :param (TF_Learning): If it is true, then algorithm will need the second persistence diagram set and it will compute feature matrices for training set and test set.
    
    :Returns:

        :output:
            (dict) A dictionary that includes the feature matrix. In the case of transfer learning, output will have training and test feature matrices separately.
            
    """
    if TF_Learning: # transfer learning option
    
        PD2=args[0]
        
        #number of diagrams in training set
        N1=len(PD1)
        #number of diagrams in test set 
        N2=len(PD2) 

        persistence_range_train = np.zeros((N1,1))
        birth_range_train = np.zeros((N1,1)) 

        persistence_range_test = np.zeros((N2,1))
        birth_range_test = np.zeros((N2,1))
        
        #find the max persistence and birth times for test set persistence diagrams
        
        for i in range(0,N1):
            per_diag = PD1[i]
            birth_range_train [i,0] = max(per_diag[:,0])
            persistence_range_train [i,0] = max(per_diag[:,1]-per_diag[:,0])

        
        #find the max persistence and birth times for test set persistence diagrams
        
        for i in range(0,N2):
            per_diag = PD2[i]
            birth_range_test [i,0] = max(per_diag[:,0])
            persistence_range_test [i,0] = max(per_diag[:,1]-per_diag[:,0])
        
        # determine the maximum bound for images of test set diagrams
        max_per_lim_test = max(persistence_range_test[:,0])+1
        max_bth_lim_test = max(birth_range_test[:,0])+1
        
        # determine the maximum bound for images of training set diagrams
        max_per_lim_train = max(persistence_range_train[:,0])+1
        max_bth_lim_train = max(birth_range_train[:,0])+1
        
        # determine the bounds of the images based on bounds of training set and test set diagrams
        # adjust the image parameters and compute images
        pers_imager = pimg.PersistenceImager()
        pers_imager.pixel_size = PS
        pers_imager.birth_range = (0,max(max_bth_lim_test,max_bth_lim_train))
        pers_imager.pers_range = (0,max(max_per_lim_test,max_per_lim_train))
        pers_imager.kernel_params = {'sigma': var}
        
        #compute persistence images of training set
        pers_img_train = [pers_imager.transform(PD1[i], skew=True) for i in np.arange(0,N1,1)]  
        
        #compute persistence images of test set
        pers_img_test = [pers_imager.transform(PD2[i], skew=True) for i in np.arange(0,N2,1)] 
        
        #generate feature matrices---------------------------------------------
        
        #training set
        feature_PI_train = np.zeros((N1,len(pers_img_train[0][:,0])*len(pers_img_train[0][0,:])))
        #test set
        feature_PI_test = np.zeros((N2,len(pers_img_test[0][:,0])*len(pers_img_test[0][0,:])))
        
        for i in range(N1):
            feature_PI_train[i,:] = pers_img_train[i].flatten()         
        for i in range(N2):
            feature_PI_test[i,:] = pers_img_test[i].flatten()         
        
        output = {}
        output['F_train'] = feature_PI_train
        output['F_test'] = feature_PI_test
        
    else: 
        output = {}
        # number of persistence diagrams
        N1=len(PD1)
        
        persistence_range = np.zeros((N1,1))
        birth_range = np.zeros((N1,1))
        
        # find a bounds of the image
        for i in range(0,N1):
            per_diag = PD1[i]
            birth_range [i,0] = max(per_diag[:,0])
            persistence_range [i,0] = max(per_diag[:,1]-per_diag[:,0]) 
        
        
        # adjust the image parameters and compute images
        pers_imager = pimg.PersistenceImager()
        pers_imager.pixel_size = PS
        pers_imager.birth_range = (0,max(birth_range[:,0])+1)
        pers_imager.pers_range = (0,max(persistence_range[:,0])+1)
        pers_imager.kernel_params = {'sigma': var}
        pers_img = [pers_imager.transform(PD1[i], skew=True) for i in np.arange(0,N1,1)]
                

        # generate feature matrix
        feature_PI=np.zeros((N1,len(pers_img[0][:,0])*len(pers_img[0][0,:])))
        for i in range(N1):
            feature_PI[i,:] = pers_img[i].flatten() 
     
    
        # plot all images or images of certain persistence diagrams
        if plot==True:
            fig = []
            if D_Img==[]:
                D_Img=np.arange(1,2,1)
            for i in range(len(D_Img)):
                plt.figure()
                pers_imager.plot_image(PD1[D_Img[i]-1], skew=True)
                fig.append(plt.gcf())
            output['figures'] = fig
        
        output['F_Matrix'] = feature_PI
    return output


def F_CCoordinates(PD,FN):
    """
    
    This code generates feature matrix to be used in machine learning applications using Carlsson Coordinates which is composed of five different functions shown in Eq. :eq:`1st_coord` - :eq:`5th_coord`.
    The first four functions are taken from Ref. :cite:`2 <Adcock2016>` and the last one is obtained from Ref. :cite:`3 <Khasawneh2018>`.
    There are two inputs to the function. These are persistence diagrams and number of coordinates that user wants to use in feature matrix.
    Algorithm will return feature matrix object array that includes feature matrices for different combinations, and total number of combinations will be :math:`\sum_{i=1}^{FN} {FN \choose i}`. 
    
    .. math:: f_{1}(PD) = \sum b_{i}(d_{i}-b_{i})
       :label: 1st_coord
    
    .. math:: f_{2}(PD) = \sum (d_{max}-d_{i})(d_{i}-b_{i})
       :label: 2nd_coord
    
    .. math:: f_{3}(PD) = \sum b_{i}^{2}(d_{i}-b_{i})^{4}
       :label: 3rd_coord
    
    .. math:: f_{4}(PD) = \sum (d_{max}-d_{i})^{2}(d_{i}-b_{i})^{4}
       :label: 4th_coord

    .. math:: f_{5}(PD) = \sum max(d_{i}-b_{i})
       :label: 5th_coord
       
    :param ndarray (PD):
        Object array that includes all persistence diagrams
    
    :param float (FN):
        Number of features
        
    :Returns:
        
        :FeatureMatrix:
                (Array) NxFN matrix that includes the features for each persistence diagram, where N is the number of persistence diagrams and FN is the number of feature chosen.
        
        :TotalNumComb: 
                (int) Number of combinations.
        
        :CombList: 
                (list) List of combinations.
    
    """

    N = len(PD) 
    
    # Create combinations for features with respect to user choice
    Combinations = []            # define a list that includes all combinations
    TotalNumComb = 0
    for i in range(0,FN):  
        poss_comb = list(combinations(range(1,FN+1), i+1))
        Combinations.append(poss_comb)
        TotalNumComb = TotalNumComb+len(poss_comb) 
    
    # Generating feature matrix that includes whole features inside of it.
    # Then this matrix will be used to create feature matrix for different combinations of features.
       
    feature = np.zeros((N,5))
    for i in range (0,N):
        PerDgm = PD[i]
        birth_time = PerDgm[:,0]
        death_time = PerDgm[:,1]
        life_time = death_time-birth_time
        max_death_time = max(death_time)
        maxdtminusdt = max_death_time-death_time[:]
        btsquare = np.square(birth_time)
        #features
        feature[i,0] = np.sum(np.multiply(birth_time,life_time))
        feature[i,1] = np.sum(np.multiply(maxdtminusdt,life_time))
        feature[i,2] = np.sum(np.multiply(btsquare,np.power(life_time,4)))
        feature[i,3] = np.sum(np.multiply(np.power(maxdtminusdt,2),np.power(life_time,4)))
        feature[i,4] = max(life_time)
        
    # faeture matrix will be modified depending on how many features user wants
    feature=feature[:,0:FN]
    
    # Create a matrix that includes feature matrix for different combinations
    FeatureMatrix = np.ndarray(shape=(TotalNumComb),dtype=object) 
    
    # Create a matrix that includes whole possible combinations for given number of feature
    CombList=np.zeros((TotalNumComb,5))
    
    increment = 0
    for j in range (0,FN):
        listofCombinations = Combinations[j]              # combinations with n number inside total number of features 
        numberincombination = len (listofCombinations[0]) # number of elements inside of the combination
        numberofcombination = len (listofCombinations)    # number of combinations
           
        # create feature matrices for all combinations
        for i in range (0,numberofcombination):
            featmat = np.zeros((N,numberincombination))   # create an array will take columns from feature matrix with FN=5 with respect to combination
            combfeat = np.array(listofCombinations[i])    # combinations
            CombList[increment,0:len(combfeat)]=combfeat
            for k in range(0, numberincombination):    
                featmat[:,k] = feature[:,combfeat[k]-1]   
            FeatureMatrix[increment] = featmat            # storing feature matrix for each combination in a object type array
            increment = increment + 1                     # increment
        
    return FeatureMatrix,TotalNumComb,CombList


def F_PSignature(PL,L_Number=[]):
    """
    
    This function takes the persistence landscape set and returns the feature matrix which is computed using path signatures :cite:`4 <Chevyrev2016,Chevyrev2020>`.
    Function takes two inputs and these are persistence landcsape set in an object array and the landscape numbers that user wants to compute their signatures.
    
    :param ndarray (PL):
        Object array that includes all landscape functions for each persistence diagram
        
    :param list (L_Number):
        Landscape numbers that user wants to use in feature matrix generation. If this parameter is not specified, algorithm will generate feature matrix using first landscapes.
        
    :Returns:

        :feature_PS:
            (Array) Nx6 matrix that includes the features for each persistence diagram, where N is the number of persistence landscape sets.
        
    
    """
    N = len(PL)
    
    #generate feature matrix
    feature_PS = np.zeros((N,6*len(L_Number)))
    
    if L_Number==[]:
        L_Number=[1]
    
    # the loop that computes the signature for selected landscape functions
    for l in range(0,len(L_Number)): 
        
        #the loop that computes the signature for each case or persistence landscape set
        for i in range(0,N):
            Landscape = PL[i].iloc[L_Number[l]-1].values[1]
            
            #node points of the landscape functions
            x = Landscape[:,0]
            y = Landscape[:,1]
            
            #define first function of path 
            g = Piecewise((0, t < 0), (t, t >= 0))
            for j in range(0,len(x)-1) :
                function_name =  'f_%d_%d_case%d' %(j+1,1,i+1)
                exec("%s = ((y[j+1]-y[j])/(x[j+1]-x[j]))*t+(y[j+1]*(x[j+1]-x[j])+x[j+1]*(y[j]-y[j+1]))/(x[j+1]-x[j])" % (function_name))
            
             
            
            lower_bound = x[0]
            upper_bound = x[-1]
            
            # S(1)
            S_1 = upper_bound - lower_bound
            
            # S(1,1)
            S_1_1 = (np.power(upper_bound,2)/2)+(np.power(lower_bound,2)/2)-(lower_bound*upper_bound)
            
            S_2 = 0
            S_1_2 = 0
            S_2_1 = 0
            S_2_2 = 0
            for j in range(0,len(x)-1):
                
                f_name =  'f_%d_%d_case%d' %(j+1,1,i+1)
                exec('global PL_function; PL_function = Piecewise((0,t<x[j]),(0,t>x[j+1]),(%s ,True))'%(f_name))
                # S(2)
                Signature_2 = integrate(diff(PL_function),(t,x[j],x[j+1]))
                S_2 = S_2 + float(Signature_2)
                
                # S(1,2)
                Signature_1_2 = integrate((t-x[0])*diff(PL_function),(t,x[j],x[j+1]))
                S_1_2 = S_1_2 + float(Signature_1_2)
                
                # S(2,1)
                Signature_2_1 = integrate((t-x[j])*diff(PL_function),(t,x[j],x[j+1]))
                S_2_1 = S_2_1 + float(Signature_2_1)
            
                # S(2,2)
                Signature_2_2 = integrate((t-x[j])*np.power(diff(PL_function),2),(t,x[j],x[j+1]))
                S_2_2 = S_2_2 + float(Signature_2_2)
                
            # storing the signatures in feature matrix
            feature_PS[i,6*l] = S_1 
            feature_PS[i,6*l+1] = S_2
            feature_PS[i,6*l+2] = S_1_1 
            feature_PS[i,6*l+3] = S_1_2
            feature_PS[i,6*l+4] = S_2_1
            feature_PS[i,6*l+5] = S_2_2
    return feature_PS


def KernelMethod(perDgm1, perDgm2, sigma):
    """
    
    This function computes the kernel for given two persistence diagram based on the formula provided in Ref. :cite:`5 <Reininghaus2015>`.
    There are three inputs and these are two persistence diagrams and the kernel scale sigma.

    :param ndarray (perDgm1):
        Object array that includes first persistence diagram set
        
    :param ndarray (perDgm2):
        Object array that includes second persistence diagram set
    
    :param float (sigma):
        Kernel scale
        
    :Returns:

        :Kernel:
            (float) The kernel value for given two persistence diagrams.

    """

    L1=len(perDgm1)
    L2=len(perDgm2)
    kernel = np.zeros((L2,L1))
    
    Kernel=0
    
    for i in range (0,L1):
        p=perDgm1[i]
        p = np.reshape(p,(2,1))
        for j in range(0,L2):
            q = perDgm2[j]
            q = np.reshape(q,(2,1))
            q_bar = np.zeros((2,1))
            q_bar[0] = q[1]
            q_bar[1] = q[0]
            dist1 = lnorm(p-q)
            dist2 = lnorm(p-q_bar)
            kernel[j,i] = np.exp(-(math.pow(dist1,2))/(8*sigma))-np.exp(-(math.pow(dist2,2))/(8*sigma))
            Kernel = Kernel+kernel[j,i]
    Kernel = Kernel*(1/(8*pi*sigma))   
    
    return Kernel