import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from ripser import ripser


#-------------Circles and Annuli---------------------------------------#
def Circle(N=100, r=1, gamma=None, seed=None):
    """
    Generate :math:`N` points in :math:`\mathbb{R}^2` from the circle centered
    at the origin with radius :math:`r`.

    If `gamma` is not `None`, then we add noise
    using a normal distribution.  Note that this means the resulting
    distribution is not bounded, so your favorite stability theorem doesn't
    immediately apply.

    Parameters:

        N
            Number of points to generate
        r
            Radius of the circle
        gamma
            Standard deviation of the normally distributed noise.
        seed
            Fixes the seed.  Good if we want to replicate results.

    :returns:
        P- An :math:`N \\times 3` numpy array with a point per row.

    """
    np.random.seed(seed)
    theta = np.random.rand(N, 1)
    theta = theta.reshape((N,))
    P = np.zeros([N, 2])

    P[:, 0] = r*np.cos(2*np.pi*theta)
    P[:, 1] = r*np.sin(2*np.pi*theta)

    if gamma is not None:
        # better be a number of some type!
        noise = np.random.normal(0, gamma, size=(N, 2))
        P += noise

    return P


def Sphere(N=100, r=1, noise=0, seed=None):
    """
    Generate :math:`N` points in :math:`\mathbb{R}^3` from the sphere centered
    at the origin with radius :math:`r`.
    If noise is set to a positive number, the points
    can be at distance :math:`r \pm` noise from the origin.

    Parameters:

        N
            Number of points to generate
        r
            Radius of the sphere
        seed
            Value for seed, or `None`.


    :returns:
        P- An :math:`N \\times 3` numpy array with a point per row.

    """
    np.random.seed(seed)

    Rvect = 2*noise*np.random.random(N) + r
    thetaVect = np.pi * np.random.random(N)
    phiVect = 2 * np.pi * np.random.random(N)

    P = np.zeros((N, 3))
    P[:, 0] = Rvect * np.sin(thetaVect) * np.cos(phiVect)
    P[:, 1] = Rvect * np.sin(thetaVect) * np.sin(phiVect)
    P[:, 2] = Rvect * np.cos(thetaVect)

    return P


def Annulus(N=200, r=1, R=2, seed=None):
    '''
    Returns point cloud sampled from uniform distribution on
    annulus in :math:`\mathbb{R}^2` of inner radius `r` and outer radius `R`

    Parameters:

        N
            Number of points to generate
        r
            Inner radius of the annulus
        R
            Outer radius of the annulus
        seed
            Fixes the seed.  Good if we want to replicate results.


    :returns:
        P - An :math:`N \\times 2` numpy array with one point per row.

    '''
    np.random.seed(seed)
    P = np.random.uniform(-R, R, [2*N, 2])
    S = P[:, 0]**2 + P[:, 1]**2
    P = P[np.logical_and(S >= r**2, S <= R**2)]
    # print np.shape(P)

    while P.shape[0] < N:
        Q = np.random.uniform(-R, R, [2*N, 2])
        S = Q[:, 0]**2 + Q[:, 1]**2
        Q = Q[np.logical_and(S >= r**2, S <= R**2)]
        P = np.append(P, Q, 0)
        # print np.shape(P)

    return P[:N, :]


#-------------Torus a la Diaconis paper--------------------------------#

def Torus(N=100, r=1, R=2,  seed=None):
    '''
    Sampling method taken from Sampling from a Manifold by Diaconis,
    Holmes and Shahshahani, arXiv:1206.6913.

    Generates torus with points

    .. math::
        x = ( R + r \cos(\\theta) )  \cos(\psi),
    .. math::
        y = ( R + r\cos(\\theta) ) \sin(\psi),
    .. math::
        z = r \sin(\\theta)

    Need to draw :math:`\\theta` with distribution

    .. math::
        g(\\theta) = (1+ r \cos(\\theta)/R ) / (2\pi)

    on :math:`0 \leq \\theta < 2\pi`, and :math:`\psi` with uniform density on :math:`[0,2\pi)`. Draw :math:`\\theta` uniformly from :math:`[0,2\pi)` and :math:`\eta` from :math:`[1-r/R,1+r/R]`.  If :math:`\eta< 1 + (r/R) \cos(\\theta)`, return :math:`\\theta`.

    Parameters:

        N
            Number of points to generate
        r
            Inner radius of the torus
        R
            Outer radius of the torus
        seed
            Value for seed, or `None`.

    :returns:
        P - An :math:`N \\times 3` numpy array with one point per row.

    '''

    np.random.seed(seed)
    psi = np.random.rand(N, 1)
    psi = 2*np.pi*psi

    outputTheta = []
    while len(outputTheta) < N:
        theta = np.random.rand(2*N, 1)
        theta = 2*np.pi*theta

        eta = np.random.rand(2*N, 1)
        eta = eta / np.pi

        fx = (1 + r/float(R)*np.cos(theta)) / (2*np.pi)

        outputTheta = theta[eta < fx]

    theta = outputTheta[:N]
    theta = theta.reshape(N, 1)

    x = (R + r*np.cos(theta)) * np.cos(psi)
    y = (R + r*np.cos(theta)) * np.sin(psi)
    z = r * np.sin(theta)
    x = x.reshape((N,))
    y = y.reshape((N,))
    z = z.reshape((N,))

    P = np.zeros([N, 3])
    P[:, 0] = x
    P[:, 1] = y
    P[:, 2] = z

    return P


#----------------------------------------------------------------------#

def Cube(N=100, diam=1, dim=2, seed=None):
    """
    Generate `N` points in :math:`\mathbb{R}^{dim}` from the box
    :math:`[0,diam]\\times[0,diam]\\times ...\\times [0,diam]`

    Parameters:

        N
            Number of points to generate
        diam
            Points are pulled from the box :math:`[0,diam]^{dim}`
        dim
            Points are embedded in :math:`\mathbb{R}^{dim}`

    :returns:
        P - An :math:`N \\times dim` numpy array with a point per row.

    """
    np.random.seed(seed)

    P = diam*np.random.random((N, dim))

    return P


#----------------------------------------------------------------------#

def Clusters(N=100,
             centers=np.array(((0, 0), (3, 3))),
             sd=1,
             seed=None):
    """
    Generate `k` clusters of points in :math:`\mathbb{R}^d`, `N` points in total, approximately evenly divided.
    The centers are given as a :math:`k \\times d` numpy array, where `centers[i,:]` is the center of the ith cluster in :math:`\mathbb{R}^d`.
    Points are drawn from a normal distribution all with the same standard deviation `sd`.

    Parameters:

         N
            Number of points to be generated
         centers
            :math:`k \\times d` numpy array, where `centers[i,:]` is the center of
            the ith cluster in :math:`\mathbb{R}^d`.
         sd
            Standard deviation of clusters.

         seed
            Fixed value for the seed, or `None`.

    :returns:

        P - An :math:`N \\times d` numpy array with a point per row.

    """

    np.random.seed(seed)

    # Dimension for embedding
    d = np.shape(centers)[1]

    # Identity matrix for covariance
    I = sd * np.eye(d)

    # Number of clusters
    k = np.shape(centers)[0]

    ptsPerCluster = N//k
    ptsForLastCluster = N//k + N % k

    for i in range(k):
        if i == k-1:
            newPts = np.random.multivariate_normal(
                centers[i, :], I, ptsForLastCluster)
        else:
            newPts = np.random.multivariate_normal(
                centers[i, :], I, ptsPerCluster)

        if i == 0:
            P = newPts[:]
        else:
            P = np.concatenate([P, newPts])

    return P


#----------------------------------------------------------------------#
def GaussianField(m, n, a=0.8, b=100):
    '''
    Returns matrix representing the 2D gaussian field made on an m x n grid

    Parameters:

        m
            Integer for length of grid in x
        n
            Integer for length of grid in y
        a
            Smoothness of the gaussian field: Positive real number less than 1.0
        b
            Spatial scaling parameter: Positive real number

    :returns:
        Z
            A 2D numpy array

    '''

    import numpy as np

    if a <= 0:
        print("Smoothness must be greater than 0 and <= 1.")
        exit()
    elif a > 1:
        print("Smoothness must be greater than 0 and <= 1.")
        exit()

    if b <= 0:
        print("Scaling must be positive.")
        exit()

    def cov_fun(x): return np.exp((np.sqrt(x[0]**2 + x[1]**2)**(2 * a))/(-b))

    tx = np.arange(0, m, 1)
    ty = np.arange(0, n, 1)

    rows = np.zeros((m, n))
    cols = rows

    for i in range(0, n):
        for j in range(0, m):
            rows[j, i] = cov_fun([tx[j]-tx[0], ty[i]-ty[0]])
            cols[j, i] = cov_fun([tx[j]-tx[0], ty[i]-ty[0]])

    """diff_tx = tx[:, np.newaxis] - tx[0]
    diff_ty = ty[:, np.newaxis] - ty[0]
    cov_values = cov_fun([diff_tx, diff_ty])
    rows.T[:, :] = cov_values
    cols.T[:, :] = cov_values"""

    second = cols[:, ::-1]
    second = second[:, :-1]
    third = cols[::-1, :]
    third = third[:-1, :]
    fourth = rows[::-1, ::-1]
    fourth = fourth[:-1, :-1]

    top = np.concatenate((rows, second), axis=1)
    bottom = np.concatenate((third, fourth), axis=1)
    BlkCirc_row = np.concatenate((top, bottom), axis=0)

    lam = np.fft.fft2(BlkCirc_row, axes=(0, 0)).real / ((2*m - 1)*(2*n - 1))

    if len(lam[np.where(lam < 0)]) != 0:
        if abs(np.min(lam[np.where(lam < 0)])) > 10**(-15):
            print('Could not find positive definite embedding!')
            exit()
        else:
            lam[np.where(lam < 0)] = 0
    lam = np.sqrt(lam)

    comp = 0.00000001*np.multiply(np.random.randn(2*m-1, 2*n-1),
                                  complex(np.random.randn(1), np.random.randn(1)))

    F = np.fft.fft2(np.multiply(lam, comp))
    F = F[0:m-1, 0:n-1]
    field1 = F.real
    field2 = F.imag

    return field1


#----------------------------------------------------------------------#
def Gaussians(centers, variances, amplitudes=None, resolution=200):
    '''
    Returns matrix representing the 2D gaussians made with given centers, variances and amplitudes

    Parameters:

        centers
            Numpy array of lists as [center x, center y] for each required peak
        variances
            Numpy array of variances for each required peak
        amplitudes
            Numpy array of amplitudes for each peak. If length is less than provided centers/variances, amplitude will be considered 1.0
        resolution
            Number of points used to generate meshgrid in both x and y

    :returns:
        Z
            A 2D numpy array

    '''

    import numpy as np

    if len(centers) != len(variances):
        if len(centers) > len(variances):
            print("Please specify all centers or reduce the supplied variances.")
            exit()
        if len(centers) < len(variances):
            print("Please specify all variances or reduce the supplied centers.")
            exit()

    x = np.linspace(min(centers[:, 0]) - 5 * max(variances), max(
        centers[:, 0]) + 5 * max(variances), resolution, endpoint=True)
    y = np.linspace(min(centers[:, 1]) - 5 * max(variances), max(
        centers[:, 1]) + 5 * max(variances), resolution, endpoint=True)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    if amplitudes is None:
        amplitudes = np.ones(len(variances))
    elif len(amplitudes) > len(centers):
        print("Please specify all centers/variances or reduce the supplied amplitudes.")
        exit()
    elif len(amplitudes) < len(centers):
        amplitudes = np.concatenate(
            (amplitudes, np.ones(len(centers) - len(amplitudes))), axis=0)

    for center, variance, amplitude in zip(centers, variances, amplitudes):
        Z += amplitude * \
            np.exp(-((X - center[0]) ** 2 + (Y - center[1])
                   ** 2) / (2 * variance ** 2))

    return Z


#----------------------------------------------------------------------#
def Sinc(x1=-10, x2=10, y1=-10, y2=10, N1=1000, N2=1000, mu=0, sigma=0.01, seed=None):
    '''
    Returns matrix representing the 2D sinc function on grid made on x1, x2, y1, y2 of N1 and N2 length

    Parameters:

        x1
            Left x boundary
        x2
            Right x boundary
        y1
            Left y boundary
        y2
            Right y boundary
        N1
            Discretization of x
        N2
            Discretization of y
        mu 
            Mean of normal noise added
        sigma
            Variance of normal noise added


    :returns:
        f
            A numpy array with shape (N1, N2)

    '''

    import numpy as np

    np.random.seed(seed)

    x, y = np.linspace(x1, x2, N1, endpoint=True), np.linspace(
        y1, y2, N2, endpoint=True)
    X, Y = np.meshgrid(x, y)

    f = np.sinc(np.hypot(X, Y)) + np.random.normal(mu, sigma, size=X.shape)

    return f

#----------------------------------------------------------------------#


#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
#----------------Sets of data for ML-----------------------------------#
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#------------Normally distributed points in (birth,death) plane--------#
#----------------------------------------------------------------------#
def normalDiagram(N=20, mu=(2, 4), sd=1, seed=None):
    """
    Generates a diagram with points drawn from a normal distribution in the persistence diagram plane.
    Pulls `N` points from a normal distribution with mean `\mu` and standard deviation `sd`, then discards any points that are below the diagonal.  Note, however, that this does not get rid of negative birth times.

    Parameters:

     N
        Original number of points drawn for the persistence diagram.
     mu, sd
        Mean and standard deviation of the normal distribution used to generate the points.
     seed
        Used to fix the seed if passed an integer; otherwise should be `None`.

    :returns:
        A persistence diagram given as a numpy array of size :math:`K \\times 2`.

    """

    np.random.seed(seed)
    dgm = np.zeros((N, 2))
    dgm[:, 0] = np.random.normal(mu[0], sd, N).T
    dgm[:, 1] = np.random.normal(mu[1], sd, N).T

    # Get rid of points below the diagonal
    good = np.where(dgm[:, 1]-dgm[:, 0] > 0)[0]

    dgm = dgm[good, :]

    # Get rid of points with negative birth times
    good = np.where(dgm[:, 0] > 0)
    dgm = dgm[good, :]

    dgm = dgm[0, :, :]

    return dgm


def testSetClassification(N=20,
                          numDgms=(10, 10),
                          muRed=(1, 3),
                          muBlue=(2, 5),
                          sd=1,
                          permute=True,
                          seed=None):
    '''
    Generate a collection of diagrams using the normalDiagram() function for classification tests.

    Parameters:

        N
            The number of initial diagrams pulled to create each diagram.  Diagrams could end up with fewer than `N` pts as the pts drawn below the diagonal will be discarded. See normalDiagram() for more information.
        numDgms
            The number of diagrams for the collection.  Can either be an integer, in which case `numDgms` is the number of diagrams of *each type* that are generated, thus returning a data set with `2*numDgms` diagrams.  Alternatively, `numDgms` can be passed as a length two list `(n,m)` where `n` diagrams of the first type and `m` diagrams of the second type are drawn, for a total of `n+m` diagrams.
        muRed, muBlue
            The means used for the normal distribution in normalDiagram() for the two different types.
        sd
            The standard deviation for the normal distribution used for normalDiagram().
        permute
            If ``permute=True``, the data frame returned has its rows randomly permuted.  If `False`, the rows will be red type followed by blue type.
        seed
            Used to fix the seed if passed an integer; otherwise should be `None`.

    :returns:
        A pandas dataframe with columns ``['Dgm', 'mean', 'sd', 'trainingLabel']``. In this case, the entry in `trainingLabel` is -1 if the diagram was drawn from the red type, and 1 if drawn from the blue type.
    '''

    if type(numDgms) == int:
        numDgms = (numDgms, numDgms)

    columns = ['Dgm', 'mean', 'sd', 'trainingLabel']
    index = list(range(sum(numDgms)))
    DgmsDF = pd.DataFrame(columns=columns, index=index, dtype=object)

    counter = 0

    for i in range(numDgms[0]):
        if not seed is None:
            seed += 1
        dgm = normalDiagram(N=N, mu=muRed, sd=sd, seed=seed)
        data = {'Dgm': dgm, 'mean': muRed, 'sd': sd, 'trainingLabel': -1}
        DgmsDF.loc[counter] = data
        counter += 1

    for j in range(numDgms[1]):
        if not seed is None:
            seed += 1
        dgm = normalDiagram(N=N, mu=muBlue, sd=sd, seed=seed)
        data = {'Dgm': dgm, 'mean': muRed, 'sd': sd, 'trainingLabel': 1}
        DgmsDF.loc[counter] = data
        counter += 1

    # Permute the data
    if permute:
        DgmsDF = DgmsDF.reindex(np.random.permutation(DgmsDF.index))

    return DgmsDF


# ------Experiment testing regression----------------------------
# ---------------LINEAR---------------------------------------
# -----------------------------------------------

def testSetRegressionLine(N=20,
                          numDgms=40,
                          muStart=(1, 3),
                          muEnd=(2, 5),
                          sd=1,
                          permute=True,
                          seed=None):
    '''
    Generate a collection of diagrams with means distributed along a line using the normalDiagram() function for regression tests.

    Parameters:

     N
        The number of initial points pulled to create each diagram.  Diagrams could end up with fewer than `N` pts as the pts drawn below the diagonal will be discarded. See normalDiagram() for more information.
     numDgms
        The number of diagrams for the collection given as an integer.
     muStart, muEnd
        The means used for the normal distribution in normalDiagram() are evenly spread along the line segment spanned by `muStart` and `muEnd`.
     sd
        The standard deviation for the normal distribution used for normalDiagram().
     permute
        If ``permute=True``, the data frame returned has its rows randomly permuted.  If `False`, the rows will be be sorted by the location of the means.
     seed
        Used to fix the seed if passed an integer; otherwise should be `None`.

    :returns:
        A pandas dataframe with columns ``['Dgm', 'mean', 'sd', 'trainingLabel']``.  In this case, `trainingLabel` is the distance from the mean used for that persistence diagram to `muStart`.

    '''

    columns = ['Dgm', 'mean', 'sd', 'trainingLabel']
    index = range(numDgms)
    DgmsDF = pd.DataFrame(columns=columns, index=index)

    t = np.random.random((numDgms, 1))
    centers = np.array((muStart))*t + np.array((muEnd))*(1-t)

    for i in index:
        if not seed is None:
            seed += 1
        mu = centers[i, :]
        dgm = normalDiagram(N=N, mu=mu, sd=sd, seed=seed)
        distToStart = euclidean(muStart, mu)
        data = {'Dgm': dgm, 'mean': mu, 'sd': sd, 'trainingLabel': distToStart}
        DgmsDF.loc[counter] = data

    # Permute the data
    if permute:
        DgmsDF = DgmsDF.reindex(np.random.permutation(DgmsDF.index))

    return DgmsDF


# ------Experiment testing regression----------------------------
# ---------------2D-ball around center-----------------------
# -----------------------------------------------

def testSetRegressionBall(N=20,
                          numDgms=40,
                          muCenter=(1, 3),
                          sd=1,
                          permute=True,
                          seed=None):
    '''
    Generate a collection of diagrams with means distributed normally using the normalDiagram() function; used for regression tests.

    Parameters:

         N
            The number of initial diagrams pulled to create each diagram.  Diagrams could end up with fewer than `N` pts as the pts drawn below the diagonal will be discarded. See normalDiagram() for more information.
         numDgms
            The number of diagrams for the collection given as an integer.
         muCenter
            The means used for the normal distribution in normalDiagram() are drawn from the normal distribution with mean `muCenter`.
         sd
            The standard deviation for the normal distribution used for normalDiagram(), as well as for the standard deviation passed to normalDiagram().
         permute
            If ``permute=True``, the data frame returned has its rows randomly permuted.  If `False`, the rows will be be sorted by the location of the means.
         seed
            Used to fix the seed if passed an integer; otherwise should be `None`.

    :returns:
        A pandas dataframe with columns ``['Dgm', 'mean', 'sd', 'trainingLabel']``.  In this case, `trainingLabel` is the distance from the mean used for that persistence diagram to `muCenter`.

    '''

    columns = ['Dgm', 'mean', 'sd', 'trainingLabel']
    index = range(numDgms)
    DgmsDF = pd.DataFrame(columns=columns, index=index)

    centers = np.random.normal(loc=muCenter, scale=sd, size=(numDgms, 2))

    for i in index:
        if not seed is None:
            seed += 1
        mu = centers[i, :]
        dgm = normalDiagram(N=N, mu=mu, sd=sd, seed=seed)
        distToStart = euclidean(muCenter, mu)
        data = {'Dgm': dgm, 'mean': mu, 'sd': sd, 'trainingLabel': distToStart}
        DgmsDF.loc[counter] = data

    # Permute the data
    if permute:
        DgmsDF = DgmsDF.reindex(np.random.permutation(DgmsDF.index))

    return DgmsDF


#------------------------------------------------------------#

def testSetManifolds(numDgms=50,
                     numPts=300,
                     permute=True,
                     seed=None,
                     verbose=False
                     ):
    '''
    Generates a collection of diagrams from different underlying topological spaces.  This set is useful for testing classification tasks.

    The types of underlying spaces with their entry in the `trainingLabel` column is as follows. Each function uses the default values (except for the number of points) unless otherwise noted.
        - **Torus**: A torus embedded in :math:`\mathbb{R}^3` using the function Torus().
        - **Annulus**: An annulus generated with default inputs of Annulus().
        - **Cube**: Points drawn uniformly from the cube :math:`[0,1]^3 \subset \mathbb{R}^3` using the function Cube().
        - **3Cluster**: Points are drawn using Clusters() with centers `[0,0], [0,1.5], [1.5,0]` with `sd = 0.05`.
        - **3Clusters of 3Clusters**: Points are drawn with 9 different centers, which can be loosely grouped into three groups of three; again uses Clusters() with `sd = 0.05`. The centers are `[0,0], [0,1.5], [1.5,0]`; this set rotated 45 degrees and shifted up by 4; and the first set shifted right 3 and up 4.
        - **Sphere**: Points drawn from a sphere using Sphere() with `noise = .05`.

    Parameters:

         numDgms
            The number of diagrams generated of each type. The resulting dataset will have `6*numDgms` diagrams.
         numPts
            The number of points in each point cloud.
         permute
            If ``permute=True``, the data frame returned has its rows randomly permuted.  If `False`, the rows will be red type followed by blue type.
         seed
            Used to fix the seed if passed an integer; otherwise should be `None`.

    :returns:
        A pandas DataFrame with columns ``['Dgm0', 'Dgm1', 'trainingLabel']``.  The `trainingLabel` row has entries with labels given as the boldface above.

    '''

    columns = ['Dgm0', 'Dgm1', 'trainingLabel']
    index = range(6*numDgms)
    DgmsDF = pd.DataFrame(columns=columns, index=index)

    counter = 0

    if type(seed) == int:
        fixSeed = True
    else:
        fixSeed = False

    # -
    if verbose:
        print('Generating torus clouds...')
    for i in range(numDgms):
        if fixSeed:
            seed += 1
        dgmOut = ripser(Torus(N=numPts, seed=seed))[
            'dgms']  # using ripser package
        data = {'Dgm0': dgmOut[0], 'Dgm1': dgmOut[1], 'trainingLabel': 'Torus'}
        DgmsDF.loc[counter] = data
        counter += 1

    # -
    if verbose:
        print('Generating annuli clouds...')
    for i in range(numDgms):
        if fixSeed:
            seed += 1
        dgmOut = ripser(Annulus(N=numPts, seed=seed))['dgms']
        data = {'Dgm0': dgmOut[0], 'Dgm1': dgmOut[1],
                'trainingLabel': 'Annulus'}
        DgmsDF.loc[counter] = data
        counter += 1

    # -
    if verbose:
        print('Generating cube clouds...')
    for i in range(numDgms):
        if fixSeed:
            seed += 1
        dgmOut = ripser(Cube(N=numPts, seed=seed))['dgms']
        data = {'Dgm0': dgmOut[0], 'Dgm1': dgmOut[1], 'trainingLabel': 'Cube'}
        DgmsDF.loc[counter] = data
        counter += 1

    # -
    if verbose:
        print('Generating three cluster clouds...')
    # Centered at (0,0), (0,5), and (5,0) with sd =1
    # Then scaled by .3 to make birth/death times closer to the other examples
    centers = np.array([[0, 0], [0, 2], [2, 0]])
    # centers = np.array( [ [0,0], [0,2], [2,0]  ])
    for i in range(numDgms):
        if fixSeed:
            seed += 1
        dgmOut = ripser(Clusters(centers=centers, N=numPts,
                                 seed=seed, sd=.05))['dgms']
        data = {'Dgm0': dgmOut[0], 'Dgm1': dgmOut[1],
                'trainingLabel': '3Cluster'}
        DgmsDF.loc[counter] = data
        counter += 1

    # -
    if verbose:
        print('Generating three clusters of three clusters clouds...')

    centers = np.array([[0, 0], [0, 1.5], [1.5, 0]])
    theta = np.pi/4
    centersUp = np.dot(centers, np.array(
        [(np.sin(theta), np.cos(theta)), (np.cos(theta), -np.sin(theta))])) + [0, 5]
    centersUpRight = centers + [3, 5]
    centers = np.concatenate((centers,  centersUp, centersUpRight))
    for i in range(numDgms):
        if fixSeed:
            seed += 1
        dgmOut = ripser(Clusters(centers=centers,
                                 N=numPts,
                                 sd=.05,
                                 seed=seed))['dgms']
        # Dgms.append([dgmOut[0],dgmOut[1]])
        data = {'Dgm0': dgmOut[0], 'Dgm1': dgmOut[1],
                'trainingLabel': '3Clusters of 3Clusters'}
        DgmsDF.loc[counter] = data
        counter += 1

    # -

    if verbose:
        print('Generating sphere clouds...')

    for i in range(numDgms):
        if fixSeed:
            seed += 1
        dgmOut = ripser(Sphere(N=numPts, noise=.05, seed=seed))['dgms']
        data = {'Dgm0': dgmOut[0], 'Dgm1': dgmOut[1],
                'trainingLabel': 'Sphere'}
        DgmsDF.loc[counter] = data
        counter += 1

    if verbose:
        print('Finished generating clouds and computing persistence.\n')

    # Permute the diagrams if necessary.
    if permute:
        DgmsDF = DgmsDF.reindex(np.random.permutation(DgmsDF.index))

    return DgmsDF
