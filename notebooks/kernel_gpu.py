@guvectorize(["void(int64[:,:,:], int64[:,:], float64, float64[:])",],"(p,m,n), (m,n), ()->(p)", target='cuda', nopython=True)
def kernel_gpu(dgms0, dgms1, s, result):
    """

    This function computes (using gpu) the multiscale heat kernel distance for an array of persistence diagrams based on the formula provided in Ref. :cite:`5 <Reininghaus2015>`.
    There are three inputs and these are two persistence diagram arrays and the kernel scale sigma.  This function should be used for large size datasets.  Please note this function will not work if you do not have a compatible GPU.
    Details for compatibility can be found `here <https://numba.readthedocs.io/en/stable/cuda/index.html>`_.

    Parameters
    ----------
    dgms0 : ndarray
        Object array that includes first persistence diagram set.
    dgms1 : ndarray
        Object array that includes second persistence diagram set.
    sigma : float
        Kernel scale.

    Returns
    -------
    result : np array
        The kernel matrix

    """
    n_train = len(dgms0)
    for i in range(n_train):
        dgm0=dgms0[i]
        dgm1 = dgms1
        kSigma0 = 0
        kSigma1 = 0
        kSigma2 = 0
        sigma = s
        for k in range(dgm0.shape[0]):
            p = dgm0[k,0:2]
            if p[0]+p[1]==-2:
                continue
            for l in range(dgm0.shape[0]):
                q = dgm0[l,0:2]
                if q[0]+q[1]==-2:
                    continue
                qc = dgm0[l, 1::-1]
                pq = (p[0] - q[0])**2 + (p[1] - q[1])**2
                pqc = (p[0] - qc[0])**2 + (p[1] - qc[1])**2
                kSigma0 += math.exp(-( pq) / (8 * sigma)) - math.exp(-(pqc) / (8 * sigma))
        for k in range(dgm1.shape[0]):
            p = dgm1[k,0:2]
            if p[0]+p[1]==-2:
                continue
            for l in range(dgm1.shape[0]):
                q = dgm1[l,0:2]
                if q[0]+q[1]==-2:
                    continue
                qc = dgm1[l, 1::-1]
                pq = (p[0] - q[0])**2 + (p[1] - q[1])**2
                pqc = (p[0] - qc[0])**2 + (p[1] - qc[1])**2
                kSigma1 += math.exp(-( pq) / (8 * sigma)) - math.exp(-(pqc) / (8 * sigma))
        for k in range(dgm0.shape[0]):
            p = dgm0[k,0:2]
            if p[0]+p[1]==-2:
                continue
            for l in range(dgm1.shape[0]):
                q = dgm1[l,0:2]
                if q[0]+q[1]==-2:
                    continue
                qc = dgm1[l, 1::-1]
                pq = (p[0] - q[0])**2 + (p[1] - q[1])**2
                pqc = (p[0] - qc[0])**2 + (p[1] - qc[1])**2
                kSigma2 += math.exp(-( pq) / (8 * sigma)) - math.exp(-(pqc) / (8 * sigma))

        kSigma0 = kSigma0/(8 * np.pi * sigma)
        kSigma1 = kSigma1/(8 * np.pi * sigma)
        kSigma2 = kSigma2/(8 * np.pi * sigma)
        result[i] = math.sqrt(kSigma1 + kSigma0-2*kSigma2)