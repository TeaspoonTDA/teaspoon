import yappi
import time
import numpy as np
import numpy as np
from scipy import special
import collections
from joblib import Parallel, delayed


def norm_cdf(x):
    return special.erfc(-x / np.sqrt(2.0)) / 2.0

def persistence(birth, pers, n=1.0):
    return pers ** n

def img_transform(pers_dgm, ps=.2):
    pers_dgm = np.copy(pers_dgm)
    pers_dgm[:, 1] = pers_dgm[:, 1] - pers_dgm[:, 0]
    pixel_size = ps
    birth_range = [min(pers_dgm[:,0]), max(pers_dgm[:,0])]
    pers_range = [min(pers_dgm[:,1]), max(pers_dgm[:,1])]

    width = birth_range[1] - birth_range[0]
    height = pers_range[1] - pers_range[0]
    resolution = (int(width/pixel_size), int(height/pixel_size))
    pers_img = np.zeros(resolution)
    n = pers_dgm.shape[0]


    bpnts = np.linspace(birth_range[0], birth_range[1] + pixel_size,
                                            resolution[0] + 1, endpoint=False, dtype=np.float64)
    ppnts = np.linspace(pers_range[0], pers_range[1] + pixel_size,
                                            resolution[1] + 1, endpoint=False, dtype=np.float64)



    wts = persistence(pers_dgm[:, 0], pers_dgm[:, 1])
    sigma = 1
    sigma = np.array([[sigma, 0.0], [0.0, sigma]], dtype=np.float64)
    sigma = np.sqrt(sigma[0][0])
    for i in range(n):
        ncdf_b = norm_cdf((bpnts - pers_dgm[i, 0]) / sigma)
        ncdf_p = norm_cdf((ppnts - pers_dgm[i, 1]) / sigma)
        curr_img = ncdf_p[None, :] * ncdf_b[:, None]
        pers_img += wts[i]*(curr_img[1:, 1:] - curr_img[:-1, 1:] - curr_img[1:, :-1] + curr_img[:-1, :-1])

    return pers_img


def ensure_iterable(pers_dgms):
    # if first entry of first entry is not iterable, then diagrams is singular and we need to make it a list of diagrams
    try:
        singular = not isinstance(pers_dgms[0][0], collections.Iterable)
    except IndexError:
        singular = False

    if singular:
        pers_dgms = [pers_dgms]
        
    return pers_dgms, singular


def transform(pers_dgms, ps=.2, n_jobs=None):
    if n_jobs is not None:
        parallelize = True
    else:
        parallelize = False
    
    # convert to a list of diagrams if necessary 
    pers_dgms, singular = ensure_iterable(pers_dgms)
    
    if parallelize:
        pers_imgs = Parallel(n_jobs=n_jobs)(delayed(img_transform)(pers_dgm, ps) for pers_dgm in pers_dgms)
    else:
        pers_imgs = [img_transform(pers_dgm, ps) for pers_dgm in pers_dgms]
    
    if singular:
        pers_imgs = pers_imgs[0]
    
    return pers_imgs


num_diagrams = 100
min_pairs = 50
max_pairs = 100

dgms = [np.random.rand(np.random.randint(min_pairs, max_pairs), 2) for _ in range(num_diagrams)]

ps=1
# start_time = time.time()
# transform(dgms, ps)
# print("Execution time in serial: %g sec." % (time.time() - start_time))

yappi.start()
start_time = time.time()
transform(dgms, ps, n_jobs=-2)
print("Execution time: %g sec." % (time.time() - start_time))

yappi.stop()

# retrieve thread stats by their thread id (given by yappi)
threads = yappi.get_thread_stats()
for thread in threads:
    print(
        "Function stats for (%s) (%d)" % (thread.name, thread.id)
    )  # it is the Thread.__class__.__name__
    yappi.get_func_stats(ctx_id=thread.id).print_all()