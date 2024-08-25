import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from ripser import ripser
import matplotlib.gridspec as gridspec
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations import DiagramSelector
from sklearn.pipeline import Pipeline
from scipy import stats
from scipy import ndimage as sp
from scipy import integrate

# Set Font
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def lattice_shape(data, plot=False):
    """
        Quantify the lattice shape of a given point cloud of center points.

        Args:
            data (array): (x,y) strike center coordinates.
            plot (bool): Boolean variable to plot the persistence diagram and lifetime histogram

        Returns:
            [varh0, sumh1] -- Array containing the 0D and 1D persistence scores for the given point cloud
    """
    if type(data) != np.ndarray:
        raise TypeError('Data needs to be a numpy array.')
    if np.shape(data) != (len(data), 2):
        raise ValueError(
            'Data needs to have dimension (n,2) where n is the number of center points in the grid.')
    if int(np.sqrt(len(data))) ** 2 != len(data):
        raise ValueError(
            'The number of data points needs to be a square number.')

    # Scale data from [-1,1]
    xmin = np.min(data[:, 0])
    xmax = np.max(data[:, 0])
    ymin = np.min(data[:, 1])
    ymax = np.max(data[:, 1])
    data = np.array(
        [((data[:, 0] - xmin) / (xmax - xmin)) * 2 - 1, ((data[:, 1] - ymin) / (ymax - ymin)) * 2 - 1]).transpose()

    n = np.sqrt(len(data))
    diagrams = ripser(data)['dgms']
    diagram0 = diagrams[0]  # zero dimensional persistence diagram
    diagram1 = diagrams[1]  # one dimensional persistence diagram
    bi0, de0 = np.array(diagram0.T[0]), np.array(diagram0.T[1])
    bi1, de1 = np.array(diagram1.T[0]), np.array(diagram1.T[1])
    bi0 = np.delete(bi0, -1)  # Delete Infinite Class
    de0 = np.delete(de0, -1)
    top = max(max(de0), max(de1))
    # find bounds for persistence diagram using top variable

    li0 = de0 - bi0
    li1 = de1 - bi1

    # Compute normalized measures
    varh0 = np.var(li0)
    norm_varh0 = 4 * varh0
    sumh1 = np.sum(li1)
    norm_sumh1 = (1 / (2 * (np.sqrt(2) - 1) * (n - 1))) * sumh1

    varh0 = norm_varh0
    sumh1 = norm_sumh1

    if plot:
        # Plot Persistence Diagram
        TextSize = 30
        MS = 20  # marker size
        plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 2)
        ax = plt.subplot(gs[0:1, 1:2])
        plt.xticks(np.arange(0, 1, step=0.2), size=TextSize)
        plt.yticks(np.arange(0, 1, step=0.2), size=TextSize)
        plt.xlabel('Birth', size=TextSize)
        plt.ylabel('Death', size=TextSize)
        plt.plot(bi0, de0, 'b.', markersize=MS, label='$H_0$', alpha=0.5)
        plt.plot(bi1, de1, 'r.', markersize=MS, label='$H_1$', alpha=0.5)
        plt.plot([-top, top * 10], [-top, top * 10], 'k--')
        plt.xlim(-0.02, 1.2 * top)
        plt.ylim(-0.02, 1.2 * top)
        plt.title('Persistence Diagram', size=TextSize)
        plt.legend(loc='lower right', fontsize=TextSize - 5, markerscale=1.3)

        # Plot Histograms
        ax = plt.subplot(gs[0:1, 0:1])
        plt.hist(de0, bins='rice', orientation='horizontal',
                 label='$H_0$', color='blue')
        plt.hist(de1, bins='rice', orientation='horizontal',
                 label='$H_1$', color='red')
        plt.xticks(size=TextSize)
        plt.yticks(np.arange(0, 1, step=0.2), size=TextSize)
        plt.xlabel('Count', size=TextSize)
        plt.ylabel('Death', size=TextSize)
        plt.ylim(-0.02, 1.2 * top)
        plt.title(f'Histogram', size=TextSize)
        plt.legend(loc='lower right', fontsize=TextSize - 5, markerscale=0.2)

        plt.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(wspace=0.5)
        plt.tight_layout()
        plt.show()

    return [varh0, sumh1]


def feature_depth(nom_image, exp_image, nfeat, plot=False):
    """
        Quantify the striking depths of a given texture scan.

        Args:
            nom_image (array): 2D image of the nominal surface (scaled between 0-1)
            exp_image (array): 2D image of the experimental surface (scaled between 0-1)
            nfeat (int): Expected number of features in the image
            plot (bool): Variable to plot the persistence diagram and lifetime histograms

        Returns:
            (float): depth_score -- The earth movers distance score as a percentage
    """
    if type(nom_image) != np.ndarray:
        raise TypeError('Nominal image needs to be a numpy array.')
    if type(exp_image) != np.ndarray:
        raise ValueError('Experimental image needs to be a numpy array.')
    nom_image = nom_image.astype(np.float64)
    exp_image = exp_image.astype(np.float64)

    # Compute experimental and nominal 0D persistence diagrams
    print('Computing Nominal Persistence Distribution')
    pd = generate_ph(nom_image, 0)
    pd = pd.astype(np.float64)
    print('Nominal Persistence Computed')

    bi0 = np.array(pd[:, 0])
    de0 = np.array(pd[:, 1])
    li0 = np.array(de0 - bi0)
    nom_li0 = li0

    print('Computing Experimental Persistence Distribution')
    pd = generate_ph(exp_image, 0)
    pd = pd.astype(np.float64)
    print('Experimental Persistence Computed')

    bi0 = np.array(pd[:, 0])
    de0 = np.array(pd[:, 1])
    li0 = np.array(de0 - bi0)
    coords = np.transpose(np.array([bi0, li0]))
    pts = filter_outliers(coords, nfeat)
    bi0 = pts[:, 0]
    exp_li0 = pts[:, 1]

    if plot:
        plt.figure(figsize=(18, 5))
        gs = gridspec.GridSpec(1, 3)

        # Plot nominal distribution
        numbins = 10
        TextSize = 35
        MS = 15  # marker size
        maxli = max(nom_li0)
        top = max(maxli, 1)
        ax = plt.subplot(gs[0:1, 0:1])
        plt.hist(nom_li0, bins=np.linspace(0, 1, num=numbins), orientation='horizontal', color='blue',
                 label='$H_0$', density=True)
        plt.xticks(size=TextSize - 2)
        plt.yticks(
            np.around(np.arange(0, 1.1 * top, step=(top / 5)), 2), size=TextSize - 2)
        plt.ylabel('Lifetime', size=TextSize)
        plt.xlabel('Probability Density', size=TextSize)
        plt.ylim(-0.02, 1.2 * top)
        plt.legend(loc='best', fontsize=TextSize - 8, markerscale=0)
        plt.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(wspace=0.5)
        plt.title(f'Nominal', size=TextSize)

        # Plot experimental distribution
        plt.subplot(gs[0:1, 1:2])
        maxli = max(exp_li0)
        top = max(maxli, 1)
        plt.hist(exp_li0, bins='rice', orientation='horizontal', color='blue',
                 label='$H_0$', density=True)
        plt.xticks(size=TextSize - 2)
        plt.yticks(
            np.around(np.arange(0, 1.1 * top, step=(top / 4)), 2), size=TextSize - 2)
        plt.ylabel('Lifetime', size=TextSize)
        plt.xlabel('Probability Density', size=TextSize)
        plt.ylim(-0.02, 1.2 * top)
        plt.legend(loc='best', fontsize=TextSize - 8, markerscale=0)
        plt.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(wspace=0.5)
        plt.title(f'Experimental', size=TextSize)

        plt.subplot(gs[0:1, 2:3])
        plt.plot(bi0, exp_li0, 'b.', markersize=MS, label='$H_0$', alpha=0.5)
        plt.xlabel('Birth', size=TextSize)
        plt.ylabel('Lifetime', size=TextSize)
        plt.xlim(-0.02, 1.2 * top)
        plt.ylim(-0.02, 1.2 * top)
        plt.xticks(
            np.around(np.arange(0, 1.1 * top, step=(top / 4)), 2), size=TextSize - 2)
        plt.yticks(
            np.around(np.arange(0, 1.1 * top, step=(top / 4)), 2), size=TextSize - 2)
        plt.title(f'Persistence Diagram', size=TextSize)
        plt.legend(loc='best', fontsize=TextSize - 10, markerscale=1.3)
        plt.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(wspace=0.5)
        plt.tight_layout()
        plt.show()
    return np.round((1 - stats.wasserstein_distance(nom_li0, exp_li0)) * 100, 2)


def feature_roundness(nom_image, exp_image, nfeat, width, num_steps=50, plot=False):
    """
        Quantify the strike roundness of a given PVST texture scan.

        Args:
            nom_image (array): 2D image of the nominal surface (scaled between 0-1)
            exp_image (array): 2D image of the experimental surface (scaled between 0-1)
            nfeat (int): Expected number of features in the image
            width (float): image width in millimeters
            num_steps (int): Number of data points for the roundness plot
            plot (bool): Variable to plot the persistence diagram and lifetime histograms

        Returns:
            (float): roundness_score -- The earth movers distance score as a percentage
    """
    if type(nom_image) != np.ndarray:
        raise TypeError('Nominal image needs to be a numpy array.')
    if type(exp_image) != np.ndarray:
        raise ValueError('Experimental image needs to be a numpy array.')
    if type(nfeat) != int:
        raise TypeError('Number of features should be an integer.')
    if (type(width) != float) and (type(width != int)):
        raise TypeError('width should be a number in millimeters.')

    # Convert images to correct types
    nom_image = nom_image.astype(np.float64)
    exp_image = exp_image.astype(np.float64)

    # Find experimental image reference height
    pd0 = generate_ph(exp_image, 0)
    ref = find_ref_height(nfeat, pd0)
    print(f'Reference Height: {np.round(ref, 5)}')

    # Loop over feature height (0-1) in specified number of steps
    emd = []
    step = 1
    for T in np.linspace(0, 1, num_steps):

        # Binarize image at threshold T
        binary = nom_image < T
        im_size = len(nom_image)

        # Compute distance transform of binarized image at T
        # Convert distances to physical units
        dt_im = sp.distance_transform_edt(binary)
        dt_im = dt_im * width / im_size
        del binary

        # Compute nominal image 1D persistence
        nom_dgm = np.array(generate_ph(dt_im, 1))
        nom_dgm.astype(np.float64)
        lifetimes = nom_dgm[:, 1] - nom_dgm[:, 0]
        nom_li = np.array(lifetimes)

        # Shift experimental image by reference, binarize and distance transform
        binary = exp_image < T + ref
        im_size = len(exp_image)
        dt_im = sp.distance_transform_edt(binary)
        dt_im = dt_im * width / im_size
        del binary

        # Compute experimental 1D persistence
        exp_dgm = np.array(generate_ph(dt_im, 1))
        exp_dgm.astype(np.float64)
        lifetimes = exp_dgm[:, 1] - exp_dgm[:, 0]
        lifetimes = np.array(lifetimes)

        # Filter noise features from experimental persistence diagram
        if len(lifetimes) > nfeat:
            hist_noise = np.histogram(lifetimes, bins='rice')

            cutoffmin1 = 0
            for n, item in enumerate(hist_noise):
                if item[0] > nfeat:
                    cutoffmin1 = hist_noise[1][n + 1]

            lifetimes = lifetimes * (lifetimes > cutoffmin1)
        lifetimes = lifetimes[np.nonzero(lifetimes)]
        exp_li = lifetimes

        # Compute earth movers distance
        if len(nom_li) > 0 and len(exp_li) > 0:
            dist = np.array([np.round(T, 2), np.round(
                stats.wasserstein_distance(nom_li, exp_li), 4)])
            emd.append(dist)
            print(f'Step {step}/{num_steps}: (T, EMD) = ({dist})')
        else:
            dist = np.array([np.round(T, 2), 0])
            emd.append(dist)
            print(f'Step {step}/{num_steps}: (T, EMD) = ({dist})')

        step += 1

    # Compute roundness score and plot roundness curve
    emd = np.array(emd)
    roundness_score = integrate.simpson(emd[:, 1], x=emd[:, 0]) / (1 - ref)
    if plot:
        plt.figure(figsize=(6, 6))
        plt.plot(emd[:, 0], emd[:, 1], 'r')
        plt.plot([-0.02, 1], [0, 0], 'k--', linewidth=0.5)
        plt.plot([0, 0], [-0.02, 1.2 * np.max(emd[:, 1])],
                 'k--', linewidth=0.5)
        plt.xticks(np.round(np.linspace(0, 1, 4), 2), fontsize=30)
        plt.yticks(np.round(np.linspace(0, 0.25, 4), 2), fontsize=30)
        plt.xlim(-0.02, 1)
        plt.ylim(-0.02, 1.2 * np.max(emd[:, 1]))
        plt.xlabel('Threshold', fontsize=30)
        plt.ylabel('EMD', fontsize=30)
        plt.title(f'Roundness Plot', fontsize=35)
        plt.tight_layout()
        plt.show()
    return roundness_score


def filter_outliers(points, num_features):
    '''
        Function to filter outliers from a persistence diagrams by removing bars from the lifetime histogram
        with a quantity larger than the expected total number of features.
    '''
    # Compute histogram
    hist_noise = np.histogram(points[:, 1], bins='rice')
    cutoffmin1 = 0
    remaining = np.sum(hist_noise[0])
    # If a bar is larger than feat, increase the noise cutoff
    for n, item in enumerate(hist_noise[0]):
        if item > num_features and remaining > num_features:
            cutoffmin1 = hist_noise[1][n + 1]
            remaining -= item

    # Remove points from the persistence diagram with a lifetime below the cutoff
    points = points[np.where(points[:, 1] > cutoffmin1)]
    print(f'Cutoff Lifetime: {cutoffmin1}')
    return points


def find_ref_height(feat, per_diag):
    '''
        Function to filter a persistence diagram down to the specified number of features by removing noise,
        and filtering the birth time back until the specified number of features remain. The average birth time
        is returned to give a reference height for the image.
    '''
    # Set birth-death bounds
    cutoff_bmax = 1
    cutoff_bmin = 0
    cutoff_dmin = 0
    pd = per_diag

    # Check if persistence diagram can be filtered
    if len(pd) <= feat and len(pd) > 0:
        z = pd[:, 0]
    elif len(pd) == 0:
        z = 0
    else:
        # Filter persistence diagram
        while len(pd) > feat:
            pd = per_diag
            lifetime1 = pd[:, 1] - pd[:, 0]
            if np.std(lifetime1) > 0:
                histogram = np.histogram(lifetime1)
                for n, c in enumerate(histogram):
                    # Filter noise based on histogram bars (death min)
                    if c[0] > feat:
                        cutoff_dmin = histogram[1][n + 1]
                # Filter by bounds
                pd = pd[cutoff_bmax > pd[:, 0]]
                pd = pd[pd[:, 0] > cutoff_bmin]
                pd = pd[(pd[:, 1] - pd[:, 0]) > cutoff_dmin]

                # Pull maximum birth back and iterate
                cutoff_bmax -= 0.001
                z = pd[:, 0]
            else:
                z = pd[:, 0]
                break
    # No features? return minimum birth
    if len(z) == 0:
        z = np.min(per_diag[:, 0])
    # Return average feature birth height
    return np.mean(z)


def generate_ph(img, dim):
    """
        Description: This function computes sub-level persistent homology on an image
        using cubical ripser.

        Parameters
        ----------
        img: array
        Image array to compute persistence on.

        dim: int
        Desired dimension of persistence.
    Returns
    -------
    Persistence pairs of desired dimension on the input image.

    """

    pipe = Pipeline(
        [
            ("cub_pers", CubicalPersistence(homology_dimensions=dim, n_jobs=-2)),
            ("finite_diags", DiagramSelector(use=True, point_type="finite")),
        ]
    )
    pdgm = pipe.fit_transform(np.array([img]))
    pdgm = pdgm[0].astype(np.float64)

    return pdgm
