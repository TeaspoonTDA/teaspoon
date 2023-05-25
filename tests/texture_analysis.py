import unittest
from teaspoon.SP import texture_analysis as ta
import numpy as np
from scipy.stats import multivariate_normal

class ta_module(unittest.TestCase):

    def test_perfect_square_lattice(self):
        np.random.seed(48824)
        n = 10
        x = np.linspace(-1, 1, n) + np.random.uniform(-0.1,0.1, n)
        y = np.linspace(-1, 1, n) + np.random.uniform(-0.1,0.1, n)


        xv, yv = np.meshgrid(x, y)
        xv = xv.reshape(-1, 1)
        yv = yv.reshape(-1, 1)

        data = np.column_stack((xv, yv))
        scores = ta.lattice_shape(data)
        self.assertAlmostEqual(scores[0], 0.01458, delta=0.001)
        self.assertAlmostEqual(scores[1], 0.65979, delta=0.001)

    def test_depth(self):
        # Generate a grid of sample xy pairs
        n = 500
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xv, yv = np.meshgrid(x, y)
        xv = xv.reshape(-1, 1)
        yv = yv.reshape(-1, 1)
        xy = np.column_stack((xv, yv))
        nom = 0
        exp = 0
        ind = 0

        # Generate Grid of Strike Locations
        grid_size = 4
        xc = np.linspace(-0.8, 0.8, grid_size)
        yc = np.linspace(-0.8, 0.8, grid_size)
        xvc, yvc = np.meshgrid(xc, yc)
        xvc = xvc.reshape(-1, 1)
        yvc = yvc.reshape(-1, 1)
        xyc = np.column_stack((xvc, yvc))


        # Generate 3D surface from gaussian distribution and scale between 0-1
        for point in xyc:
            ind += 1
            nom += multivariate_normal.pdf(xy, mean=np.array([point[0], point[1]]), cov=np.diag(np.array([0.1, 0.1]) ** 2))
            if ind == 6:
                exp += multivariate_normal.pdf(xy, mean=np.array([point[0], point[1]]), cov=np.diag(np.array([0.13, 0.13]) ** 2))
            else:
                exp += multivariate_normal.pdf(xy, mean=np.array([point[0], point[1]]), cov=np.diag(np.array([0.1, 0.1]) ** 2))

        nom = nom.reshape(n, n).astype(np.float64)
        nom = -1.0 * nom + np.max(nom)
        nom = nom / np.max(nom) # scale between 0-1

        exp = exp.reshape(n, n).astype(np.float64)
        exp = -1.0 * exp + np.max(exp)
        exp = exp / np.max(exp) # scale between 0-1



        # Generate nominal and experimental images by adding noise and scaling from 0-1 again
        np.random.seed(48824)
        nom_im = nom
        exp_im = exp + np.random.normal(scale=0.01, size=[n,n])
        exp_im = (exp_im - np.min(exp_im))/(np.max(exp_im) - np.min(exp_im)) # Scale between 0-1

        # Compute the depth score
        score = ta.feature_depth(nom_im, exp_im, 16, plot=False)
        self.assertAlmostEqual(score, 93.12, delta=0.01)

    def test_roundness(self):
        # Generate a grid of sample xy pairs
        n = 500
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xv, yv = np.meshgrid(x, y)
        xv = xv.reshape(-1, 1)
        yv = yv.reshape(-1, 1)
        xy = np.column_stack((xv, yv))
        nom = 0
        exp = 0
        ind = 0

        # Generate Grid of Strike Locations
        grid_size = 4
        xc = np.linspace(-0.8, 0.8, grid_size)
        yc = np.linspace(-0.8, 0.8, grid_size)
        xvc, yvc = np.meshgrid(xc, yc)
        xvc = xvc.reshape(-1, 1)
        yvc = yvc.reshape(-1, 1)
        xyc = np.column_stack((xvc, yvc))


        # Generate 3D surface from gaussian distribution and scale between 0-1
        for point in xyc:
            ind += 1
            nom += multivariate_normal.pdf(xy, mean=np.array([point[0], point[1]]), cov=np.diag(np.array([0.1, 0.1]) ** 2))
            exp += multivariate_normal.pdf(xy, mean=np.array([point[0], point[1]]), cov=np.diag(np.array([0.2, 0.1]) ** 2))

        nom = nom.reshape(n, n).astype(np.float64)
        nom = -1.0 * nom + np.max(nom)
        nom = nom / np.max(nom) # scale between 0-1

        exp = exp.reshape(n, n).astype(np.float64)
        exp = -1.0 * exp + np.max(exp)
        exp = exp / np.max(exp) # scale between 0-1



        # Generate nominal and experimental images by adding noise and scaling from 0-1 again
        np.random.seed(48824)
        nom_im = nom
        exp_im = exp + np.random.normal(scale=0.1, size=[n,n])
        exp_im = (exp_im - np.min(exp_im))/(np.max(exp_im) - np.min(exp_im)) # Scale between 0-1

        # Compute the roundness score
        score = ta.feature_roundness(nom_im, exp_im, 1, 2.5, num_steps=50, plot=False)
        self.assertAlmostEqual(score, 0.11744985, delta=0.001)



if __name__ == '__main__':
    unittest.main()







