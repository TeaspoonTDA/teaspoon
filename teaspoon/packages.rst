Packages
========


.. toctree::
   :caption: Packages:

.. autosummary::

   teaspoon.MakeData.PointCloud
   teaspoon.ML.feature_functions
   teaspoon.ML.tents
   teaspoon.TDA.Distance
   teaspoon.TDA.Draw
   teaspoon.TDA.Persistence

.. automodule:: teaspoon.MakeData.PointCloud
   :members: Circle, Sphere, Annulus, Torus, Cube, Clusters, normalDiagram, testSetClassification, testSetRegressionLine, testSetRegressionBall, testSetManifolds
.. automodule:: teaspoon.ML.feature_functions
   :members: tent, sub2ind, ind2sub, quad_cheb1, quad_legendre, bary_weights, bary_diff_matrix, interp_polynomial
.. automodule:: teaspoon.ML.tents
   :members: tent, build_G, TentML, getPercentScore
.. automodule:: teaspoon.TDA.Distance
   :members: dgmDist_Hera
.. automodule:: teaspoon.TDA.Draw
   :members: drawDgm, drawPtCloud
.. automodule:: teaspoon.TDA.Persistence
   :members: prepareFolders, readPerseusOutput, readRipserString, readRipserOutput, VR_Ripser, writePointCloudFileForPerseus, VR_Perseus, distMat_Ripser, distMat_Perseus, writeMatrixFileForPerseus, Cubical_Perseus, minPers, maxPers, maxBirth, minBirth, minPersistenceSeries, maxPersistenceSeries, minBirthSeries, maxBirthSeries, removeInfiniteClasses
