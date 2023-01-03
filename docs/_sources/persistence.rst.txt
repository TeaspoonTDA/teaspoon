Persistence
============

.. automodule:: teaspoon.TDA.Persistence
   :members: minPers, maxPers, maxBirth, minBirth, minPersistenceSeries, maxPersistenceSeries, minBirthSeries, maxBirthSeries, removeInfiniteClasses

The following example computes the minimum and maximum birth times, as well as the maximum persistence::

  from ripser import ripser
  from teaspoon.TDA.Draw import drawDgm
  from teaspoon.MakeData.PointCloud import Torus
  from teaspoon.TDA.Persistence import maxBirth, minBirth, maxPers

  numPts = 500
  seed = 0

  # Generate Torus
  t = Torus(N=numPts,seed = seed)

  # Compute persistence diagrams
  PD1 = ripser(t,2)['dgms'][1]

  print('Maximum Birth: ', maxBirth(PD1))
  print('Minimum Birth: ', minBirth(PD1))
  print('Max Persistence: ', maxPers(PD1))

The output of this code is::

  Maximum Birth:  1.0100464820861816
  Minimum Birth:  0.17203105986118317
  Max Persistence:  1.3953008949756622


The following example computes the minimum and maximum birth times and the
maximum persistence across all persistence diagrams in the specified column of
the DataFrame::

  from teaspoon.MakeData.PointCloud import testSetManifolds
  from teaspoon.TDA.Persistence import maxBirthSeries, minBirthSeries, maxPersistenceSeries

  df = testSetManifolds(numDgms = 1, numPts = 500, seed=0)

  print('Maximum Birth of all diagrams: ', maxBirthSeries(df['Dgm1']))
  print('Minimum Birth of all diagrams: ', minBirthSeries(df['Dgm1']))
  print('Max Persistence of all diagrams: ', maxPersistenceSeries(df['Dgm1']))

The output of this code is::

  Maximum Birth of all diagrams:  1.2070081233978271
  Minimum Birth of all diagrams:  0.028233738616108894
  Max Persistence of all diagrams:  1.6290252953767776
