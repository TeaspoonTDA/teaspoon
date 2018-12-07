Contributing
============

While I'm sure this workflow will change as we go, here are some links to get you started.

- A nice introduction to git: `Git Immersion <http://gitimmersion.com/>`_
- Other packages to install in order to get full functionality out of teaspoon:
    - `Ripser <https://github.com/Ripser/ripser>`_. Code by Ulrich Bauer for computing persistent homology of a point cloud or distance matrix.
    - `Perseus <http://people.maths.ox.ac.uk/nanda/perseus/index.html>`_. Code by Vidit Nanda for computing persistent homology of point clouds, cubical complexes, and distance matrices.
    - `Hera <https://bitbucket.org/grey_narn/hera>`_. Code by Michael Kerber, Dmitriy Morozov, and Arnur Nigmetov for computing bottleneck and Wasserstein distances.
- Please put issues in the `issue tracker <https://gitlab.msu.edu/TSAwithTDA/teaspoon/issues>`_

**Basic workflow**

When you're going to start messing with something, create a branch for it.  This will likely just be done on your local machine.  When you think it's ready to be merged into the master branch, go to the merge request page on gilab and submit your merge request.

Discussions, comments, and updates can be done on the gitlab merge request page.

**Documentation**

Automatic documentation is being done with `doxygen <www.doxygen.org>`_.  Info to get you started is available in their `manual <http://www.stack.nl/~dimitri/doxygen/manual/index.html>`_.  The config file here is `Doxyfile` in the top level of the folder.  So, compiling the documentation should be as easy as running::

  doxygen /path/to/Doxyfile

Compiled documentation goes into the `doc` folder.  Point your browser to `doc/html/index.html` to see the result.  Don't forget to refresh your browser after you compile the documentation since most browsers (coughchromecough) rely heavily on caching and won't automatically update the page for you, even after you've just opened it.

**Before you push**

- Have you run doxygen to update the documentation?
- Have you incremented the version number?

**Questions, comments, and other issues**

...should be sent to `Liz Munch <mailto:muncheli@egr.msu.edu>`_.
