Welcome to teaspoon's documentation page!
==========================================

This page provides a summary of the functionality of teaspoon as well as documentation for the various topologicval signal processing modules:

This package provides wrappers for using various persistent homology and other Topological Data Analysis software packages through python.



Table of Contents
*******************

.. toctree::
   :maxdepth: 1
   :numbered:
   
   Getting Started <installation.rst>
   Persistent Homology of Networks Module (PHN) <PHN.rst>
   Dynamic Systems Library (DynSysLib) Module <DynSysLib.rst>
   Parameter Selection Module <parameter_selection.rst>
   Information Module <information.rst>
   Machine Learning (ML) Module <ML.rst>
   Topological Data Analaysis (TDA) Module <TDA.rst>
   Topological Signal Processing (TSP) Module <TSP.rst>
   Make Data (MakeData) Module <MakeData.rst>



Collaborators
***************

The code is an compilation of work done by `Elizabeth Munch <http://www.elizabethmunch.com>`_ along with her students and collaborators.  People who have contributed to teaspoon include:

- `Firas Khasawneh <http://www.firaskhasawneh.com>`_
- Jesse Berwald
- Brian Bollen
- Audun Myers
- Melih Yesilli
- Sarah Tymochko


Current Issues and To-do Items
*********************************************

This is a list of to do items:

	- Finish updating documentation
	- Get autodoc with sphinx to work
	- Make pip install-able
	- Remove any redundant function (e.g. Takens' Embedding)

This is a list of the current issues for Sphinx:

	- Failing to autodoc the modules in PD_Classification
	- Failing to autodoc the LandscapesParameterDucket class from Base.py
	- PointCloud.py is failing to autodoc from the MakeData module
	- There is no sphinx docstring documentation in the adaptivePart.py code.


Contributing
*********************************************
Contributions are more than welcome! There are lots of opportunities for potential projects, so please get in touch if you would like to help out. Everything from code to notebooks to examples and documentation are all equally valuable so please don't feel you can't contribute. To contribute please fork the project make your changes and submit a pull request. We will do our best to work through any issues with you and get your code merged into the main branch.

If contributing to the documentation, the following python packages needs to be pip installed as shown:

	- pip install PersistenceImages
	- pip install TermColor
	- pip install sphinxcontrib-bibtex

Also, please make sure your sphinx installation is up-to-date and that you are running **make html** from the teaspoon directory.

Contact Information
********************
Liz Munch: muncheli@egr.msu.edu
