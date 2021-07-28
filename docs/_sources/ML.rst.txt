Persistence Diagram Featurization
==================================

Persistence diagram can not be directly used in the machine learning algorithms.
Therefore, there are several methods that are proposed to extract features from persistence diagrams.
Some these methods are :ref:`persistence_images`:cite:`1 <Adams2017>`, :ref:`carlsson_coordinates`:cite:`2 <Adcock2016,Khasawneh2018>`, :ref:`persistence_landscapes`:cite:`3 <Bubenik2017>`, :ref:`path_signatures`:cite:`4 <Chevyrev2016,Chevyrev2020>`, and kernel method :cite:`5 <Reininghaus2015>`.
In this toolbox, we provide the documentation for the codes of these five methods.
We used these methods to extract features from persistence diagrams of cutting signals to diagnose chatter in machining.
Our data set is available in Ref. :cite:`2 <Khasawneh2019>`.
One can refer to :cite:`6 <Yesilli2019>` for more details.

.. toctree::
   :maxdepth: 4

   Featurization <F_PD.rst>
   Classification <CL.rst>

References
###########
.. bibliography:: references.bib
   :style: plain
