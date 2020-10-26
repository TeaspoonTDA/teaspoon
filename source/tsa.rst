Time Series Analysis (TSA) Tools
====================================



Takens' Embedding
******************


The Takens' embedding algorhtm reconstructs the state space from a single time series with delay-coordinate embedding as shown in the animation below.

.. image:: figures/takens_embedding_gif.gif
   :class: with-shadow float-center
   :scale: 35


Figure: Takens' embedding animation for a simple time series and embedding of dimension two. This embeds the 1-D signal into n=2 dimensions with delayed coordinates.

.. rst-class::  clear-both



.. automodule:: teaspoon.SP.tsa_tools
    :members: takens
    :noindex:

Permutation Sequence Generation
**********************************


.. image:: figures/permutation_sequence_animation.gif
   :class: with-shadow float-center
   :scale: 45


Figure: Permutation sequence animation for a simple time series and permutations of dimension three.

.. rst-class::  clear-both

.. automodule:: teaspoon.SP.tsa_tools
    :members: permutation_sequence
    :noindex:

k Nearest Neighbors
***********************

.. automodule:: teaspoon.SP.tsa_tools
    :members: k_NN
    :noindex:


