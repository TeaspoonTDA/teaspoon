import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..', 'teaspoon', 'ML'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..'))

# -- mocking packages -----------------------------------------------------

from unittest import mock
sys.modules['ripser'] = mock.Mock()


# -- Project information -----------------------------------------------------

project = 'teaspoon'
copyright = '2020, Munch'
author = 'Elizabeth Munch'
release = '0.1.0'


# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc', 
              'sphinx.ext.coverage', 
              'sphinx.ext.napoleon',
              'sphinx_rtd_theme',
              'matplotlib.sphinxext.mathmpl',
              'matplotlib.sphinxext.plot_directive',
              'sphinx.ext.autosummary',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.mathjax',
              'sphinx.ext.ifconfig',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'sphinxcontrib.bibtex',]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_logo = 'teaspoon.png'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False,

}
html_static_path = ['_static']
# enable numbering figures
numfig = True
