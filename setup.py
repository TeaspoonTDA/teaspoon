from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
      name='teaspoon',
      version='0.4.08',
      description='Python wrappers for various applied topology related code.',
      long_description=long_description,

      url = 'http://www.elizabethmunch.com',

      #Author details
      author='Elizabeth Munch',
      author_email='muncheli@msu.edu',

      # License
      license = 'GNU GPL',


      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
              # How mature is this project? Common values are
              #   3 - Alpha
              #   4 - Beta
              #   5 - Production/Stable
              'Development Status :: 3 - Alpha',

              # This project is intended for mathematicians and researchers
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: Mathematics',

              # License info
              'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

              # Supported versions of python
              # Can include multiple, however currently only tested on Liz's system, Python 3.5.2
              # 'Programming Language :: Python :: 3',
              # 'Programming Language :: Python :: 3.3',
              # 'Programming Language :: Python :: 3.4',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.5.2',
              ],



      keywords='persistent homology, TDA, topological data analysis',

      packages=find_packages(exclude=['contrib', 'docs', 'tests']),


    install_requires=[
          'ripser',
          'numpy',
          'matplotlib',
          # 'os',
          # 'sys',
          'scipy',
          # 'subprocess',
          # 'time',
          'sklearn',
          'pandas'
      ],

      include_package_data=True,
      zip_safe=False
      )

# Notes: See the sample project at https://github.com/pypa/sampleproject for more setup info
