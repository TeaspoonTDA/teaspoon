import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="teaspoon", # Replace with your own username
    version="1.3.1",
    author="Elizabeth Munch and Firas Khasawneh",
    author_email="muncheli@msu.edu",
    description="A Topological Signal Processing Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lizliz/teaspoon",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = [
        'cycler==0.11.0',
        'fonttools==4.34.4',
        'joblib==1.1.0',
        'kiwisolver==1.4.4',
        'matplotlib==3.5.2',
        'networkx==2.8.5',
        'numpy==1.23.1',
        'packaging==21.3',
        'Pillow==9.2.0',
        'POT==0.8.2',
        'pyentrp==0.7.1',
        'pyparsing==3.0.9',
        'python-dateutil==2.8.2',
        'scikit-learn==1.1.1',
        'scipy==1.8.1',
        'six==1.16.0',
        'sklearn==0.0',
        'threadpoolctl==3.1.0'
    ]
)
