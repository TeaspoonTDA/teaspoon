import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="teaspoon", # Replace with your own username
    version="1.0.0",
    author="Elizabeth Munch and Firas Khasawneh",
    author_email="author@example.com",
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
)
