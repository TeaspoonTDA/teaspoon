from setuptools import setup
from setuptools import find_packages


with open("README.md", "r") as fh:
    ld = fh.read()

if __name__ == "__main__":
    setup(packages = find_packages(),
    long_description = ld,
    dependency_links = ["git+https://github.com/shizuo-kaji/CubicalRipser_3dim.git"])