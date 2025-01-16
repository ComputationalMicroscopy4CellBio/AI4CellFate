from setuptools import setup, find_packages

setup(
    name='cellfate', # Name of the package
    version='0.1',
    packages=find_packages(where="src"),  # Locate packages inside src
    package_dir={"": "src"},  # Specifying src as the root package directory
    install_requires=[],  
)
