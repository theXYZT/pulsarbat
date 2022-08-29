"""The setup script."""

from setuptools import setup

setup(
    name="pulsarbat",
    install_requires=[
        "astropy >= 5.1",
        "numpy >= 1.22",
        "scipy >= 1.9",
        "baseband >= 4.1.1",
        "dask[array] >= 2022.8.1",
    ],
    tests_require=["pytest >= 7.1.2", "cloudpickle >= 2.1.0"],
)
