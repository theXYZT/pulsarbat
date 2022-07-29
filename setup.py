"""The setup script."""

from setuptools import setup

setup(
    name="pulsarbat",
    install_requires=[
        "astropy >= 5.1",
        "numpy >= 1.23",
        "scipy >= 1.8",
        "baseband >= 4.1.1",
        "dask[array] >= 2022.6.1",
    ],
    tests_require=["pytest >= 7.1.2"],
)
