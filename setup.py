"""The setup script."""

from setuptools import setup

setup(
    name="pulsarbat",
    install_requires=[
        "astropy >= 5.1",
        "numpy >= 1.23",
        "scipy >= 1.9",
        "baseband >= 4.1.1",
        "dask[array] >= 2022.9.2",
    ],
    tests_require=["pytest >= 7.1.3", "cloudpickle >= 2.2.0"],
)
