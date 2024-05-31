"""The setup script."""

from setuptools import setup

setup(
    name="pulsarbat",
    install_requires=[
        "astropy >= 6.1",
        "numpy >= 1.26",
        "scipy >= 1.13",
        "baseband >= 4.1.3",
        "dask[array] >= 2024.5",
    ],
    tests_require=["pytest >= 8.2.1", "cloudpickle >= 3.0.0"],
)
