"""The setup script."""

from setuptools import setup

setup(
    name="pulsarbat",
    install_requires=[
        "astropy >= 5.2",
        "numpy >= 1.23",
        "scipy >= 1.10",
        "baseband >= 4.1.2",
        "dask[array] >= 2023.4.1",
    ],
    tests_require=["pytest >= 7.3.1", "cloudpickle >= 2.2.1"],
)
