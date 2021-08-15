=========
pulsarbat
=========

.. image:: https://img.shields.io/pypi/v/pulsarbat.svg
        :target: https://pypi.python.org/pypi/pulsarbat

.. image:: https://img.shields.io/pypi/pyversions/pulsarbat.svg
        :target: https://pypi.python.org/pypi/pulsarbat

.. image:: https://github.com/theXYZT/pulsarbat/workflows/CI/badge.svg
        :target: https://github.com/theXYZT/pulsarbat/actions

.. image:: https://codecov.io/gh/theXYZT/pulsarbat/branch/master/graph/badge.svg?token=Ia6qdZNhHE
        :target: https://codecov.io/gh/theXYZT/pulsarbat

.. image:: https://readthedocs.org/projects/pulsarbat/badge/?version=latest
        :target: https://pulsarbat.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Pulsarbat (Pulsar Baseband Analysis Tools) is a package for analysis of baseband observations of pulsars.

Documentation can be found at: https://pulsarbat.readthedocs.io

Requirements
------------

`pulsarbat` has the following strict requirements:

- Python_ 3.8 or later
- Numpy_ 1.20 or later
- Scipy_ 1.6 or later
- Astropy_ 4.2 or later
- Baseband_ 4.0.3 or later

For optional features, `pulsarbat` also depends on:

- Dask_ 2021.2 or later: Lazy execution and "embarassingly parallel" workflows.

.. _Python: http://www.python.org/
.. _Numpy: https://www.numpy.org/
.. _Scipy: https://scipy.org/
.. _Astropy: https://www.astropy.org/
.. _Baseband: https://baseband.readthedocs.io/
.. _Dask: https://dask.org/

Installation
------------

You can install pulsarbat along with all optional dependencies via::

    pip install pulsarbat[all]


License
-------

Pulsarbat is licensed under the GNU General Public License v3.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
