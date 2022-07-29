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


``pulsarbat`` (PULSAR Baseband Analysis Tools) is a Python package for analysis of radio baseband signals. Although this package has a special focus on radio pulsar astronomy, it can be easily used to work with other types of radio astronomical observations (such as fast radio bursts, quasars, and so on) or any time-frequency data, in general. ``pulsarbat`` provides:

* Signals: Standardized containers for signal data.
* Functions/Transforms for manipulating signals.
* Easy integration with Dask_ for lazily executed workflows or managing large workloads
  in an "embarassingly parallel" manner.

.. _Dask: https://dask.org/

Documentation can be found at: https://pulsarbat.readthedocs.io


Quickstart
----------

Install the latest version of ``pulsarbat``:

.. code-block:: console

    $ pip install pulsarbat

To use ``pulsarbat`` in a project:

.. code-block:: python

    import pulsarbat as pb


Citing
------

For next release, there should be a Zenodo citation!


License
-------

``pulsarbat`` is licensed under the GNU General Public License v3.
