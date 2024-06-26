=========
pulsarbat
=========

.. container::

    |PyPI Status| |Python Versions| |Actions Status| |Coverage Status| |Documentation Status| |Zenodo|

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

``pulsarbat`` has a DOI via Zenodo: https://doi.org/10.5281/zenodo.6934355

This DOI link represents all versions, and will always resolve to the latest one.
Use the following Bibtex entry to cite this work:

.. code-block:: bibtex

    @software{pulsarbat,
      author       = {Nikhil Mahajan and Rebecca Lin},
      title        = {pulsarbat: PULSAR Baseband Analysis Tools},
      year         = {2023},
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.6934355},
      url          = {https://doi.org/10.5281/zenodo.6934355}
    }


License
-------

``pulsarbat`` is licensed under the GNU General Public License v3.


.. |PyPI Status| image:: https://img.shields.io/pypi/v/pulsarbat.svg
    :target: https://pypi.python.org/pypi/pulsarbat
    :alt: PyPI Status

.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/pulsarbat.svg
    :target: https://pypi.python.org/pypi/pulsarbat
    :alt: Python versions supported

.. |Actions Status| image:: https://github.com/theXYZT/pulsarbat/workflows/Tests/badge.svg
    :target: https://github.com/theXYZT/pulsarbat/actions
    :alt: GitHub Actions Status

.. |Coverage Status| image:: https://codecov.io/gh/theXYZT/pulsarbat/branch/master/graph/badge.svg?token=Ia6qdZNhHE
    :target: https://codecov.io/gh/theXYZT/pulsarbat
    :alt: Coverage Status

.. |Documentation Status| image:: https://readthedocs.org/projects/pulsarbat/badge/?version=latest
    :target: https://pulsarbat.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Zenodo| image:: https://zenodo.org/badge/194818440.svg
    :target: https://zenodo.org/badge/latestdoi/194818440
    :alt: Zenodo DOI
