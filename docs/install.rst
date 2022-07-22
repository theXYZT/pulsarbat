:nosearch:

============
Installation
============

``pulsarbat`` has the following dependencies:

- **Python 3.9 or later**
- `Astropy <https://www.astropy.org/>`_ 5.1 or later
- `Numpy <https://www.numpy.org/>`_ 1.23 or later
- `Scipy <https://scipy.org/>`_ 1.8.1 or later
- `Baseband <https://baseband.readthedocs.io/>`_ 4.1.1 or later
- `Dask <https://dask.org/>`_ 2022.6.1 or later: with Dask Array.


Released version
----------------

To install the latest released version of ``pulsarbat`` with `pip`_, run:

.. code-block:: console

    $ pip install pulsarbat

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


Development version
-------------------

To install the latest development version of ``pulsarbat``, you can clone the `public repository <https://github.com/theXYZT/pulsarbat>`_ and install the package:

.. code-block:: console

    $ git clone git://github.com/theXYZT/pulsarbat
    $ cd pulsarbat
    $ pip install -e .

The ``-e`` flag installs the package in editable mode which allows you to update the package at any time via:

.. code-block:: console

    $ git pull


Testing
-------

``pulsarbat`` uses the ``pytest`` testing package. You can test the development version of the package from the source directory with:

.. code-block:: console

    $ pytest
