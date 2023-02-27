:html_theme.sidebar_secondary.remove:
:nosearch:

============
Installation
============

Pulsarbat has the following dependencies:

- **Python 3.9 or later**
- `Astropy <https://www.astropy.org/>`_ 5.2 or later
- `Numpy <https://www.numpy.org/>`_ 1.23 or later
- `Scipy <https://scipy.org/>`_ 1.10 or later
- `Baseband <https://baseband.readthedocs.io/>`_ 4.1.1 or later
- `Dask <https://dask.org/>`_ 2023.2.1 or later: with Dask Array.


Released version
----------------

To install the latest released version of Pulsarbat with `pip`_, run:

.. code-block:: console

    $ pip install pulsarbat

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


Development version
-------------------

To install the latest development version of Pulsarbat, you can clone the `public repository <https://github.com/theXYZT/pulsarbat>`_ and install the package:

.. code-block:: console

    $ git clone git://github.com/theXYZT/pulsarbat
    $ cd pulsarbat
    $ pip install -e .

The ``-e`` flag installs the package in editable mode which allows you to update the package at any time via:

.. code-block:: console

    $ git pull


Testing
-------

Pulsarbat uses `pytest`_ for testing. You can test the development version of the package from the source directory with:

.. code-block:: console

    $ pytest

.. _pytest: https://docs.pytest.org/
