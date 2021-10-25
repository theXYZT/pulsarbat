.. highlight:: shell

============
Installation
============

Requirements
------------

`pulsarbat` has the following strict requirements:

- **Python 3.8 or later**
- `Astropy <https://www.astropy.org/>`_ 4.3 or later
- `Numpy <https://www.numpy.org/>`_ 1.21 or later
- `Scipy <https://scipy.org/>`_ 1.7 or later
- `Baseband <https://baseband.readthedocs.io/>`_ 4.0.3 or later

For optional features, `pulsarbat` also depends on:

- `Dask <https://dask.org/>`_ 2021.10 or later: Lazy execution and
  "embarassingly parallel" workflows.


Stable release
--------------

To install pulsarbat with `pip`_ along with optional dependencies, run:

.. code-block:: console

    $ pip install pulsarbat[all]

This is the preferred method to install pulsarbat, as it will always
install the most recent stable release. If you don't have `pip`_
installed, this `Python installation guide`_ can guide you through the
process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The source code for pulsarbat can be downloaded from the `Github repo`_.

You can clone the public repository:

.. code-block:: console

    $ git clone git://github.com/theXYZT/pulsarbat

Then install with:

.. code-block:: console

    $ cd pulsarbat
    $ pip install .[all]


.. _Github repo: https://github.com/theXYZT/pulsarbat
