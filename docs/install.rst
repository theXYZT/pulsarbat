.. highlight:: shell

============
Installation
============

Requirements
------------

`pulsarbat` has the following strict requirements:

- **`Python <https://www.python.org/>`_ 3.8 or later**
- `Astropy <https://www.astropy.org/>`_ 4.2 or later
- `Numpy <https://www.numpy.org/>`_ 1.19 or later
- `Scipy <https://scipy.org/>`_ 1.15 or later
- `Baseband <https://baseband.readthedocs.io/>`_ 4.0.3 or later

For optional features, `pulsarbat` also depends on:

- `Dask <https://dask.org/>`_ 2020.12.0 or later: Lazy execution and
  "embarassingly parallel" workflows.


Stable release
--------------

To install pulsarbat with `pip`_, run:

.. code-block:: console

    $ pip install pulsarbat

This is the preferred method to install pulsarbat, as it will always
install the most recent stable release. If you don't have `pip`_
installed, this `Python installation guide`_ can guide you through the
process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The source code for pulsarbat can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/theXYZT/pulsarbat

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/theXYZT/pulsarbat/tarball/master

Once you have a copy of the source, you can install it with either:

.. code-block:: console

    $ python setup.py install

or

.. code-block:: console

    $ pip install .


.. _Github repo: https://github.com/theXYZT/pulsarbat
.. _tarball: https://github.com/theXYZT/pulsarbat/tarball/master
