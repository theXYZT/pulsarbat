
.. image:: https://img.shields.io/pypi/v/pulsarbat.svg
        :target: https://pypi.python.org/pypi/pulsarbat

.. image:: https://img.shields.io/travis/theXYZT/pulsarbat.svg
        :target: https://travis-ci.org/theXYZT/pulsarbat

.. image:: https://readthedocs.org/projects/pulsarbat/badge/?version=latest
        :target: https://pulsarbat.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

==================
What is pulsarbat?
==================

Pulsarbat (PULSAR Baseband Analysis Tools) is a package for analysis of baseband observations of pulsars.

| The source code can be found on GitHub: https://github.com/theXYZT/pulsarbat
| For documentation, please go to: https://pulsarbat.readthedocs.io

Pulsarbat is licensed under the GNU General Public License v3.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

============
Installation
============

Requirements
------------

Pulsarbat requires:

- `Astropy <https://www.astropy.org/>`_ v4.0 or later
- `Numpy <https://www.numpy.org/>`_ v1.17 or later
- `Baseband <https://github.com/mhvk/baseband>`_ v3.1.0 or later

Stable release
--------------

To install pulsarbat, run this command in your terminal:

.. code-block:: console

    $ pip install pulsarbat

This is the preferred method to install pulsarbat, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for pulsarbat can be downloaded from the `Github repo`_.

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
