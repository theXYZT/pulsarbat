:nosearch:

``pulsarbat``
=============

    :Release: |release|
    :Date: |today|

``pulsarbat`` (PULSAR Baseband Analysis Tools) is a Python package for analysis of radio baseband signals. Although this package has a special focus on radio pulsar astronomy, it can also be used to work with other types of radio astronomical observations (such as fast radio bursts, quasars, and so on) or any time-frequency data, in general. ``pulsarbat`` provides:

* Signals: Standardized containers for signal data.
* Functions/Transforms for manipulating signals.
* Easy integration with Dask_ for lazily executed workflows or managing large workloads
  in an "embarassingly parallel" manner.

The source code can be found on GitHub: https://github.com/theXYZT/pulsarbat

.. _Dask: https://dask.org/


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


.. toctree::
   :maxdepth: 1
   :hidden:

   install
   tutorial/index
   reference/index
   development
   changelog
