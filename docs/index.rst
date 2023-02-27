:html_theme.sidebar_secondary.remove:
:nosearch:

=========
Pulsarbat
=========

    :Release: |release|
    :Date: |today|

Pulsarbat (PULSAR Baseband Analysis Tools) is a Python package for analysis of radio baseband signals. Although this package has a special focus on radio pulsar astronomy, it can also be used to work with other types of radio astronomical observations (such as fast radio bursts, quasars, and so on) or any time-frequency data, in general. Pulsarbat provides:

* Signals: Standardized containers for signal data.
* Functions/Transforms for manipulating signals.
* Easy integration with Dask_ for lazily executed workflows or managing large workloads
  in an "embarassingly parallel" manner.

The source code can be found on GitHub: https://github.com/theXYZT/pulsarbat

.. _Dask: https://dask.org/


Quickstart
----------

Install the latest version of Pulsarbat:

.. code-block:: console

    $ pip install pulsarbat

To use Pulsarbat in a project:

.. code-block:: python

    import pulsarbat as pb


Citing
------

Pulsarbat has a DOI via Zenodo: https://doi.org/10.5281/zenodo.6934355

This DOI link represents all versions, and will always resolve to the latest one.
Use the following Bibtex entry to cite this work:

.. code-block:: plaintext

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

Pulsarbat is licensed under the GNU General Public License v3.


.. toctree::
   :maxdepth: 1
   :hidden:

   install
   user_guide/index
   reference/index
   development
   changelog
