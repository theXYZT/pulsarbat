:html_theme.sidebar_secondary.remove:
:nosearch:

===========
Development
===========

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given. This page assumes the reader has some familiarity with contributing to open-source Python projects using GitHub.

Pulsarbat could always use more documentation, especially in the form of worked examples.

For bug reports and feature requests, create an issue on GitHub here: https://github.com/theXYZT/pulsarbat/issues


Developer Workflow
------------------

To contribute changes (fixing bugs, adding features), we follow a typical `GitHub workflow <https://docs.github.com/en/get-started/quickstart/github-flow>`_:

* Create a personal fork of the repository.
* Create a branch (preferrably, with an informative name).
* Make changes, test your contributions and document them!
* Open a pull request.
* Iterate until changes pass various linters and checks.
* Work through code review until your PR is accepted and merged.


Deploying
---------

A reminder for the maintainers on how to deploy a release:

* Make sure all changes are committed.
* Update changelog in ``HISTORY.rst``.
* Update package version in ``pulsarbat/__init__.py`` either manually or
  using ``bump2version``.
* Create a tagged commit with tag: ``vX.Y.Z`` and push tags to origin.
  A tagged commit should automatically publish the package to PyPI via
  Github Actions.
* Create a release on Github on the tagged commit (this will trigger Zenodo).


Useful Things
-------------

This section is an informal reference for contributors and maintainers on how
to do some development-related things.

Testing and Coverage
^^^^^^^^^^^^^^^^^^^^

From the root directory of the repository, run:

.. code-block:: console

    $ pytest --cov-report=html --cov=pulsarbat

Then open ``htmlcov/index.html`` in your browser to look at test coverage.

Building Docs
^^^^^^^^^^^^^

From ``docs/``, run:

.. code-block:: console

    $ make clean
    $ make html && python -m http.server --directory _build/html 9000

Then open ``http://localhost:9000/`` in your browser to access the built docs.
