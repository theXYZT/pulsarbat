:nosearch:

===========
Development
===========

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given. This page assumes the reader has some familiarity with contributing to open-source Python projects using GitHub.

``pulsarbat`` could always use more documentation, especially in the form of worked examples.

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

A reminder for the maintainers on how to deploy a release. A tagged commit will automatically trigger a release and publish the package to PyPI. For now, we use the ``bump2version`` package to do this. Make sure all your changes are committed (including an entry in `HISTORY.rst`).

Then run:

.. code-block:: console

    $ bumpversion patch  # possible: major / minor / patch
    $ git push
    $ git push --tags
