============
Installation
============

.. _install:

There are two common ways to install vsscale.

The first is to install the latest release build through `pypi <https://pypi.org/project/vsscale/>`_.

You can use pip to do this, as demonstrated below:


.. code-block:: console

    pip install vsscale --no-cache-dir -U

This ensures that any previous versions will be overwritten and vsscale will be upgraded if you had already previously installed it.

------------------

The second method is to build the latest version from git.

This will be less stable, but will feature the most up-to-date features, as well as accurately reflect the documentation.

.. code-block:: console

    pip install git+https://github.com/Irrational-Encoding-Wizardry/vs-scale.git -U

It's recommended you use a release version over building from git
unless you require new functionality only available upstream.
