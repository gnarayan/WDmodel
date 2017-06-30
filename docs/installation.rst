Installing WDmodel
------------------
.. highlight:: guess

This document will step you through getting the ``WDmodel`` package installed
on your system.

.. toctree::
   :maxdepth: 3

* :ref:`Installation instructions <install>`
   - :ref:`Get python <python>`
   - :ref:`Get the code <code>`
   - :ref:`Install everything <package>`
   - :ref:`Get auxillary pysynphot files <synphot>`
   - :ref:`Install the code  <finalize>`

* :ref:`Some extra notes <notes>`


.. _install:

=========================
Installation Instructions
=========================

Here's a minimal set of instructions to get up and running. We will eventually
get this package up on PyPI and conda-forge, and that should make this even
easier.

.. _python:

0. Install python:
~~~~~~~~~~~~~~~~~~
We recommend using the anaconda python distribution to run this package. If you
don't have it already, follow the instructions `here
<https://conda.io/docs/install/quick.html#linux-miniconda-install>`__
    
**Make sure you added the conda/bin dir to your path!**

If you elect to use your system python, or some other distribution, we will
assume you know what you are doing, and you can, skip ahead. 

.. _code:

1. Get the code:
~~~~~~~~~~~~~~~~

Clone this repository

.. code-block:: console

   git clone https://github.com/gnarayan/WDmodel.git
   cd WDmodel

.. _package:

2. Install everything:
~~~~~~~~~~~~~~~~~~~~~~

 a. Create a new environment from specification

    .. code-block:: console

      conda env create -f docs/env/conda_environment_py[27|35]_[osx64|i686].yml

 *or*  
    
 b. Create a new environment from scratch

  .. code-block:: console

    cp docs/env/condarc.example ~/.condarc
    conda create -n WDmodel
    source activate WDmodel
    conda install --yes --file dependencies_py[27|35].txt

.. _synphot:

3. Get the latest HST CDBS files:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are available over FTP from
[ftp://archive.stsci.edu/pub/hst/pysynphot/]

Untar them wherever you like, and set the ``PYSYN_CDBS`` environment variable

.. code-block:: console

 export PYSYN_CDBS=place_you_untarred_the_files

.. _finalize:

4. Install the package [optional]:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

  python setup.py install


.. _notes:

=====
Extra
=====

The instructions should be enough to get up and running, even without ``sudo``
privileges. There's a few edge cases on cluster environments though. These
notes may help:

.. toctree::
   :maxdepth: 2 

   extra

