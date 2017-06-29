Using WDmodel
-------------
.. highlight:: guess

This document will help you get comfortable using the ``WDmodel`` package.

.. toctree::
   :maxdepth: 3

* :ref:`Usage <usage>`
   - :ref:`Get data  <data>`
   - :ref:`Running single threaded <singlethread>`
   - :ref:`Running with MPI  <mpipool>`

* :ref:`Useful options <argparse>`
   - :ref:`Quick analysis <quicklook>`
   - :ref:`Initializing the fitter <init>`
   - :ref:`Configuring the sampler <samptype>`
   - :ref:`Resuming the fit <resume>`


.. _usage:

=====
Usage
=====

This is the TL;DR version to get up and running.

.. _data:

1. GET THE DATA
~~~~~~~~~~~~~~~

Instructions will be available here when the paper is accepted. In the meantime
there's a single test object in the spectroscopy directory. If you want more,
Write your own HST proposal! :-P

.. _singlethread:

2. Run a fit single threaded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   ./fit_WDmodel.py --specfile data/spectroscopy/yourfavorite.flm

This option is single threaded and slow, but useful to testing or quick
exploratory analysis.

A more reasonable way to run things fast is to use mpi.

.. _mpipool:

3. Run as an MPI process
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   mpirun -np 8 fit_WDmodel.py --mpi --specfile=file.flm [--ignorephot]

Note that ``--mpi`` **MUST** be specified in the options to
``fit_WDmodel.py`` and you must start the process with ``mpirun``


.. _argparse:

======================
Useful runtime options
======================

There's a large number of command line options to the fitter, and most of it's
aspects can be configured. Some options make sense in concert with others, and
here's a short summary of use cases.

.. _quicklook:

Quick looks
~~~~~~~~~~~

The spectrum can be trimmed prior to fitting with the ``--trimspec``
option. You can also blotch over gaps and cosmic rays if your reduction
was sloppy, and you just need a quick fit, but it's better to do this
manually.

If there is no photometry data for the object, the fitter will barf
unless ``--ignorephot`` is specified explicitly, so you know that the
parameters are only constrained by the spectroscopy.

The fitter runs a MCMC to explore the posterior distribution of the model
parameters given the data. If you are running with the above two options,
chances are you are at the telescope, getting spectra, and doing quick look
reductions, and you just want a rough idea of temperature and surface gravity
to decide if you should get more signal, and eventually get HST photometry. The
MCMC is overkill for this purpose so you can ``--skipmcmc``, in which case,
you'll get results using minuit. They'll be biased, and the errors will
probably be too small, but they give you a ballpark estimate.

If you do want to use the MCMC anyway, you might like it to be faster. You can
choose to use only every nth point in computing the log likelihood with
``--everyn`` - this is only intended for testing purposes, and should probably
not be used for any final analysis. Note that the uncertainties increase as
you'd expect with fewer points. 

.. _init:

Setting the initial state
~~~~~~~~~~~~~~~~~~~~~~~~~

The fitter really runs minuit to refine initial supplied guesses for
parameters. Every now at then, the guess prior to running minuit is so far off
that you get rubbish out of minuit. This can be fixed by explictly supplying a
better initial guess. Of course, if you do that, you might wonder why even
bother with minuit, and may wish to skip it entirely. This can be disabled with
the ``--skipminuit`` option.  If ``--skipminuit`` is used, a dl guess **MUST**
be specified.

All of the parameter files can be supplied via a JSON parameter file
supplied via the ``--param_file`` option, or using individual parameter
options. An example parameter file is available in the module directory.

.. _samptype:

Configuring the sampler
~~~~~~~~~~~~~~~~~~~~~~~

You can change the sampler type (``-samptype``), number of chain temperatures
(``--ntemps``), number of walkers (``--nwalkers``), burn in steps
(``--nburnin``), production steps (``--nprod``), and proposal scale for the
MCMC (``--ascale``). You can also thin the chain (``--thin``) and discard some
fraction of samples from the start (``--discard``). The default sampler is the
ensemble sampler from the :py:mod:`emcee` package. For a more conservative
approach, we recommend the ptsampler with ``ntemps=5``, ``nwalkers=100`,
``nprod=5000`` (or more).

.. _resume:

Resuming the fit
~~~~~~~~~~~~~~~~

If the sampling needs to be interrupted, or crashes for whatever reason, the
state is saved every 100 steps, and the sampling can be restarted with
``--resume``. Note that you must have run at least the burnin and 100 steps for
it to be possible to resume, and the state of the data, parameters, or chain
configuration should not be changed externally (if they need to be use
``--redo`` and rerun the fit). You can increase the length of the chain, and
chain the visualization options when you ``--resume`` but the state of
everything else is restored.

You can get a summary of all available options with ``--help``
