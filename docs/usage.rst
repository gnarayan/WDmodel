Useful runtime options:
-----------------------

The spectrum can be trimmed prior to fitting with the ``--trimspec``
option. You can also blotch over gaps and cosmic rays if your reduction
was sloppy, and you just need a quick fit, but it's better to do this
manually.

If there is no photometry data for the object, the fitter will barf
unless ``--ignorephot`` is specified explicitly, so you know that the
parameters are only constrained by the spectroscopy.

The fitter runs minuit to refine initial supplied guesses for teff,
logg, av and dl. This can be disabled with the ``--skipminuit`` option.
If ``--skipminuit`` is used, a dl guess **MUST** be specified.

All of the parameter files can be supplied via a JSON parameter file
supplied via the ``--param_file`` option, or using individual parameter
options. An example parameter file is available in the module directory.

You can change the sampler type (``-samptype``), number of chain temperatures
(``--ntemps``), number of walkers (``--nwalkers``), burn in steps
(``--nburnin``), production steps (``--nprod``), and proposal scale for the
MCMC (``--ascale``). You can also thin the chain (``--thin``) and discard some
fraction of samples from the start (``--discard``).

If the sampling needs to be interrupted, or crashes for whatever reason, the
state is saved every 100 steps, and the sampling can be restarted with
``--resume``. Note that you must have run at least the burnin and 100 steps for
it to be possible to resume, and the state of the data, parameters, or chain
configuration should not be changed externally (if they need to be use
``--redo`` and rerun the fit). You can increase the length of the chain, and
chain the visualization options when you ``--resume`` but the state of
everything else is restored.

You can also choose to use only every nth point in computing the log likelihood
with ``--everyn`` - this is only intended for testing purposes, and should
probably not be used for any final analysis. Note that the uncertainties
increase as you'd expect with fewer points. 

You can get a summary of all available options with ``--help``
