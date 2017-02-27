# WDmodel

__Copyright 2017- Gautham Narayan (gnarayan@noao.edu)__

## DA White Dwarf model atmosphere code for CALSPEC recalibration

Infers intrinsic Tlusty model params & extrinsic reddening params from DA White
Dwarf spectra and HST Photometry HST photometry is through GO programs 12967
(cycle 20),  13711 (cycle 22)

Imperfectly flux-calibrated Spectra are modelled with the DA white dwarf
atmosphere, reddened with an extinction law and the residuals are modelled as
the uncertainties + a Gaussian process with a Exponential Squared Kernel, with
a length scale that is fixed to 5000 Angstroms by default, or bounded to be
large.

A list of packages needed to run this code is available in requirements.txt
pip install -r requirements.txt

Very much in beta - caveat emptor

## To run (minimally):
`fit_WDmodel.py --specfile=file.flm [--ignorephot]`  

This option is single threaded and slow, but useful to testing or quick
exploratory analysis.

A more reasonable way to run things fast is to use mpi. 

`mpirun -np 8 fit_WDmodel.py mpi --specfile=file.flm [--ignorephot]`  

Note that `mpi` __MUST__ be the first option after `fit_WDmodel.py` and you
must start the process with `mpirun`

You can also run the MCMC again with different valies for `nburnin`, `nprod`
and `nwalkers`. This involves running the fitter with the regular
`fit_WDmodel.py`  at least upto the MCMC stage to produce all the input files,
and then restoring the inputs to the MCMC stage and running with
`mpifit_WDmodel.py`.

`fit_WDmodel.py --specfile=file.flm [--ignorephot] --skipmcmc`  
`mpirun -np 8 mpifit_WDmodel.py --specfile=file.flm [--ignorephot]`  

### Some useful options

The spectrum can be trimmed prior to fitting with the `--trimspec` option. You
can also blotch over gaps and cosmic rays if your reduction was sloppy, and you
just need a quick fit, but it's better to do this manually.

The fitter runs minuit to refine initial supplied guesses for teff, logg, av
and dl. This can be disabled with the `--skipminuit` option.

All of the parameter files can be supplied via a JSON parameter file supplied
via the `--param_fil`e option, or using individual parameter options. An example
parameter file is available in the module directory. 

You can change the number of walkers, burn in steps, production steps, and
proposal scale for the MCMC. You can also choose to use only every nth point in
computing the loglikelihood - this is only intended for testing purposes, and
should probably not be used for any final analysis. Note that the
uncertainities increase as you'd expect with fewer points.

You can get a summary of all available options with `--help`


## TODO:
* Add in inference from photometry
* More testing with a full testing suite
* Infer true HST WFC3 zeropoints using spectra + photometry of three primary standards (GD71, GD153, G191B2b)
* Add Rauch model atmospheres for comparison with Tlusty
* All of the documentation 
* setup.py
* Push to PyPI


You can read the first version of our analysis of four of the Cycle 20 objects
[here] (http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1603.03825)

That analysis used custom IDL routines from Jay Holberg (U. Arizona) to infer
DA intrinsic parameters and custom python code to fit the reddening parameters.
This code is inteded to (significantly) improve on that analysis
