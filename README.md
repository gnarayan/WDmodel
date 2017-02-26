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

A more reasonable way to run things fast is to use mpi. This involves running
the fitter upto the MCMC stage to produce all the input files, and then
skipping the stage in favor of running it over mpi.


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

### There are many optional arguments to configure the fitter

`optional arguments:
  -h, --help            show this help message and exit
  --param_file PARAM_FILE
                        Specify parameter config JSON file (default: None)

spectrum:
  Spectrum options

  --specfile SPECFILE   Specify spectrum to fit (default: None)
  --trimspec BLUELIM REDLIM
                        Trim spectrum to wavelength range (default: (None,
                        None))
  --blotch              Blotch the spectrum to remove gaps/cosmic rays before
                        fitting? (default: False)

photometry:
  Photometry options

  --photfile PHOTFILE   Specify file containing photometry lookup table for
                        objects (default: data/WDphot_C22.dat)
  --reddeningmodel REDDENINGMODEL
                        Specify functional form of reddening law (default:
                        od94)
  --ignorephot          Ignores missing photometry and does the fit with just
                        the spectrum (default: False)

model:
  Model options. Modify using --param_file or CL. CL overrides. Caveat
  emptor.

  --teff TEFF           Specify param teff value (default: 35000.0)
  --teff_fix TEFF_FIX   Specify if param teff is fixed (default: False)
  --teff_scale TEFF_SCALE
                        Specify param teff scale/step size (default: 2000.0)
  --teff_bounds LOWERLIM UPPERLIM
                        Specify param teff bounds (default: [16000.0,
                        90000.0])
  --logg LOGG           Specify param logg value (default: 7.8)
  --logg_fix LOGG_FIX   Specify if param logg is fixed (default: False)
  --logg_scale LOGG_SCALE
                        Specify param logg scale/step size (default: 0.1)
  --logg_bounds LOWERLIM UPPERLIM
                        Specify param logg bounds (default: [7.0, 9.5])
  --av AV               Specify param av value (default: 0.2)
  --av_fix AV_FIX       Specify if param av is fixed (default: False)
  --av_scale AV_SCALE   Specify param av scale/step size (default: 0.05)
  --av_bounds LOWERLIM UPPERLIM
                        Specify param av bounds (default: [0.0, 2.0])
  --rv RV               Specify param rv value (default: 3.1)
  --rv_fix RV_FIX       Specify if param rv is fixed (default: True)
  --rv_scale RV_SCALE   Specify param rv scale/step size (default: 0.18)
  --rv_bounds LOWERLIM UPPERLIM
                        Specify param rv bounds (default: [1.7, 5.1])
  --dl DL               Specify param dl value (default: None)
  --dl_fix DL_FIX       Specify if param dl is fixed (default: False)
  --dl_scale DL_SCALE   Specify param dl scale/step size (default: 10.0)
  --dl_bounds LOWERLIM UPPERLIM
                        Specify param dl bounds (default: [1e-07, 10000000.0])
  --fwhm FWHM           Specify param fwhm value (default: None)
  --fwhm_fix FWHM_FIX   Specify if param fwhm is fixed (default: False)
  --fwhm_scale FWHM_SCALE
                        Specify param fwhm scale/step size (default: 0.5)
  --fwhm_bounds LOWERLIM UPPERLIM
                        Specify param fwhm bounds (default: [0.1, 25.0])
  --sigf SIGF           Specify param sigf value (default: 0.01)
  --sigf_fix SIGF_FIX   Specify if param sigf is fixed (default: False)
  --sigf_scale SIGF_SCALE
                        Specify param sigf scale/step size (default: 0.005)
  --sigf_bounds LOWERLIM UPPERLIM
                        Specify param sigf bounds (default: [1e-09, 1000.0])
  --tau TAU             Specify param tau value (default: 5000.0)
  --tau_fix TAU_FIX     Specify if param tau is fixed (default: True)
  --tau_scale TAU_SCALE
                        Specify param tau scale/step size (default: 100.0)
  --tau_bounds LOWERLIM UPPERLIM
                        Specify param tau bounds (default: [200.0, 10000.0])

mcmc:
  MCMC options

  --skipminuit          Skip Minuit fit - make sure to specify dl guess
                        (default: False)
  --skipmcmc            Skip MCMC - if you skip both minuit and MCMC, simply
                        prepares files (default: False)
  --ascale ASCALE       Specify proposal scale for MCMC (default: 2.0)
  --nwalkers NWALKERS   Specify number of walkers to use (0 disables MCMC)
                        (default: 200)
  --nburnin NBURNIN     Specify number of steps for burn-in (default: 50)
  --nprod NPROD         Specify number of steps for production (default: 1000)
  --everyn EVERYN       Use only every nth point in data for computing
                        likelihood - useful for testing. (default: 1)
  --discard DISCARD     Specify percentage of steps to be discarded (default:
                        5)

viz:
  Visualization options

  -b BALMERLINES [BALMERLINES ...], --balmerlines BALMERLINES [BALMERLINES ...]
                        Specify Balmer lines to visualize [1:7] (default: [1,
                        2, 3, 4, 5, 6])
  --ndraws NDRAWS       Specify number of draws from posterior to overplot for
                        model (default: 21)

output:
  Output options

  -o OUTDIR, --outdir OUTDIR
                        Specify a custom output directory. Default is
                        CWD+objname/ subdir (default: None)
  --redo                Clobber existing fits (default: False`


## TODO:
* Add in inference from photometry
* More testing with a full testing suite
* Infer true HST WFC3 zeropoints using spectra + photometry of three primary standards (GD71, GD153, G191B2b)
* Add Rauch model atmospheres for comparison with Tlusty
* All of the documentation 
* setup.py
* Push to PyPI


You can read the first version of our analysis of four of the Cycle 20 objects here:
[link] (http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1603.03825)

That analysis used custom IDL routines from Jay Holberg (U. Arizona) to infer
DA intrinsic parameters and custom python code to fit the reddening parameters.
This code is inteded to (significantly) improve on that analysis
