# WDmodel

__Copyright 2017- Gautham Narayan (gnarayan@noao.edu)__

## DA White Dwarf model atmosphere code for CALSPEC recalibration

Infers intrinsic Tlusty model params & extrinsic reddening params from DA White
Dwarf spectra and HST Photometry HST photometry is through GO programs 12967
(cycle 20),  13711 (cycle 22)

Imperfectly flux-calibrated Spectra are modeled with the DA white dwarf
atmosphere, reddened with an extinction law and the residuals are modeled as
the uncertainties + a Gaussian process with a Exponential Squared Kernel, with
a length scale that is fixed to 5000 Angstroms by default, or bounded to be
large.

Very much in beta - caveat emptor

______

## Installation Instructions

### get the code:
Clone this repository

* `git clone https://github.com/gnarayan/WDmodel.git`

### install miniconda/anaconda/astroconda if you haven't already:
Follow instructions [here](https://astroconda.readthedocs.io/en/latest/)

* `source activate astroconda`

(Make sure you added the conda/bin dir to your path!)

### install eigen3:
if it isn't on your system - for OS X do:

* `brew install eigen`

or on a linux system with apt:

* `apt-get install libeigen3-dev`

or compile it from [source](http://eigen.tuxfamily.org/index.php?title=Main_Page)

### Get the latest HST CDBS files:
These are available over FTP from [ftp://archive.stsci.edu/pub/hst/pysynphot/]

Untar them, and set the `PYSYN_CDBS` environment variable

* `export PYSYN_CDBS=place_you_untarred_the_files`


### cd to directory you git cloned:
* `cd WDmodel`

### install other requirements:
* `pip install -r requirements.txt`

### GET THE DATA:
Will be available here when the paper is accepted. In the meantime there's a
single test object in the spectroscopy directory. If you want more, Write your
own HST proposal! :-P

### run a fit single threaded:
* `./fit_WDmodel.py --specfile data/spectroscopy/yourfavorite.flm`

This option is single threaded and slow, but useful to testing or quick
exploratory analysis.

A more reasonable way to run things fast is to use mpi.

### Install OpenMPI and mpi4py:
* `apt-get install openmpi-bin`

* `pip install mpi4py`


### Run as an MPI process:
* `mpirun -np 8 fit_WDmodel.py mpi --specfile=file.flm [--ignorephot]`

Note that `mpi` __MUST__ be the first option after `fit_WDmodel.py` and you
must start the process with `mpirun`

______

## Notes from installing on the Odyssey cluster at Harvard
These may be of use to get the code up and running with MPI on some other
cluster. Good luck. 

### git clone the repository and install anaconda/astroconda/miniconda
Follow the regular instructions above until install eigen3

### Install everything but emcee, george and celerite
* `pip install (everything but emcee, george and celerite)`

### Follow this process
Odyssey uses the lmod system for module management, like many other clusters
You can `module spider openmpi` to find what the openmpi modules
There may also be an eigen module someplace, in which case use that instead of
the conda version

* `conda install eigen` or "install" (i.e. copy the headers someplace) from
  source as in the normal instructions
* `module load gcc/6.3.0-fasrc01 openmpi/2.0.2.40dc0399-fasrc01`
* `wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-2.0.0.tar.gz`
* `tar xvzf mpi4py-2.0.0.tar.gz`
* `cd mpi4py-2.0.0`
* `python setup.py build --mpicc=$(which mpicc)`
* `python setup.py build_exe --mpicc="$(which mpicc) --dynamic"`
* `python setup.py install`
* `cd ..`
* `pip install emcee`
* `pip install george --global-option=build_ext --global-option=-I/path/to/eigen3`
* `pip install celerite --global-option=build_ext --global-option=-I/path/to/eigen3`

`bin/make_batch_scripts.py` can setup batch SLURM scripts that you can submit with `sbatch`
______

## Some useful options:

The spectrum can be trimmed prior to fitting with the `--trimspec` option. You
can also blotch over gaps and cosmic rays if your reduction was sloppy, and you
just need a quick fit, but it's better to do this manually.

If there is no photometry data for the object, the fitter will barf unless
`--ignorephot` is specified explicitly, so you know that the parameters are
only constrained by the spectroscopy.

The fitter runs minuit to refine initial supplied guesses for teff, logg, av
and dl. This can be disabled with the `--skipminuit` option. If `--skipminuit` is
used, a dl guess __MUST__ be specified.

All of the parameter files can be supplied via a JSON parameter file supplied
via the `--param_file` option, or using individual parameter options. An example
parameter file is available in the module directory.

You can change the number of walkers, burn in steps, production steps, and
proposal scale for the MCMC. You can also choose to use only every nth point in
computing the log likelihood with `--everyn` - this is only intended for
testing purposes, and should probably not be used for any final analysis. Note
that the uncertainities increase as you'd expect with fewer points.

Occasionally, you might see very low-acceptance fractions, accompanied with
weird spikes in the fsig parameter. We've noticed this on the odyssey cluster,
but aren't yet sure why it is happening. It appears to be the HODLR solver that
is the default. The problem goes away completely with `--usebasic`, which forces
the basic solver, at the cost of runtime. We have not seen the problem, with
the exact same data and settings on a linux desktop or Macbook Pro.

You can get a summary of all available options with `--help`
______

## TODO:
* More testing with a full testing suite
* Add Rauch model atmospheres for comparison with Tlusty
* All of the documentation
* setup.py
* Push to PyPI


You can read the first version of our analysis of four of the Cycle 20 objects
[here](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1603.03825)

That analysis used custom IDL routines from Jay Holberg (U. Arizona) to infer
DA intrinsic parameters and custom python code to fit the reddening parameters.
This code is inteded to (significantly) improve on that analysis
