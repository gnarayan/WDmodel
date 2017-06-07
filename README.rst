WDmodel
=======

**Copyright 2017- Gautham Narayan (gnarayan@noao.edu)**

DA White Dwarf model atmosphere code for CALSPEC recalibration
--------------------------------------------------------------

Infers intrinsic Tlusty model params & extrinsic reddening params from
DA White Dwarf spectra and HST Photometry HST photometry is through GO
programs 12967 (cycle 20), 13711 (cycle 22)

Imperfectly flux-calibrated Spectra are modeled with the DA white dwarf
atmosphere, reddened with an extinction law and the residuals are modeled with
a Gaussian process.

Very much in beta - caveat emptor

--------------

Installation Instructions
-------------------------

1. Get the code:
~~~~~~~~~~~~~~~~

Clone this repository

-  ``git clone https://github.com/gnarayan/WDmodel.git``


2. Get the python environment configured:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We recommend using the anaconda python distribution.

If you elect to use your system python, or some other distribution, skip ahead to step3_.

1. To install with miniconda/anaconda, follow instructions `here <https://conda.io/docs/install/quick.html#linux-miniconda-install>`__

**Make sure you added the conda/bin dir to your path!**

2. Edit your condarc to have channels for all the packages. 
  
We've included an example version which you can copy to your home directory,
else edit your own appropriately. Note that in an ideal world, the same package
version from a different channel will work identically. The world is seldom
ideal.

- ``cp WDmodel/docs/condarc.example ~/.condarc``

3. Create a new environment from specification

- ``conda env create -f WDmodel/docs/conda_environment_py27.yml``

You can now skip over step 5!

*or*  
    
Create a new environment from scratch

- ``conda create -n WDmodel``
- ``source activate WDmodel``

*or*

Else if you want astroconda, follow the instructions `here <https://astroconda.readthedocs.io/en/latest/>`__

-  ``source activate astroconda``


3. Get the latest HST CDBS files:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _step3:

These are available over FTP from
[ftp://archive.stsci.edu/pub/hst/pysynphot/]

Untar them, and set the ``PYSYN_CDBS`` environment variable

-  ``export PYSYN_CDBS=place_you_untarred_the_files``


4. cd to directory you git cloned:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, we got email about this step, so we are including it explictly.

-  ``cd WDmodel``
  

5. Install other requirements:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you did not create your environment with the ``conda_environment_py27.yml``
file, then you will need to install the other requirements.

Install all the packages with conda

- ``conda install --yes --file requirements.txt``

*or*

- install eigen3 headers and your favorite flavor of mpi. See the notes_ at end.
- ``pip install -r requirements.txt``


6. GET THE DATA:
~~~~~~~~~~~~~~~~

Instructions will be available here when the paper is accepted. In the meantime
there's a single test object in the spectroscopy directory. If you want more,
Write your own HST proposal! :-P


7. Run a fit single threaded:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``./fit_WDmodel.py --specfile data/spectroscopy/yourfavorite.flm``

This option is single threaded and slow, but useful to testing or quick
exploratory analysis.

A more reasonable way to run things fast is to use mpi.


8. Run as an MPI process:
~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``mpirun -np 8 fit_WDmodel.py --mpi --specfile=file.flm [--ignorephot]``

Note that ``--mpi`` **MUST** be specified in the options to
``fit_WDmodel.py`` and you must start the process with ``mpirun``

--------------


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

--------------

Some extra notes: 
-----------------
.. _notes: 

If you followed the installation process detailed above, you shouldn't need
these notes.

Installing eigen3:
~~~~~~~~~~~~~~~~~~

if eigen3 isn't on your system, install it with conda:

-  ``conda install -c conda-forge eigen``

or for OS X do:

-  ``brew install eigen``

or on a linux system with apt:

-  ``apt-get install libeigen3-dev``

or compile it from `source <http://eigen.tuxfamily.org/index.php?title=Main_Page>`__


Installing OpenMPI and mpi4py:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if no mpi is on your system, install it with conda (along with mpi4py)

- ``conda install -c mpi4py mpich mpi4py``

or for OS X do:

- ``brew install [mpich|mpich2|open-mpi]``

on a linux system with apt:

-  ``apt-get install openmpi-bin``

and if you had to resort to brew or apt, then finish with: 

-  ``pip install mpi4py``


Notes from installing on the Odyssey cluster at Harvard:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These may be of use to get the code up and running with MPI on some
other cluster. Good luck.

Odyssey uses the lmod system for module management, like many other clusters
You can ``module spider openmpi`` to find what the openmpi modules. 

The advantage to using this is distributing your computation over multiple
nodes. The disadvantage is that you have to compile mpi4py yourself against
the cluster mpi.

-  ``module load gcc/6.3.0-fasrc01 openmpi/2.0.2.40dc0399-fasrc01``
-  ``wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-2.0.0.tar.gz``
-  ``tar xvzf mpi4py-2.0.0.tar.gz``
-  ``cd mpi4py-2.0.0``
-  ``python setup.py build --mpicc=$(which mpicc)``
-  ``python setup.py build_exe --mpicc="$(which mpicc) --dynamic"``
-  ``python setup.py install``

Note that if the cluster has eigen3 include files already, you might want to
compile celerite against them, instead of the conda version. To do that:

-  ``pip install celerite --global-option=build_ext --global-option=-I/path/to/eigen3``


--------------

TODO:
-----

-  More testing with a full testing suite
-  Add Rauch model atmospheres for comparison with Tlusty
-  All of the documentation
-  setup.py
-  Push to PyPI

You can read the first version of our analysis of four of the Cycle 20
objects
`here <http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1603.03825>`__

That analysis used custom IDL routines from Jay Holberg (U. Arizona) to
infer DA intrinsic parameters and custom python code to fit the
reddening parameters. This code is intended to (significantly) improve
on that analysis
