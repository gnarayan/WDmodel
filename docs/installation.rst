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

    - ``conda env create -f WDmodel/docs/conda_environment_py27_[osx64|i686].yml``

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