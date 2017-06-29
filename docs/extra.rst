Some extra notes on installation
--------------------------------

If you followed the installation process detailed above, you shouldn't need
these notes, but they are provided for users who may be running on environments
they do not manage themselves.

.. toctree::
   :maxdepth: 2

* :ref:`Installing eigen3 without conda <eigen>`
* :ref:`Installing OpenMPI and mpi4py without conda <mpi>`
* :ref:`Installing on a cluster <cluster>`


.. _eigen:

Installing eigen3 without conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If eigen3 isn't on your system, and installing it with conda didn't work

For OS X do:

.. code-block:: console

 brew install eigen

or on a linux system with apt:

.. code-block:: console

 apt-get install libeigen3-dev

or compile it from `source <http://eigen.tuxfamily.org/index.php?title=Main_Page>`__


Note that if you do install it in a custom location, you may have to compile
celerite yourself.

.. code-block:: console
 
 pip install celerite --global-option=build_ext --global-option=-I/path/to/eigen3

.. _mpi:

Installing OpenMPI and mpi4py without conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if no mpi is on your system, and installing it with conda didn't work

For OS X do:

.. code-block:: console
 
 brew install [mpich|mpich2|open-mpi]

on a linux system with apt:

.. code-block:: console

 apt-get install openmpi-bin

and if you had to resort to brew or apt, then finish with: 

.. code-block:: console

 pip install mpi4py

.. _cluster:

Notes from installing on the Odyssey cluster at Harvard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These may be of use to get the code up and running with MPI on some
other cluster. Good luck.

Odyssey uses the lmod system for module management, like many other clusters
You can ``module spider openmpi`` to find what the openmpi modules. 

The advantage to using this is distributing your computation over multiple
nodes. The disadvantage is that you have to compile mpi4py yourself against
the cluster mpi.

.. code-block:: console

  module load gcc/6.3.0-fasrc01 openmpi/2.0.2.40dc0399-fasrc01
  wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-2.0.0.tar.gz
  tar xvzf mpi4py-2.0.0.tar.gz
  cd mpi4py-2.0.0
  python setup.py build --mpicc=$(which mpicc)
  python setup.py build_exe --mpicc="$(which mpicc) --dynamic"
  python setup.py install

