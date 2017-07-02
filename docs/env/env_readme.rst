Setting up an environment vs setting up a known good environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``env`` folder contains files to help get you setup using a consistent
environment with all packages specified.

The ``requirements_py[27|36].txt`` files contains a list of required python
packages and known working versions for each. They differ from the
``dependencies_py[27|36].txt`` files in the root directory in that those files
specify packages and version ranges, rather than exact versions, to allow conda
to resolve dependecies and pull updated versions.

Of course, the environment really needs more than just python packages, while
``pip`` only manages python packages. The conda environment files,
``conda_environment_py[27|37]_[osx64|i686].yml`` files can be used to create
conda environments with exact versions of all the packages for python 2.7 or
3.6 on OS X or linux. This is the most reliable way to recreate the entire
environment.
