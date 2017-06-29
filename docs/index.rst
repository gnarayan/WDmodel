.. WDmodel documentation master file, created by
   sphinx-quickstart on Sun May 28 12:08:59 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to WDmodel's documentation!
===================================

.. automodule:: WDmodel

``WDmodel`` is a DA White Dwarf model atmosphere fitting code. It fits observed
spectrophotometry of DA White Dwarfs to infer intrinsic model atmosphere
parameters in the presence of dust and correlated spectroscopic flux
calibration errors, thereby determining full SEDs for the white dwarf. It's
primary scientific purpose is to establish a network of faint (V = 16.5--19
mag) standard stars, suitable for LSST and other wide-field photometric
surveys, and tied to HST and the CALSPEC system, defined by the three primary
standards, GD71, GD153 and G191B2B.

This document will help get you up and running with the ``WDmodel`` package. 

For the most part, you can simply execute code in grey boxes to get things up
and running, and ignore the text initially. Come back to it when you need help,
or to configure the fitter.

.. toctree::
   :maxdepth: 3
   :caption: Contents
   :titlesonly:

   Installation <installation>
   Usage <usage>
   Package documentation <source/modules.rst>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
