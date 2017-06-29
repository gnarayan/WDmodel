WDmodel
=======

**Copyright 2017- Gautham Narayan (gnarayan@noao.edu)**

About
-----

This is a DA White Dwarf model atmosphere fitting code. It fits observed
spectrophotometry of DA White Dwarfs to infer intrinsic model atmosphere
parameters in the presence of dust and correlated spectroscopic flux
calibration errors, thereby determining full SEDs for the white dwarf. It's
primary scientific purpose is to establish a network of faint (V = 16.5--19
mag) standard stars, suitable for LSST and other wide-field photometric
surveys, and tied to HST and the CALSPEC system, defined by the three primary
standards, GD71, GD153 and G191B2B.

Click on the badge below for installation and usage instructions, as well as
package documentation:

|docs|

The code has been tested on Python 2.7 and 3.6 on both OS X (El Capitan and
Sierra) and Liniux (Debian-derivatives). Send us email or open an issue if you
need help!

We're working on a publication with the results of our Cycle 22 and Cycle 20
data, while ramping up for Cycle 25! A full data release of Cycle 20 and 22 HST
data, and ground-based imaging and spectroscopy will accompany the publication.
Look for an updated link here!

TODO
----

-  More testing with a full testing suite
-  Add Rauch model atmospheres for comparison with Tlusty
-  setup.py
-  Push to PyPI
-  PUBLISH!

You can read the first version of our analysis of four of the Cycle 20
objects
`here <http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1603.03825>`__

That analysis used custom IDL routines from Jay Holberg (U. Arizona) to
infer DA intrinsic parameters and custom python code to fit the
reddening parameters. This code is intended to (significantly) improve
on that analysis


.. |docs| image:: http://readthedocs.org/projects/wdmodel/badge/?version=latest
    :alt: Documentation Status
    :scale: 120%
    :target: http://wdmodel.readthedocs.io/en/latest/?badge=latest
