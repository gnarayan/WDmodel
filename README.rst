WDmodel
=======

**Copyright 2017- Gautham Narayan (gnarayan@noao.edu)**

About
-----
|github| |license| |docs|

``WDmodel`` is a DA White Dwarf model atmosphere fitting code. It fits observed
spectrophotometry of DA White Dwarfs to infer intrinsic model atmosphere
parameters in the presence of dust and correlated spectroscopic flux
calibration errors, thereby determining full SEDs for the white dwarf. Its
primary scientific purpose is to establish a network of faint (V = 16.5--19
mag) standard stars, suitable for LSST and other wide-field photometric
surveys, and tied to HST and the CALSPEC system, defined by the three primary
standards, GD71, GD153 and G191B2B.

Click on the badges above  for code, licensing and documentation.

.. |github| image:: https://img.shields.io/badge/Github-gnarayan%2FWDmodel-blue.svg
    :alt: Github Link
    :target: http://github.com/gnarayan/WDmodel

.. |license| image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
    :alt: GPLv3 License
    :target: http://www.gnu.org/licenses/gpl-3.0

.. |docs| image:: http://readthedocs.org/projects/wdmodel/badge/?version=latest
    :alt: Documentation Status
    :target: http://wdmodel.readthedocs.io/en/latest/?badge=latest

Compatability
-------------

The code has been tested on Python 2.7 and 3.6 on both OS X (El Capitan and
Sierra) and Linux (Debian-derivatives). Send us email or open an issue if you
need help!

Analysis
--------

We're working on a publication with the results from our combined Cycle 22 and
Cycle 20 data, while ramping up for Cycle 25! A full data release of Cycle 20
and 22 HST data, and ground-based imaging and spectroscopy will accompany the
publication.  Look for an updated link here!

You can read the first version of our analysis of four of the Cycle 20
objects
`here <http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1603.03825>`__

That analysis was intended as a proof-of-concept and used custom IDL routines
from Jay Holberg (U. Arizona) to infer DA intrinsic parameters and custom
python code to fit the reddening parameters. This code is intended to
(significantly) improve on that analysis.

TODO
----

-  More testing with a full testing suite
-  Add Rauch model atmospheres for comparison with Tlusty
-  Push to PyPI
-  PUBLISH!

