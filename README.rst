WDmodel
=======

**Copyright 2017- Gautham Narayan (gsnarayan@gmail.com)**

About
-----
|github| |license| |docs| |doi| 

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
    
.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1699215.svg
   :target: https://doi.org/10.5281/zenodo.1699215

Compatibility
-------------
|travis| |python| |coveralls| 

The code has been tested on Python 3.6 on both OS X (El Capitan, Sierra and High Sierra) and Linux (Debian-derivatives). 
Python 3.6 is highly recommended. 
Python 2.7 should work, but consider upgrading because this package's dependencies are dropping 2.7 support. 
Send us email or open an issue if you need help!

.. |travis| image:: https://travis-ci.org/gnarayan/WDmodel.svg?branch=master
    :alt: Travis badge
    :target: https://travis-ci.org/gnarayan/WDmodel

.. |python| image:: https://img.shields.io/badge/python-3.6-blue.svg
    :alt: Python badge
    :target: https://www.python.org/

.. |coveralls| image:: https://coveralls.io/repos/github/gnarayan/WDmodel/badge.svg?branch=master
    :alt: Coveralls badge
    :target: https://coveralls.io/github/gnarayan/WDmodel?branch=master

Analysis
--------

We've published the results from our combined Cycle 22 and Cycle 20 data, while
ramping up for Cycle 25! 

You can read about the analysis in 
`Narayan et al., 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJS..241...20N/abstract>`__

and the data in
`Calamida et al., 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...872..199C/abstract>`__

The data from Cycle 20 and 22 are available
`on Zenodo <https://doi.org/10.5281/zenodo.2032365>`__

You can read also the first version of our preliminary analysis of four of the Cycle 20
objects
`here <http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1603.03825>`__

That analysis was intended as a proof-of-concept and used custom IDL routines
from Jay Holberg (U. Arizona) to infer DA intrinsic parameters and custom
python code to fit the reddening parameters. This code is intended to
(significantly) improve on that analysis.
