# WDmodel

Copyright 2016- Gautham Narayan (gnarayan@noao.edu)

DA White Dwarf model atmosphere code for CALSPEC recalibration

Infers intrinsic Tlusty model params & extrinsic reddening params from DA White
Dwarf spectra and HST Photometry HST photometry is through GO programs 12967
(cycle 20),  13711 (cycle 22)

Imperfectly flux-calibrated Spectra are modelled with the DA white dwarf
atmosphere, reddened with an extinction law and the residuals are modelled as
the uncertainties + a Gaussian process with a Exponential Squared Kernel, with
a length scale that is bounded to be large.

A list of packages needed to run this code is available in requirements.txt
pip install -r requirements.txt

Very much in alpha - caveat emptor

TODO:
- Add in inference from photometry
- More testing with a full testing suite
- Infer true HST WFC3 zeropoints using spectra + photometry of three primary standards (GD71, GD153, G191B2b)
- Add Rauch model atmospheres for comparison with Tlusty
- All of the documentation 
- setup.py
- Push to PyPI


You can read the first version of our analysis of four of the Cycle 20 objects here:
http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1603.03825

That analysis used custom IDL routines from Jay Holberg (U. Arizona) to infer
DA intrinsic parameters and custom python code to fit the reddening parameters.
This code is inteded to (significantly) improve on that analysis
