# WDmodel

Copyright 2016- Gautham Narayan (gnarayan@noao.edu)

DA White Dwarf model atmosphere code for CALSPEC recalibration

Infers intrinsic Tlusty model params & extrinsic reddening params from DA White Dwarf spectra and HST Photometry
HST photometry is through GO programs 12967 (cycle 20),  13711 (cycle 22)

Very much in alpha - caveat emptor
At present, this simply models the spectrum to infer Teff, logg and Av

TODO:
- More reliable extraction of Balmer lines 
- More robust treatment of covariance with george
- Add in inference from photometry
- More testing with a full testing suite
- Infer true HST WFC3 zeropoints using spectra + photometry of three primary standards (GD71, GD153, G191B2b)
- Add Rauch model atmospheres for comparison with Tlusty
- Code needs signficant refactoring
- All of the documentation 


You can read the first version of our analysis of four of the Cycle 20 objects here:
http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1603.03825

That analysis used custom IDL routines from Jay Holberg (U. Arizona) to infer DA intrinsic parameters
and custom python code to fit the reddening parameters. This code is inteded to (significantly) 
improve on that analysis
