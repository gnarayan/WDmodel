#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Runs some tests for coveralls on the WDmodel package. Just checks that the
functions execute, not that the output is sane.
"""
import sys
import numpy as np
import WDmodel.WDmodel
import WDmodel.io

def main():
    model = WDmodel.WDmodel.WDmodel()
    TEFF = 42757.
    LOGG = 7.732
    AV   = 0.01
    FWHM = 3.
    WAVE = np.arange(3000., 9001., 1.)

    model._get_model(TEFF, LOGG, wave=WAVE, log=True)

    model._get_model_nosp(TEFF, LOGG, wave=WAVE) 
    model._get_model_nosp(TEFF, LOGG, wave=WAVE, log=True)

    model._get_red_model(TEFF, LOGG, AV, WAVE)
    model._get_red_model(TEFF, LOGG, AV, WAVE, log=True)

    model._get_obs_model(TEFF, LOGG, AV, FWHM,  WAVE, log=True)
    
    _, testspec = model._get_full_obs_model(TEFF, LOGG, AV, FWHM,  WAVE, log=True)

    model._wave_test(testspec.wave[0:1])

    BADTEFF = 9000.
    BADLOGG = 6.5

    model.get_model(BADTEFF, BADLOGG, strict=False)

    model.get_red_model(TEFF, LOGG, AV, wave=WAVE)
    model.get_red_model(TEFF, LOGG, AV, wave=WAVE, log=True)

    model.get_obs_model(TEFF, LOGG, AV, FWHM,  wave=WAVE, log=True)

    model.extract_spectral_line(testspec.wave, testspec.flux, line=2)

    fn = 'out/test/test/test_mcmc.hdf5'
    WDmodel.io.read_mcmc(fn)

    return




    


if __name__=='__main__':
    sys.exit(main())
    
