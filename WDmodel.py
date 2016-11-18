#!/usr/bin/env python
import sys
import os
import warnings
warnings.simplefilter('once')
import numpy as np
import h5py
import scipy.interpolate as spinterp


class WDmodel:
    def __init__(self, grid_file=None, grid_name=None):
        """
        constructs a white dwarf model atmosphere object
        Virtually nore of the attributes should be used directly
        since it is trivially possible to break the model by redefining them
        Access to them is best through the functions connected to the models
        """
        lno     = [   1    ,   2     ,    3     ,    4    ,   5      ,  6      ]
        lines   = ['alpha' , 'beta'  , 'gamma'  , 'delta' , 'zeta'   , 'eta'   ]
        H       = [6562.857, 4861.346, 4340.478 ,4101.745 , 3970.081 , 3889.056]
        #D       = [ 100.0  ,  130.0  ,  125.0   ,  75.0   ,  50.0    ,  27.0   ]
        D       = [ 130.0  ,  170.0  ,  125.0   ,  75.0   ,  50.0    ,  27.0   ]
        eps     = [  10.0  ,   10.0  ,   10.0   ,   8.0   ,   5.0    ,   3.0   ]
        self._lines = dict(zip(lno, zip(lines, H, D, eps)))
        # we're passing grid_file so we know which model to init 
        self.__init__tlusty(grid_file=grid_file)


    def __init__tlusty(self, grid_file=None, grid_name=None):
        """
        Initalize the Tlusty Model <grid_name> from the grid file <grid_file>
        """
        self._grid_file = 'TlustyGrids.hdf5'
        if grid_file is not None:
            if os.path.exists(grid_file):
                self._grid_file = grid_file
        _grids     = h5py.File(self._grid_file, 'r')
        
        # the original IDL SAV file Tlusty grids were annoyingly broken up by wavelength
        # this was because the authors had different wavelength spacings
        # since they didn't feel that the contiuum needed much spacing anyway
        # and "wanted to save disk space"
        # and then their old IDL interpolation routine couldn't handle the variable spacing
        # so they broke up the grids
        # So really, the original IDL SAV files were annoyingly broken up by wavelength because old dudes
        # We have concatenated these "large" arrays because we don't care about disk space
        # This grid is called "default", but the orignals also exist (because duplicates are great)
        # and you can pass grid_name to use them if you choose to
        self._default_grid = 'default'
        if grid_name is None:
            grid_name = self._default_grid
        try:
            grid = _grids[grid_name]
        except KeyError,e:
            message = 'Grid %s not found in grid_file %s. Accepted values are (%s)'%(grid_name, self._grid_file,\
                    ','.join(_grids.keys()))
            raise ValueError(message)

        self._grid_name = grid_name
        self._wave  = grid['wave'].value.astype('float64')
        self._ggrid = grid['ggrid'].value
        self._tgrid = grid['tgrid'].value
        _flux  = grid['flux'].value.astype('float64')
        self._fluxnorm = 1.
        
        # pre-init the interpolation and do it in log-space
        # note that we do the interpolation in log-log
        # this is because the profiles are linear, redward of the Balmer break in log-log
        # and the regular grid interpolator is just doing linear interpolation under the hood
        self._model = spinterp.RegularGridInterpolator((np.log10(self._wave), self._ggrid, self._tgrid),\
                np.log10(_flux))        
        _grids.close()


    def _get_xi(self, teff, logg, wave):
        """
        returns the formatted points for interpolation, xi
        """
        lwave = np.log10(wave)
        nwave = len(lwave)
        xi    = np.dstack((lwave, [logg]*nwave, [teff]*nwave))[0]
        return xi


    def _get_model(self, xi, log=False):
        if log:
            return self._model(xi)
        return (10.**self._model(xi))


    @classmethod
    def _wave_test(cls, wave):
        """
        Checks if the wavelengths passed are sane
        """
        if len(wave) == 0:
            message = 'Wavelengths not specified.'
            raise ValueError(message)

        if not np.all(wave > 0):
            message = 'Wavelengths are not all positive'
            raise ValueError(message)

        if len(wave) == 1:
            return 

        dwave = np.diff(wave)
        if not(np.all(dwave > 0) or np.all(dwave < 0)):
            message = 'Wavelength array is not monotonic'
            raise ValueError(message)

        
    def get_model(self, teff, logg, wave=None, log=False, strict=True):
        """
        Returns the model for some teff, logg at wavelengths wave
        If not specified, wavelengths are from 3000-9000A

        Checks inputs for consistency and calls _get_xi(), _get_model()
        If you need the model repeatedly for slightly different parameters, use those functions directly
        """
        if wave is None:
            wave = self._wave

        wave = np.atleast_1d(wave)
        self._wave_test(wave)

        teff = float(teff)
        logg = float(logg)

        if not ((teff >= self._tgrid.min()) and (teff <= self._tgrid.max())):
            message = 'Temperature out of model range'
            if strict:
                raise ValueError(message)
            else:
                warnings.warn(message,RuntimeWarning)
                teff = min([self._tgrid.min(), self._tgrid.max()], key=lambda x:abs(x-teff))

        if not ((logg >= self._ggrid.min()) and (logg <= self._ggrid.max())):
            message = 'Surface gravity out of model range'
            if strict:
                raise ValueError(message)
            else:
                warnings.warn(message,RuntimeWarning)
                logg = min([self._ggrid.min(), self._ggrid.max()], key=lambda x:abs(x-logg))

        outwave = wave[((wave >= self._wave.min()) & (wave <= self._wave.max()))]

        if len(outwave) > 0:
            xi = self._get_xi(teff, logg, outwave)
            outflux = self._get_model(xi,log=log)
            return outwave, outflux
        else:
            message = 'No valid wavelengths'
            raise ValueError(message)
            

    def _normalize_model(self, spec, log=False):
        """
        Imprecise normalization for visualization only
        If you want to normalize model and data, get the model, and the data,
        compute the integrals and use the ratio to properly normalize
        """
        wave = spec.wave
        flux = spec.flux
        ind = np.where((self._wave >= wave.min()) & (self._wave <= wave.max()))[0]
        modelmedian = np.median(self._model.values[ind,:,:])
        if not log:
            modelmedian = 10.**modelmedian    
        datamedian  = np.median(flux)
        self._fluxnorm = modelmedian/datamedian
        return self._fluxnorm

    @classmethod
    def _get_indices_in_range(cls, w, WA, WB, W0=None):
        if W0 is None:
            W0 = WA + (WB-WA)/2.
        ZE  = np.where((w >= WA) & (w <= WB))
        return W0, ZE


    def _get_line_indices(self, w, line):
        """
        Returns the central wavelength, and _indices_ of the line profile 
        The widths of the profile are predefined in the _lines attribute
        """
        _, W0, WID, DW = self._lines[line]
        WA  = W0 - WID - DW
        WB  = W0 + WID + DW
        #wred  = w[((W0 + WID) <  w < WB)]        
        #wblue = w[(WA < w < (W0 - WID))] 
        #wbluecore = w[(W0 - WID) < w < W0]
        #wredcore  = W[(W0 < w < (W0 + WID))]
        #if len(wblue)==0 or len(wred) == 0 or len(wbluecore) == 0 or len(wredcore) == 0:
        #    message = 'Spectrum does not adequately cover line %s'%line
        #    raise ValueError(message)
        return self._get_indices_in_range(w, WA, WB, W0=W0)



    def _extract_from_indices(self, w, f, ZE, df=None):
        """
        Returns the wavelength and flux of the line using the indices ZE which can be determined by _get_line_indices
        Optionally accepts the noise vector to extract as well
        """
        if df is None:
            return w[ZE], f[ZE]
        else:
            return w[ZE], f[ZE], df[ZE]
        


    def extract_spectral_line(self, w, f, line, df=None):
        """
        extracts a section of a line, fits a straight line to the flux outside of the line core
        to model the continuum, and then divides it out
        accepts the wavelength and the flux, and the line name (alpha, beta, gamma, delta, zeta, eta)
        returns the wavelength and normalized flux of the line
        """
        try:
            _, W0, WID, DW = self._lines[line]
        except KeyError, e:
            message = 'Line name %s is not valid. Must be one of (%s)' %(str(line), ','.join(self._lines.keys())) 
            raise ValueError(message)

        w  = np.atleast_1d(w)
        self._wave_test(w)

        f  = np.atleast_1d(f)
        if w.shape != f.shape:
            message = 'Shape mismatch between wavelength and flux arrays'
            raise ValueError(message)

        if df is not None:
            df = np.atleast_1d(df)
            if w.shape != df.shape:
                message = 'Shape mismatch between wavelength and fluxerr arrays'
                raise ValueError(message)

        return self._extract_spectral_line(w, f, ZE, df=df)


    def _extract_spectral_line(self, w, f, line, df=None):
        """
        Same as extract_spectral_line() except no testing
        Used internally to extract the spectral line for the model 
        """
        W0, ZE = self._get_line_indices(w,  line)
        return self._extract_from_indices(w, f, ZE, df=df)


    # these are implemented for compatibility with python's pickle
    # which in turn is required to make the code work with multiprocessing
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    __call__ = get_model
        

