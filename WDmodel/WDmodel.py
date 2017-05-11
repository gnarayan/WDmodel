""" DA White Dwarf Model Atmosphere Clas s"""
import warnings
import numpy as np
from . import io
from astropy import units as u
import extinction
from scipy.ndimage.filters import gaussian_filter1d


class WDmodel(object):
    """
    Base class defines the routines to generate and work with DA White Dwarf
    model spectra. Requires the grid file - TlustyGrids.hdf5, or a custom
    user-specified grid. Look at the package level help for description on the
    grid file. There are various convenience methods that begin with an
    underscore (_) that will  not be imported by default These are intended for
    internal use, and do not have the sanity checking of the public methods.
    """

    def __init__(self, grid_file=None, grid_name=None, rvmodel='od94'):
        """
        constructs a white dwarf model atmosphere object
        Virtually none of the attributes should be used directly
        since it is trivially possible to break the model by redefining them
        Access to them is best through the functions connected to the models
        """
        lno     = [   1    ,   2     ,    3     ,    4    ,   5      ,  6      ]
        lines   = ['alpha' , 'beta'  , 'gamma'  , 'delta' , 'zeta'   , 'eta'   ]
        H       = [6562.857, 4861.346, 4340.478 ,4101.745 , 3970.081 , 3889.056]
        D       = [ 130.0  ,  170.0  ,  125.0   ,  75.0   ,   50.0   ,   27.0  ]
        eps     = [  10.0  ,   10.0  ,   10.0   ,   8.0   ,    5.0   ,    3.0  ]
        self._lines = dict(zip(lno, zip(lines, H, D, eps)))
        # we're passing grid_file so we know which model to init
        self._fwhm_to_sigma = np.sqrt(8.*np.log(2.))
        self.__init__tlusty(grid_file=grid_file)
        self.__init__rvmodel(rvmodel=rvmodel)


    def __init__tlusty(self, grid_file=None, grid_name=None):
        """
        Initialize the Tlusty Model <grid_name> from the grid file <grid_file>
        """
        self._fluxnorm = 1. #LEGACY CRUFT

        ingrid = io.read_model_grid(grid_file, grid_name)
        self._grid_file, self._grid_name, self._wave, self._ggrid, self._tgrid, self._flux = ingrid
        self._lwave = np.log10(self._wave, dtype=np.float64)
        self._lflux  = np.log10(self._flux.T)
        self._ntemp = len(self._tgrid)
        self._ngrav = len(self._ggrid)
        self._nwave = len(self._wave)


    def __init__rvmodel(self, rvmodel='od94'):
        if rvmodel == 'ccm89':
            self._law = extinction.ccm89
        elif rvmodel == 'od94':
            self._law = extinction.odonnell94
        elif rvmodel == 'f99':
            self._law = extinction.fitzpatrick99
        else:
            message = 'Unknown reddening law {}'.format(rvmodel)
            raise ValueError(message)


    def extinction(self, wave, av, rv=3.1):
        """ Return the extinction for Av, Rv at wavelengths wave (Angstrom)"""
        return self._law(wave, av, rv, unit='aa')


    def reddening(self, wave, flux, av, rv=3.1):
        """ Apply extinction Av, Rv to flux values flux at wavelengths wave (Angstrom) in place"""
        return extinction.apply(self.extinction(wave, av, rv), flux, inplace=True)


    def _get_model(self, teff, logg, wave=None, log=False):
        """
        Returns the model flux given temperature and logg at wavelengths wave
        """
        thigh = self._ntemp - 1
        tlow = 0
        while(thigh > tlow + 1) :
            mid_ind = int(np.floor((thigh-tlow)/2))+tlow
            if (teff > self._tgrid[mid_ind]) :
                tlow  = mid_ind
            else :
                thigh = mid_ind

        ghigh = self._ngrav - 1
        glow = 0
        while(ghigh > glow + 1) :
            mid_ind = int(np.floor((ghigh-glow)/2))+glow
            if (logg > self._ggrid[mid_ind]) :
                glow  = mid_ind
            else :
                ghigh = mid_ind

        tfac  = (teff - self._tgrid[tlow])/(self._tgrid[thigh] - self._tgrid[tlow])
        gfac  = (logg - self._ggrid[glow])/(self._ggrid[ghigh] - self._ggrid[glow])
        otfac = 1.-tfac

        v00   = self._lflux[tlow,  glow,  :]
        v01   = self._lflux[tlow,  ghigh, :]
        v10   = self._lflux[thigh, glow,  :]
        v11   = self._lflux[thigh, ghigh, :]
        out  = (v00*otfac + v10*tfac)*(1.-gfac) + (v01*otfac + v11*tfac)*gfac

        if wave is not None:
            out = np.interp(np.log10(wave), self._lwave, out)

        if log:
            return out

        return (10.**out)


    def _get_red_model(self, teff, logg, av, wave, rv=3.1, log=False):
        """
        Returns the reddened model flux given teff, logg, av, rv and
        wavelengths
        """
        mod = self._get_model(teff, logg, wave, log=log)
        if log:
            mod = 10.**mod
        mod = self.reddening(wave, mod, av, rv=rv)
        if log:
            mod = np.log10(mod)
        return mod


    def _get_obs_model(self, teff, logg, av, fwhm, wave, rv=3.1, log=False, pixel_scale=1.):
        """
        Returns the observed model flux given teff, logg, av, rv, fwhm
        (for Gaussian instrumental broadening) and wavelengths
        """
        mod = self._get_model(teff, logg, wave, log=log)
        if log:
            mod = 10.**mod
        mod = self.reddening(wave, mod, av, rv=rv)
        gsig = fwhm/self._fwhm_to_sigma * pixel_scale
        mod = gaussian_filter1d(mod, gsig, order=0, mode='nearest')
        if log:
            mod = np.log10(mod)
        return mod


    def _get_full_obs_model(self, teff, logg, av, fwhm, wave, rv=3.1, log=False, pixel_scale=1.):
        """
        Convenience function that does the same thing as _get_obs_model, but
        also returns the full SED without any instrumental broadening applied
        """
        mod  = self._get_model(teff, logg)
        mod  = self.reddening(self._wave, mod, av, rv=rv)
        omod = np.interp(np.log10(wave), self._lwave, np.log10(mod))
        omod = 10.**omod
        gsig = fwhm/self._fwhm_to_sigma * pixel_scale
        omod = gaussian_filter1d(omod, gsig, order=0, mode='nearest')
        if log:
            omod = np.log10(omod)
            mod  = np.log10(mod)
        mod = np.rec.fromarrays((self._wave, mod), names='wave,flux')
        return omod, mod


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
        Returns the model (wavelength and flux) for some teff, logg at wavelengths wave
        If not specified, wavelengths are from 3000-9000A

        Checks inputs for consistency and calls _get_model() If you
        need the model repeatedly for slightly different parameters, use those
        functions directly
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
            outflux = self._get_model(teff, logg, outwave, log=log)
            return outwave, outflux
        else:
            message = 'No valid wavelengths'
            raise ValueError(message)


    def get_red_model(self, teff, logg, av, rv=3.1, wave=None, log=False, strict=True):
        """
        Returns the model (wavelength and flux) for some teff, logg av, rv with
        the reddening law at wavelengths wave If not specified,
        wavelengths are from 3000-9000A Applies reddening that is specified to
        the spectrum (the model has no reddening by default)

        Checks inputs for consistency and calls get_model() If you
        need the model repeatedly for slightly different parameters, use
        _get_red_model directly.
        """
        modwave, modflux = self.get_model(teff, logg, wave=wave, log=log, strict=strict)
        av = float(av)
        rv = float(rv)

        if log:
            modflux = 10.**modflux
        modflux = self.reddening(modwave, modflux, av, rv=rv)
        if log:
            modflux = np.log10(modflux)
        return modwave, modflux


    def get_obs_model(self, teff, logg, av, fwhm, rv=3.1, wave=None,\
            log=False, strict=True, pixel_scale=1.):
        """
        Returns the model (wavelength and flux) for some teff, logg av, rv with
        the reddening law at wavelengths wave If not specified,
        wavelengths are from 3000-9000A Applies reddening that is specified to
        the spectrum (the model has no reddening by default)

        Checks inputs for consistency and calls get_red_model() If you
        need the model repeatedly for slightly different parameters, use
        _get_obs_model directly
        """
        modwave, modflux = self.get_red_model(teff, logg, av, rv=rv,\
                wave=wave, log=log, strict=strict)
        if log:
            modflux = 10.**modflux
        gsig = fwhm/self._fwhm_to_sigma * pixel_scale
        modflux = gaussian_filter1d(modflux, gsig, order=0, mode='nearest')
        if log:
            modflux = np.log10(modflux)
        return modwave, modflux


    def _normalize_model(self, spec, log=False):
        """
        Imprecise normalization for visualization only
        If you want to normalize model and data, get the model, and the data,
        compute the integrals and use the ratio to properly normalize
        """
        wave = spec.wave
        flux = spec.flux
        ind = np.where((self._wave >= wave.min()) & (self._wave <= wave.max()))[0]
        modelmedian = np.median(self._model.values[:,:, ind])
        if not log:
            modelmedian = 10.**modelmedian
        datamedian  = np.median(flux)
        self._fluxnorm = modelmedian/datamedian
        return self._fluxnorm


    @classmethod
    def _get_indices_in_range(cls, w, WA, WB, W0=None):
        """
        Accepts a wavelength array, and blue and redlimits, and returns the
        indices in the array that are between the limits
        """
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
        except KeyError:
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

        return self._extract_spectral_line(w, f, line, df=df)


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
