# -*- coding: UTF-8 -*-
"""
DA White Dwarf Atmosphere Models and SED generator.

Model grid originally from J. Holberg using I. Hubeny's Tlusty code (v202) and
custom Synspec routines, repackaged into HDF5 by G. Narayan.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import warnings
import numpy as np
from . import io
import extinction
import scipy.interpolate as spinterp
from scipy.ndimage.filters import gaussian_filter1d
from six.moves import zip

__all__=['WDmodel']

class WDmodel(object):
    """
    DA White Dwarf Atmosphere Model and SED generator

    Base class defines the routines to generate and work with DA White Dwarf
    model spectra. Requires a grid file of DA White Dwarf atmospheres. This
    grid file is included along with the package - ``TlustyGrids.hdf5`` - and is
    the default.

    Parameters
    ----------
    grid_file : str, optional
        Filename of the HDF5 grid file to read. See
        :py:func:`WDmodel.io.read_model_grid` for format of the grid file.
        Default is ``TlustyGrids.hdf5``, included with the :py:mod:`WDmodel`
        package.
    grid_name : str, optional
        Name of the HDF5 group containing the white dwarf model atmosphere
        grids in ``grid_file``. Default is ``default``
    rvmodel : ``{'ccm89','od94','f99'}``, optional
        Specify parametrization of the reddening law. Default is ``'od94'``.

        =========== ==================================================
          rvmodel                 parametrization
        =========== ==================================================
        ``'ccm89'`` Cardelli, Clayton and Mathis (1989, ApJ, 345, 245)
        ``'od94'``  O'Donnell (1994, ApJ, 422, 158)
        ``'f99'``   Fitzpatrick (1999, PASP, 111, 63)
        =========== ==================================================

    Attributes
    ----------
    _lines : dict
        dictionary mapping Hydrogen Balmer series line names to line number,
        central wavelength in Angstrom, approximate line width and continuum
        region width around line. Used to extract Balmer lines from spectra for
        visualization.
    _grid_file : str
        Filename of the HDF5 grid file that was read.
    _grid_name : str
        Name of the HDF5 group containing the white dwarf model atmosphere
    _wave : array-like
        Array of model grid wavelengths in Angstroms, sorted in ascending order
    _ggrid : array-like
        Array of model grid surface gravity values in dex, sorted in ascending order
    _tgrid : array-like
        Array of model grid temperature values in Kelvin, sorted in ascending order
    _nwave : int
        Size of the model grid wavelength array, ``_wave``
    _ngrav : int
        Size of the model grid surface gravity array, ``_ggrid``
    _ntemp : int
        Size of the model grid temperature array, ``_tgrid``
    _flux : array-like
        Array of model grid fluxes, shape ``(_nwave, _ntemp, _ngrav)``
    _lwave : array-like
        Array of model grid ``log10`` wavelengths for interpolation
    _lflux : array-like
        Array of model grid ``log10`` fluxes for interpolation, shape ``(_ntemp, _ngrav, _nwave)``
    _law : extinction function corresponding to ``rvmodel``

    Returns
    -------
    out : :py:class:`WDmodel.WDmodel.WDmodel` instance

    Raises
    ------
    ValueError
        If the supplied rvmodel is unknown

    Notes
    -----
        Virtually none of the attributes should be used directly since it is
        trivially possible to break the model by redefining them. Access to
        them is best through the functions connected to the models.

        A custom user-specified grid file can be specified. See
        :py:func:`WDmodel.io.read_model_grid` for the format of the grid file.

        Uses :py:class:`scipy.interpolate.RegularGridInterpolator` to
        interpolate the models.

        The class contains various convenience methods that begin with an
        underscore (_) that will  not be imported by default. These are intended
        for internal use, and do not have the sanity checking and associated
        overhead of the public methods.
    """

    def __init__(self, grid_file=None, grid_name=None, rvmodel='od94'):
        lno     = [   1    ,   2     ,    3     ,    4    ,   5      ,  6      ]
        lines   = ['alpha' , 'beta'  , 'gamma'  , 'delta' , 'zeta'   , 'eta'   ]
        H       = [6562.857, 4861.346, 4340.478 ,4101.745 , 3970.081 , 3889.056]
        D       = [ 130.0  ,  170.0  ,  125.0   ,  75.0   ,   50.0   ,   27.0  ]
        eps     = [  10.0  ,   10.0  ,   10.0   ,   8.0   ,    5.0   ,    3.0  ]
        self._lines = dict(list(zip(lno, list(zip(lines, H, D, eps)))))
        # we're passing grid_file so we know which model to init
        self._fwhm_to_sigma = np.sqrt(8.*np.log(2.))
        self.__init__tlusty(grid_file=grid_file, grid_name=grid_name)
        self.__init__rvmodel(rvmodel=rvmodel)


    def __init__tlusty(self, grid_file=None, grid_name=None):
        ingrid = io.read_model_grid(grid_file, grid_name)
        self._grid_file, self._grid_name, self._wave, self._ggrid, self._tgrid, _flux = ingrid
        self._lwave = np.log10(self._wave, dtype=np.float64)
        self._lflux = np.log10(_flux.T)
        self._ntemp = len(self._tgrid)
        self._ngrav = len(self._ggrid)
        self._nwave = len(self._wave)

        # pre-init the interpolation and do it in log-space
        # note that we do the interpolation in log-log
        # this is because the profiles are linear, redward of the Balmer break in log-log
        # and the regular grid interpolator is just doing linear interpolation under the hood
        self._model = spinterp.RegularGridInterpolator((self._tgrid, self._ggrid),\
                self._lflux)

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
        """
        Return the extinction for ``av``, ``rv`` at wavelengths ``wave``

        Uses the extinction function corresponding to the ``rvmodel``
        parametrization set as
        :py:attr:`WDmodel.WDmodel.WDmodel._law` to calculate the
        extinction as a function of wavelength (in Angstroms),
        :math:`A_{\lambda}`.

        Parameters
        ----------
        wave : array-like
            Array of wavelengths in Angstrom at which to compute extinction,
            sorted in ascending order
        av : float
            Extinction in the V band, :math:`A_V`
        rv : float, optional
            The reddening law parameter, :math:`R_V`, the ration of the V band
            extinction :math:`A_V` to the reddening between the B and V bands,
            :math:`E(B-V)`. Default is ``3.1``, appropriate for stellar SEDs in the
            Milky Way.

        Returns
        -------
        out : array-like
            Extinction at wavelengths ``wave`` for ``av`` and ``rv``

        Notes
        -----
            ``av`` should be >= 0.
        """
        return self._law(wave, av, rv, unit='aa')


    def reddening(self, wave, flux, av, rv=3.1):
        """
        Redden a 1-D spectrum with extinction

        Uses the extinction function corresponding to the ``rvmodel``
        parametrization set in
        :py:func:`WDmodel.WDmodel.WDmodel._WDmodel__init__rvmodel` to calculate the
        extinction as a function of wavelength (in Angstroms),
        :math:`A_{\lambda}`.

        Parameters
        ----------
        wave : array-like
            Array of wavelengths in Angstrom at which to compute extinction,
            sorted in ascending order
        flux : array-like
            Array of fluxes at ``wave`` at which to apply extinction
        av : float
            Extinction in the V band, :math:`A_V`
        rv : float, optional
            The reddening law parameter, :math:`R_V`, the ration of the V band
            extinction :math:`A_V` to the reddening between the B and V bands,
            :math:`E(B-V)`. Default is ``3.1``, appropriate for stellar SEDs in the
            Milky Way.

        Returns
        -------
        out : array-like
            The reddened spectrum

        Notes
        -----
            ``av`` and ``flux`` should be >= 0.
        """
        return extinction.apply(self.extinction(wave, av, rv), flux, inplace=True)


    def _get_model_nosp(self, teff, logg, wave=None, log=False):
        """
        Returns the model flux given ``teff`` and ``logg`` at wavelengths
        ``wave``

        Simple 3-D interpolation of model grid. Computes unreddened,
        unnormalized, unconvolved, interpolated model flux. Not used, but
        serves as check of output of interpolation of
        :py:class:`scipy.interpolate.RegularGridInterpolator` output.

        Parameters
        ----------
        teff : float
            Desired model white dwarf atmosphere temperature (in Kelvin)
        logg : float
            Desired model white dwarf atmosphere surface gravity (in dex)
        wave : array-like, optional
            Desired wavelengths at which to compute the model atmosphere flux.
            If not supplied, the full model wavelength grid is returned.
        log : bool, optional
            Return the log10 flux, rather than the flux.

        Returns
        -------
        flux : array-like
            Interpolated model flux at ``teff``, ``logg`` and wavelengths ``wave``

        Notes
        -----
            Inputs ``teff``, ``logg`` and ``wave`` must be within the bounds of the grid.
            See :py:attr:`WDmodel.WDmodel.WDmodel._wave`,
            :py:attr:`WDmodel.WDmodel.WDmodel._ggrid`,
            :py:attr:`WDmodel.WDmodel.WDmodel._tgrid`, for grid locations and
            limits.

            This restriction is not imposed here for performance reasons, but
            is implicitly set by routines that call this method. The user is
            expected to verify this condition if this method is used outside
            the context of the :py:mod:`WDmodel` package. Caveat emptor.
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


    def _get_model(self, teff, logg, wave=None, log=False):
        """
        Returns the model flux given ``teff`` and ``logg`` at wavelengths ``wave``

        Simple 3-D interpolation of model grid. Computes unreddened,
        unnormalized, unconvolved, interpolated model flux. Uses
        :py:class:`scipy.interpolate.RegularGridInterpolator` to generate the
        interpolated model. This output has been tested against
        :py:func:`WDmodel.WDmodel.WDmodel._get_model_nosp`.

        Parameters
        ----------
        teff : float
            Desired model white dwarf atmosphere temperature (in Kelvin)
        logg : float
            Desired model white dwarf atmosphere surface gravity (in dex)
        wave : array-like, optional
            Desired wavelengths at which to compute the model atmosphere flux.
            If not supplied, the full model wavelength grid is returned.
        log : bool, optional
            Return the log10 flux rather than the flux.
        
        Returns
        -------
        flux : array-like
            Interpolated model flux at ``teff``, ``logg`` and wavelengths ``wave``.

        Notes
        -----
            Inputs ``teff``, ``logg`` and ``wave`` must be within the bounds of
            the grid.  See :py:attr:`WDmodel.WDmodel.WDmodel._wave`,
            :py:attr:`WDmodel.WDmodel.WDmodel._ggrid`,
            :py:attr:`WDmodel.WDmodel.WDmodel._tgrid`, for grid locations and
            limits.
        """
        xi = (teff, logg)
        out = self._model(xi)
        if wave is not None:
            out = np.interp(np.log10(wave), self._lwave, out)

        if log:
            return out

        return (10.**out)


    def _get_red_model(self, teff, logg, av, wave, rv=3.1, log=False):
        """
        Returns the reddened model flux given ``teff``, ``logg``, ``av``, ``rv`` at
        wavelengths ``wave``

        Uses :py:func:`WDmodel.WDmodel.WDmodel._get_model` to get the
        unreddened model, and reddens it with
        :py:func:`WDmodel.WDmodel.WDmodel.reddening`

        Parameters
        ----------
        teff : float
            Desired model white dwarf atmosphere temperature (in Kelvin)
        logg : float
            Desired model white dwarf atmosphere surface gravity (in dex)
        av : float
            Extinction in the V band, :math:`A_V`
        wave : array-like
            Desired wavelengths at which to compute the model atmosphere flux.
        rv : float, optional
            The reddening law parameter, :math:`R_V`, the ration of the V band
            extinction :math:`A_V` to the reddening between the B and V bands,
            :math:`E(B-V)`. Default is 3.1, appropriate for stellar SEDs in the
            Milky Way.
        log : bool, optional
            Return the log10 flux, rather than the flux (what's actually interpolated)

        Returns
        -------
        flux : array-like
            Interpolated model flux at ``teff``, ``logg`` with reddening parametrized
            by ``av``, ``rv`` at wavelengths ``wave``
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
        Returns the observed model flux given ``teff``, ``logg``, ``av``, ``rv``,
        ``fwhm`` (for Gaussian instrumental broadening) and wavelengths ``wave``

        Uses :py:func:`WDmodel.WDmodel.WDmodel._get_model` to get the
        unreddened model, and reddens it with
        :py:func:`WDmodel.WDmodel.WDmodel.reddening` and convolves it with a
        Gaussian kernel using
        :py:func:`scipy.ndimage.filters.gaussian_filter1d`

        Parameters
        ----------
        teff : float
            Desired model white dwarf atmosphere temperature (in Kelvin)
        logg : float
            Desired model white dwarf atmosphere surface gravity (in dex)
        av : float
            Extinction in the V band, :math:`A_V`
        fwhm : float
            Instrumental FWHM in Angstrom
        wave : array-like
            Desired wavelengths at which to compute the model atmosphere flux.
        rv : float, optional
            The reddening law parameter, :math:`R_V`, the ration of the V band
            extinction :math:`A_V` to the reddening between the B and V bands,
            :math:`E(B-V)`. Default is 3.1, appropriate for stellar SEDs in the
            Milky Way.
        log : bool, optional
            Return the log10 flux, rather than the flux (what's actually interpolated)
        pixel_scale : float, optional
            Jacobian of the transformation between wavelength in Angstrom and
            pixels. In principle, this should be a vector, but virtually all
            spectral reduction packages resample the spectrum onto a uniform
            wavelength scale that is close to the native pixel scale of the
            spectrograph. Default is ``1.``

        Returns
        -------
        flux : array-like
            Interpolated model flux at ``teff``, ``logg`` with reddening parametrized
            by ``av``, ``rv`` and broadened by a Gaussian kernel defined by ``fwhm`` at
            wavelengths ``wave``

        Notes
        -----
            ``fwhm`` and ``pixel_scale`` must be > 0
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
        Returns the observed model flux given ``teff``, ``logg``, ``av``, ``rv``,
        ``fwhm`` (for Gaussian instrumental broadening) at wavelengths, ``wave`` as
        well as the full SED.

        Convenience function that does the same thing as
        :py:func:`WDmodel.WDmodel.WDmodel._get_obs_model`, but also returns the
        full SED without any instrumental broadening applied, appropriate for
        synthetic photometry.

        Uses :py:func:`WDmodel.WDmodel.WDmodel._get_model` to get the
        unreddened model, and reddens it with
        :py:func:`WDmodel.WDmodel.WDmodel.reddening` and convolves it with a
        Gaussian kernel using
        :py:func:`scipy.ndimage.filters.gaussian_filter1d`

        Parameters
        ----------
        teff : float
            Desired model white dwarf atmosphere temperature (in Kelvin)
        logg : float
            Desired model white dwarf atmosphere surface gravity (in dex)
        av : float
            Extinction in the V band, :math:`A_V`
        fwhm : float
            Instrumental FWHM in Angstrom
        wave : array-like
            Desired wavelengths at which to compute the model atmosphere flux.
        rv : float, optional
            The reddening law parameter, :math:`R_V`, the ration of the V band
            extinction :math:`A_V` to the reddening between the B and V bands,
            :math:`E(B-V)`. Default is ``3.1``, appropriate for stellar SEDs in the
            Milky Way.
        log : bool, optional
            Return the log10 flux, rather than the flux (what's actually interpolated)
        pixel_scale : float, optional
            Jacobian of the transformation between wavelength in Angstrom and
            pixels. In principle, this should be a vector, but virtually all
            spectral reduction packages resample the spectrum onto a uniform
            wavelength scale that is close to the native pixel scale of the
            spectrograph. Default is ``1.``

        Returns
        -------
        flux : array-like
            Interpolated model flux at ``teff``, ``logg`` with reddening
            parametrized by ``av``, ``rv`` and broadened by a Gaussian kernel
            defined by ``fwhm`` at wavelengths ``wave``
        mod : :py:class:`numpy.recarray` with ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
            Full model SED at ``teff``, ``logg`` with reddening parametrized by
            ``av``, ``rv``

        Notes
        -----
            ``fwhm`` and ``pixel_scale`` must be > 0
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
        names=str('wave,flux')
        mod = np.rec.fromarrays((self._wave, mod), names=names)
        return omod, mod


    @classmethod
    def _wave_test(cls, wave):
        """
        Raises an error if wavelengths are not valid

        Parameters
        ----------
        wave : array-like
            Array of wavelengths to test for validity

        Raises
        ------
        ValueError
            If wavelength array is empty, has negative values, or is not monotonic
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
        Returns the model flux given ``teff`` and ``logg`` at wavelengths ``wave``

        Wraps :py:func:`WDmodel.WDmodel.WDmodel._get_model` adding checking of
        inputs.

        Parameters
        ----------
        teff : float
            Desired model white dwarf atmosphere temperature (in Kelvin)
        logg : float
            Desired model white dwarf atmosphere surface gravity (in dex)
        wave : array-like, optional
            Desired wavelengths at which to compute the model atmosphere flux.
            If not supplied, the full model wavelength grid is returned.
        log : bool, optional
            Return the log10 flux, rather than the flux
        strict: bool, optional
            If strict, ``teff`` and ``logg`` out of model grid range raise a
            :py:exc:`ValueError`, otherwise raise a :py:exc:`RuntimeWarning`
            and set ``teff``, ``logg`` to the nearest grid value.

        Returns
        -------
        wave : array-like
            Valid output wavelengths
        flux : array-like
            Interpolated model flux at ``teff``, ``logg`` and wavelengths ``wave``

        Raises
        ------
        ValueError
            If ``teff`` or ``logg`` are out of range of model grid and ``strict`` is True
            or
            if there are any invalid wavelengths, or the requested wavelengths
            to do not overlap with the model grid

        Notes
        -----
            Unlike the corresponding private methods, the public methods implement
            checking of the inputs and returns the wavelengths in addition to the
            flux. Internally, we only use the private methods as the inputs only
            need to be checked once, and their state is not altered anywhere after.
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
        Returns the reddened model flux given ``teff``, ``logg``, ``av``, ``rv`` at
        wavelengths ``wave``

        Uses :py:func:`WDmodel.WDmodel.WDmodel.get_model` to get the unreddened
        model, and reddens it with :py:func:`WDmodel.WDmodel.WDmodel.reddening`

        Parameters
        ----------
        teff : float
            Desired model white dwarf atmosphere temperature (in Kelvin)
        logg : float
            Desired model white dwarf atmosphere surface gravity (in dex)
        av : float
            Extinction in the V band, :math:`A_V`
        rv : float, optional
            The reddening law parameter, :math:`R_V`, the ration of the V band
            extinction :math:`A_V` to the reddening between the B and V bands,
            :math:`E(B-V)`. Default is ``3.1``, appropriate for stellar SEDs in the
            Milky Way.
        wave : array-like, optional
            Desired wavelengths at which to compute the model atmosphere flux.
            If not supplied, the full model wavelength grid is returned.
        log : bool, optional
            Return the log10 flux, rather than the flux (what's actually interpolated)
        strict: bool, optional
            If strict, ``teff`` and ``logg`` out of model grid range raise a
            :py:exc:`ValueError`, otherwise raise a :py:exc:`RuntimeWarning`
            and set ``teff``, ``logg`` to the nearest grid value.

        Returns
        -------
        wave : array-like
            Valid output wavelengths
        flux : array-like
            Interpolated model flux at ``teff``, ``logg`` with reddening parametrized
            by ``av``, ``rv`` at wavelengths ``wave``

        Raises
        ------
        ValueError
            If ``av  < 0`` or ``rv`` not in ``[1.7, 5.1]``
        """
        modwave, modflux = self.get_model(teff, logg, wave=wave, log=log, strict=strict)
        av = float(av)
        rv = float(rv)

        if av < 0:
            message = 'Av must be positive'
            raise ValueError(message)

        if rv < 1.7 or rv > 5.1:
            message = 'Rv must be in [1.7, 5.1]'
            raise ValueError(message)

        if log:
            modflux = 10.**modflux
        modflux = self.reddening(modwave, modflux, av, rv=rv)
        if log:
            modflux = np.log10(modflux)
        return modwave, modflux


    def get_obs_model(self, teff, logg, av, fwhm, rv=3.1, wave=None,\
            log=False, strict=True, pixel_scale=1.):
        """
        Returns the observed model flux given ``teff``, ``logg``, ``av``,
        ``rv``, ``fwhm`` (for Gaussian instrumental broadening) and wavelengths
        ``wave``

        Uses :py:func:`WDmodel.WDmodel.WDmodel.get_red_model` to get the
        reddened model and convolves it with a Gaussian kernel using
        :py:func:`scipy.ndimage.filters.gaussian_filter1d`

        Parameters
        ----------
        teff : float
            Desired model white dwarf atmosphere temperature (in Kelvin)
        logg : float
            Desired model white dwarf atmosphere surface gravity (in dex)
        av : float
            Extinction in the V band, :math:`A_V`
        fwhm : float
            Instrumental FWHM in Angstrom
        rv : float, optional
            The reddening law parameter, :math:`R_V`, the ration of the V band
            extinction :math:`A_V` to the reddening between the B and V bands,
            :math:`E(B-V)`. Default is ``3.1``, appropriate for stellar SEDs in
            the Milky Way.
        wave : array-like, optional
            Desired wavelengths at which to compute the model atmosphere flux.
            If not supplied, the full model wavelength grid is returned.
        log : bool, optional
            Return the log10 flux, rather than the flux (what's actually interpolated)
        strict: bool, optional
            If strict, ``teff`` and ``logg`` out of model grid range raise a
            :py:exc:`ValueError`, otherwise raise a :py:exc:`RuntimeWarning` and set
            ``teff``, ``logg`` to the nearest grid value.
        pixel_scale : float, optional
            Jacobian of the transformation between wavelength in Angstrom and
            pixels. In principle, this should be a vector, but virtually all
            spectral reduction packages resample the spectrum onto a uniform
            wavelength scale that is close to the native pixel scale of the
            spectrograph. Default is ``1.``

        Returns
        -------
        wave : array-like
            Valid output wavelengths
        flux : array-like
            Interpolated model flux at ``teff``, ``logg`` with reddening
            parametrized by ``av``, ``rv`` broadened by a Gaussian kernel
            defined by ``fwhm`` at wavelengths ``wave``

        Notes
        -----
            ``fwhm`` and ``pixel_scale`` must be ``> 0``
        """
        modwave, modflux = self.get_red_model(teff, logg, av, rv=rv,\
                wave=wave, log=log, strict=strict)
        if log:
            modflux = 10.**modflux

        fwhm = float(fwhm)
        pixel_scale = float(pixel_scale)
        if fwhm <= 0:
            message = 'FWHM must be strictly positive'
            raise ValueError(message)

        if pixel_scale <= 0:
            message = 'Pixel scale must be strictly positive'
            raise ValueError(message)

        gsig = fwhm/self._fwhm_to_sigma * pixel_scale
        modflux = gaussian_filter1d(modflux, gsig, order=0, mode='nearest')
        if log:
            modflux = np.log10(modflux)
        return modwave, modflux


    @classmethod
    def _get_indices_in_range(cls, wave, WA, WB, W0=None):
        """
        Returns indices of wavelength between blue and red wavelength limits
        and the central wavelength

        Parameters
        ----------
        wave : array-like
            Wavelengths array from which to extract indices
        WA : float
            blue limit of wavelengths to extract
        WB : float
            red limit of wavelenghts to extract
        W0 : float or None, optional
            ``None`` or a central wavelength of range ``[WA, WB]`` to return.
            If ``None``, the central wavelength is computed, else the input is
            simply returned.

        Returns
        -------
        W0 : float
            central wavelength of range ``[WA, WB]``
        ZE : array-like
            indices of ``wave`` in range ``[WA, WB]``
        """
        if W0 is None:
            W0 = WA + (WB-WA)/2.
        ZE  = np.where((wave >= WA) & (wave <= WB))
        return W0, ZE


    def _get_line_indices(self, wave, line):
        """
        Returns the central wavelength and indices of wavelength corresponding
        to a hydrogen Balmer line

        Parameters
        ----------
        wave : array-like
            Wavelengths array from which to extract indices
        line : ``{'alpha', 'beta', 'gamma', 'delta', 'zeta', 'eta'}``
            Name of hydrogen Balmer line to extract.
            Properties are pre-defined in :py:attr:`WDmodel.WDmodel.WDmodel._lines`

        Returns
        -------
        W0 : float
            central wavelength of line
        ZE : array-like
            indices of `wave` of line

        Notes
        -----
        No checking of input - will throw :py:exc:`KeyError` if line is not
        accepted value
        """
        _, W0, WID, DW = self._lines[line]
        WA  = W0 - WID - DW
        WB  = W0 + WID + DW
        return self._get_indices_in_range(wave, WA, WB, W0=W0)


    def _extract_from_indices(self, w, f, ZE, df=None):
        """
        Extracts slices of multiple arrays for the same set of indices.

        Convenience function to extract elements of wavelength ``w``, flux ``f``
        and optionally flux uncertainty ``df`` using indices ``ZE``

        Parameters
        ----------
        w : array-like
            Wavelength array from which to extract indices ``ZE``
        f : array-like
            Flux array from which to extract indices ``ZE``
        ZE : array-like
            indices to extract
        df : None or array-like, optional
            If array-like, extracted elements of this array are also returned

        Returns
        -------
        w : array-like
            elements of input wavelength array at indices ``ZE``
        f : array-like
            elements of input flux array at indices ``ZE``
        [df] : array-like
            elements of input flux uncertainty array at indices ``ZE`` if
            optional input ``df`` is supplied
        """
        if df is None:
            return w[ZE], f[ZE]
        else:
            return w[ZE], f[ZE], df[ZE]


    def extract_spectral_line(self, w, f, line, df=None):
        """
        Extracts slices of multiple arrays corresponding to a hydrogen Balmer
        line

        Convenience function to extract elements of wavelength ``w``, flux ``f``
        and optionally flux uncertainty ``df`` for a hydrogen Balmer line. Wraps
        :py:func:`WDmodel.WDmodel.WDmodel._extract_spectral_line` adding
        checking of inputs.

        Parameters
        ----------
        w : array-like
            Wavelength array from which to extract elements corresponding to
            hydrogen Balmer ``line``
        f : array-like
            Flux array from which to extract elements corresponding to hydrogen
            Balmer ``line``
        line : ``{'alpha', 'beta', 'gamma', 'delta', 'zeta', 'eta'}``
            Name of hydrogen Balmer line to extract.
            Properties are pre-defined in :py:attr:`WDmodel.WDmodel.WDmodel._lines`
        df : None or array-like, optional
            If array-like, extracted elements of this array are also returned

        Returns
        -------
        w : array-like
            elements of input wavelength array for hydrogen Balmer feature
            ``line``
        f : array-like
            elements of input flux array for hydrogen Balmer feature ``line``
        [df] : array-like
            elements of input flux uncertainty array for hydrogen Balmer
            feature ``line`` if optional input ``df`` is supplied

        Raises
        ------
        ValueError
            If ``line`` is not one of the first six of the Balmer series
            or 
            If wavelengths are invalid
            of 
            If there's a difference in shape of any of the arrays
        """
        try:
            _, W0, WID, DW = self._lines[line]
        except KeyError:
            message = "Line name {} is not valid. Must be one of ({})".format(str(line),\
                        ','.join(list(self._lines.keys())))
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
        Extracts slices of multiple arrays corresponding to a hydrogen Balmer
        line

        Convenience function to extract elements of wavelength ``w``, flux ``f``
        and optionally flux uncertainty ``df`` for a hydrogen Balmer ``line``.
        Wraps :py:func:`WDmodel.WDmodel.WDmodel._get_line_indices` and
        :py:func:`WDmodel.WDmodel.WDmodel._extract_from_indices`, both of which
        have their own reasons for existence as well.

        Parameters
        ----------
        w : array-like
            Wavelength array from which to extract elements corresponding to
            hydrogen Balmer ``line``
        f : array-like
            Flux array from which to extract elements corresponding to hydrogen
            Balmer ``line``
        line : ``{'alpha', 'beta', 'gamma', 'delta', 'zeta', 'eta'}``
            Name of hydrogen Balmer line to extract.
            Properties are pre-defined in :py:attr:`WDmodel.WDmodel.WDmodel._lines`
        df : None or array-like, optional
            If array-like, extracted elements of this array are also returned

        Returns
        -------
        w : array-like
            elements of input wavelength array for hydrogen Balmer feature
            ``line``
        f : array-like
            elements of input flux array for hydrogen Balmer feature ``line``
        [df] : array-like
            elements of input flux uncertainty array for hydrogen Balmer
            feature ``line`` if optional input ``df`` is supplied

        Notes
        -----
        Same as :py:func:`WDmodel.WDmodel.WDmodel.extract_spectral_line`
        without checking of inputs and therefore corresponding overhead. Used
        internally.
        """
        W0, ZE = self._get_line_indices(w,  line)
        return self._extract_from_indices(w, f, ZE, df=df)


    # these are implemented for compatibility with python's pickle
    # which in turn is required to make the code work with MPI
    def __getstate__(self):
        return self.__dict__


    def __setstate__(self, d):
        self.__dict__.update(d)


    __call__ = get_model
