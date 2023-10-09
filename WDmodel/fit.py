# -*- coding: UTF-8 -*-
"""
Core data processing and fitting/sampling routines
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import sys
import warnings
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.signal as scisig
from scipy.stats import norm
from iminuit import Minuit
from astropy.constants import c as _C
import emcee
import h5py
import six.moves.cPickle as pickle
from clint.textui import progress
from six.moves import map
from six.moves import range
from six.moves import zip
progress.STREAM = sys.stdout
from . import io
from . import passband
from . import likelihood
from . import mossampler


def polyfit_continuum(continuumdata, wave):
    """
    Fit a polynomial to the DA white dwarf continuum to normalize it - purely
    for visualization purposes

    Parameters
    ----------
    continuumdata : :py:class:`numpy.recarray`
        The continuum data.
        Must have ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
        Produced by running the spectrum through
        :py:func:`WDmodel.fit.orig_cut_lines` and extracting the pre-defined
        lines in the :py:class:`WDmodel.WDmodel.WDmodel` instance.
    wave : array-like
        The full spectrum wavelength array on which to interpolate the
        continuum model

    Returns
    -------
    cont_model : :py:class:`numpy.recarray`
        The continuum model
        Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``

    Notes
    -----
        Roughly follows the algorithm described by the SDSS SSPP for a global
        continuum fit. Fits a red side and blue side at 5500 A separately to
        get a smooth polynomial representation. The red side uses a degree 5
        polynomial and the blue side uses a degree 9 polynomial. Then "splices"
        them together - I don't actually know how SDSS does this, but we simply
        assert the two bits are the same function - and fits the full continuum
        to a degree 9 polynomial.
    """

    cwave = continuumdata.wave
    cflux = continuumdata.flux
    cdflux = continuumdata.flux_err
    w = 1./cdflux**2.

    # divide the wavelengths into a blue and red side at 5500 Angstroms
    maskblue = (cwave <= 5500.)
    maskred  = (cwave  > 5500.)
    outblue  = ( wave <= 5500.)
    outred   = ( wave  > 5500.)

    mublue = []
    mured  = []
    # fit a degree 9 polynomial to the blueside
    if len(cwave[maskblue]) > 0 and len(wave[outblue]) > 0:
        coeffblue = poly.polyfit(cwave[maskblue], cflux[maskblue], deg=9, w=w[maskblue])
        mublue = poly.polyval(cwave[maskblue], coeffblue)

    # fit a degree 5 polynomial to the redside
    if len(cwave[maskred]) > 0 and len(wave[outred]) > 0:
        coeffred = poly.polyfit(cwave[maskred], cflux[maskred], deg=5, w=w[maskred])
        mured = poly.polyval(cwave[maskred], coeffred)

    # splice the two fits together
    mu = np.hstack((mublue,mured))

    # fit a degree 9 polynomial to the spliced continuum
    coeff = poly.polyfit(cwave, mu, deg=9, w=w)

    # get the continuum model at the requested wavelengths
    out = poly.polyval(wave, coeff)
    names=str("wave,flux")
    cont_model = np.rec.fromarrays([wave, out], names=names)
    return cont_model


def orig_cut_lines(spec, model):
    """
    Cut out the hydrogen Balmer spectral lines defined in
    :py:class:`WDmodel.WDmodel.WDmodel` from the spectrum.

    The masking of Balmer lines is basic, and not very effective at high
    surface gravity or low temperature, or in the presence of non hydrogen
    lines. It's used to get a roughly masked set of data suitable for continuum
    detection, and is effective in the context of our ground-based
    spectroscopic followup campaign for HST GO 12967 and 13711 programs.

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator

    Returns
    -------
    linedata : :py:class:`numpy.recarray`
        The observations of the spectrum corresponding to the hydrogen Balmer
        lines. 
        Has ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8'), ('line_mask', 'i4'), (line_ind', 'i4')]``
    continuumdata : :py:class:`numpy.recarray`
        The continuum data. Has ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``

    Notes
    -----
        Does a coarse cut to remove hydrogen absorption lines from DA white
        dwarf spectra The line centroids, and widths are fixed and defined with
        the model grid This is insufficient, and particularly at high surface
        gravity and low temperatures the lines are blended. This routine is
        intended to provide a rough starting point for the process of continuum
        determination.
    """

    wave    = spec.wave
    flux    = spec.flux
    fluxerr = spec.flux_err
    balmerwaveindex = {}
    line_wave     = np.array([], dtype='float64', ndmin=1)
    line_flux     = np.array([], dtype='float64', ndmin=1)
    line_fluxerr  = np.array([], dtype='float64', ndmin=1)
    line_number   = np.array([], dtype='int', ndmin=1)
    line_ind      = np.array([], dtype='int', ndmin=1)
    for x in range(1,7):
        W0, ZE = model._get_line_indices(wave, x)
        # save the central wavelengths and spectrum specific indices for each line
        # we don't need this for the fit, but we do need this for plotting
        x_wave, x_flux, x_fluxerr = model._extract_from_indices(wave, flux, ZE, df=fluxerr)
        balmerwaveindex[x] = W0, ZE
        line_wave    = np.hstack((line_wave, x_wave))
        line_flux    = np.hstack((line_flux, x_flux))
        line_fluxerr = np.hstack((line_fluxerr, x_fluxerr))
        line_number  = np.hstack((line_number, np.repeat(x, len(x_wave))))
        line_ind     = np.hstack((line_ind, ZE[0]))
    # continuum data is just the spectrum with the Balmer lines removed
    continuumdata  = (np.delete(wave, line_ind), np.delete(flux, line_ind), np.delete(fluxerr, line_ind))
    linedata = (line_wave, line_flux, line_fluxerr, line_number, line_ind)
    names=str('wave,flux,flux_err,line_mask,line_ind')
    linedata = np.rec.fromarrays(linedata, names=names)
    names=str('wave,flux,flux_err')
    continuumdata = np.rec.fromarrays(continuumdata, names=names)
    return linedata, continuumdata


def blotch_spectrum(spec, linedata):
    """
    Automagically remove cosmic rays and gaps from spectrum

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    linedata : :py:class:`numpy.recarray`
        The observations of the spectrum corresponding to the hydrogen Balmer
        lines. 
        Must have 
        ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8'), ('line_mask', 'i4'), ('line_ind','i4')]``
        Produced by :py:func:`orig_cut_lines`

    Returns
    -------
    spec : :py:class:`numpy.recarray`
        The blotched spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``

    Notes
    -----
        Some spectra have nasty cosmic rays or gaps in the data. This routine
        does a reasonable job blotching these by Wiener filtering the spectrum,
        marking features that differ significantly from the local variance in
        the region, and replace them with the filtered values. The hydrogen
        Balmer lines are preserved, so if your gap/cosmic ray lands on a line
        it will not be filtered. Additionally, filtering has edge effects, and
        these data are preserved as well. If you do blotch the spectrum, it is
        highly recommended that you use the bluelimit and redlimit options to
        trim the ends of the spectrum. Note that the spectrum will be rejected
        if it has flux or flux errors that are not finite or below zero. This
        is often the case with cosmic rays and gaps, so you will likely
        have to do some manual removal of these points.  
            
        YOU SHOULD PROBABLY PRE-PROCESS YOUR DATA YOURSELF BEFORE FITTING IT
        AND NOT BE LAZY! THIS ROUTINE ONLY EXISTS TO FIT QUICK LOOK SPECTRUM AT
        THE TELESCOPE, BEFORE FINAL REDUCTIONS!
    """

    message = 'You have requested the spectrum be blotched. You should probably do this by hand. Caveat emptor.'
    warnings.warn(message, UserWarning)

    window = 151
    blueend = spec.flux[0:window]
    redend  = spec.flux[-window:]

    # wiener filter the spectrum
    med_filt = scisig.wiener(spec.flux, mysize=window)
    diff = np.abs(spec.flux - med_filt)

    # calculate the running variance with the same window
    sigma = scisig.medfilt(diff, kernel_size=window)

    # the sigma is really a median absolute deviation
    scaling = norm.ppf(3/4.)
    sigma/=scaling

    mask = (diff > 5.*sigma)

    # clip the bad outliers from the spectrum
    spec.flux[mask] = med_filt[mask]

    # restore the original lines, so that they aren't clipped
    saveind = linedata.line_ind
    spec.flux[saveind] = linedata.flux
    spec.flux[0:window] = blueend
    spec.flux[-window:] = redend
    return spec


def rebin_spec_by_int_factor(spec, f=1):
    """
    Rebins a spectrum by an integer factor f

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    f : int, optional
        an integer factor to rebin the spectrum by. Default is ``1`` (no rebinning)

    Returns
    -------
    rspec : :py:class:`numpy.recarray`
        The rebinned spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``

    Notes
    -----
        If the spectrum is not divisible by f, the edges are trimmed by discarding
        the remainder measurements from both ends. If the remainder itself is odd, the
        extra measurement is discarded from the blue side.
    """

    f    = int(f)
    if f <= 1:
        return spec
    nwave     = len(spec.wave)
    rnwave    = nwave//f
    # if the spectrum is not perfectly divisible by the f, cut the remainder out
    # divide it evenly between blue and red
    remainder = nwave%f
    ncut      = remainder//2
    icut_blue = ncut + (remainder%2)
    icut_red  = nwave - ncut
    ispec     = spec[icut_blue:icut_red]
    rf        = [(np.mean(ispec.wave[f*x:f*x+f]),\
                  np.average(ispec.flux[f*x:f*x+f],\
                             weights=1./ispec.flux_err[f*x:f*x+f]**2.,\
                             returned=True))\
                 for x in range(rnwave)]
    rwave, rf = list(zip(*rf))
    rflux, rw = list(zip(*rf))
    rw = np.array(rw)
    rflux_err = 1./(rw**0.5)
    rwave = np.array(rwave)
    rflux  = np.array(rflux)
    names=str('wave,flux,flux_err')
    rspec = np.rec.fromarrays((rwave, rflux, rflux_err),names=names)
    return rspec


def pre_process_spectrum(spec, bluelimit, redlimit, model, params,\
        lamshift=0., vel=0., rebin=1, blotch=False, rescale=False):
    """
    Pre-process the spectrum before fitting

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    bluelimit : None or float
        Trim wavelengths bluer than this limit. Uses the bluest wavelength of spectrum if ``None``
    redlimit : None or float
        Trim wavelengths redder than this limit. Uses the reddest wavelength of spectrum if ``None``
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator
    params : dict
        A parameter dict such as that produced by
        :py:func:`WDmodel.io.read_params`
        Will be modified to adjust the spectrum normalization parameters ``dl``
        limits if ``rescale`` is set
    lamshift : float, optional
        Apply a flat wavelength shift to the spectrum. Useful if the target was
        not properly centered in the slit, and the shift is not correlated with
        wavelength. Default is ``0``.
    vel : float, optional
        Apply a velocity shift to the spectrum. Default is ``0``.
    rebin : int, optional 
        Integer factor by which to rebin the spectrum.
        Default is ``1`` (no rebinning).
    blotch : bool, optional
        Attempt to remove cosmic rays and gaps from spectrum. Only to be used
        for quick look analysis at the telescope.
    rescale : bool, optional
        Rescale the spectrum to make the median noise ``~1``. Has no effect on
        fitted parameters except spectrum flux normalization parameter ``dl``
        but makes residual plots, histograms more easily interpretable as they
        can be compared to an ``N(0, 1)`` distribution.

    Returns
    -------
    spec : :py:class:`numpy.recarray`
        The spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``


    See Also
    --------
    :py:func:`orig_cut_lines`
    :py:func:`blotch_spectrum`
    :py:func:`rebin_spec_by_int_factor`
    :py:func:`polyfit_continuum`
    """

    # Test that the array is monotonic
    model._wave_test(spec.wave)

    out_params = io.copy_params(params)

    # offset and blueshift/redshift the spectrum
    if lamshift != 0.:
        spec.wave += lamshift
    if vel != 0.:
        spec.wave *= (1. +(vel*1000./_C.value))

    # clip the spectrum to whatever range is requested
    if bluelimit is None:
        bluelimit = spec.wave.min()

    if bluelimit > 0:
        bluelimit = float(bluelimit)
    else:
        bluelimit = spec.wave.min()

    if redlimit is None:
        redlimit = spec.wave.max()

    if redlimit > 0:
        redlimit = float(redlimit)
    else:
        redlimit = spec.wave.max()

    if rescale:
        _, scalemask = model._get_indices_in_range(spec.wave, bluelimit, redlimit)
        scale_factor = np.median(spec.flux_err[scalemask])
        if scale_factor == 0:
            message = 'Uhhhh. This spectrum has incredibly weird errors with a median of 0'
            raise RuntimeError(message)
        message = "Scaling the spectrum by {:.10g}".format(scale_factor)
        print(message)
        spec.flux /= scale_factor
        spec.flux_err /= scale_factor
        if out_params['dl']['value'] is not None:
            out_params['dl']['value'] *= (scale_factor)**0.5
        lb, ub = out_params['dl']['bounds']
        out_params['dl']['bounds'] = (lb*(scale_factor**0.5), ub*(scale_factor**0.5))
        out_params['dl']['scale'] *= (scale_factor)**0.5
    else:
        scale_factor = 1.

    # if requested, rebin the spectrum
    if rebin != 1:
        spec = rebin_spec_by_int_factor(spec, f=rebin)

    # get a coarse mask of line and continuum
    linedata, continuumdata  = orig_cut_lines(spec, model)
    if blotch:
        spec = blotch_spectrum(spec, linedata)

    # get a coarse estimate of the full continuum
    # this isn't used for anything other than cosmetics
    cont_model = polyfit_continuum(continuumdata, spec.wave)

    # trim the outputs to the requested length
    _, usemask = model._get_indices_in_range(spec.wave, bluelimit, redlimit)
    spec = spec[usemask]
    cont_model = cont_model[usemask]

    _, usemask = model._get_indices_in_range(linedata.wave, bluelimit, redlimit)
    linedata = linedata[usemask]

    _, usemask = model._get_indices_in_range(continuumdata.wave, bluelimit, redlimit)
    continuumdata = continuumdata[usemask]
    return spec, cont_model, linedata, continuumdata, scale_factor, out_params


def quick_fit_spec_model(spec, spec2, model, params):
    """
    Does a quick fit of the spectrum to get an initial guess of the fit parameters

    Uses iminuit to do a rough diagonal fit - i.e. ignores covariance.
    For simplicity, also fixed FWHM and Rv (even when set to be fit).
    Therefore, only teff, logg, av, dl are fit for (at most).
    This isn't robust, but it's good enough for an initial guess.

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator
    params : dict
        A parameter dict such as that produced by
        :py:func:`WDmodel.io.read_params`

    Returns
    -------
    migrad_params : dict
        The output parameter dictionary with updated initial guesses stored in
        the ``value`` key. Same format as ``params``.

    Raises
    ------
    RuntimeError
        If all of ``teff, logg, av, dl`` are set as fixed - there's nothing to fit.
    RuntimeWarning
        If :py:func:`minuit.Minuit.migrad` or :py:func:`minuit.Minuit.hesse` indicate that the fit is unreliable

    Notes
    -----
        None of the starting values for the parameters maybe ``None`` EXCEPT ``c``.
        This refines the starting guesses, and determines a reasonable value for ``c``
    """

    teff0 = params['teff']['value']
    logg0 = params['logg']['value']
    av0   = params['av']['value']
    dl0   = params['dl']['value']

    # we don't actually fit for these values
    rv   = params['rv']['value']
    fwhm = params['fwhm']['value']
    fwhm2 = params['fwhm2']['value']

    fix_teff = params['teff']['fixed']
    fix_logg = params['logg']['fixed']
    fix_av   = params['av']['fixed']
    fix_dl   = params['dl']['fixed']

    if all((fix_teff, fix_logg, fix_av, fix_dl)):
        message = "All of teff, logg, av, dl are marked as fixed - nothing to fit."
        raise RuntimeError(message)

    pixel_scale = 1./np.median(np.gradient(spec.wave))
    pixel_scale2 = 1./np.median(np.gradient(spec2.wave))

    if dl0 is None:
        # only dl and fwhm are allowed to have None as input values
        # fwhm will get set to a default fwhm if it's None
        mod = model._get_obs_model(teff0, logg0, av0, fwhm, spec.wave, rv=rv, pixel_scale=pixel_scale)
        mod2 = model._get_obs_model(teff0, logg0, av0, fwhm2, spec2.wave, rv=rv, pixel_scale=pixel_scale2)
        c0   = np.mean([spec.flux.mean()/mod.mean(), spec2.flux.mean()/mod2.mean()])
        dl0 = (1./(4.*np.pi*c0))**0.5

    teff_scale = params['teff']['scale']
    logg_scale = params['logg']['scale']
    av_scale   = params['av']['scale']
    dl_scale   = params['dl']['scale']

    teff_bounds = params['teff']['bounds']
    logg_bounds = params['logg']['bounds']
    av_bounds   = params['av']['bounds']
    dl_bounds   = params['dl']['bounds']

    # ignore the covariance and define a simple chi2 to minimize
    def chi2(teff, logg, av, dl):
        mod = model._get_obs_model(teff, logg, av, fwhm, spec.wave, rv=rv, pixel_scale=pixel_scale)
        mod2 = model._get_obs_model(teff, logg, av, fwhm2, spec2.wave, rv=rv, pixel_scale=pixel_scale2)
        mod *= (1./(4.*np.pi*(dl)**2.))
        mod2 *= (1./(4.*np.pi*(dl)**2.))
        chi2 = np.sum(((spec.flux-mod)/spec.flux_err)**2.) + np.sum(((spec2.flux-mod2)/spec2.flux_err)**2.)
        return chi2

    # use minuit to refine our starting guess
    m = Minuit(chi2, teff=teff0, logg=logg0, av=av0, dl=dl0,\
                fix_teff=fix_teff, fix_logg=fix_logg, fix_av=fix_av, fix_dl=fix_dl,\
                error_teff=teff_scale, error_logg=logg_scale, error_av=av_scale, error_dl=dl_scale,\
                limit_teff=teff_bounds, limit_logg=logg_bounds, limit_av=av_bounds, limit_dl=dl_bounds,\
                print_level=1, pedantic=True, errordef=1)

    outfnmin, outpar = m.migrad()

    result = m.values
    errors = m.errors
    # duplicate the input dictionary and update
    migrad_params = io.copy_params(params)
    for param in result:
        migrad_params[param]['value'] = result[param]
        migrad_params[param]['scale'] = errors[param]

    if outfnmin['is_valid']:
        # if minuit works, then try computing the Hessian
        try:
            m.hesse()
            # if we sucessfully computed the Hessian, then update the parameters
            #
            result = m.values
            errors = m.errors
            for param in result:
                migrad_params[param]['value'] = result[param]
                migrad_params[param]['scale'] = errors[param]
        except RuntimeError:
            message = "Something seems to have gone wrong with Hessian matrix computation. You should probably stop."
            warnings.warn(message, RuntimeWarning)
    else:
        # there's some objects for which minuit will fail - just return the original params then
        migrad_params = io.copy_params(params)

        # if the minuit fill failed, and we didn't get an initial dl guess,
        # just use the failed result value as a start this has the dubious
        # value of being better than None
        if migrad_params['dl']['value'] is None:
            migrad_params['dl']['value'] = result['dl']

        message = "Something seems to have gone wrong refining parameters with migrad. You should probably stop."
        warnings.warn(message, RuntimeWarning)
    return migrad_params


def fix_pos(pos, free_param_names, params):
    """
    Ensures that the initial positions of the :py:mod:`emcee` walkers are out of bounds

    Parameters
    ----------
    pos : array-like
        starting positions of all the walkers, such as that produced by 
        :py:func:`emcee:utils.sample_ball`
    free_param_names : iterable
        names of parameters that are free to float. Names must correspond to keys in ``params``.
    params : dict
        A parameter dict such as that produced by
        :py:func:`WDmodel.io.read_params`

    Returns
    -------
    pos : array-like
        starting positions of all the walkers, fixed to guarantee that they are
        within ``bounds`` defined in ``params``

    Notes
    -----
        :py:func:`emcee.utils.sample_ball` creates random walkers that may be
        initialized out of bounds.  These walkers get stuck as there is no step
        they can take that will make the change in loglikelihood finite.  This
        makes the chain appear strongly correlated since all the samples of one
        walker are at a fixed location. This resolves the issue by assuming
        that the parameter ``value``  was within ``bounds`` to begin with. This
        routine does not do any checking of types, values or bounds. This check
        is done by :py:func:`WDmodel.io.get_params_from_argparse` before the
        fit. If you setup the fit using an external code, you should check
        these values.

    See Also
    --------
    :py:func:`emcee.utils.sample_ball`
    :py:func:`WDmodel.io.get_params_from_argparse`
    """

    for i, name in enumerate(free_param_names):
        lb, ub = params[name]['bounds']
        p0     = params[name]['value']
        std    = params[name]['scale']
        # take a 5 sigma range
        lr, ur = (p0-5.*std, p0+5.*std)
        ll = max(lb, lr, 0.)
        ul = min(ub, ur)
        ind = np.where((pos[:,i] <= ll) | (pos[:,i] >= ul))
        nreplace = len(pos[:,i][ind])
        pos[:,i][ind] = np.random.rand(nreplace)*(ul - ll) + ll
    return pos


def hyper_param_guess(spec, phot, model, pbs, params):
    """
    Makes a guess for the parameter ``mu`` after the initial fit by
    :py:func:`quick_fit_spec_model`

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    phot : :py:class:`numpy.recarray`
        The photometry of ``objname`` with ``dtype=[('pb', 'str'), ('mag', '<f8'), ('mag_err', '<f8')]``
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator
    pbs : dict
        Passband dictionary containing the passbands corresponding to
        phot.pb` and generated by :py:func:`WDmodel.passband.get_pbmodel`.
    params : dict
        A parameter dict such as that produced by
        :py:func:`WDmodel.io.read_params`

    Returns
    -------
    out_params : dict
        The output parameter dictionary with an initial guess for ``mu`` 

    Notes
    -----
        Uses the initial guess of parameters from the spectrum fit by
        :py:func:`quick_fit_spec_model` to construct an initial guess of the
        SED, and computes ``mu`` (which looks like a distance modulus, but also
        includes a normalization for the radius of the DA white dwarf, and it's
        radius) as the median difference between the observed and synthetic
        photometry.
    """
    out_params = io.copy_params(params)
    # mu has a user supplied guess - do nothing
    if params['mu']['value'] is not None:
        return out_params

    # restore the minuit best fit model
    teff = params['teff']['value']
    logg = params['logg']['value']
    av   = params['av']['value']
    rv   = params['rv']['value']
    fwhm = params['fwhm']['value']
    pixel_scale = 1./np.median(np.gradient(spec.wave))
    _, model_spec = model._get_full_obs_model(teff, logg, av, fwhm, spec.wave,\
            rv=rv, pixel_scale=pixel_scale)

    # update the mu guess if we don't have one, or the parameter isn't fixed
    if params['mu']['value'] is None or (not params['mu']['fixed']):
        # get the median between observed and model mags for the minuit model
        if phot is not None:
            model_mags = passband.get_model_synmags(model_spec, pbs)
            mu0_guess  = np.median(phot.mag - model_mags.mag)
        else:
            mu0_guess = 0.
        out_params['mu']['value'] = mu0_guess

    return out_params


def fit_model(spec, spec2, phot, model, covmodel, covmodel2, pbs, params,\
            objname, outdir, specfile,\
            phot_dispersion=0.,\
            samptype='ensemble', ascale=2.0,\
            ntemps=1, nwalkers=300, nburnin=50, nprod=1000, everyn=1, thin=1, pool=None,\
            resume=False, redo=False):
    """
    Core routine that models the spectrum using the white dwarf model and a
    Gaussian process with a stationary kernel to account for any flux
    miscalibration, sampling the posterior using a MCMC.

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum with ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    phot : :py:class:`numpy.recarray`
        The photometry of ``objname`` with ``dtype=[('pb', 'str'), ('mag', '<f8'), ('mag_err', '<f8')]``
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator
    pbs : dict
        Passband dictionary containing the passbands corresponding to
        phot.pb` and generated by :py:func:`WDmodel.passband.get_pbmodel`.
    params : dict
        A parameter dict such as that produced by
        :py:func:`WDmodel.io.read_params`
    objname : str
        object name - used to save output with correct name
    outdir : str
        controls where the chain file s written
    specfile : str
        Used in the title, and to set the name of the ``outfile``
    phot_dispersion : float, optional
        Excess photometric dispersion to add in quadrature with the
        photometric uncertainties ``phot.mag_err``. Use if the errors are
        grossly underestimated. Default is ``0.``
    samptype : ``{'ensemble', 'pt', 'gibbs'}``
        Which sampler to use. The default is ``ensemble``.
    ascale : float
        The proposal scale for the sampler. Default is ``2.``
    ntemps : int
        The number of temperatures to run walkers at. Only used if ``samptype``
        is in ``{'pt','gibbs'}`` and set to ``1.`` for ``ensemble``. See a
        short summary `review
        <https://en.wikipedia.org/wiki/Parallel_tempering>`_ for details.
        Default is ``1.``
    nwalkers : int
        The number of `Goodman and Weare walkers
        <http://msp.org/camcos/2010/5-1/p04.xhtml>`_. Default is ``300``.
    nburnin : int
        The number of steps to discard as burn-in for the Markov-Chain. Default is ``500``.
    nprod : int
        The number of production steps in the Markov-Chain. Default is ``1000``.
    everyn : int, optional
        If the posterior function is evaluated using only every nth
        observation from the data, this should be specified. Default is ``1``.
    thin : int
        Only save every ``thin`` steps to the output Markov Chain. Useful, if
        brute force way of reducing correlation between samples.
    pool : None or :py:class`emcee.utils.MPIPool`
        If running with MPI, the pool object is used to distribute the
        computations among the child process
    resume : bool
        If ``True``, restores state and resumes the chain for another ``nprod`` iterations.
    redo : bool
        If ``True``, and a chain file and state file exist, simply clobbers them.

    Returns
    -------
    free_param_names : list
        names of parameters that were fit for. Names correspond to keys in
        ``params`` and the order of parameters in ``samples``.
    samples : array-like
        The flattened Markov Chain with the parameter positions.
        Shape is ``(ntemps*nwalkers*nprod, nparam)``
    samples_lnprob : array-like
        The flattened log of the posterior corresponding to the positions in
        ``samples``. Shape is ``(ntemps*nwalkers*nprod, 1)``
    everyn :  int 
        Specifies sampling of the data used to compute the posterior. Provided
        in case we are using ``resume`` to continue the chain, and this value
        must be restored from the state file, rather than being supplied as a
        user input.
    shape : tuple
        Specifies the shape of the un-flattened chain.
        ``(ntemps, nwalkers, nprod, nparam)``
        Provided in case we are using ``resume`` to continue the chain, and
        this value must be restored from the state file, rather than being
        supplied as a user input.

    Raises
    ------
    RuntimeError
        If ``resume`` is set without the chain having been run in the first
        place.

    Notes
    -----
        Uses an Ensemble MCMC (implemented by emcee) to generate samples from
        the posterior. Does a short burn-in around the initial guess model
        parameters - either :py:mod:`minuit` or user supplied values/defaults.
        Model parameters may be frozen/fixed. Parameters can have bounds
        limiting their range. Then runs a full production change. Chain state
        is saved after every 100 production steps, and may be continued after
        the first 100 steps if interrupted or found to be too short. Progress
        is indicated visually with a progress bar that is written to STDOUT.

    See Also
    --------
    :py:mod:`WDmodel.likelihood`
    :py:mod:`WDmodel.covariance`
    """

    outfile = io.get_outfile(outdir, specfile, '_mcmc.hdf5', check=True, redo=redo, resume=resume)
    if not resume:
        # create a HDF5 file to hold the chain data
        outf = h5py.File(outfile, 'w')
    else:
        # restore some attributes from the HDF5 file to make sure the state is consistent
        try:
            outf = h5py.File(outfile, 'a')
            chain = outf['chain']
        except (IOError, OSError, KeyError) as e:
            message = '{}\nMust run fit to generate mcmc chain before attempting to resume'
            raise RuntimeError(message)

        ntemps      = chain.attrs["ntemps"]
        nwalkers    = chain.attrs["nwalkers"]
        everyn      = chain.attrs["everyn"]
        thin        = chain.attrs["thin"]
        samptype    = chain.attrs["samptype"]
        ascale      = chain.attrs["ascale"]

    # create a state file to periodically save the state of the chain
    statefile = io.get_outfile(outdir, specfile, '_state.pkl', redo=redo)

    # setup the likelihood function
    lnlike = likelihood.setup_likelihood(params)
    nparam   = lnlike.vector_size

    # get the starting position and the scales for each parameter
    init_p0  = lnlike.get_parameter_dict()
    p0       = list(init_p0.values())
    free_param_names = list(init_p0.keys())
    std = [params[x]['scale'] for x in free_param_names]

    # create a sample ball
    pos = emcee.utils.sample_ball(p0, std, size=ntemps*nwalkers)
    pos = fix_pos(pos, free_param_names, params)
    if samptype != 'ensemble':
        pos = pos.reshape(ntemps, nwalkers, nparam)

    if everyn != 1:
        inspec = spec[::everyn]
    else:
        inspec = spec
    if everyn != 1:
        inspec2 = spec2[::everyn]
    else:
        inspec2 = spec2

    # even if we only take every nth sample, the pixel scale is the same
    pixel_scale = 1./np.median(np.gradient(spec.wave))
    pixel_scale2 = 1./np.median(np.gradient(spec2.wave))

    # configure the posterior function
    lnpost = likelihood.WDmodel_Posterior(inspec,inspec2, phot, model, covmodel,covmodel2, pbs, lnlike,\
            pixel_scale=pixel_scale, pixel_scale2=pixel_scale2, phot_dispersion=phot_dispersion)

    # setup the sampler
    if samptype == 'ensemble':
        sampler = emcee.EnsembleSampler(nwalkers, nparam, lnpost,\
                a=ascale,  pool=pool)
        ntemps = 1
    else:
        logpkwargs = {'prior':True}
        loglkwargs = {'likelihood':True}
        sampler = mossampler.MOSSampler(ntemps, nwalkers, nparam, lnpost, lnpost,\
                a=ascale, pool=pool, logpkwargs=logpkwargs, loglkwargs=loglkwargs)

    # use James Guillochon's MOSSampler "gibbs"(-ish) implementation
    gibbs = False
    if samptype == 'gibbs':
        gibbs = True
    sampler_kwargs = {}
    if samptype != 'ensemble':
        inpos = pos.reshape(ntemps*nwalkers, nparam)
        if pool is None:
            lnprob0 = list(map(lnpost, inpos))
        else:
            lnprob0 = pool.map(lnpost, inpos)

        lnprob0 = np.array(lnprob0)
        lnprob0 = lnprob0.reshape(ntemps, nwalkers)
        sampler_kwargs = {'gibbs':gibbs,'lnprob0':lnprob0}
    sampler_kwargs['thin'] = thin

    # do a short burn-in
    if not resume:
        with progress.Bar(label="Burn-in", expected_size=nburnin, hide=False) as bar:
            bar.show(0)
            j = 0
            for i, result in enumerate(sampler.sample(pos, iterations=thin*nburnin, **sampler_kwargs)):
                if (i+1)%thin == 0:
                    bar.show(j+1)
                    j+=1

        # find the MAP position after the burnin
        samples        = sampler.flatchain
        samples_lnprob = sampler.lnprobability
        map_samples        = samples.reshape(ntemps, nwalkers, nburnin, nparam)
        map_samples_lnprob = samples_lnprob.reshape(ntemps, nwalkers, nburnin)
        max_ind        = np.argmax(map_samples_lnprob)
        max_ind        = np.unravel_index(max_ind, (ntemps, nwalkers, nburnin))
        max_ind        = tuple(max_ind)
        p1        = map_samples[max_ind]

        # reset the sampler
        sampler.reset()

        lnlike.set_parameter_vector(p1)
        message = "\nMAP Parameters after Burn-in"
        print(message)
        for k, v in lnlike.get_parameter_dict().items():
            message = "{} = {:f}".format(k,v)
            print(message)

        # set walkers to start production at final burnin state
        pos = result[0]

        # setup incremental chain saving
        chain = outf.create_group("chain")

        # last saved position of the chain
        laststep = 0

        # save some other attributes about the chain
        chain.attrs["nwalkers"] = nwalkers
        chain.attrs["nparam"]   = nparam
        chain.attrs["everyn"]   = everyn
        chain.attrs["ascale"]   = ascale
        chain.attrs["samptype"] = samptype
        chain.attrs["ntemps"]   = ntemps
        chain.attrs["thin"]     = thin
        chain.attrs["nprod"]    = nprod
        chain.attrs["laststep"] = laststep

        # save the parameter names corresponding to the chain
        free_param_names = np.array([str(x) for x in free_param_names])
        dt = free_param_names.dtype.str.lstrip('|').replace('U','S')
        chain.create_dataset("names",data=free_param_names.astype(np.string_), dtype=dt)

        # save the parameter configuration as well
        # this is redundant, since it is saved in the JSON file, but having it one place is nice
        names = lnlike.get_parameter_names(include_frozen=True)
        names = np.array([str(x) for x in names])
        dt = names.dtype.str.lstrip('|').replace('U','S')
        par_grp = outf.create_group("params")
        par_grp.create_dataset("names",data=names.astype(np.string_), dtype=dt)
        for param in params:
            this_par = par_grp.create_group(param)
            this_par.attrs["value"]  = params[param]["value"]
            this_par.attrs["fixed"]  = params[param]["fixed"]
            this_par.attrs["scale"]  = params[param]["scale"]
            this_par.attrs["bounds"] = params[param]["bounds"]

        # production
        dset_chain  = chain.create_dataset("position",(ntemps*nwalkers*nprod,nparam),maxshape=(None,nparam))
        dset_lnprob = chain.create_dataset("lnprob",(ntemps*nwalkers*nprod,),maxshape=(None,))
    else:
        # if we are resuming, we only need to make sure we can write the
        # resumed chain to the chain file (i.e. the arrays are properly
        # resized)
        laststep    = chain.attrs["laststep"]
        dset_chain  = chain["position"]
        dset_lnprob = chain["lnprob"]
        dset_chain.resize((ntemps*nwalkers*(laststep+nprod),nparam))
        dset_lnprob.resize((ntemps*nwalkers*(laststep+nprod),))
        chain.attrs["nprod"] = laststep+nprod

        # and that we have the state of the chain when we ended
        try:
            with open(statefile, 'rb') as f:
                pickle_kwargs = {}
                if sys.version_info[0] > 2:
                    pickle_kwargs['encoding'] = 'latin-1'
                position, lnpost, rstate = pickle.load(f, **pickle_kwargs)
        except (IOError, OSError) as e:
            message = '{}\nMust run fit to generate mcmc chain state pickle before attempting to resume'.format(e)
            raise RuntimeError(message)

        if samptype in ('pt', 'gibbs'):
            # PTsampler doesn't include rstate0 in the release version of emcee
            # this is apparently fixed on git, but not in release yet it is
            # included in run_mcmc which simply sets the sampler random_state
            # attribute - do the same thing here
            sampler.random_state = rstate
        else:
            sampler_kwargs['rstate0']=rstate
        sampler_kwargs['lnprob0']=lnpost
        pos = position

        message = "Resuming from iteration {} for {} steps".format(laststep, nprod)
        print(message)

    # write to disk before we start
    outf.flush()

    # since we're going to save the chain in HDF5, we don't need to save it in memory elsewhere
    sampler_kwargs['storechain']=False

    # run the production chain
    with progress.Bar(label="Production", expected_size=laststep+nprod, hide=False) as bar:
        bar.show(laststep)
        j = laststep
        for i, result in enumerate(sampler.sample(pos, iterations=thin*nprod, **sampler_kwargs)):
            if (i+1)%thin != 0:
                continue
            position = result[0]
            lnpost   = result[1]
            position = position.reshape((-1, nparam))
            lnpost   = lnpost.reshape(ntemps*nwalkers)
            dset_chain[ntemps*nwalkers*j:ntemps*nwalkers*(j+1),:] = position
            dset_lnprob[ntemps*nwalkers*j:ntemps*nwalkers*(j+1)] = lnpost

            # save state every 100 steps
            if (j+1)%100 == 0:
                # make sure we know how many steps we've taken so that we can resize arrays appropriately
                chain.attrs["laststep"] = j+1
                outf.flush()

                # save the state of the chain
                with open(statefile, 'wb') as f:
                    pickle.dump(result, f, 2)

            bar.show(j+1)
            j+=1

        # save the final state of the chain and nprod, laststep
        chain.attrs["nprod"]    = laststep+nprod
        chain.attrs["laststep"] = laststep+nprod
        with open(statefile, 'wb') as f:
            pickle.dump(result, f, 2)

    # save the acceptance fraction
    if resume:
        if "afrac" in list(chain.keys()):
            del chain["afrac"]
        if samptype != 'ensemble':
            if "tswap_afrac" in list(chain.keys()):
                del chain["tswap_afrac"]
    chain.create_dataset("afrac", data=sampler.acceptance_fraction)
    if samptype != 'ensemble' and ntemps > 1:
        chain.create_dataset("tswap_afrac", data=sampler.tswap_acceptance_fraction)

    samples         = np.array(dset_chain)
    samples_lnprob  = np.array(dset_lnprob)

    # finalize the chain file, close it and close the pool
    outf.flush()
    outf.close()
    if pool is not None:
        pool.close()

    # find the MAP value after production
    map_samples = samples.reshape(ntemps, nwalkers, laststep+nprod, nparam)
    map_samples_lnprob = samples_lnprob.reshape(ntemps, nwalkers, laststep+nprod)
    max_ind = np.argmax(map_samples_lnprob)
    max_ind = np.unravel_index(max_ind, (ntemps, nwalkers, laststep+nprod))
    max_ind = tuple(max_ind)
    p_final = map_samples[max_ind]
    lnlike.set_parameter_vector(p_final)
    message = "\nMAP Parameters after Production"
    print(message)

    for k, v in lnlike.get_parameter_dict().items():
        message = "{} = {:f}".format(k,v)
        print(message)
    message = "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))
    print(message)

    # return the parameter names of the chain, the positions, posterior, and the shape of the chain
    return  free_param_names, samples, samples_lnprob, everyn, (ntemps, nwalkers, laststep+nprod, nparam)


def get_fit_params_from_samples(param_names, samples, samples_lnprob, params,\
        ntemps=1, nwalkers=300, nprod=1000, discard=5):
    """
    Get the marginalized parameters from the sample chain

    Parameters
    ----------
    param_names : list
        names of parameters that were fit for. Names correspond to keys in
        ``params`` and the order of parameters in ``samples``.
    samples : array-like
        The flattened Markov Chain with the parameter positions.
        Shape is ``(ntemps*nwalkers*nprod, nparam)``
    samples_lnprob : array-like
        The flattened log of the posterior corresponding to the positions in
        ``samples``. Shape is ``(ntemps*nwalkers*nprod, 1)``
    params : dict
        A parameter dict such as that produced by
        :py:func:`WDmodel.io.read_params`
    ntemps : int
        The number of temperatures chains were run at. Default is ``1.``
    nwalkers : int
        The number of `Goodman and Weare walkers
        <http://msp.org/camcos/2010/5-1/p04.xhtml>`_ used in the fit. Default
        is ``300``.
    nprod : int
        The number of production steps in the Markov-Chain. Default is ``1000``.
    discard : int
        percentage of nprod steps from the start of the chain to discard in
        analyzing samples

    Returns
    -------
    mcmc_params : dict
        The output parameter dictionary with updated parameter estimates,
        errors and a scale. 
        ``params``.
    out_samples : array-like
        The flattened Markov Chain with the parameter positions with the first
        ``%discard`` tossed.
    out_ samples_lnprob : array-like
        The flattened log of the posterior corresponding to the positions in
        ``samples`` with the first ``%discard`` samples tossed.

    See Also
    --------
    :py:func:`fit_model`
    """

    ndim = len(param_names)

    in_samp   = samples.reshape(ntemps, nwalkers, nprod, ndim)
    in_lnprob = samples_lnprob.reshape(ntemps, nwalkers, nprod)

    # discard the first %discard steps from all the walkers
    nstart    = int(np.ceil((discard/100.)*nprod))
    in_samp   = in_samp[:,:,nstart:,:]
    in_lnprob = in_lnprob[:,:,nstart:]

    # reflatten
    in_samp = in_samp.reshape((-1, ndim))
    in_lnprob = in_lnprob.reshape(in_samp.shape[0])

    # only select entries with finite log posterior
    # if this isn't all, something is wrong
    mask = np.isfinite(in_lnprob)

    # update the parameter dict
    for i, param in enumerate(param_names):
        x = in_samp[mask,i]
        q_16, q_50, q_84 = np.percentile(x, [16., 50., 84.])
        params[param]['value']  = q_50
        params[param]['bounds'] = (q_16, q_84)
        params[param]['errors_pm'] = (q_84 - q_50, q_50 - q_16)
        params[param]['scale']  = float(np.std(x))

    # make sure the output for fixed parameters is fixed
    fixed_params = set(params.keys()) - set(param_names)
    for param in fixed_params:
        if params[param]['fixed']:
            params[param]['scale'] = 0.
            params[param]['errors_pm'] = (0., 0.)
        else:
            # this should never happen, unless the state of the files was changed
            message = "Huh.... {} not marked as fixed but was not fit for...".format(param)
            print(message)
    return params, in_samp[mask,:], in_lnprob[mask]
