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
from clint.textui import progress
progress.STREAM = sys.stdout
from . import io
from . import passband
from . import likelihood
from . import mossampler


def polyfit_continuum(continuumdata, wave):
    """
    Accepts continuumdata: a tuple with wavelength, flux and flux error derived
    from the spectrum with the lines roughly masked and wave: an array of
    wavelengths on which to derive the continuum.

    Roughly follows the algorithm described by the SDSS SSPP for a global
    continuum fit. Fits a red side and blue side at 5500 A separately to get a
    smooth polynomial representation. The red side uses a degree 5 polynomial
    and the blue side uses a degree 9 polynomial. Then splices them together -
    I don't actually know how SDSS does this - and fits the full continuum to a
    degree 9 polynomial.

    Returns a recarray with wave, continuum flux derived from the polyfit.
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
    cont_model = np.rec.fromarrays([wave, out], names='wave,flux')
    return cont_model


def orig_cut_lines(spec, model):
    """
    Does a coarse cut to remove hydrogen absorption lines from DA white dwarf
    spectra The line centroids, and widths are fixed and defined with the model
    grid This is insufficient, and particularly at high log(g) and low
    temperatures the lines are blended, and better masking is needed.  This
    routine is intended to provide a rough starting point for that process.

    Accepts a spectrum and the model
    returns a recarray with the data on the absorption lines
    (wave, flux, fluxerr, Balmer line number for use as a mask)

    and coarse continuum data - the part of the spectrum that's not masked as lines
    (wave, flux, fluxerr)

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
    linedata = (line_wave, line_flux, line_fluxerr, line_number)

    linedata = np.rec.fromarrays(linedata, names='wave,flux,flux_err,line_mask')
    continuumdata = np.rec.fromarrays(continuumdata, names='wave,flux,flux_err')

    return linedata, continuumdata


def blotch_spectrum(spec, linedata):
    """
    Accepts a recarray spectrum, spec, and a tuple linedata, such as can be
    produced by orig_cut_lines, and blotches it

    Some spectra have nasty cosmic rays or giant gaps in the data This routine
    does a reasonable job blotching these by Wiener filtering the spectrum,
    marking features that differ significantly from the local variance in the
    region And replace them with the filtered values

    The lines specified in linedata are preserved, so if your gap/cosmic ray
    lands on a line it will not be filtered. Additionally, filtering has edge
    effects, and these data are preserved as well. If you do blotch the
    spectrum, it is highly recommended that you use the bluelimit and redlimit
    options to trim the ends of the spectrum.

    YOU SHOULD PROBABLY PRE-PROCESS YOUR DATA YOURSELF BEFORE FITTING IT AND
    NOT BE LAZY!

    Returns the blotched spectrum
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
    saveind = linedata[-1]
    spec.flux[saveind] = linedata[1]
    spec.flux[0:window] = blueend
    spec.flux[-window:] = redend
    return spec


def rebin_spec_by_int_factor(spec, f=1):
    """
    Rebins a spectrum by an integer factor f

    Accepts
        spec: recarray spectrum (wave, flux, flux_err)
        f: an integer factor to rebin the spectrum by

    If the spectrum is not divisible by f, the edges are trimmed by discarding
    the remainder measurements from both ends. If the remainder itself is odd, the
    extra measurement is discarded from the blue side.

    Returns the rebinned recarray spectrum
    """
    f    = int(f)
    if f <= 1:
        return spec
    nwave     = len(spec.wave)
    rnwave    = nwave/f
    # if the spectrum is not perfectly divisible by the f, cut the remainder out
    # divide it evenly between blue and red
    remainder = nwave%f
    ncut      = remainder/2
    icut_blue = ncut + (remainder%2)
    icut_red  = nwave - ncut
    ispec     = spec[icut_blue:icut_red]
    rf        = [(np.mean(ispec.wave[f*x:f*x+f]),\
                  np.average(ispec.flux[f*x:f*x+f],\
                             weights=1./ispec.flux_err[f*x:f*x+f]**2.,\
                             returned=True))\
                 for x in range(rnwave)]
    rwave, rf = zip(*rf)
    rflux, rw = zip(*rf)
    rw = np.array(rw)
    rflux_err = 1./(rw**0.5)
    rwave = np.array(rwave)
    rflux  = np.array(rflux)
    rspec = np.rec.fromarrays((rwave, rflux, rflux_err),names='wave,flux,flux_err')
    return rspec


def pre_process_spectrum(spec, bluelimit, redlimit, model, params,\
        lamshift=0., vel=0., rebin=1, blotch=False, rescale=False):
    """
    Accepts
        spec: recarray spectrum (wave, flux, flux_err)
        bluelimit, redlimit: the wavelength limits in Angstrom
        model: a WDmodel instance
        params: the input fit parameters - only modified if rescale
        lamshift: applies a flat wavelength offset to the spectrum wavelengths
        vel: applies a velocity shift to the spectrum wavelengths
        rebin: integer factor to rebin the spectrum by. Does not correlate bins.
        blotch: not very robust removal of gaps, cosmic rays and other weird reduction defects
        rescale: rescale the spectrum to make the noise ~1

    Returns the (optionally blotched) spectrum, the continuum model for the
    spectrum, and the extracted line and continuum data for visualization

    Applies an offset lamshift to the spectrum wavelengths if non-zero. This is
    useful to correct slit centering errors with wide slits, which result in
    flat wavelength calibration shifts

    Applies any velocity shift to the spectrum wavelengths if non-zero.
    Blueshifts are negative, redshifts are positive.

    Does a coarse extraction of Balmer lines in the optical, (optionally)
    blotches the data, builds a continuum model for visualization purposes, and
    trims the spectrum the red and blue limits

    Rescales the spectrum to make the noise~1. If so, also changes the bounds
    and scale on dl, and any supplied initial guess

    Note that the continuum model and spectrum are the same length and both
    respect blue/red limits.
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
    if bluelimit > 0:
        bluelimit = float(bluelimit)
    else:
        bluelimit = spec.wave.min()

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
        print "Scaling the spectrum by {:.10g}".format(scale_factor)
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


def quick_fit_spec_model(spec, model, params, rvmodel='od94'):
    """
    Does a quick fit of the spectrum to get an initial guess of the fit parameters
    This isn't robust, but it's good enough for an initial guess
    None of the starting values for the parameters maybe None EXCEPT c
    This refines the starting guesses, and determines a reasonable value for c

    Accepts
        spectrum: recarray spectrum with wave, flux, flux_err
        model: WDmodel.WDmodel instance
        params: dict of parameters with keywords value, fixed, bounds for each

    Uses iminuit to do a rough diagonal fit - i.e. ignores covariance
    For simplicity, also fixed FWHM and Rv
    Therefore, only teff, logg, av, dl are fit for (at most)

    Returns best guess parameters and errors
    """
    teff0 = params['teff']['value']
    logg0 = params['logg']['value']
    av0   = params['av']['value']
    dl0   = params['dl']['value']

    # we don't actually fit for these values
    rv   = params['rv']['value']
    fwhm = params['fwhm']['value']

    fix_teff = params['teff']['fixed']
    fix_logg = params['logg']['fixed']
    fix_av   = params['av']['fixed']
    fix_dl   = params['dl']['fixed']

    if all((fix_teff, fix_logg, fix_av, fix_dl)):
        message = "All of teff, logg, av, dl are marked as fixed - nothing to fit."
        raise RuntimeError(message)

    pixel_scale = 1./np.median(np.gradient(spec.wave))

    if dl0 is None:
        # only dl and fwhm are allowed to have None as input values
        # fwhm will get set to a default fwhm if it's None
        mod = model._get_obs_model(teff0, logg0, av0, fwhm, spec.wave, rv=rv, rvmodel=rvmodel, pixel_scale=pixel_scale)
        c0   = spec.flux.mean()/mod.mean()
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
        mod = model._get_obs_model(teff, logg, av, fwhm, spec.wave, rv=rv, rvmodel=rvmodel, pixel_scale=pixel_scale)
        mod *= (1./(4.*np.pi*(dl)**2.))
        chi2 = np.sum(((spec.flux-mod)/spec.flux_err)**2.)
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
    emcee.utils.sample_ball doesn't care about bounds but is really convenient to init the walker positions

    Accepts
        pos: list of starting positions for all walkers produced by emcee.utils.sample_ball
        free_param_names: list of the names of free parameters
        param: dict of parameters with keywords value, fixed, bounds for each

    Assumes that the parameter values p0 was within bounds to begin with
    Takes p0 -/+ 5sigma or lower/upper bounds as the lower/upper limits whichever is higher/lower

    Returns pos with out of bounds positions fixed to be within bounds
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


def hyper_param_guess(spec, phot, model, pbs, params, rvmodel='od94'):
    """
    Makes a guess for mu after the initial minuit fit

    Accepts
        spec: recarray spectrum with wave, flux, flux_err
        phot: recarray of photometry with passband pb, magnitude mag, magnitude err mag_err
        model: WDmodel.WDmodel instance
        pbs: dict of throughput models for each passband with passband name as key
        params: dict of parameters with keywords value, fixed, bounds, scale for each
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
            rv=rv, rvmodel=rvmodel, pixel_scale=pixel_scale)

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


def fit_model(spec, phot, model, covmodel, pbs, params,\
            objname, outdir, specfile,\
            rvmodel='od94', phot_dispersion=0.,\
            samptype='ensemble', ascale=2.0,\
            ntemps=1, nwalkers=300, nburnin=50, nprod=1000, everyn=1, thin=1, pool=None,\
            redo=False):
    """
    Models the spectrum using the white dwarf model and a Gaussian process with
    an exponential squared kernel to account for any flux miscalibration

    Accepts
        spec: recarray spectrum with wave, flux, flux_err
        phot: recarray of photometry with passband pb, magnitude mag, magnitude err mag_err
        model: WDmodel.WDmodel instance
        covmodel: WDmodel.covmodel.WDmodel_CovModel instance
        pbs: dict of throughput models for each passband with passband name as key
        params: dict of parameters with keywords value, fixed, bounds, scale for each

    Uses an Ensemble MCMC (implemented by emcee) to generate samples from the
    posterior. Does a short burn-in around the initial guess model parameters -
    either minuit or user supplied values/defaults. Model parameters may be
    frozen/fixed. Parameters can have bounds limiting their range.

    The WDmodel_Posterior class implements additional priors on
    parameters. See there for details.

    pool controls if the process is run with MPI or single threaded.  If pool
    is an MPIPool object and the process is started with mpirun, the tasks are
    divided amongst the MPI processes.

    Incrementally saves the chain if run single-threaded
    """

    lnlike = likelihood.setup_likelihood(params)
    nparam   = lnlike.vector_size

    # get the starting position and the scales for each parameter
    init_p0  = lnlike.get_parameter_dict()
    p0       = init_p0.values()
    free_param_names = init_p0.keys()
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

    # even if we only take every nth sample, the pixel scale is the same
    pixel_scale = 1./np.median(np.gradient(spec.wave))

    # configure the posterior function
    lnpost = likelihood.WDmodel_Posterior(inspec, phot, model, covmodel, rvmodel, pbs, lnlike,\
            pixel_scale=pixel_scale, phot_dispersion=phot_dispersion)

    # setup the sampler
    if samptype == 'ensemble':
        sampler = emcee.EnsembleSampler(nwalkers, nparam, lnpost,\
                a=ascale,  pool=pool)
        thin = 1
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
            lnprob0 = map(lnpost, inpos)
        else:
            lnprob0 = pool.map(lnpost, inpos)

        lnprob0 = np.array(lnprob0)
        lnprob0 = lnprob0.reshape(ntemps, nwalkers)
        sampler_kwargs = {'gibbs':gibbs, 'thin':thin, 'lnprob0':lnprob0}

    # do a short burn-in
    print "Burn-in"
    pos, _,  _ = sampler.run_mcmc(pos, thin*nburnin, **sampler_kwargs)

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
    print "\nMAP Parameters after Burn-in"
    for k, v in lnlike.get_parameter_dict().items():
        print "{} = {:f}".format(k,v)

    # adjust the walkers to start around the MAP position from burnin
    pos = emcee.utils.sample_ball(p1, std, size=ntemps*nwalkers)
    pos = fix_pos(pos, free_param_names, params)
    if samptype != 'ensemble':
        pos = pos.reshape(ntemps, nwalkers, nparam)

    # create a HDF5 file to hold the chain data
    outfile = io.get_outfile(outdir, specfile, '_mcmc.hdf5',check=True, redo=redo)

    # setup incremental chain saving
    outf = h5py.File(outfile, 'w')
    chain = outf.create_group("chain")

    # save some other attributes about the chain
    chain.attrs["nwalkers"] = nwalkers
    chain.attrs["nprod"]    = nprod
    chain.attrs["nparam"]   = nparam
    chain.attrs["everyn"]   = everyn
    chain.attrs["ascale"]   = ascale
    chain.attrs["samptype"] = samptype
    chain.attrs["ntemps"]   = ntemps
    chain.attrs["thin"]     = thin

    # save the parameter names corresponding to the chain
    free_param_names = np.array(free_param_names)
    dt = free_param_names.dtype.str.lstrip('|')
    chain.create_dataset("names",data=free_param_names, dtype=dt)

    # save the parameter configuration as well
    # this is redundant, since it is saved in the JSON file, but having it one place is nice
    names = lnlike.get_parameter_names(include_frozen=True)
    names = np.array(names)
    dt = names.dtype.str.lstrip('|')
    par_grp = outf.create_group("params")
    par_grp.create_dataset("names",data=names, dtype=dt)
    for param in params:
        this_par = par_grp.create_group(param)
        this_par.attrs["value"]  = params[param]["value"]
        this_par.attrs["fixed"]  = params[param]["fixed"]
        this_par.attrs["scale"]  = params[param]["scale"]
        this_par.attrs["bounds"] = params[param]["bounds"]

    # write to disk before we start
    outf.flush()

    # since we're going to save the chain in HDF5, we don't need to save it in memory elsewhere
    sampler_kwargs['storechain']=False

    # production
    dset_chain  = chain.create_dataset("position",(ntemps*nwalkers*nprod,nparam),maxshape=(None,nparam))
    dset_lnprob = chain.create_dataset("lnprob",(ntemps*nwalkers*nprod,),maxshape=(None,))
    with progress.Bar(label="Production", expected_size=nprod, hide=False) as bar:
        bar.show(0)
        for i, result in enumerate(sampler.sample(pos, iterations=nprod, **sampler_kwargs)):
            position = result[0]
            lnpost   = result[1]
            rstate   = result[2]
            position = position.reshape((-1, nparam))
            lnpost   = lnpost.reshape(ntemps*nwalkers)
            dset_chain[ntemps*nwalkers*i:ntemps*nwalkers*(i+1),:] = position
            dset_lnprob[ntemps*nwalkers*i:ntemps*nwalkers*(i+1)] = lnpost
            if (i > 0) & (i%100 == 0):
                outf.flush()
            bar.show(i+1)

    # save the acceptance fraction
    chain.create_dataset("afrac", data=sampler.acceptance_fraction)
    if samptype != 'ensemble':
        chain.create_dataset("tswap_afrac", data=sampler.tswap_acceptance_fraction)

    # TODO save the rstate of the chain to allow us to restore state and
    # increase length of chain if we want

    samples         = np.array(dset_chain)
    samples_lnprob  = np.array(dset_lnprob)
    outf.flush()
    outf.close()
    if pool is not None:
        pool.close()

    # find the MAP value after production
    map_samples = samples.reshape(ntemps, nwalkers, nprod, nparam)
    map_samples_lnprob = samples_lnprob.reshape(ntemps, nwalkers, nprod)
    max_ind = np.argmax(map_samples_lnprob)
    max_ind = np.unravel_index(max_ind, (ntemps, nwalkers, nprod))
    max_ind = tuple(max_ind)
    p_final = map_samples[max_ind]
    lnlike.set_parameter_vector(p_final)
    print "\nMAP Parameters after Production"

    for k, v in lnlike.get_parameter_dict().items():
        print "{} = {:f}".format(k,v)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    return  free_param_names, samples, samples_lnprob


def get_fit_params_from_samples(param_names, samples, samples_lnprob, params,\
        ntemps=1, nwalkers=300, nprod=1000, discard=5):
    """
    Get the marginalized parameters from the sample chain

    Accepts
        param_names: ordered vector of parameter names, corresponding to each of the dimensions of sample
        samples: flat chain (nwalker*nprod, ndim) array of walker positions
        samples_lnprob: flat chain (nwalker*nprod) array of log likelihood corresponding to sampler position
        params: dict of parameters with keywords value, fixed, bounds for each

        The following keyword arguments should be consistent with the call to fit_model/mpifit_model
            ntemps: number of temperatures
            nwalkers: number of walkers
            nprod:  number of steps

        Finally, even with the burn-in, the walkers may be tightly correlated
        initially, so the discard keyword allows a percentage of the nprod
        steps to be discarded.
            discard: percentage of nprod steps to discard

    Returns dictionary with the marginalized parameter values and errors,
    filtered flat chain of sampler position, filtered flat chain of log
    likelihood corresponding to sampler position

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

    mask = np.isfinite(in_lnprob)

    for i, param in enumerate(param_names):
        x = in_samp[mask,i]
        q_16, q_50, q_84 = np.percentile(x, [16., 50., 84.])
        params[param]['value']  = q_50
        params[param]['bounds'] = (q_16, q_84)
        params[param]['errors_pm'] = (q_84 - q_50, q_50 - q_16)
        params[param]['scale']  = float(np.std(x))

    fixed_params = set(params.keys()) - set(param_names)
    for param in fixed_params:
        if params[param]['fixed']:
            params[param]['scale'] = 0.
            params[param]['errors_pm'] = (0., 0.)
        else:
            # this should never happen, unless we did something stupid between fit_WDmodel and mpifit_WDmodel
            print "Huh.... {} not marked as fixed but was not fit for...".format(param)
    return params, in_samp[mask,:], in_lnprob[mask]
