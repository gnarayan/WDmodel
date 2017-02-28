import time
import warnings
warnings.simplefilter('once')
import os
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.stats as scistat
import scipy.signal as scisig
from iminuit import Minuit
import emcee
import h5py
from clint.textui import progress
from .WDmodel import WDmodel
from . import io
from . import likelihood


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
    scaling = scistat.norm.ppf(3/4.)
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


def pre_process_spectrum(spec, bluelimit, redlimit, blotch=False):
    """
    Accepts a recarray spectrum, spec, blue and red limits, and an optional
    keyword blotch

    Does a coarse extraction of Balmer lines in the optical, (optionally)
    blotches the data, builds a continuum model for visualization purposes, and
    trims the spectrum the red and blue limits

    Returns the (optionally blotched) spectrum, the continuum model for the
    spectrum, and the extracted line and continuum data for visualization

    Note that the continuum model and spectrum are the same length and both
    respect blue/red limits.
    """

    # Test that the array is monotonic 
    WDmodel._wave_test(spec.wave)
    model = WDmodel()

    # get a coarse mask of line and continuum
    linedata, continuumdata  = orig_cut_lines(spec, model)
    if blotch:
        spec = blotch_spectrum(spec, linedata)

    # get a coarse estimate of the full continuum
    # this isn't used for anything other than cosmetics
    cont_model = polyfit_continuum(continuumdata, spec.wave)
    
    # clip the spectrum to whatever range is requested
    if bluelimit > 0:
        bluelimit = float(bluelimit)
    else:
        bluelimit = spec.wave.min()
    
    if redlimit > 0:
        redlimit = float(redlimit)
    else:
        redlimit = spec.wave.max()
    
    # trim the outputs to the requested length
    usemask = ((spec.wave >= bluelimit) & (spec.wave <= redlimit))
    spec = spec[usemask]
    cont_model = cont_model[usemask]

    usemask = ((linedata.wave >= bluelimit) & (linedata.wave <= redlimit))
    linedata = linedata[usemask]

    usemask = ((continuumdata.wave >= bluelimit) & (continuumdata.wave <= redlimit))
    continuumdata = continuumdata[usemask]

    return spec, cont_model, linedata, continuumdata


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

    if dl0 is None:
        # only dl and fwhm are allowed to have None as input values
        # fwhm will get set to a default fwhm if it's None
        mod = model._get_obs_model(teff0, logg0, av0, fwhm, spec.wave, rv=rv, rvmodel=rvmodel)
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
        mod = model._get_obs_model(teff, logg, av, fwhm, spec.wave, rv=rv, rvmodel=rvmodel)
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
 
    # there's some objects for which minuit will fail
    if outfnmin['is_valid']:
        try:
            m.hesse()
        except RuntimeError:
            message = "Something seems to have gone wrong with Hessian matrix computation. You should probably stop."
            warnings.warn(message, RuntimeWarning)
    else:
        message = "Something seems to have gone wrong refining parameters with migrad. You should probably stop."
        warnings.warn(message, RuntimeWarning)

    result = m.values
    errors = m.errors
    # duplicate the input dicrionary and update
    migrad_params = params.copy()
    for param in result:
        migrad_params[param]['value'] = result[param]
        migrad_params[param]['scale'] = errors[param]

    return migrad_params


#**************************************************************************************************************

# make a local copy of the loglikelihood function
loglikelihood = likelihood.loglikelihood

#**************************************************************************************************************

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
        ll = max(lb, lr)
        ul = min(ub, ur)
        ind = np.where((pos[:,i] <= ll) & (pos[:,i] >= ul))
        nreplace = len(pos[:i][ind])
        pos[:,i][ind] = np.random.rand(nreplace)*(ul - ll) + ll

    return pos


def fit_model(spec, phot, model, params,\
            objname, outdir, specfile,\
            rvmodel='od94',\
            ascale=2.0, nwalkers=300, nburnin=50, nprod=1000, everyn=1,\
            redo=False, excludepb=None):
    """
    Models the spectrum using the white dwarf model and a gaussian process with
    an exponential squared kernel to account for any flux miscalibration

    TODO: add modeling the phot

    Accepts
        spec: recarray spectrum with wave, flux, flux_err
        phot: recarray of photometry ith passband pb, magnitude mag, magintude err mag_err
        model: WDmodel.WDmodel instance
        params: dict of parameters with keywords value, fixed, bounds for each

    Uses an Ensemble MCMC (implemented by emcee) to generate samples from the
    posterior. Does a short burn-in around the initial guess model parameters -
    either minuit or user supplied values/defaults. Model parameters may be
    frozen/fixed. Parameters can have bounds limiting their range.

    The prior is a tophat on all parameters, preventing them from going out of
    range. The WDmodel.likelihood class can implement additional priors on
    parameters.

    Incrementally saves the chain. 
    """

    # parse the params to create a dictionary and init the WDmodel.likelihood.WDmodel_Likelihood instance
    setup_args = {}
    bounds     = []
    scales     = {}
    fixed      = {}
    for param in likelihood._PARAMETER_NAMES:
        setup_args[param] = params[param]['value']
        bounds.append(params[param]['bounds'])
        scales[param] = params[param]['scale']
        fixed[param] = params[param]['fixed']

    setup_args['bounds'] = bounds
    lnprob = likelihood.WDmodel_Likelihood(**setup_args)

    # freeze any parameters that we want fixed
    for param, val in fixed.items():
        if val:
            print "Freezing {}".format(param)
            lnprob.freeze_parameter(param)

    nparam   = lnprob.vector_size

    # get the starting position and the scales for each parameter
    init_p0  = lnprob.get_parameter_dict()
    p0       = init_p0.values()
    free_param_names = init_p0.keys()
    std = [scales[x] for x in free_param_names]

    if nwalkers==0:
        print "nwalkers set to 0. Not running MCMC"
        return

    # create a sample ball 
    pos = emcee.utils.sample_ball(p0, std, size=nwalkers)
    pos = fix_pos(pos, free_param_names, params)

    if everyn != 1:
        spec = spec[::everyn]

    # exclude passbands that we want excluded 
    pbnames = np.unique(phot.pb) 
    if excludepb is not None:
        pbnames = list(set(pbnames) - set(excludepb))

    # TODO get the passbands
    pbmodel = None

    # setup the sampler
    sampler = emcee.EnsembleSampler(nwalkers, nparam, loglikelihood,\
            a=ascale, args=(spec, phot, model, rvmodel, pbmodel, lnprob)) 

    # do a short burn-in
    if nburnin > 0:
        print "Burn-in"
        pos, prob, state = sampler.run_mcmc(pos, nburnin, storechain=False)
        sampler.reset()
        lnprob.set_parameter_vector(pos[np.argmax(prob)])
        print "\nParameters after Burn-in"
        for k, v in lnprob.get_parameter_dict().items():
            print "{} = {:f}".format(k,v)

        # init a new set of walkers around the maximum likelihood position from the burn-in
        burnin_p0 = pos[np.argmax(prob)]
        pos = emcee.utils.sample_ball(burnin_p0, std, size=nwalkers)
        burnin_params = params.copy()
        for i, key in enumerate(free_param_names):
            burnin_params[key]['value'] = burnin_p0[i]
        pos = fix_pos(pos, free_param_names, burnin_params)

    # create a HDF5 file to hold the chain data
    outfile = io.get_outfile(outdir, specfile, '_mcmc.hdf5')
    if os.path.exists(outfile) and (not redo):
        message = "Output file %s already exists. Specify --redo to clobber."%outfile
        raise IOError(message)

    # setup incremental chain saving
    outf = h5py.File(outfile, 'w')
    chain = outf.create_group("chain")
    dset_chain  = chain.create_dataset("position",(nwalkers*nprod,nparam),maxshape=(None,nparam))
    dset_lnprob = chain.create_dataset("lnprob",(nwalkers*nprod,),maxshape=(None,))

    # save some other attributes about the chain
    chain.create_dataset("nwalkers", data=nwalkers)
    chain.create_dataset("nprod", data=nprod)
    chain.create_dataset("nparam", data=nparam)
    chain.create_dataset("everyn",data=everyn)

    # save the parameter names corresponding to the chain 
    free_param_names = np.array(free_param_names)
    dt = free_param_names.dtype.str.lstrip('|')
    chain.create_dataset("names",data=free_param_names, dtype=dt)
    
    # save the parameter configuration as well
    names = lnprob.get_parameter_names(include_frozen=True)
    names = np.array(names)
    dt = names.dtype.str.lstrip('|')
    par_grp = outf.create_group("params")
    par_grp.create_dataset("names",data=names, dtype=dt)
    for param in params:
        this_par = par_grp.create_group(param)
        this_par.create_dataset("value",data=params[param]["value"])
        this_par.create_dataset("fixed",data=params[param]["fixed"])
        this_par.create_dataset("scale",data=params[param]["scale"])
        this_par.create_dataset("bounds",data=params[param]["bounds"])

    # production
    with progress.Bar(label="Production", expected_size=nprod) as bar:
        for i, result in enumerate(sampler.sample(pos, iterations=nprod)):
            position = result[0]
            lnpost   = result[1]
            dset_chain[nwalkers*i:nwalkers*(i+1),:] = position
            dset_lnprob[nwalkers*i:nwalkers*(i+1)] = lnpost
            outf.flush()
            bar.show(i+1)

    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    samples         = np.array(dset_chain)
    samples_lnprob  = np.array(dset_lnprob)
    outf.close()

    return  free_param_names, samples, samples_lnprob


def mpi_fit_model(spec, phot, model, params,\
            objname, outdir, specfile,\
            rvmodel='od94',\
            ascale=2.0, nwalkers=300, nburnin=50, nprod=1000, everyn=1, pool=None,\
            redo=False, excludepb=None):
    """
    Models the spectrum using the white dwarf model and a gaussian process with
    an exponential squared kernel to account for any flux miscalibration

    TODO: add modeling the phot

    Accepts
        spec: recarray spectrum with wave, flux, flux_err
        phot: recarray of photometry ith passband pb, magnitude mag, magintude err mag_err
        model: WDmodel.WDmodel instance
        params: dict of parameters with keywords value, fixed, bounds for each

    Uses an Ensemble MCMC (implemented by emcee) to generate samples from the
    posterior. Does a short burn-in around the initial guess model parameters -
    either minuit or user supplied values/defaults. Model parameters may be
    frozen/fixed. Parameters can have bounds limiting their range.

    The prior is a tophat on all parameters, preventing them from going out of
    range. The WDmodel.likelihood class can implement additional priors on
    parameters.

    This version is identical to the fit_model() routine except for the
    incremental chain saving It's intended for use with mpifit_WDmodel.py since
    with MPI, we do not want the pool sitting idle, while master is making
    incremental writes to disk.

    """

    # parse the params to create a dictionary and init the WDmodel.likelihood.WDmodel_Likelihood instance
    setup_args = {}
    bounds     = []
    scales     = {}
    fixed      = {}
    for param in likelihood._PARAMETER_NAMES:
        setup_args[param] = params[param]['value']
        bounds.append(params[param]['bounds'])
        scales[param] = params[param]['scale']
        fixed[param] = params[param]['fixed']

    setup_args['bounds'] = bounds
    lnprob = likelihood.WDmodel_Likelihood(**setup_args)

    # freeze any parameters that we want fixed
    for param, val in fixed.items():
        if val:
            print "Freezing {}".format(param)
            lnprob.freeze_parameter(param)

    nparam   = lnprob.vector_size

    # get the starting position and the scales for each parameter
    init_p0  = lnprob.get_parameter_dict()
    p0       = init_p0.values()
    free_param_names = init_p0.keys()
    std = [scales[x] for x in free_param_names]

    if nwalkers==0:
        print "nwalkers set to 0. Not running MCMC"
        return

    # create a sample ball 
    pos = emcee.utils.sample_ball(p0, std, size=nwalkers)
    pos = fix_pos(pos, free_param_names, params)

    # only use every n'th point - useful for testing since we want speedup
    if everyn != 1:
        spec = spec[::everyn]

    # exclude passbands that we want excluded 
    pbnames = np.unique(phot.pb) 
    if excludepb is not None:
        pbnames = list(set(pbnames) - set(excludepb))

    pbmodel = None
    # setup the sampler
    sampler = emcee.EnsembleSampler(nwalkers, nparam, loglikelihood,\
            a=ascale, args=(spec, phot, model, rvmodel, pbmodel, lnprob), pool=pool) 

    # do a short burn-in
    if nburnin > 0:
        print "Burn-in"
        pos, prob, state = sampler.run_mcmc(pos, nburnin, storechain=False)
        sampler.reset()
        lnprob.set_parameter_vector(pos[np.argmax(prob)])
        print "\nParameters after Burn-in"
        for k, v in lnprob.get_parameter_dict().items():
            print "{} = {:f}".format(k,v)

        # init a new set of walkers around the maximum likelihood position from the burn-in
        burnin_p0 = pos[np.argmax(prob)]
        pos = emcee.utils.sample_ball(burnin_p0, std, size=nwalkers)
        burnin_params = params.copy()
        for i, key in enumerate(free_param_names):
            burnin_params[key]['value'] = burnin_p0[i]
        pos = fix_pos(pos, free_param_names, burnin_params)

    # create a HDF5 file to hold the chain data
    outfile = io.get_outfile(outdir, specfile, '_mcmc.hdf5')
    if os.path.exists(outfile) and (not redo):
        message = "Output file %s already exists. Specify --redo to clobber."%outfile
        raise IOError(message)

    # setup chain saving
    # NOTE THAT THIS IS NOT INCREMENTAL WITH MPI
    # We don't want the entire pool sitting around while master is buys writing to disk
    outf = h5py.File(outfile, 'w')
    chain = outf.create_group("chain")

    # save some other attributes about the chain
    chain.create_dataset("nwalkers", data=nwalkers)
    chain.create_dataset("nprod", data=nprod)
    chain.create_dataset("nparam", data=nparam)
    chain.create_dataset("everyn",data=everyn)

    free_param_names = np.array(free_param_names)
    dt = free_param_names.dtype.str.lstrip('|')
    chain.create_dataset("names",data=free_param_names, dtype=dt)
    
    # save the parameter configuration as well
    names = lnprob.get_parameter_names(include_frozen=True)
    names = np.array(names)
    dt = names.dtype.str.lstrip('|')
    par_grp = outf.create_group("params")
    par_grp.create_dataset("names",data=names, dtype=dt)
    for param in params:
        this_par = par_grp.create_group(param)
        this_par.create_dataset("value",data=params[param]["value"])
        this_par.create_dataset("fixed",data=params[param]["fixed"])
        this_par.create_dataset("scale",data=params[param]["scale"])
        this_par.create_dataset("bounds",data=params[param]["bounds"])

    # production
    start = time.clock()
    print "Started at : {:f}".format(start)
    pos, prob, state = sampler.run_mcmc(pos, nprod)
    end = time.clock()
    print "Stopped at : {:f}".format(end)
    print "Done in {:f} minutes".format((end-start)/60.)

    # save the chain and 
    dset_chain  = chain.create_dataset("position",data=sampler.flatchain)
    dset_lnprob = chain.create_dataset("lnprob",data=sampler.flatlnprobability)

    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    samples         = np.array(dset_chain)
    samples_lnprob  = np.array(dset_lnprob)
    outf.close()

    return  free_param_names, samples, samples_lnprob


def get_fit_params_from_samples(param_names, samples, samples_lnprob, params, nwalkers=300, nprod=1000, discard=5): 
    """
    Get the margnialized parameters from the sample chain

    Accepts
        param_names: ordered vector of parameter names, corresponding to each of the dimensions of sample
        samples: flat chain (nwalker*nprod, ndim) array of walker positions
        samples_lnprob: flat chain (nwalker*nprod) array of log likelihood corresponding to sampler position
        params: dict of parameters with keywords value, fixed, bounds for each

        The following keyword arguments should be consistent with the call to fit_model/mpifit_model
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

    in_samp   = samples.reshape(nwalkers, nprod, ndim)
    in_lnprob = samples_lnprob.reshape(nwalkers, nprod)

    # discard the first %discard steps from all the walkers
    nstart    = int(np.ceil((discard/100.)*nprod))
    in_samp   = in_samp[:,nstart:,:]
    in_lnprob = in_lnprob[:,nstart:]

    in_samp = in_samp.reshape((-1, ndim))
    in_lnprob = in_lnprob.reshape(in_samp.shape[0])
        
    mask = np.isfinite(in_lnprob)

    result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(in_samp[mask,:], [16, 50, 84], axis=0)))
    for i, param in enumerate(param_names):
        params[param]['value']  = result[i][0]
        params[param]['bounds'] = (result[i][0] - result[i][2], result[i][1] + result[i][0])
        params[param]['errors_pm'] = (result[i][1], result[i][2])
        scale = float(np.std(in_samp[mask,i]))
        params[param]['scale']  = scale
    fixed_params = set(params.keys()) - set(param_names)
    for param in fixed_params:
        if params[param]['fixed']:
            params[param]['scale'] = 0.
            params[param]['errors_pm'] = (0., 0.)
        else:
            # this should never happen, unless we did something stupid between fit_WDmodel and mpifit_WDmodel
            print "Huh.... {} not marked as fixed but was not fit for...".format(param)
    return params, in_samp[mask,:], in_lnprob[mask]
