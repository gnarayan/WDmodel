import sys
import warnings
warnings.simplefilter('once')
import os
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.stats as scistat
import scipy.signal as scisig
import scipy.optimize as sciopt
from astropy.convolution import Gaussian1DKernel, convolve
from iminuit import Minuit
import george
import emcee
import h5py
from clint.textui import progress
from .WDmodel import WDmodel
from . import likelihood
from . import viz


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

    cwave, cflux, cdflux = continuumdata
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
    returns a tuple with the data on the absorption lines
    (wave, flux, fluxerr, Balmer line number, index from original spectrum)

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
    linedata = (line_wave, line_flux, line_fluxerr, line_number, line_ind)
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
    
    # trim the spectrum to the requested length
    usemask = ((spec.wave >= bluelimit) & (spec.wave <= redlimit))
    spec = spec[usemask]
    cont_model = cont_model[usemask]

    return spec, cont_model, linedata, continuumdata


#**************************************************************************************************************

def quick_fit_spec_model(spec, model, fwhm, rv=3.1, rvmodel='od94'):
    """
    Does a quick fit of the spectrum to get an initial guess of the spectral parameters
    This isn't robust, but it's good enough for an initial guess
    The guess defines the regions of the spectrum that are actually used for each line in the full fit
    These regions are only weak functions of Teff, logg
    It beats hard-coded pre-defined regions which are only valid for some Teff, logg
    """

    nparam = 5
    
    # hardcode an initial guess that's somewhere near the mean of the sample
    teff_guess = 35000.
    logg_guess = 7.8
    av_guess   = 0.1
    fwhm_guess = fwhm
    mod = model._get_obs_model(teff_guess, logg_guess, av_guess, fwhm_guess, spec.wave, rv=rv, rvmodel=rvmodel)
    c_guess    = spec.flux.mean()/mod.mean()

    teff_scale = 2000.
    logg_scale = 0.1
    av_scale   = 0.1
    c_scale    = 10
    fwhm_scale = 1.

    teff_bounds = (17000,80000)
    logg_bounds = (7.,9.499999)
    av_bounds   = (0.,0.5)
    fwhm_bounds = (1., max(fwhm_guess, 20.))

    def chi2(teff, logg, av, c, fwhm):
        mod = model._get_obs_model(teff, logg, av, fwhm, spec.wave)

        # smooth the model, and extract the section that overlays the model
        # since we smooth the full model, computed on the full wavelength range of the spectrum
        # and then extract the subset range that overlaps with the data
        # we avoid any edge effects with smoothing at the end of the range
        mod *= c
        return np.sum(((spec.flux-mod)/spec.flux_err)**2.)

    m = Minuit(chi2, teff=teff_guess, logg=logg_guess, av=av_guess, c=c_guess, fwhm=fwhm_guess,\
                error_teff=teff_scale, error_logg=logg_scale, error_av=av_scale, error_c=c_scale, error_fwhm=fwhm_scale,\
                limit_teff=teff_bounds, limit_logg=logg_bounds, limit_av=av_bounds, limit_fwhm=fwhm_bounds,\
                print_level=1)

    m.migrad()
    result = m.args

    #p0 = np.ones(nparam).tolist()
    #bounds = []
    #p0[0] = teff_guess
    #p0[1] = logg_guess
    #p0[2] = av_guess
    #p0[3] = c_guess
    #bounds.append((17000,80000))
    #bounds.append((7.,9.499999))
    #bounds.append((0.,0.5))
    #bounds.append((None, None))

    #nll = likelihood.nll
    # do a quick fit with minimize to get a decent starting guess
    #result = sciopt.minimize(nll, p0, bounds=bounds, args=(spec, model, kernel))

    return result


#**************************************************************************************************************

def fit_model(spec, phot,\
            objname, outdir, specfile,\
            rv=3.1, rvmodel='od94', fwhm=4.,\
            nwalkers=200, nburnin=500, nprod=2000, nthreads=1,\
            redo=False):

    # we set the output file based on the spectrum name, since we can have multiple spectra per object
    outfile = os.path.join(outdir, os.path.basename(objname.replace('.flm','.mcmc.hdf5')))
    if os.path.exists(outfile) and (not redo):
        message = "Output file %s already exists. Specify --redo to clobber."%outfile
        raise IOError(message)

    nparam = 3

    wave    = spec.wave
    flux    = spec.flux
    fluxerr = spec.flux_err

    outf = h5py.File(outfile, 'w')
    dset_spec = outf.create_group("spec")
    dset_spec.create_dataset("wave",data=wave)
    dset_spec.create_dataset("flux",data=flux)
    dset_spec.create_dataset("fluxerr",data=fluxerr)


    # init the model, and determine the coarse normalization to match the spectrum
    model = WDmodel()
    
    # do a quick, not very robust fit to create the quantities we need to store
    # get a reasonable starting position for the chains 
    # and set the wavelength thresholds for each line
    result = quick_fit_spec_model(spec, model, fwhm, rv=rv, rvmodel=rvmodel)
    print result
    teff, logg, av, c, fwhm = result
    # init a simple Gaussian 1D kernel to smooth the model to the resolution of the instrument
    gsig     = fwhm/np.sqrt(8.*np.log(2.))
    kernel   = Gaussian1DKernel(gsig)
    quick_fit_result = teff, logg, av, c, kernel
    fig = viz.plot_spectrum_fit(spec, objname, specfile, model, quick_fit_result, rv=rv, rvmodel=rvmodel)
    fig.show()
    outf.close()

    sys.exit(-1)

    if nwalkers==0:
        print "nwalkers set to 0. Not running MCMC"
        return

    # setup the sampler
    pos = [result.x + 1e-1*np.random.randn(nparam) for i in range(nwalkers)]
    if nthreads > 1:
        print "Multiproc"
        sampler = emcee.EnsembleSampler(nwalkers, nparam, lnprob,\
                threads=nthreads, args=(wave, model, kernel, balmerlinedata)) 
    else:
        sampler = emcee.EnsembleSampler(nwalkers, nparam, lnprob, args=(wave, model, kernel, balmerlinedata)) 

    # do a short burn-in
    if nburnin > 0:
        print "Burn-in"
        pos, prob, state = sampler.run_mcmc(pos, nburnin, storechain=False)
        sampler.reset()
        pos = pos[np.argmax(prob)] + 1e-2 * np.random.randn(nwalkers, nparam)

    # setup incremental chain saving
    # TODO need to test this alongside multiprocessing
    outf = h5py.File(outfile, 'a')
    chain = outf.create_group("chain")
    dset_chain = chain.create_dataset("position",(nwalkers*nprod,3),maxshape=(None,3))

    # production
    with progress.Bar(label="Production", expected_size=nprod) as bar:
        for i, result in enumerate(sampler.sample(pos, iterations=nprod)):
            position = result[0]
            dset_chain[nwalkers*i:nwalkers*(i+1),:] = position
            outf.flush()
            bar.show(i+1)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    samples = np.array(dset_chain)
    outf.close()
    return model, samples, kernel, balmerlinedata


