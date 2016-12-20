#!/usr/bin/env python
import sys
import os
import warnings
warnings.simplefilter('once')
import argparse
from clint.textui import progress
import numpy as np
import scipy.optimize as op
import scipy.interpolate as scinterp
import scipy.integrate as scinteg
import scipy.signal as scisig
import scipy.stats as scistat
import h5py
import george
import emcee
import WDmodel
from astropy import units as u
from astropy.convolution import convolve, Gaussian1DKernel
from specutils.extinction import extinction, reddening
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties as FM
from matplotlib import rc
from matplotlib.mlab import rec2txt
import corner
#rc('text', usetex=True)
#rc('font', family='serif')
#rc('ps', usedistiller='xpdf')
#rc('text.latex', preamble = ','.join('''\usepackage{amsmath}'''.split()))
#**************************************************************************************************************

def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """
    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree,1,degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree,1,count-1)

    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0-degree,count+degree+degree-1,dtype='int')
    else:
        kv = np.array([0]*degree + range(count-degree+1) + [count-degree]*degree,dtype='int')

    # Calculate query range
    u = np.linspace(periodic,(count-degree),n)

    # Calculate result
    arange = np.arange(len(u))
    points = np.zeros((len(u),cv.shape[1]))
    for i in xrange(cv.shape[1]):
                points[arange,i] = scinterp.splev(u, (kv,cv[:,i],degree))
    return points


#**************************************************************************************************************

def bspline_continuum(continuumdata, wave):
    cwave, cflux, cdflux = continuumdata
    cv = zip(cwave, cflux)
    points = bspline(cv, n=len(wave))
    cbspline = np.rec.fromrecords(points, names='wave,flux')
    mu = np.interp(wave, cbspline.wave, cbspline.flux)
    return wave, mu, None


#**************************************************************************************************************

def gp_continuum(continuumdata, bsw, bsf, wave, scalemin=500):
    cwave, cflux, cdflux = continuumdata
    kernel= np.median(cdflux)**2.*george.kernels.ExpSquaredKernel(scalemin)
    if bsw is not None:
        f = scinterp.interp1d(bsw, bsf, kind='linear', fill_value='extrapolate')
        def mean_func(x):
            x = np.array(x).ravel()
            return f(x)
    else:
        mean_func = 0.

    gp = george.GP(kernel=kernel, mean=mean_func)
    gp.compute(cwave, cdflux)
    pars, result = gp.optimize(cwave, cflux, cdflux,\
                    bounds=((None, np.log(3*np.median(cdflux)**2.)),\
                            (np.log(scalemin),np.log(100000)) ))
    mu, cov = gp.predict(cflux, wave)
    return wave, mu, cov


#**************************************************************************************************************

def lnprior(theta):
    teff, logg, av  = theta
    if 17000. < teff < 80000. and 7.0 < logg < 9.5 and 0. <= av <= 0.5:
        return 0.
    return -np.inf


def lnprob(theta, wave, model, kernel, balmer):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, wave, model, kernel, balmer)


def lnlike(theta, wave, model, kernel, balmerlinedata):
    teff, logg, av  = theta
    xi = model._get_xi(teff, logg, wave)
    mod = model._get_model(xi)

    # redden the model
    bluening = reddening(wave*u.Angstrom, av, r_v=3.1, model='od94')
    mod*=bluening

    # smooth the model, and extract the section that overlays the model
    # since we smooth the full model, computed on the full wavelength range of the spectrum
    # and then extract the subset range that overlaps with the data
    # we avoid any edge effects with smoothing at the end of the range
    smoothedmod = convolve(mod, kernel)
    (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu, cov) = balmerlinedata

    cwave = np.delete(wave, save_ind)
    cflux = np.delete(smoothedmod, save_ind)
    #mod_cflux = np.interp(line_wave, cwave, cflux)
    _, mod_cflux, _ = bspline_continuum((cwave, cflux, None), line_wave)
    _, smoothedfn = model._extract_from_indices(wave, smoothedmod, line_ind)

    smoothedfn*=(line_cflux/mod_cflux)
    #return gp.lnlikelihood(datafn - smoothedfn)
    return -0.5*np.sum(((line_flux-smoothedfn)/line_fluxerr)**2.)


def nll(*args):
    return -lnlike(*args)


#**************************************************************************************************************

def orig_cut_lines(spec, model):
    wave    = spec.wave
    flux    = spec.flux
    fluxerr = spec.flux_err
    balmerwaveindex = {}
    line_wave     = np.array([], dtype='float64', ndmin=1)
    line_flux     = np.array([], dtype='float64', ndmin=1)
    line_fluxerr  = np.array([], dtype='float64', ndmin=1)
    line_number   = np.array([], dtype='int', ndmin=1)
    line_ind      = np.array([], dtype='int', ndmin=1)
    save_ind      = np.array([], dtype='int', ndmin=1)
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
        save_ind     = np.hstack((save_ind, ZE[0]))
    # continuum data is just the spectrum with the Balmer lines removed
    continuumdata  = (np.delete(wave, save_ind), np.delete(flux, save_ind), np.delete(fluxerr, save_ind))
    linedata = (line_wave, line_flux, line_fluxerr, line_number, line_ind)
    return linedata, continuumdata, save_ind


#**************************************************************************************************************

def pre_process_spectrum(specfile, smooth, bluelimit, redlimit, balmerlines):
    """
    reads the input spectrum
    builds the continuum model
    extracts the lines
    """
    spec = read_spec(specfile)

    # remove any NaNs
    ind = np.where((np.isnan(spec.wave)==0) & (np.isnan(spec.flux)==0) & (np.isnan(spec.flux_err)==0))
    spec = spec[ind]

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
    mask = ((spec.wave >= bluelimit) & (spec.wave <= redlimit))
    spec = spec[mask]

    # Test that the array is monotonic 
    WDmodel.WDmodel._wave_test(spec.wave)
    model = WDmodel.WDmodel()

    balmer = np.atleast_1d(balmerlines).astype('int')
    balmer.sort()

    if smooth is None:
        spectable = read_spectable()
        shortfile = os.path.basename(specfile).replace('-total','')
        if shortfile.startswith('test'):
            message = 'Spectrum filename indicates this is a test - using default resolution 4.0'
            warnings.warn(message, RuntimeWarning)
            smooth = 8.0
        else:
            mask = (spectable.specname == shortfile)
            if len(spectable[mask]) != 1:
                message = 'Could not find an entry for this spectrum in the spectable file - using default resolution 4.0'
                warnings.warn(message, RuntimeWarning)
                smooth = 8.0
            else:
                smooth = spectable[mask].fwhm
    else:
        message = 'Smoothing factor specified on command line - overridng spectable file'
        warnings.warn(message, RuntimeWarning)
    print('Using smoothing factor %.2f'%smooth)
    
    scaling = scistat.norm.ppf(3/4.)
    linedata, continuumdata, saveind = orig_cut_lines(spec, model)

    # TODO make this a parameter 
    window1 = 7
    window2 = 151
    blueend = spec.flux[0:window2]
    redend  = spec.flux[-window2:]

    med_filt2 = scisig.wiener(spec.flux, mysize=window2)

    diff = np.abs(spec.flux - med_filt2)
    sigma = scisig.medfilt(diff, kernel_size=window2)

    sigma/=scaling
    mask = (diff > 5.*sigma)
    
    # clip the bad outliers from the spectrum
    spec.flux[mask] = med_filt2[mask]
    
    # restore the original lines, so that they aren't clipped
    spec.flux[saveind] = linedata[1]
    spec.flux[0:window2] = blueend
    spec.flux[-window2:] = redend

    # re-extract the continuum with the bad outliers hopefully masked out fully 
    continuumdata  = (np.delete(spec.wave, saveind), np.delete(spec.flux, saveind), np.delete(spec.flux_err, saveind))

    # create a smooth version of the spectrum to refine line detection
    med_filt1 = scisig.wiener(spec.flux, mysize=window1)

    # get a coarse estimate of the full continuum
    # fit a bspline to the continuum data
    # this is going to be oscillatory because of all the white noise
    # but it's good enough to serve as the mean function over the lines
    # then use a Gaussian Process to constrain the continuum
    bsw, bsf, _     = bspline_continuum(continuumdata, spec.wave)
    gpw, gpf, gpcov = gp_continuum(continuumdata, bsw, bsf, spec.wave)
    
    # compute the difference between the GP and the smooth spectrum
    pdiff = np.abs(gpf - med_filt1)/gpf
    # select where difference is within 1% of continuum
    # TODO make this a parameter 
    mask = (pdiff*100 <= 0.5)
    
    lineparams = [(x, model._get_line_indices(spec.wave, x)) for x in balmer]
    lineno, lineparams = zip(*lineparams)
    linecentroids, _  = zip(*lineparams)
    lineno = np.array(lineno)
    linecentroids = np.array(linecentroids)
    linelimits = {}
    for x, W0 in zip(lineno, linecentroids):
        delta_lambda = (spec.wave[mask] - W0)
        blueind  = (delta_lambda < 0)
        redind   = (delta_lambda > 0)
        if len(delta_lambda[blueind]) == 0 or len(delta_lambda[redind]) == 0:
            # spectrum not blue/red enough for this line
            continue 
        bluelim  = delta_lambda[blueind].argmax()
        redlim   = delta_lambda[redind].argmin()
        bluewave = spec.wave[mask][blueind][bluelim]
        redwave  = spec.wave[mask][redind][redlim]
        
        lineind = ((spec.wave >= bluewave) & (spec.wave <= redwave))
        signalind = (pdiff[lineind]*100. > 1.)
        if len(pdiff[lineind][signalind]) <= 5:
            print "Not enough signal ",W0
            continue

        linelimits[x] = (W0, bluewave, redwave)
    balmerwaveindex = {}
    line_wave     = np.array([], dtype='float64', ndmin=1)
    line_flux     = np.array([], dtype='float64', ndmin=1)
    line_fluxerr  = np.array([], dtype='float64', ndmin=1)
    line_number   = np.array([], dtype='int', ndmin=1)
    line_ind      = np.array([], dtype='int', ndmin=1)
    save_ind      = np.array([], dtype='int', ndmin=1)
    for x in range(1,7):
        if x in linelimits:
            (W0, bluewave, redwave) = linelimits[x]
            WO, ZE = model._get_indices_in_range(spec.wave,  bluewave, redwave, W0=W0)
            uZE = np.setdiff1d(ZE[0], save_ind)
            save_ind     = np.hstack((save_ind, uZE))
            x_wave, x_flux, x_fluxerr = model._extract_from_indices(spec.wave, spec.flux, (uZE), df=spec.flux_err)
        else:
            W0, ZE = model._get_line_indices(spec.wave, x)
            uZE = np.setdiff1d(ZE[0], save_ind)
            save_ind     = np.hstack((save_ind, uZE))
            continue

        balmerwaveindex[x] = W0, uZE
        line_wave    = np.hstack((line_wave, x_wave))
        line_flux    = np.hstack((line_flux, x_flux))
        line_fluxerr = np.hstack((line_fluxerr, x_fluxerr))
        line_number  = np.hstack((line_number, np.repeat(x, len(x_wave))))
        line_ind     = np.hstack((line_ind, uZE))
    # continuum data is just the spectrum with the Balmer lines removed
    continuumdata  = (np.delete(spec.wave, save_ind), np.delete(spec.flux, save_ind), np.delete(spec.flux_err, save_ind))
    linedata = (line_wave, line_flux, line_fluxerr, line_number, line_ind)
    balmer = sorted(linelimits.keys())
    return spec, linedata, continuumdata, save_ind, balmer, smooth, balmerwaveindex




#**************************************************************************************************************

def quick_fit_model(spec,linedata, continuumdata, save_ind,  balmer, model, kernel):
    """
    Does a quick fit of the spectrum to get an initial guess of the spectral parameters
    This isn't robust, but it's good enough for an initial guess
    The guess defines the regions of the spectrum that are actually used for each line in the full fit
    These regions are only weak functions of Teff, logg
    It beats hard-coded pre-defined regions which are only valid for some Teff, logg
    """

    nparam = 3

    wave    = spec.wave
    flux    = spec.flux
    fluxerr = spec.flux_err

    cwave, cflux, cdflux = continuumdata
    (line_wave, line_flux, line_fluxerr, line_number, line_ind) = linedata
    # do a quick Gaussian Process Fit to model the continuum of the specftrum
    gp = george.GP(kernel=george.kernels.ExpSquaredKernel(10))
    gp.compute(cwave, cdflux)
    pars, result = gp.optimize(cwave, cflux, cdflux, bounds=((10,100000),))
    mu, cov = gp.predict(cflux, wave)
    line_cflux , line_cov = gp.predict(cflux, line_wave)

    balmerlinedata = (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu, cov)

    p0 = np.ones(nparam).tolist()
    bounds = []
    p0[0] = 40000.
    p0[1] = 7.5
    p0[2] = 0.1
    bounds.append((17000,80000))
    bounds.append((7.,9.499999))
    bounds.append((0.,0.5))

    # do a quick fit with minimize to get a decent starting guess
    result = op.minimize(nll, p0, args=(wave, model, kernel, balmerlinedata), bounds=bounds)
    print result
    return balmerlinedata, continuumdata, result


#**************************************************************************************************************

def fit_model(objname, spec, linedata, continuumdata, save_ind, balmer=None, rv=3.1, rvmodel='od94', smooth=4., photfile=None,\
            nwalkers=200, nburnin=500, nprod=2000, nthreads=1, outdir=os.getcwd(), redo=False):

    outfile = os.path.join(outdir, os.path.basename(objname.replace('.flm','.mcmc.hdf5')))
    if os.path.exists(outfile) and (not redo):
        print("Output file already exists. Specify --redo to clobber.")
        sys.exit(0)

    nparam = 3


    #TODO - do something with the photometry to fit Av
    if photfile is not None:
        phot = read_phot(photfile)
        # set the likelihood functions here 
    else:
        phot = None
        # set the likelihood functions here 

    wave    = spec.wave
    flux    = spec.flux
    fluxerr = spec.flux_err
    outf = h5py.File(outfile, 'w')
    dset_spec = outf.create_group("spec")
    dset_spec.create_dataset("wave",data=wave)
    dset_spec.create_dataset("flux",data=flux)
    dset_spec.create_dataset("fluxerr",data=fluxerr)

    # init a simple Gaussian 1D kernel to smooth the model to the resolution of the instrument
    gsig     = smooth*(0.5/(np.log(2.)**0.5))
    kernel   = Gaussian1DKernel(gsig)

    # init the model, and determine the coarse normalization to match the spectrum
    model = WDmodel.WDmodel()
    data = {}
    
    # bundle the line dnd continuum data so we don't have to extract it every step in the MCMC
    if balmer is None:
        balmer = np.arange(1, 7)
    else:
        balmer = np.array(sorted(balmer))
    
    # do a quick, not very robust fit to create the quantities we need to store
    # get a reasonable starting position for the chains 
    # and set the wavelength thresholds for each line
    balmerlinedata, continuumdata, result = quick_fit_model(spec, linedata, continuumdata, save_ind, balmer, model, kernel)

    (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu, cov) = balmerlinedata
    (cwave, cflux, cdflux) = continuumdata

    dset_lines = outf.create_group("lines")
    dset_lines.create_dataset("line_wave",data=line_wave)
    dset_lines.create_dataset("line_flux",data=line_flux)
    dset_lines.create_dataset("line_fluxerr",data=line_fluxerr)
    dset_lines.create_dataset("line_number",data=line_number)
    dset_lines.create_dataset("line_cflux",data=line_cflux)
    dset_lines.create_dataset("line_cov",data=line_cov)
    dset_lines.create_dataset("line_ind",data=line_ind)
    dset_lines.create_dataset("save_ind",data=save_ind)

    # note that we bundle mu and cov (the continuum model and error) with balmerlinedata
    # but save it with continuumdata
    # the former makes sense for fitting
    # the latter is a more logical structure 
    dset_continuum = outf.create_group("continuum")
    dset_continuum.create_dataset("con_wave", data=cwave)
    dset_continuum.create_dataset("con_flux", data=cflux)
    dset_continuum.create_dataset("con_fluxerr", data=cdflux)
    dset_continuum.create_dataset("con_model", data=mu)
    dset_continuum.create_dataset("con_cov", data=cov)
    outf.close()
    

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
    nstep = 100
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


#**************************************************************************************************************

def plot_spectrum_fit(objname, spec,  model, samples, kernel, balmerlinedata, bwi):
    
    font  = FM(size='small')
    font2 = FM(size='x-small')
    font3 = FM(size='large')
    font4 = FM(size='medium')

    wave = spec.wave
    flux = spec.flux
    fluxerr = spec.flux_err

    fig = plt.figure(figsize=(10,8))
    ax_spec  = fig.add_axes([0.075,0.4,0.85,0.55])
    ax_resid = fig.add_axes([0.075, 0.1,0.85, 0.25])
    ax_acorr = fig.add_axes([0.5,0.675,0.4,0.25])

    fig2 = plt.figure(figsize=(10,5))
    ax_lines = fig2.add_subplot(1,2,1)
    ax_corr  = fig2.add_subplot(1,2,2)

    ax_spec.errorbar(wave, flux, fluxerr, color='grey', alpha=0.5, capsize=0, linestyle='-', marker='None')
    if samples is not None:
        expsamples = np.ones(samples.shape)
        expsamples[:, 1] = np.exp(samples[:, 1])
        #crap_a, crap_tau, teff_mcmc, logg_mcmc, av_mcmc  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        #                                                    zip(*np.percentile(samples, [16, 50, 84],
        #                                                    axis=0)))
        teff_mcmc, logg_mcmc, av_mcmc  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                            zip(*np.percentile(samples, [16, 50, 84],
                                                            axis=0)))

        print teff_mcmc
        print logg_mcmc
        print av_mcmc


        (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu, cov) = balmerlinedata

        _, modelflux = model.get_model(teff_mcmc[0], logg_mcmc[0], wave=wave, strict=False)
        _, modelhi   = model.get_model(teff_mcmc[0]+teff_mcmc[2], logg_mcmc[0]+logg_mcmc[2], wave=wave, strict=False)
        _, modello   = model.get_model(teff_mcmc[0]-teff_mcmc[1], logg_mcmc[0]-logg_mcmc[1], wave=wave, strict=False)

        # redden the model
        bluening   = reddening(wave*u.Angstrom, av_mcmc[0], r_v=3.1, model='od94')
        blueninglo = reddening(wave*u.Angstrom, av_mcmc[0]+av_mcmc[2], r_v=3.1, model='od94')
        blueninghi = reddening(wave*u.Angstrom, av_mcmc[0]-av_mcmc[1], r_v=3.1, model='od94')
        modelflux*=bluening
        modelhi  *=blueninghi
        modello  *=blueninglo

        smoothed   = convolve(modelflux, kernel)
        smoothedhi = convolve(modelhi, kernel)
        smoothedlo = convolve(modello, kernel)

        cwave = np.delete(wave, save_ind)
        _, smoothedc,   _  = bspline_continuum((cwave, np.delete(smoothed, save_ind), None), wave)
        _, smoothedchi, _  = bspline_continuum((cwave, np.delete(smoothedhi, save_ind), None), wave)
        _, smoothedclo, _  = bspline_continuum((cwave, np.delete(smoothedlo, save_ind), None), wave)
        smoothed*=mu/smoothedc
        smoothedhi*=mu/smoothedchi
        smoothedlo*=mu/smoothedclo
        
        model_lines    = smoothed[line_ind]
        model_lines_hi = smoothedhi[line_ind]
        model_lines_lo = smoothedlo[line_ind]

        ax_spec.fill(np.concatenate([wave, wave[::-1]]), np.concatenate([smoothedhi, smoothedlo[::-1]]),\
                alpha=0.5, fc='grey', ec='None')
        ax_spec.plot(wave, smoothed, color='red', linestyle='-',marker='None')
        ax_resid.errorbar(wave, flux-smoothed, fluxerr, linestyle='-', marker=None, capsize=0, color='grey', alpha=0.5)

        for l in np.unique(line_number):
            m = (line_number == l)
            try:
                W0, uZE = bwi[l]
            except KeyError, e:
                print l, e
                continue 
            ax_resid.errorbar(line_wave[m], (line_flux[m]-smoothed[line_ind][m]), line_fluxerr[m], linestyle='-',\
                    marker=None, capsize=0, color='black', alpha=0.7)
        
            ax_acorr.acorr(line_flux[m] - smoothed[line_ind][m], label=str(l), usevlines=False, linestyle='-',marker='None')
            ax_lines.fill_between(line_wave[m]-W0,\
                    ((model_lines_lo)/line_cflux)[m]+0.2*line_number[m],\
                    ((model_lines_hi)/line_cflux)[m]+0.2*line_number[m],\
                    facecolor='purple', alpha=0.5, interpolate=True)
            ax_lines.fill_between(line_wave[m]-W0,\
                    ((line_flux-line_fluxerr)/line_cflux)[m]+0.2*line_number[m],\
                    ((line_flux+line_fluxerr)/line_cflux)[m]+0.2*line_number[m],\
                    facecolor='grey', alpha=0.5, interpolate=True)
            ax_lines.plot(line_wave[m]-W0, line_flux[m]/line_cflux[m] + 0.2*line_number[m],color='k',ls='-',lw=2)
            ax_lines.plot(line_wave[m]-W0, model_lines[m]/line_cflux[m] + 0.2*line_number[m], color='r',ls='-',alpha=0.7)

        lags, c, _, _ = ax_corr.xcorr(line_flux/line_cflux, model_lines/line_cflux,\
                    maxlags=21, normed=True, color='k', linestyle='-')
        spacing = np.median(np.diff(wave))
        lagind = c.argmax()
        offset = -lags[lagind]*spacing
        ax_corr.axvline(lags[lagind],color='red',lw=2)
        ax_corr.set_title("Cross-Corr Offset: %.3f"%offset)
        ax_corr.set_xlabel('Lag~(\AA)')
        ax_corr.set_ylabel('Normed Corr Coeff')
        ax_lines.set_title('Lines vs Models')
        ax_lines.set_xlabel('Delta Wavelength~(\AA)')
        ax_lines.set_ylabel('Normalized Line Flux')


        ax_acorr.legend(frameon=False)

    for l in np.unique(line_number):
        m = (line_number == l)
        ax_spec.errorbar(line_wave[m], line_flux[m], line_fluxerr[m],\
                    color='black', capsize=0, linestyle='-', marker='None',alpha=0.7)

    ax_resid.set_xlabel('Wavelength~(\AA)',fontproperties=font3, ha='center')
    ax_spec.set_ylabel('Normalized Flux', fontproperties=font3)
    ax_resid.set_ylabel('Fit Residual Flux', fontproperties=font3)
    return fig, fig2


#**************************************************************************************************************

def plot_continuum_fit(objname, spec, continuumdata, balmerlinedata):

    font  = FM(size='small')
    font2 = FM(size='x-small')
    font3 = FM(size='large')
    font4 = FM(size='medium')

    wave = spec.wave
    flux = spec.flux
    fluxerr = spec.flux_err
    cwave, cflux, cdflux = continuumdata
    (line_wave, line_flux, line_fluxerr, line_number, line_cflux, line_cov, line_ind, save_ind, mu, cov) = balmerlinedata

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax1.errorbar(spec.wave, spec.flux, yerr=spec.flux_err, marker='None',\
                        linestyle='-', color='grey', alpha=0.3, capsize=0)
    ax1.plot(spec.wave, spec.flux, marker='None', linestyle='-', color='black', alpha=0.8)
    ax1.plot(cwave, cflux, marker='.', linestyle='None', color='blue', ms=1)
    ax1.plot(spec.wave, mu, marker='None', linestyle='--', color='red')

    ax2.errorbar(spec.wave, spec.flux/mu, yerr=spec.flux_err/mu, marker='None',\
            linestyle='-', color='grey', alpha=0.3, capsize=0)
    ax2.plot(spec.wave, spec.flux/mu, marker='None', linestyle='-', color='black', alpha=0.8)

    ax2.set_xlabel('Wavelength~(\AA)',fontproperties=font3, ha='center')
    ax1.set_ylabel('Flux', fontproperties=font3)
    ax2.set_ylabel('Norm Flux', fontproperties=font3)
    ax1.set_title(objname)
    plt.tight_layout()
    return fig


#**************************************************************************************************************

def plot_model(objname, spec,  model, samples, kernel, continuumdata, balmerlinedata, bwi, outdir=os.getcwd(), discard=5):

    outfilename = os.path.join(outdir, os.path.basename(objname.replace('.flm','.pdf')))
    with PdfPages(outfilename) as pdf:
        fig = plot_continuum_fit(objname, spec, continuumdata, balmerlinedata)
        pdf.savefig(fig)
        fig, fig2 =  plot_spectrum_fit(objname, spec,  model, samples, kernel, balmerlinedata, bwi)
        pdf.savefig(fig)
        pdf.savefig(fig2)


        #labels = ['Nuisance Amplitude', 'Nuisance Scale', r"$T_\text{eff}$" , r"$log(g)$", r"A$_V$"]
        labels = [r"Teff" , r"log(g)", r"A_V"]
        samples = samples[int(round(discard*samples.shape[0]/100)):]
        fig = corner.corner(samples, bins=41, labels=labels, show_titles=True,quantiles=(0.16,0.84),\
             use_math_text=True)
        pdf.savefig(fig)
        #endwith
    

#**************************************************************************************************************

def make_outdirs(dirname):
    print("Writing to outdir {}".format(dirname))
    if os.path.isdir(dirname):
        return

    try:
        os.makedirs(dirname)
    except OSError, e:
        message = '%s\nCould not create outdir %s for writing.'
        raise OSError(message)
    

#**************************************************************************************************************

def get_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--specfiles', required=True, nargs='+',\
            help="Specify spectra to fit. Can be used multiple times.")
    parser.add_argument('-b', '--balmerlines', nargs='+', type=int, default=range(1,7,1),\
            help="Specifiy Balmer lines to fit [1:7]")
    parser.add_argument('-o', '--outdir', required=False,\
            help="Specify a custom output directory. Default is CWD+objname/ subdir")
    parser.add_argument('--bluelimit', required=False, type=float,\
            help="Specify blue limit of spectrum - trim wavelengths lower")
    parser.add_argument('--redlimit', required=False, type=float,\
            help="Specify red limit of spectrum - trim wavelengths higher")
    parser.add_argument('-s', '--smooth', required=False, type=float, default=None,\
            help="Specify instrumental resolution for smoothing (FWHM, angstroms)")
    parser.add_argument('-r', '--rv', required=False, type=float, default=3.1,\
            help="Specify reddening law R_V")
    parser.add_argument('--reddeningmodel', required=False, default='od94',\
            help="Specify functional form of reddening law" )
    parser.add_argument('--photfiles', required=False, nargs='+')
    parser.add_argument('--nthreads',  required=False, type=int, default=1,\
            help="Specify number of threads to use. (>1 implies multiprocessing)")
    parser.add_argument('--nwalkers',  required=False, type=int, default=200,\
            help="Specify number of walkers to use (0 disables MCMC)")
    parser.add_argument('--nburnin',  required=False, type=int, default=200,\
            help="Specify number of steps for burn-in")
    parser.add_argument('--nprod',  required=False, type=int, default=1000,\
            help="Specify number of steps for production")
    parser.add_argument('--discard',  required=False, type=float, default=5,\
            help="Specify percentage of steps to be discarded")
    parser.add_argument('--redo',  required=False, action="store_true", default=False,\
            help="Clobber existing fits")
    args = parser.parse_args()
    balmer = args.balmerlines
    specfiles = args.specfiles
    photfiles = args.photfiles

    try:
        balmer = np.atleast_1d(balmer).astype('int')
        if np.any((balmer < 1) | (balmer > 6)):
            raise ValueError
    except (TypeError, ValueError), e:
        message = 'Invalid balmer line value - must be in range [1,6]'
        raise ValueError(message)

    if photfiles is not None:
        if len(specfiles) != len(photfiles):
            #TODO: This does not support multiple spectra per object, but it's a mess to include that in the likelihood
            # skip this problem for now until we come up a way of dealing with it
            message = 'If you are specifying photometry for fitting, number of files needs to match number of spectra'
            raise ValueError(message) 

    if args.rv < 2.1 or args.rv > 5.1:
        message = 'That Rv Value is ridiculous'
        raise ValueError(message)

    if args.smooth is not None:
        if args.smooth < 0:
            message = 'That Gaussian Smoothing FWHM value is ridiculous'
            raise ValueError(message)

    reddeninglaws = ('od94', 'ccm89', 'gcc09', 'f99', 'fm07', 'wd01', 'd03')
    if not args.reddeningmodel in reddeninglaws:
        message = 'That reddening law is not known (%s)'%' '.join(reddeninglaws) 
        raise ValueError(message)

    if args.nwalkers < 0:
        message = 'Number of walkers must be greater than zero for MCMC'
        raise ValueError(message)

    if args.nthreads <= 0:
        message = 'Number of threads must be greater than zero'
        raise ValueError(message)
    
    if args.nburnin <= 0:
        message = 'Number of walkers must be greater than zero'
        raise ValueError(message)

    if args.nprod <= 0:
        message = 'Number of walkers must be greater than zero'
        raise ValueError(message)

    if not (0 <= args.discard < 100):
        message = 'Discard must be a percentage (0-100)'
        raise ValueError(message)

    return args

#**************************************************************************************************************

def read_spec(filename):
    """
    Really quick little read spectrum from file routine
    """
    spec = np.recfromtxt(filename, names=True, dtype='float64,float64,float64')
    return spec


#**************************************************************************************************************

def read_phot(filename):
    """
    Read photometry from file - expects to have columns mag_aper magerr_aper and pb 
    Extra columns other than these three are fine
    """
    phot = np.recfromtxt(filename, names=True)
    print rec2txt(phot)
    return phot


#**************************************************************************************************************

def read_spectable():
    """
    Read spectrum resolution from a file to set instrumental smoothing
    """
    spectable = np.recfromtxt('spectable_resolution.dat', names=True)
    return spectable

#**************************************************************************************************************

def main():
    args   = get_options() 
    specfiles = args.specfiles
    photfiles = args.photfiles
    nwalkers  = args.nwalkers
    nburnin   = args.nburnin
    nprod     = args.nprod
    nthreads  = args.nthreads
    outdir    = args.outdir
    discard   = args.discard
    redo      = args.redo

    if outdir is not None:
        make_outdirs(outdir)

    for i in xrange(len(specfiles)):
        if photfiles is not None:
            photfile = photfiles[i]
        else:
            photfile = None
        specfile = specfiles[i]

        if outdir is None:
            dirname = os.path.join(os.getcwd(), os.path.basename(specfile.replace('.flm','')))
            make_outdirs(dirname)
        else:
            dirname = outdir

        spec, linedata, continuumdata, save_ind, balmer, smooth, bwi = pre_process_spectrum(specfile,\
                                args.smooth, args.bluelimit, args.redlimit, args.balmerlines)

        res = fit_model(specfile, spec, linedata, continuumdata, save_ind, balmer,\
                rv=args.rv, smooth=smooth, photfile=photfile,\
                nwalkers=nwalkers, nburnin=nburnin, nprod=nprod, nthreads=nthreads, outdir=dirname, redo=redo)
        model, samples, kernel, balmerlinedata = res
        plot_model(specfile, spec,  model, samples, kernel, continuumdata, balmerlinedata, bwi, outdir=dirname, discard=discard)
    return


#**************************************************************************************************************

if __name__=='__main__':
    main()
    #cProfile.run('main()', 'profile.dat')
    #import pstats
    #p = pstats.Stats('profile.dat')
    #p.sort_stats('cumulative','time').print_stats(20)


