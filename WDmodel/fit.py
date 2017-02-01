import warnings
warnings.simplefilter('once')
import numpy as np
import scipy.stats as scistat
import scipy.signal as scisig
import scipy.interpolate as scinterp
import george
from .WDmodel import WDmodel


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



def bspline_continuum(continuumdata, wave):
    cwave, cflux, cdflux = continuumdata
    cv = zip(cwave, cflux)
    points = bspline(cv, n=len(wave))
    cbspline = np.rec.fromrecords(points, names='wave,flux')
    mu = np.interp(wave, cbspline.wave, cbspline.flux)
    return wave, mu, None



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

def blotch_spectrum(spec, linedata, saveind):
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
    spec.flux[saveind] = linedata[1]
    spec.flux[0:window] = blueend
    spec.flux[-window:] = redend
    return spec



def pre_process_spectrum(spec, smooth, bluelimit, redlimit, balmerlines, blotch=False):
    """
    builds the continuum model
    extracts the lines
    """

    # Test that the array is monotonic 
    WDmodel._wave_test(spec.wave)
    model = WDmodel()

    balmer = np.atleast_1d(balmerlines).astype('int')
    balmer.sort()

    linedata, continuumdata, saveind = orig_cut_lines(spec, model)
    if blotch:
        spec = blotch_spectrum(spec, linedata, saveind)

    # extract the continuum 
    continuumdata  = (np.delete(spec.wave, saveind), np.delete(spec.flux, saveind), np.delete(spec.flux_err, saveind))

    # TODO make this a parameter 
    window1 = 7
    # create a smooth version of the spectrum to refine line detection
    med_filt1 = scisig.wiener(spec.flux, mysize=window1)

    # get a coarse estimate of the full continuum
    # fit a bspline to the continuum data
    # this is going to be oscillatory because of all the white noise
    # but it's good enough to serve as the mean function over the lines
    # then use a Gaussian Process to constrain the continuum
    bsw, bsf, _     = bspline_continuum(continuumdata, spec.wave)
    gpw, gpf, gpcov = gp_continuum(continuumdata, bsw, bsf, spec.wave)
    
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
    gpw  = gpw[usemask]
    gpf  = gpf[usemask]
    bsw  = bsw[usemask]
    bsf  = bsf[usemask]
    med_filt1 = med_filt1[usemask]

    # compute the difference between the GP and the smooth spectrum
    pdiff = np.abs(gpf - med_filt1)/gpf
    # select where difference is within 1% of continuum
    # TODO make this a parameter 
    mask = (pdiff*100 <= 1.)

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
