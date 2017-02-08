import warnings
warnings.simplefilter('once')
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.stats as scistat
import scipy.signal as scisig
import scipy.interpolate as scinterp
import george
from .WDmodel import WDmodel


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

    maskblue = (cwave <= 5500.)
    maskred  = (cwave  > 5500.)
    outblue  = ( wave <= 5500.)
    outred   = ( wave  > 5500.)
    
    mublue = []
    mured  = []
    if len(cwave[maskblue]) > 0 and len(wave[outblue]) > 0:
        coeffblue = poly.polyfit(cwave[maskblue], cflux[maskblue], deg=9, w=w[maskblue])
        mublue = poly.polyval(cwave[maskblue], coeffblue)

    if len(cwave[maskred]) > 0 and len(wave[outred]) > 0:
        coeffred = poly.polyfit(cwave[maskred], cflux[maskred], deg=5, w=w[maskred])
        mured = poly.polyval(cwave[maskred], coeffred)

    mu = np.hstack((mublue,mured))
    coeff = poly.polyfit(cwave, mu, deg=9, w=w)
    out = poly.polyval(wave, coeff)
    continuummodel = np.rec.fromarrays([wave, out], names='wave,flux')
    return continuummodel


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


#**************************************************************************************************************

def blotch_spectrum(spec, linedata):
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



def pre_process_spectrum(spec, bluelimit, redlimit, balmerlines, blotch=False):
    """
    builds the continuum model
    extracts the lines
    """

    # Test that the array is monotonic 
    WDmodel._wave_test(spec.wave)
    model = WDmodel()

    balmer = np.atleast_1d(balmerlines).astype('int')
    balmer.sort()

    # get a coarse mask of line and continuum
    linedata, continuumdata  = orig_cut_lines(spec, model)
    if blotch:
        spec = blotch_spectrum(spec, linedata)

    # get a coarse estimate of the full continuum
    # this isn't used for anything other than cosmetics
    bsw, bsf = polyfit_continuum(continuumdata, spec.wave)
    
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
    bsw  = bsw[usemask]
    bsf  = bsf[usemask]


#    lineparams = [(x, model._get_line_indices(spec.wave, x)) for x in balmer]
#    lineno, lineparams = zip(*lineparams)
#    linecentroids, _  = zip(*lineparams)
#    lineno = np.array(lineno)
#    linecentroids = np.array(linecentroids)
#    linelimits = {}
#    for x, W0 in zip(lineno, linecentroids):
#        delta_lambda = (spec.wave[mask] - W0)
#        blueind  = (delta_lambda < 0)
#        redind   = (delta_lambda > 0)
#        if len(delta_lambda[blueind]) == 0 or len(delta_lambda[redind]) == 0:
#            # spectrum not blue/red enough for this line
#            continue 
#        bluelim  = delta_lambda[blueind].argmax()
#        redlim   = delta_lambda[redind].argmin()
#        bluewave = spec.wave[mask][blueind][bluelim]
#        redwave  = spec.wave[mask][redind][redlim]
#        
#        lineind = ((spec.wave >= bluewave) & (spec.wave <= redwave))
#        signalind = (pdiff[lineind]*100. > 1.)
#        if len(pdiff[lineind][signalind]) <= 5:
#            print "Not enough signal ",W0
#            continue
#
#        linelimits[x] = (W0, bluewave, redwave)
#    balmerwaveindex = {}
#    line_wave     = np.array([], dtype='float64', ndmin=1)
#    line_flux     = np.array([], dtype='float64', ndmin=1)
#    line_fluxerr  = np.array([], dtype='float64', ndmin=1)
#    line_number   = np.array([], dtype='int', ndmin=1)
#    line_ind      = np.array([], dtype='int', ndmin=1)
#    save_ind      = np.array([], dtype='int', ndmin=1)
#    for x in range(1,7):
#        if x in linelimits:
#            (W0, bluewave, redwave) = linelimits[x]
#            WO, ZE = model._get_indices_in_range(spec.wave,  bluewave, redwave, W0=W0)
#            # ensure that there are no repeated wavelengths that count towards more than one line
#            # this prevents double counting 
#            uZE = np.setdiff1d(ZE[0], save_ind)
#            save_ind     = np.hstack((save_ind, uZE))
#            x_wave, x_flux, x_fluxerr = model._extract_from_indices(spec.wave, spec.flux, (uZE), df=spec.flux_err)
#        else:
#            # if this line wasn't in linelimits - i.e. didn't have enough S/N
#            # still extract the pre-defined region around it, since we do not
#            # want it to be used for continuum
#            W0, ZE = model._get_line_indices(spec.wave, x)
#            uZE = np.setdiff1d(ZE[0], save_ind)
#            save_ind     = np.hstack((save_ind, uZE))
#            continue
#
#        balmerwaveindex[x] = W0, uZE
#        line_wave    = np.hstack((line_wave, x_wave))
#        line_flux    = np.hstack((line_flux, x_flux))
#        line_fluxerr = np.hstack((line_fluxerr, x_fluxerr))
#        line_number  = np.hstack((line_number, np.repeat(x, len(x_wave))))
#        line_ind     = np.hstack((line_ind, uZE))
#    # continuum data is just the spectrum with the Balmer lines removed
#    continuumdata  = (np.delete(spec.wave, save_ind), np.delete(spec.flux, save_ind), np.delete(spec.flux_err, save_ind))
#
#    bsw, bsf, _     = polyfit_continuum(continuumdata, spec.wave)
#    gpw, gpf, gpcov = gp_continuum(continuumdata, bsw, bsf, spec.wave)
#
#    linedata = (line_wave, line_flux, line_fluxerr, line_number, line_ind)
#    balmer = sorted(linelimits.keys())
    return spec, bsw, bsf, linedata, continuumdata
