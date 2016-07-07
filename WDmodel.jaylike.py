#!/usr/bin/env python
import sys
import cProfile
import os
import argparse
import numpy as np
import h5py
import scipy.interpolate as spinterp
import scipy.stats as scistat
import scipy.optimize as op
import george
import emcee
import corner
from astropy.convolution import convolve, Gaussian1DKernel
import pysynphot as S
from specutils import Spectrum1D as speccls
from specutils.wcs import specwcs
from specutils.io import write_fits
from specutils.extinction import extinction, reddening
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties as FM
from matplotlib import rc
from matplotlib.mlab import rec2txt
rc('text', usetex=True)
rc('font', family='serif')
rc('ps', usedistiller='xpdf')



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
        D       = [ 100.0  ,  130.0  ,  125.0   ,  75.0   ,  50.0    ,  27.0   ]
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
            message = 'Grid %s not found in grid_file %s. Accepted values are (%s)'%(grid_name, self._grid_file, ','.join(_grids.keys()))
            raise ValueError(message)

        self._grid_name = grid_name
        self._wave  = grid['wave'].value.astype('float64')
        self._ggrid = grid['ggrid'].value
        self._tgrid = grid['tgrid'].value
        _flux  = grid['flux'].value.astype('float64')
        
        # pre-init the interpolation and do it in log-space
        self._model = spinterp.RegularGridInterpolator((np.log10(self._wave), self._ggrid, self._tgrid), np.log10(_flux))        
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
        return 10.**self._model(xi)


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

        
    def get_model(self, teff, logg, wave=None, log=False):
        """
        Returns the model for some teff, logg at wavelengths wave
        If not specified, wavelengths are from 3000-9000A

        Checks inputs for consistency and calls _get_xi(), _get_model()
        If you need the model repeatedly for slightly different parameters, use those functions directly
        """
        if wave is None:
            #wave = np.linspace(3000., 7000., num=4001, endpoint=True)
            wave = self._wave

        wave = np.atleast_1d(wave)
        self._wave_test(wave)

        teff = float(teff)
        logg = float(logg)

        if not ((teff >= self._tgrid.min()) and (teff <= self._tgrid.max())):
            message = 'Temperature out of model range'
            raise ValueError(message)

        if not ((logg >= self._ggrid.min()) and (logg <= self._ggrid.max())):
            message = 'Surface gravity out of model range'
            raise ValueError(message)

        outwave = wave[((wave >= self._wave.min()) & (wave <= self._wave.max()))]

        if len(outwave) > 0:
            xi = self._get_xi(teff, logg, outwave)
            outflux = self._get_model(xi,log=log)
            return outwave, outflux
        else:
            message = 'No valid wavelengths'
            raise ValueError(message)
            


    def _get_line_indices(self, w, f, line):
        """
        Returns the central wavelength, and _indices_ of blue continuum, red continuum, and profile
        """
        _, W0, WID, DW = self._lines[line]
        WA  = W0 - WID
        WB  = W0 + WID
        W1  = [WA-DW, WA+DW]
        W2  = [WB-DW, WB+DW]
        Z1  = np.where((w > W1[0]) & (w <= W1[1]))
        Z2  = np.where((w > W2[0]) & (w <= W2[1]))
        ZE  = np.where((w > W1[1]) & (w <= W2[0]))
        return W0, Z1, Z2, ZE


    def _get_line_continuum_profile(self, w, f, Z1, Z2, ZE):
        """
        Returns the wavelength and flux of the continuum, and wavelength and flux of the line core
        using the indices Z1, Z2, ZE, which can be determined by _get_line_indices
        """
        X   = np.hstack((w[Z1], w[Z2]))
        Y   = np.hstack((f[Z1], f[Z2]))
        we  = w[ZE]
        fe  = f[ZE]
        return X, Y, we, fe


    def extract_spectral_line(self, w, f, line):
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

        w = np.atleast_1d(w)
        f = np.atleast_1d(f)
        self._wave_test(w)
        if w.shape != f.shape:
            message = 'Shape mismatch between wavelength and flux arrays'
            raise ValueError(message)

        W0, Z1, Z2, ZE = self._get_line_indices(w, f, line)

        if len(w[Z1])==0 or len(w[Z2]) == 0 or len(w[ZE]) == 0:
            message = 'Spectrum does not adequately cover line %s'%line
            raise ValueError(message)

        X, Y, we, fe = self._get_line_continuum_profile(w, f, Z1, Z2, ZE)
        return self._normalize_line(X, Y, we, fe, W0)


    def _extract_spectral_line(self, w, f, line):
        """
        Same as extract_spectral_line() except no testing
        Used internally to extract the spectral line for the model 
        """
        W0, Z1, Z2, ZE = self._get_line_indices(w, f , line)
        X, Y, we, fe = self._get_line_continuum_profile(w, f, Z1, Z2, ZE)
        return self._normalize_line(X, Y, we, fe, W0)


    def _normalize_line(self, X, Y, we, fe, W0, log=False): 
        """
        Accepts the continuum wavelength and flux, X, Y
        and the line profile wavelength and flux we, fe
        and the line centroid, W0
        Fits a straight line to the continuum flux, normalizes it
        and returns the wavelength shift about the line centroid, the normalized flux
        and the normalization factor
        """
        if log:
            xl = np.log10(X)
            yl = np.log10(Y)
            out = np.polyfit(xl,yl,1)
            m, c = out[0], out[1]
            k = 10.**c
            YF  = k*(we**m)
            YC  = k*(X**m)
        else:
            out = np.polyfit(X,Y,1)
            m, c = out[0], out[1]
            YF  =m*we + c 
            YC  =m*X + c
        #log(y) = m*log(X) + c
        #y = 10**(m*log(x) + c)
        #y = 10**log(x)m *10**c 
        #y = (x**m)*k
        wn  = we - W0
        wc  = X - W0
        fn  = fe/YF
        fc  = Y/YC
        return wn, fn, YF, wc, fc, YC

                
    # these are implemented for compatibility with python's pickle
    # which in turn is required to make the code work with multiprocessing
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    __call__ = get_model
        

#**************************************************************************************************************


def lnprior(theta):
    teff, logg = theta
    if 20000. < teff < 75000. and 7.0 < logg < 9.5:
        return 0.
    return -np.inf

def lnprob(theta, wave, model, data, kernel, balmer):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, wave, model, data, kernel, balmer)


def lnlike(theta, wave, model, data, kernel, balmer):
    teff, logg = theta
    xi = model._get_xi(teff, logg, wave)
    mod = model._get_model(xi)
    smoothed = convolve(mod, kernel)
    res = []
    sig = []
    for x in balmer:
        (W0, Z1, Z2, ZE) = balmer[x]
        X, Y, we, fe = model._get_line_continuum_profile(wave, smoothed, Z1, Z2, ZE)
        smoothedwn, smoothedfn, norm, swc, wfc, norm2 = model._normalize_line(X, Y, we, fe, W0)
        datawave, datafn, datafnerr, _, _, _  = data[x]
        res += (datafn - smoothedfn).tolist()
        sig += datafnerr.tolist()
    res = np.atleast_1d(res)
    sig = np.atleast_1d(sig)
    chisqr = np.sum((res**2.)/(sig**2.))
    return -0.5*chisqr 

def nll(*args):
    return -lnlike(*args)


#**************************************************************************************************************

def fit_model(objname, spec, balmer=None, av=0., rv=3.1, rvmodel='od94', smooth=4., photfile=None):

    wave = spec.wave
    flux = spec.flux
    fluxvar = spec.fluxvar
    fluxerr = (fluxvar/3.)**0.5 #HACK HACK HACK 
    #Tom stacked three spectra but the variance he sent was the total, not the variance of the mean
    

    if photfile is not None:
        phot = read_phot(photfile)
        # set the likelihood functions here 
    else:
        phot = None
        # set the likelihood functions here 


    if balmer is None:
        balmer = np.arange(1, 7)
    else:
        balmer = balmer

    gsig     = smooth*(0.5/(np.log(2.)**0.5))
    kernel   = Gaussian1DKernel(gsig)
    model = WDmodel()
    data = {}
     
    if phot is None:
        newwave  = wave*u.Angstrom
        # yes the reddening function returns a vector than when multiplied by the flux, dereddens the object
        # because... reasons...
        bluening = reddening(newwave, av, r_v=rv, model=rvmodel)
        newflux  = flux*bluening
        newfluxerr = fluxerr*bluening
    else:
        newflux  = flux 
        newfluxerr = fluxerr

    balmerwaveindex = {}
    for x in balmer:
        W0, Z1, Z2, ZE = model._get_line_indices(wave, newflux , x)
        X, Y, we, fe = model._get_line_continuum_profile(wave, newflux, Z1, Z2, ZE)
        # we should do a test here to check that the line is actually significant
        wn, fn, norm, wc, fc, norm2  = model._normalize_line(X, Y, we, fe, W0)
        fnerr = fluxerr[ZE]/norm
        fcerr = np.hstack((fluxerr[Z1], fluxerr[Z2]))/norm2
        data[x] = wn, fn, fnerr, wc, fc, fcerr
        # cache the indices so we don't keep recomputing them - the wave array doesn't change
        balmerwaveindex[x] = (W0, Z1, Z2, ZE)

    # do a quick fit with minimize to get a decent starting guess
    result = op.minimize(nll, [40000., 7.5], args=(wave, model, data, kernel, balmerwaveindex), bounds=[(17000,80000),(7., 9.49999)])

    # setup the sampler
    ndim, nwalkers = 2, 50
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wave, model, data, kernel, balmerwaveindex)) 

    # do a short burn-in
    pos, prob, state = sampler.run_mcmc(pos, 100)
    sampler.reset()

    # production
    sampler.run_mcmc(pos, 1000)   

    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    # make a corner plot
    samples = sampler.chain[:,:, :].reshape((-1, ndim))
    plot_model(objname, spec, data, model, samples, kernel, balmerwaveindex)

#**************************************************************************************************************

def plot_model(objname, spec, data, model, samples, kernel, balmer):

    font  = FM(size='small')
    font2 = FM(size='x-small')
    font3 = FM(size='large')
    font4 = FM(size='medium')

    wave = spec.wave
    flux = spec.flux

    expsamples = np.ones(samples.shape)
    expsamples[:, 1] = np.exp(samples[:, 1])
    teff_mcmc, logg_mcmc  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

    print teff_mcmc, logg_mcmc

    outfilename = objname.replace('.flm','.pdf')
    with PdfPages(outfilename) as pdf:
        fig = plt.figure(figsize=(8,8))
        ax  = fig.add_subplot(1,1,1)

        _, modelflux = model.get_model(teff_mcmc[0], logg_mcmc[0], wave=wave)
        _, modelhi   = model.get_model(teff_mcmc[0]+teff_mcmc[2], logg_mcmc[0]+logg_mcmc[2], wave=wave)
        _, modellow  = model.get_model(teff_mcmc[0]-teff_mcmc[1], logg_mcmc[0]-logg_mcmc[1], wave=wave)
        smoothed   = convolve(modelflux, kernel)
        smoothedhi = convolve(modelhi, kernel)
        smoothedlo = convolve(modellow, kernel)


        for line in data:
            wn, fn, fnerr, wc, fc, fcerr = data[line]

            offset = 0.2*line-0.2
            (W0, Z1, Z2, ZE) = balmer[line]
            X, Y, we, fe = model._get_line_continuum_profile(wave, smoothed, Z1, Z2, ZE)
            smoothedwn, smoothedfn, norm, swc, sfc, norm2  = model._normalize_line(X, Y, we, fe, W0)

            X, Y, we, fe = model._get_line_continuum_profile(wave, smoothedhi, Z1, Z2, ZE)
            smoothedwnhi, smoothedfnhi, norm, swchi, sfchi, norm2hi  = model._normalize_line(X, Y, we, fe, W0)

            X, Y, we, fe = model._get_line_continuum_profile(wave, smoothedlo, Z1, Z2, ZE)
            smoothedwnlo, smoothedfnlo, norm, swclo, sfclo, norm2lo  = model._normalize_line(X, Y, we, fe, W0)

            ax.fill(np.concatenate([smoothedwnhi, smoothedwnlo[::-1]]), np.concatenate([smoothedfnhi, smoothedfnlo[::-1]])+offset,\
                    alpha=0.5, fc='grey', ec='None')
            ax.errorbar(wn, fn+offset, fnerr, capsize=0, linestyle='-', lw=0.5, color='k', marker='None')
            bluewing = (wc <= wn.min())
            redwing  = (wc >= wn.min())
            ax.errorbar(wc[bluewing], fc[bluewing]+offset, fcerr[bluewing], capsize=0, linestyle='-', lw=0.5, color='k', marker='None')
            ax.errorbar(wc[redwing], fc[redwing]+offset, fcerr[redwing], capsize=0, linestyle='-', lw=0.5, color='k', marker='None')
            ax.axhline(1.+offset,color='grey',linestyle='-.', alpha=0.3, xmin=wc.min(), xmax=wc.max())
            ax.plot(smoothedwn, smoothedfn+offset, 'r-', alpha=0.75, marker='None')

            linename = model._lines[line][0]
            ax.annotate(r"H$_"+"\\"+linename+"$", xy=(wn[-1], offset+fn[-1]), xycoords='data', xytext= (8.,0.),\
                     textcoords="offset points", fontproperties=font3,  ha='left', va="bottom")
            indzero = np.abs(wn).argmin()
            ax.annotate(str(W0)+'\AA', xy = (wn[indzero], offset+fn[indzero]),  xycoords='data',\
                  xytext= (0.,-8.), textcoords="offset points", fontproperties=font4,  ha='center', va="top")

        ax.set_xlabel('$\Delta$Wavelength~(\AA)',fontproperties=font3, ha='center')
        ax.set_ylabel('Normalized Flux', fontproperties=font3)
        pdf.savefig(fig)
        fig = corner.corner(samples, bins=41, labels=["$T_eff$", r"$log(g)$"], show_titles=True,quantiles=(0.16,0.84),\
             use_math_text=True)
        pdf.savefig(fig)
    #endwith
    


#**************************************************************************************************************

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specfiles', required=True, nargs='+')
    parser.add_argument('--photfiles', required=False, nargs='+')
    parser.add_argument('-a', '--av', required=True, type=float, default=0.)
    parser.add_argument('-r', '--rv', required=False, type=float, default=3.1)
    parser.add_argument('--reddeningmodel', required=False, default='od94')
    parser.add_argument('-b', '--balmerlines', nargs='+', type=int, default=range(1,7,1))
    parser.add_argument('-s', '--smooth', required=False, type=float, default=4.)
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

    if args.av < 0 or args.av > 5.:
        message = 'That Av value is ridiculous'
        raise ValueError(message)

    if args.rv < 2.1 or args.rv > 5.1:
        message = 'That Rv Value is ridiculous'
        raise ValueError(message)

    if args.smooth < 0:
        message = 'That Gaussian Smoothing FWHM value is ridiculous'
        raise ValueError(message)

    reddeninglaws = ('od94', 'ccm89', 'gcc09', 'f99', 'fm07', 'wd01', 'd03')
    if not args.reddeningmodel in reddeninglaws:
        message = 'That reddening law is not known (%s)'%' '.join(reddeninglaws) 
        raise ValueError(message)

    return args


#**************************************************************************************************************

def read_spec(filename):
    """
    Really quick little read spectrum from file routine
    """
    spec = np.recfromtxt(filename, names='wave,flux,fluxvar', dtype='float64,float64,float64')
    WDmodel._wave_test(spec.wave)
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
def main():
#    test()
    args   = get_options() 
    balmer = np.atleast_1d(args.balmerlines).astype('int')
    specfiles = args.specfiles
    photfiles = args.photfiles
    
    for i in xrange(len(specfiles)):
        if photfiles is not None:
            photfile = photfiles[i]
        else:
            photfile = None
        specfile = specfiles[i]
        spec = read_spec(specfile)
    
        fit_model(specfile, spec, balmer, av=args.av, rv=args.rv, smooth=args.smooth, photfile=photfile)    

#**************************************************************************************************************

def test(grid_name='default'):
    model = WDmodel(grid_name=grid_name)
    fig   = plt.figure(figsize=(10,5))
    ax1   = fig.add_subplot(1,1,1)
    t = np.arange(20000., 85000, 5000.)
    l = np.arange(7, 9., 0.5)
    for logg in l:
        for teff in t:
            wave, flux  = model(teff, logg, log=False)
            ax1.plot(wave, flux, 'k-')
    for x in model._lines.keys():
        line, W0, WID, DW = model._lines[x]
        WA  = W0 - WID
        WB  = W0 + WID
        W1  = [WA-DW, WA+DW]
        W2  = [WB-DW, WB+DW]
        ax1.axvline(W1[0],ls='-.', color='blue')
        ax1.axvline(W1[1],ls='--', color='blue')
        ax1.axvline(W2[0],ls='--', color='red')
        ax1.axvline(W2[1],ls='-.', color='red')
    #ax1.set_xlim((3800., 7100.))
    #ax1.set_ylim((3*1E6, 4.*1E8))
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.ion()
    plt.tight_layout()


#**************************************************************************************************************



if __name__=='__main__':
    cProfile.run('main()', 'profile.dat')
    import pstats
    p = pstats.Stats('profile.dat')
    p.sort_stats('cumulative','time').print_stats(20)


