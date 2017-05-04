import numpy as np
from celerite.modeling import Model
from scipy.stats import norm, halfcauchy
from . import io
from .pbmodel import get_model_synmags


def setup_likelihood(params):
    """
    Setup the WDmodel_Likelihood function based on the parameter dictionary, params
    Accepts:
        params: dict of parameters with keywords value, fixed, bounds, scale for each
    Returns
        WDmodel_Likelihood instance
    """
    # parse the params to create a dictionary and init the
    # WDmodel.likelihood.WDmodel_Likelihood instance
    setup_args = {}
    bounds     = []
    fixed      = {}
    for param in io._PARAMETER_NAMES:
        setup_args[param] = params[param]['value']
        bounds.append(params[param]['bounds'])
        fixed[param] = params[param]['fixed']

    setup_args['bounds'] = bounds

    # configure the likelihood function
    lnlike = WDmodel_Likelihood(**setup_args)

    # freeze any parameters that we want fixed
    for param, val in fixed.items():
        if val:
            print "Freezing {}".format(param)
            lnlike.freeze_parameter(param)
    return lnlike


class WDmodel_Likelihood(Model):
    """
    Needs a dictionary structure with parameter names and optionally a keyword
    "bounds" to set the bounds on the parameters

    Subclasses celerite.modeling.Model to implement a likelihood model for the
    WD atmosphere This allows parameters to be frozen/thawed dynamically based
    on cmdline args/a param file

    Implements the likelihood by constructing the model at the parameter
    values, and modeling the residuals with a Rational Quadratic kernel

    Implements a lnprior function, in addition to Model's log_prior function

    To use it, construct an object from the class with the kwargs dictionary
    and an initial guess, and freeze/thaw parameters as needed. Then write a
    function that wraps get_value() and lnprior() and sample the posterior
    however you like.
    """
    parameter_names = io._PARAMETER_NAMES

    def get_value(self, spec, phot, model, rvmodel, covmodel, pbs, pixel_scale=1., phot_dispersion=0.):
        """
        Returns the log likelihood of the data given the model
        """
        if phot is None:
            phot_chi = 0.
            mod = model._get_obs_model(self.teff, self.logg, self.av, self.fwhm,\
                    spec.wave, rv=self.rv, rvmodel=rvmodel, pixel_scale=pixel_scale)
        else:
            mod, full = model._get_full_obs_model(self.teff, self.logg, self.av, self.fwhm,\
                    spec.wave, rv=self.rv, rvmodel=rvmodel, pixel_scale=pixel_scale)
            mod_mags = get_model_synmags(full, pbs, mu=self.mu)
            phot_res = phot.mag - mod_mags.mag
            phot_chi = np.sum(phot_res**2./((phot.mag_err**2.)+(phot_dispersion**2.)))

        mod *= (1./(4.*np.pi*(self.dl)**2.))
        res = spec.flux - mod
        return covmodel.lnlikelihood(spec.wave, res, spec.flux_err, self.fsig, self.tau, self.fw) - (phot_chi/2.)


class WDmodel_Posterior(object):
    """
    Class to compute the posterior, given the data, and model
    The class contains the data
        spec: the recarray spectrum (wave, flux, flux_err)
        phot: the recarray photometry (pb, mag, mag_err)
    and the model
        model: a WDmodel instance to get the model spectrum in the presence
        of reddening and through some instrument
        covmodel: a WDmodel_CovModel instance
        rvmodel: The form of the reddening law to be used to redden the spectrum
        pbs: a model of the throughput of the different passbands
        lnlike: a WDmodel_Likelihood instance that can return the log likelihood

    Implements an lnprior function which imposes weakly informative priors on
    the parameters - broad guassians on most parameters except:
        - the prior on Av is the glos prior.
        - the prior on Rv is a Gaussian with mean 3.1 and sigma 0.18 - derived from
            Schlafly et al from PS1 - note that they report 3.31, but they aren't
            really measuring E(B-V) with PS1. Their sigma should be consistent despite
            the different filter set.
        - the prior on fsig, fw is half-Cauchy since we don't want it to be less than zero
            there's also a cutoff to prevent it from getting too close to zero,
            where the covariance matrix becomes non-inertvible with the
            approximate solver
            (TODO: check if we still need this with fw)
        - there is no prior on tau (i.e. a flat prior). TODO.
        - fwhm has a cut on value below which the spectrum isn't being convolved anymore
            if the fwhm is that low, then we would need a higher resolution grid...

    Note that the likelihood function has a logprior() function, inherited from
    celerite modeling, which imposes the boundscheck, so technically all priors
    are the produce of a tophat and the prior described above. This is not an
    issue for any parameter of any object, since even the bound on Av is set
    negative.

    Call returns the sum of log likelihood and log prior - the log posterior
    """
    def __init__(self, spec, phot, model, covmodel, rvmodel, pbs, lnlike, pixel_scale=1., phot_dispersion=0.):
        self.spec      = spec
        self.wavescale = spec.wave.ptp()
        self.phot      = phot
        self.model     = model
        self.covmodel  = covmodel
        self.rvmodel   = rvmodel
        self.pbs       = pbs
        self._lnlike   = lnlike
        self.pixscale  = pixel_scale
        self.phot_dispersion = phot_dispersion
        init_p0 = lnlike.get_parameter_dict(include_frozen=True)
        self.p0 = init_p0


    def __call__(self, theta, prior=False, likelihood=False):
        """
        Returns the posterior of the parameters theta, given the data
        Accepts two optional key word arguments
            prior: Return the prior instead of the posterior
            likelihood: Return the likelihood instead of the posterior
        This allows a quick assesment with a flat prior (i.e. no prior) if we want an MLE
        """
        self._lnlike.set_parameter_vector(theta)
        out = self._lnprior()
        if not np.isfinite(out):
            return -np.inf
        if prior:
            return out

        loglike = self._lnlike.get_value(self.spec, self.phot, self.model, self.rvmodel, self.covmodel, self.pbs,\
                pixel_scale=self.pixscale, phot_dispersion=self.phot_dispersion)
        if likelihood:
            return loglike

        out += loglike
        return out


    def lnlike(self, theta):
        """
        Returns the likelihood of the model given the data Note that this
        cannot be used when fitting multi-threaded because instancemethods
        cannot be pickled
        """
        self._lnlike.set_parameter_vector(theta)
        out = self._lnlike.get_value(self.spec, self.phot, self.model, self.rvmodel, self.covmodel, self.pbs,\
                pixel_scale=self.pixscale, phot_dispersion=self.phot_dispersion)
        return out


    def _lnprior(self):
        lp = self._lnlike.log_prior()
        # check if out of bounds
        if not np.isfinite(lp):
            return -np.inf
        else:
            out = 0.

            # put some weak priors on intrinsic WD  parameters

            # normal on teff
            teff  = self._lnlike.get_parameter('teff')
            teff0 = self.p0['teff']
            out += norm.logpdf(teff, teff0, 10000.)

            # normal on logg
            logg  = self._lnlike.get_parameter('logg')
            logg0 = self.p0['logg']
            out += norm.logpdf(logg, logg0, 1.)

            # this implements the glos prior on Av
            av = self._lnlike.get_parameter('av')
            avtau   = 0.4
            avsdelt = 0.1
            wtexp   = 1.
            wtdelt  = 0.5
            sqrt2pi = np.sqrt(2.*np.pi)
            normpeak = (wtdelt/sqrt2pi)/(2.*avsdelt) + wtexp/avtau
            pav = (wtdelt/sqrt2pi)*np.exp((-av**2.)/(2.*avsdelt**2.))/(2.*avsdelt) +\
                    wtexp*np.exp(-av/avtau)/avtau
            pav /= normpeak
            out += np.log(pav)

            # this is from the Schlafly et al analysis from PS1
            rv  = self._lnlike.get_parameter('rv')
            out += norm.logpdf(rv, 3.1, 0.18)

            # normal on dl
            dl  = self._lnlike.get_parameter('dl')
            dl0 = self.p0['dl']
            out += norm.logpdf(dl, dl0, 1000.)

            fwhm  = self._lnlike.get_parameter('fwhm')
            # The FWHM is converted into a gaussian sigma for convolution.
            # That convolution kernel is truncated at 4 standard deviations by default.
            # If twice that scale is less than 1 pixel, then we're not actually modifying the data.
            # This is what sets the hard lower bound on the data, not fwhm=0.
            gsig  = (fwhm/self.model._fwhm_to_sigma)*self.pixscale
            if 8.*gsig < 1.:
                return -np.inf
            # normal on fwhm
            fwhm0 = self.p0['fwhm']
            out += norm.logpdf(fwhm, fwhm0, 8.)

            # half-Cauchy on the kernel amplitudes (both stationary and white)
            fsig = self._lnlike.get_parameter('fsig')
            out += halfcauchy.logpdf(fsig, loc=0, scale=3)

            fw = self._lnlike.get_parameter('fw')
            out += halfcauchy.logpdf(fw, loc=0, scale=3)

            # normal on mu
            mu  = self._lnlike.get_parameter('mu')
            mu0 = self.p0['mu']
            out += norm.logpdf(mu, mu0, 10.)
            return out


    def lnprior(self, theta):
        """
        Extends the log_prior() is implemented in celerite.modeling.Model which
        returns -inf for out of bounds and 0 otherwise, by adding in priors on
        the other parameters

        TODO: Allow the user to specify a custom filename with a function for
        the prior
        """
        self._lnlike.set_parameter_vector(theta)
        return self._lnprior()
