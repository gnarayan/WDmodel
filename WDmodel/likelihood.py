import numpy as np
from celerite.modeling import Model
from scipy.stats import norm
from george import GP, HODLRSolver
from george.kernels import ExpSquaredKernel
from . import io
from .pbmodel import get_model_synmags

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
    This imposes a prior on Rv, and on Av.  The prior on Av is the glos prior.
    The prior on Rv is a Gaussian with mean 3.1 and sigma 0.18 - derived from
    Schlafly et al from PS1 - note that they report 3.31, but they aren't
    really measuring E(B-V) with PS1. Their sigma should be consistent despite
    the different filter set.

    To use it, construct an object from the class with the kwargs dictionary
    and an initial guess, and freeze/thaw parameters as needed. Then write a
    function that wraps get_value() and lnprior() and sample the posterior
    however you like.
    """
    parameter_names = io._PARAMETER_NAMES

    def get_value(self, spec, phot, model, rvmodel, pbs, pixel_scale=1., phot_dispersion=0.):
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

        kernel = (self.sigf**2.)*ExpSquaredKernel(self.tau)
        gp = GP(kernel, mean=0., solver=HODLRSolver)
        try:
            gp.compute(spec.wave, spec.flux_err)
        except ValueError:
            return -np.inf

        return gp.lnlikelihood(res, quiet=True) - (phot_chi/2.)


    def lnprior(self):
        """
        Extends the log_prior() is implemented in celerite.modeling.Model which
        returns -inf for out of bounds and 0 otherwise, by adding in priors on Rv, Av

        We have no priors outside the bounds (i.e. weak top hat) on the other parameters

        TODO: Allow the user to specify a custom filename with a function for the prior
        """
        lp = self.log_prior()
        theta = self.get_parameter_vector()
        if not np.isfinite(lp):
            return -np.inf
            # even if the user passes bounds that are physically unrealistic, we
            # need to keep all the quantities strictly >= 0.
        elif np.any((theta < 0.)):
            return -np.inf
        else:
            # this is from the Schlafly et al analysis from PS1
            out = norm.logpdf(self.rv, 3.1, 0.18)

            # this implements the glos prior on Av
            avtau   = 0.4
            avsdelt = 0.1
            wtexp   = 1.
            wtdelt  = 0.5
            sqrt2pi = np.sqrt(2.*np.pi)
            normpeak = (wtdelt/sqrt2pi)/(2.*avsdelt) + wtexp/avtau
            pav = (wtdelt/sqrt2pi)*np.exp((-self.av**2.)/(2.*avsdelt**2.))/(2.*avsdelt) +\
                    wtexp*np.exp(-self.av/avtau)/avtau
            pav /= normpeak
            out += np.log(pav)
            return out


class WDmodel_Posterior(object):
    """
    Class to compute the posterior, given the data, and model
    The class contains the data
        spec: the recarray spectrum (wave, flux, flux_err)
        phot: the recarray photometry (pb, mag, mag_err)
    and the model
        model: a WDmodel() instance to get the model spectrum in the presence
        of reddening and through some instrument
        rvmodel: The form of the reddening law to be used to redden the spectrum
        pbs: a model of the throughput of the different passbands
        lnlike: a WDmodel_Likelihood instance that can return the log prior and log likelihood

    Call returns the log posterior
    """
    def __init__(self, spec, phot, model, rvmodel, pbs, lnlike, pixel_scale=1., phot_dispersion=0.):
        self.spec    = spec
        self.phot    = phot
        self.model   = model
        self.rvmodel = rvmodel
        self.pbs     = pbs
        self.lnlike  = lnlike
        self.pixscale= pixel_scale
        self.phot_dispersion = phot_dispersion

    def __call__(self, theta):
        self.lnlike.set_parameter_vector(theta)
        out = self.lnlike.lnprior()
        if not np.isfinite(out):
            return -np.inf
        out += self.lnlike.get_value(self.spec, self.phot, self.model, self.rvmodel, self.pbs,\
                pixel_scale=self.pixscale, phot_dispersion=self.phot_dispersion)
        return out

    def lnlike(self, theta):
        self.lnlike.set_parameter_vector(theta)
        out = self.lnlike.get_value(self.spec, self.phot, self.model, self.rvmodel, self.pbs,\
                pixel_scale=self.pixscale, phot_dispersion=self.phot_dispersion)
        return out

    def lnprior(self, theta):
        self.lnlike.set_parameter_vector(theta)
        out = self.lnlike.lnprior()
        return out
