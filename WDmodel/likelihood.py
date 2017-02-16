import numpy as np
from celerite.modeling import Model
from scipy.stats import norm
from george import GP, HODLRSolver
from george.kernels import RationalQuadraticKernel, WhiteKernel

class WDmodel_Likelihood(Model):
    """
    Needs a dictionary structure with parameter names and optionally a keyword
    "bounds" to set the bounds on the parameters

    Subclasses celerite.modeling.Model to implement a likelihood model for the
    WD atmoshere This allows parameters to be frozen/thawed dynamically based
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
    and an intial guess, and freeze/thaw parameters as needed. Then write a
    function that wraps get_value() and lnprior() and sample the posterior
    however you like. 
    """
    parameter_names = ("teff", "logg", "av", "rv", "c", "fwhm", "sigf", "alpha", "tau")

    def get_value(self, spec, model, rvmodel):
        mod = model._get_obs_model(self.teff, self.logg, self.av, self.fwhm, spec.wave, rv=self.rv, rvmodel=rvmodel)
        mod *= self.c
        res = spec.flux - mod
        kernel = (self.sigf**2.)*RationalQuadraticKernel(self.alpha, self.tau)
        gp = GP(kernel, mean=0.)
        try:
            gp.compute(spec.wave, spec.flux_err)
        except ValueError:
            return -np.inf
        #TODO - add the photometry here
        return gp.lnlikelihood(res, quiet=True) 

    def lnprior(self):

        # log prior is implemented in celerite.modeling.Model
        # it returns -inf for out of bounds and 0 otherwise
        lp = self.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        else:
            # this is from the Schalfly et al analysis from PS1
            out = norm.logpdf(self.rv, 3.1, 0.18)

            # this implements the glos prior on Av
            avtau = 0.4
            avsdelt = 0.1
            wtexp = 1.
            wtdelt = 0.5
            sqrt2pi = np.sqrt(2.*np.pi)
            normpeak = (wtdelt/sqrt2pi)/(2.*avsdelt) + wtexp/avtau
            pav = (wtdelt/sqrt2pi)*np.exp((-self.av**2.)/(2.*avsdelt**2.))/(2.*avsdelt) +\
                    wtexp*np.exp(-self.av/avtau)/avtau
            pav /= normpeak
            out += np.log(pav)
            return out


