import numpy as np
from celerite.modeling import Model
from scipy.stats import norm
from george import GP
from george.kernels import ExpSquaredKernel

# Declare this tuple to init the likelihood model, and to preserve order of parameters
_PARAMETER_NAMES = ("teff", "logg", "av", "rv", "dl", "fwhm", "sigf", "tau")

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
    parameter_names = _PARAMETER_NAMES

    def get_value(self, spec, model, rvmodel, n):
        """
        Returns the log likelihood of the data given the model
        """
        mod = model._get_obs_model(self.teff, self.logg, self.av, self.fwhm, spec.wave, rv=self.rv, rvmodel=rvmodel)
        mod *= (1./(4.*np.pi*(self.dl)**2.))
        res = spec.flux - mod
        kernel = (self.sigf**2.)*ExpSquaredKernel(self.tau)
        gp = GP(kernel, mean=0.)
        try:
            gp.compute(spec.wave[::n], spec.flux_err[::n])
        except ValueError:
            return -np.inf
        #TODO - add the photometry here
        return gp.lnlikelihood(res[::n], quiet=True) 


    def lnprior(self):
        """
        Extends the log_prior() is implemented in celerite.modeling.Model which
        returns -inf for out of bounds and 0 otherwise, by adding in priors on Rv, Av

        We have no priors outside the bounds (i.e. weak top hat) on the other parameters

        TODO: Allow the user to specify a custom filename with a function for the prior
        """
        lp = self.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        else:
            # this is from the Schalfly et al analysis from PS1
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


def loglikelihood(theta, spec, model, rvmodel , lnprob, everyn):
    lnprob.set_parameter_vector(theta)
    out = lnprob.lnprior()
    if not np.isfinite(out):
        return -np.inf
    out += lnprob.get_value(spec, model, rvmodel, everyn)
    return out
