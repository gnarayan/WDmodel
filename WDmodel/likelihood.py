import numpy as np
from celerite.modeling import Model
from george import GP, HODLRSolver
from george.kernels import RationalQuadraticKernel

class WDmodel_Likelihood(Model):
    parameter_names = ("teff", "logg", "av", "rv", "c", "fwhm", "var_scale", "alpha", "tau")

    def get_value(self, spec, model, rvmodel):
        mod = model._get_obs_model(self.teff, self.logg, self.av, self.fwhm, spec.wave, rv=self.rv, rvmodel=rvmodel)
        mod *= self.c
        res = spec.flux - mod
        kernel = self.var_scale * RationalQuadraticKernel(self.alpha, self.tau)
        gp = GP(kernel, mean=0., solver=HODLRsolver)
        gp.compute(spec.wave, spec.flux_err)
        return gp.lnlikelihood(res, quiet=True)






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


def lnlike(theta, spec, model, kernel):
    teff, logg, av, c  = theta

    mod = model._get_red_model(teff, logg, av, spec.wave)

    # smooth the model, and extract the section that overlays the model
    # since we smooth the full model, computed on the full wavelength range of the spectrum
    # and then extract the subset range that overlaps with the data
    # we avoid any edge effects with smoothing at the end of the range
    smoothedmod = convolve(mod, kernel, boundary='extend')
    smoothedmod *= c
    return -0.5*np.sum(((spec.flux-smoothedmod)/spec.flux_err)**2.)


def nll(*args):
    return -lnlike(*args)

