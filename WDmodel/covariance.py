import numpy as np
from george import GP, HODLRSolver, BasicSolver
import george.kernels

class WDmodel_CovModel(object):
    """
    Class defines the covariance model structure, and functions to get the
    likelihood given the wavelength, residuals, flux_err, and the kernel
    hyperparameters. This is defined so the kernel is only set in a single
    location.
    """
    def __init__(self, errscale, covtype='ExpSquared', nleaf=500, tol=1e-12, usehodlr=True):
        """
        Sets the covariance model and covariance model scale
        Accepts
            errscale: characteristic scale of the spectrum flux_err. The
            hyperparameter fsig is reported as a fraction of this number
            covtype: type of covariance model
                choices are White, ExpSquared, Matern32, Matern52, Exp

                All choices except White are represented by three paramters -
                fsig, tau, fw: the kernel hyperparameters defining the amplitude and
                    scale of the stationary kernel, and the white noise
                White just uses fw

        These options are only used if usehodlr is set
            nleaf: mimimum matrix block size for the HODLR solver
            tol: tolerance for the HODLR solver
            usehodlr: use the HODLR solver over the Basic Solver

        Returns
            a WDmodel_CovModel instance
        """

        # if we rescale the problem, errscale should be 1.
        # if not, it is the median error of the data
        self._errscale = errscale
        self._tol  = tol

        # configure the solver
        if usehodlr:
            message = "Using HODLR solver with tol={:g}, nleaf={:n}".format(tol, nleaf)
            self._solver = HODLRSolver
            self._solverkwargs = {'nleaf':nleaf, 'tol':tol}
            self._computekwargs = {'seed':1}
        else:
            message = "Using Basic solver"
            self._solver = BasicSolver
            self._solverkwargs = {}
            self._computekwargs = {}
        print message

        # configure the kernel
        self._ndim = 3
        self._k1   = george.kernels.ConstantKernel # amplitude of the covariance kernel
        self._k3   = george.kernels.ConstantKernel # amplitude of the white noise kernel 
        if covtype == 'White':
            self._k1 = None
            self._k2 = None
            self._ndim = 1
        elif covtype == 'ExpSquared':
            self._k2 = george.kernels.ExpSquaredKernel
        elif covtype == 'Matern32':
            self._k2 = george.kernels.Matern32Kernel
        elif covtype == 'Matern52':
            self._k2 = george.kernels.Matern52Kernel
        elif covtype == 'Exp':
            self._k2 = george.kernels.ExpKernel
        else:
            message = 'Do not understand kernel type {}'.format(covtype)
            raise RuntimeError(message)


    def lnlikelihood(self, wave, res, flux_err, fsig, tau, fw):
        """
        Return the lnlikelihood given the data, model and hyperparameters
        Accepts
            wave: the wavelength array
            res: the residuals between flux and the model flux
            flux_err: the uncertainty on the flux measurements - added to the
            diagonal of the covariance matrix
            fsig, tau, fw: the kernel hyperparameters defining the amplitude and
            scale of the stationary kernel, and the white noise
        """
        gp = self.getgp(wave, flux_err, fsig, tau, fw)
        # TODO - we should probably add a factor based on tau here as well
        if ((fsig*self._errscale)**2. < 10.*self._tol):
            return -np.inf
        return gp.lnlikelihood(res, quiet=True)


    def predict(self, wave, res, flux_err, fsig, tau, fw, mean_only=False):
        """
        Return the prediction for residuals given the data, model and hyperparameters
        Accepts
            wave: the wavelength array
            res: the residuals between flux and the model flux
            flux_err: the uncertainty on the flux measurements - added to the
            diagonal of the covariance matrix
            fsig, tau, fw: the kernel hyperparameters defining the amplitude and
            scale of the stationary kernel, and the white noise
        Returns
            wres: array of the predicted residuals
            cov: the full covariance matrix of the observations

        """
        gp = self.getgp(wave, flux_err, fsig, tau, fw)
        return gp.predict(res, wave, mean_only)


    def optimize(self, wave, res, flux_err, fsig, tau, fw, bounds, dims=None):
        """
        Optimize the kernel hyperparameters given the data The george
        documentation describes the call to optimize as "not terribly robust"

        Accepts
            wave: the wavelength array
            res: the residuals between flux and the model flux
            flux_err: the uncertainty on the flux measurements - added to the
            diagonal of the covariance matrix
            fsig, tau, fw: the kernel hyperparameters defining the amplitude and
            scale of the stationary kernel, and the white noise
            bounds: sequence of tuples with lower, upper bounds for each parameter
            dims: (optional) array of parameter indices to optimize
        Returns
            pars: list of optimized parameters
            result: scipy.optimize.minimze object

        """
        gp = self.getgp(wave, flux_err, fsig, tau, fw)
        pars, result = gp.optimize(wave, res, flux_err, dims=dims, sort=False, verbose=True)
        return pars, result


    def getgp(self, wave, flux_err, fsig, tau, fw):
        """
        Returns the GP object, given the locations of the model observation
        locations, uncertainties, and the hyperparameters
        Accepts
            wave: the wavelength array
            flux_err: the uncertainty on the flux measurements - added to the
            diagonal of the covariance matrix
            fsig, tau, fw: the kernel hyperparameters defining the amplitude and
            scale of the stationary kernel, and the white noise
        Returns
            gp: the george GP object
        """
        kernel = self._k3((fw*self._errscale)**2.)
        if self._ndim != 1:
            kernel = self._k1((fsig*self._errscale)**2.)*self._k2(tau) + kernel
        gp = GP(kernel, mean=0., solver=self._solver, **self._solverkwargs)
        gp.compute(wave, flux_err, sort=False, **self._computekwargs)
        return gp
