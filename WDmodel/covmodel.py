from george import GP, HODLRSolver
import george.kernels

class WDmodel_CovModel(object):
    """
    Class defines the covariance model structure, and functions to get the
    likelihood given the wavelength, residuals, flux_err, and the kernel
    hyperparameters. This is defined so the kernel is only set in a single
    location.
    """
    def __init__(self, covtype='White'):

        self.k1   = george.kernels.ConstantKernel
        self.ndim = 2
        if covtype == 'White':
            self.k1 = george.kernels.WhiteKernel
            self.k2 = None
            self.ndim = 1
        elif covtype == 'ExpSquared':
            self.k2 = george.kernels.ExpSquaredKernel
        elif covtype == 'Matern32':
            self.k2 = george.kernels.Matern32Kernel
        elif covtype == 'Matern52':
            self.k2 = george.kernels.Matern52Kernel
        elif covtype == 'Exp':
            self.k2 = george.kernels.ExpKernel
        else:
            message = 'Do not understand kernel type {}'.format(covtype)
            raise RuntimeError(message)


    def lnlikelihood(self, wave, res, flux_err, sigf, tau):
        """
        Return the lnlikelihood given the data, model and hyperparameters
        Accepts
            wave: the wavelength array
            res: the residuals between flux and the model flux
            flux_err: the uncertainty on the flux measurements - added to the
            diagonal of the covariance matrix
            sigf, tau: the kernel hyperparameters defining the amplitude and
            scale of the stationary kernel
        """
        gp = self.getgp(wave, flux_err, sigf, tau)
        return gp.lnlikelihood(res, quiet=True)


    def predict(self, wave, res, flux_err, sigf, tau):
        """
        Return the prediction for residuals given the data, model and hyperparameters
        Accepts
            wave: the wavelength array
            res: the residuals between flux and the model flux
            flux_err: the uncertainty on the flux measurements - added to the
            diagonal of the covariance matrix
            sigf, tau: the kernel hyperparameters defining the amplitude and
            scale of the stationary kernel
        Returns
            wres: array of the predicted residuals
            cov: the full covariance matrix of the observations

        """
        gp = self.getgp(wave, flux_err, sigf, tau)
        return gp.predict(res, wave)


    def optimize(self, wave, res, flux_err, sigf, tau, bounds, dims=None):
        """
        Optimize the kernel hyperparameters given the data The george
        documentation describes the call to optimize as "not terribly robust"

        Accepts
            wave: the wavelength array
            res: the residuals between flux and the model flux
            flux_err: the uncertainty on the flux measurements - added to the
            diagonal of the covariance matrix
            sigf, tau: the kernel hyperparameters defining the amplitude and
            scale of the stationary kernel
            bounds: sequence of tuples with lower, upper bounds for each parameter
            dims: (optional) array of parameter indices to optimize
        Returns
            pars: list of optimized parameters
            result: scipy.optimize.minimze object

        """
        gp = self.getgp(wave, flux_err, sigf, tau)
        pars, result = gp.optimize(wave, res, flux_err, dims=dims)
        return pars, result


    def getgp(self, wave, flux_err, sigf, tau):
        """
        Returns the GP object, given the locations of the model observation
        locations, uncertainties, and the hyperparameters
        Accepts
            wave: the wavelength array
            flux_err: the uncertainty on the flux measurements - added to the
            diagonal of the covariance matrix
            sigf, tau: the kernel hyperparameters defining the amplitude and
            scale of the stationary kernel
        Returns
            gp: the george GP object
        """
        if self.ndim == 1:
            kernel = self.k1(sigf**2.)
        else:
            kernel = self.k1(sigf**2.)*self.k2(tau)
        gp = GP(kernel, mean=0., solver=HODLRSolver)
        gp.compute(wave, flux_err)
        return gp
