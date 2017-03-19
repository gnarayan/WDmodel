from george import GP, HODLRSolver
from george.kernels import ExpSquaredKernel

class WDmodel_CovModel(object):
    """
    Class defines the covariance model structure, and functions to get the
    likelihood given the wavelength, residuals, flux_err, and the kernel
    hyperparameters. This is defined so the kernel is only set in a single
    location.
    """
    @classmethod
    def lnlikelihood(cls, wave, res, flux_err, sigf, tau):
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
        gp = cls.getgp(wave, flux_err, sigf, tau)
        return gp.lnlikelihood(res, quiet=True)

    @classmethod
    def predict(cls, wave, res, flux_err, sigf, tau):
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
        gp = cls.getgp(wave, flux_err, sigf, tau)
        return gp.predict(res, wave)

    @staticmethod
    def getgp(wave, flux_err, sigf, tau):
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
        kernel = (sigf**2.)*ExpSquaredKernel(tau)
        gp = GP(kernel, mean=0., solver=HODLRSolver)
        gp.compute(wave, flux_err)
        return gp
