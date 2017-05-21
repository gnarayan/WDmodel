import numpy as np
import celerite
import warnings

class WDmodel_CovModel(object):
    """
    Class defines the covariance model structure, and functions to get the
    likelihood given the wavelength, residuals, flux_err, and the kernel
    hyperparameters. This is defined so the kernel is only set in a single
    location.
    """
    def __init__(self, errscale, covtype='Matern32', tol=1e-12):
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
            tol: tolerance for the Matern kernel approximation

        Returns
            a WDmodel_CovModel instance
        """

        # if we rescale the problem, errscale should be 1.
        # if not, it is the median error of the data
        self._errscale = errscale
        self._tol  = tol
        self._covtype = covtype

        message = "Using celerite with Cholesky Solver"
        print message

        # configure the kernel
        self._ndim = 3
        self._k2   = celerite.terms.JitterTerm # amplitude of the white noise kernel
        if covtype == 'White':
            self._k1 = None
            self._ndim = 1
        elif covtype == 'Matern32':
            self._k1 = celerite.terms.Matern32Term
        elif covtype == 'SHO':
            self._k1 = celerite.terms.SHOTerm
            self._logQ = np.log(1./np.sqrt(2.))
        else:
            message = 'Do not understand kernel type {}'.format(covtype)
            warnings.warn(message, RuntimeWarning)
            self._k1 = celerite.terms.Matern32Term
            self._covtype = 'Matern32'


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
        return gp.log_likelihood(res)


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
        return_cov = not(mean_only)
        return gp.predict(res, wave, return_cov)


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
        log_sigma_fw = np.log(fw*self._errscale)
        kw = self._k2(log_sigma_fw)
        if self._ndim != 1:
            log_sigma_fsig = np.log(fsig*self._errscale)
            if self._covtype == 'Matern32':
                log_rho = np.log(tau)
                ku = self._k1(log_sigma_fsig, log_rho, eps=self._tol)
            else:
                log_omega0 = np.log((2.*np.pi)/tau)
                ku = self._k1(log_sigma_fsig, self._logQ, log_omega0)
            kernel = ku + kw
        else:
            kernel = kw

        gp = celerite.GP(kernel, mean=0.)
        gp.compute(wave, flux_err, check_sorted=False)
        return gp
