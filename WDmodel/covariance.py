# -*- coding: UTF-8 -*-
"""
Parametrizes the noise of the spectrum fit using a Gaussian process.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import celerite
import warnings

class WDmodel_CovModel(object):
    """
    Parametrizes the noise of the spectrum fit using a Gaussian process.

    This class models the covariance of the spectrum fit using a stationary
    Gaussian process conditioned on the spectrum flux residuals and spectrum
    flux uncertainties. The class allows the kernel of the Gaussian process to
    be set in a single location. A few different stationary kernels are
    supported. These choices are defined in :py:mod:`celerite.terms`.

    Parameters
    ----------
    errscale : float
        Chracteristic scale of the spectrum flux uncertainties. The kernel
        amplitude hyperparameters are reported as fractions of this number. If
        the spectrum flux is rescaled, this must be set appropriately to get
        the correct uncertainties. The :py:mod:`WDmodel` package uses the
        median of the spectrum flux uncertainty internally.
    covtype : {'Matern32', 'SHO', 'Exp', 'White} str
        The model to use to parametrize the covariance.  Choices are defined in
        :py:mod:`celerite.terms` All choices except 'White' parametrize the
        covariance using a stationary kernel with a characteristic amplitude
        `fsig` and scale `tau` + a white noise component with amplitude `fw`.
        Only the white noise component is used to condition the Gaussian
        process if covtype is 'White'. If not specified or unknown, 'Matern32'
        is used and a RuntimeWarning is raised.
    coveps : float
        If covtype is 'Matern32' a :py:class:`celerite.terms.Matern32Term` is
        used to approximate a Matern32 kernel with precision `coveps`. The
        default is 1e-12. Ignored if any other `covtype` is specified.

    Attributes
    ----------
    _errscale : float
        The input `errscale`
    _covtype : str
        The input `covtype`
    _coveps : float
        The input `coveps`
    _ndim : int
        The dimensionality of kernel used to parametrize the covariance
    _k1 : None or a term instance from :py:mod:`celerite.terms`
        The non-trivial stationary component of the kernel
    _k2 : :py:class:`celerite.terms.JitterTerm`
        The white noise component of the kernel
    _logQ : float, conditional
        1/sqrt(2) - only set if `covtype` is 'SHO'

    Returns
    -------
    A :py:class:`WDmodel.covariance.WDmodel_CovModel` instance

    Notes
    -----
        Virtually none of the attributes should be used directly since it is
        trivially possible to break the model by redefining them. Access to
        them is best through the functions connected to the models.
    """
    def __init__(self, errscale, covtype='Matern32', coveps=1e-12):
        # if we rescale the problem, errscale should be 1.
        # if not, it is the median error of the data
        self._errscale = errscale
        self._coveps  = coveps
        self._covtype = covtype

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
        elif covtype == 'Exp':
            self._k1 = celerite.terms.RealTerm
        else:
            message = 'Do not understand kernel type {}'.format(covtype)
            warnings.warn(message, RuntimeWarning)
            self._k1 = celerite.terms.Matern32Term
            self._covtype = 'Matern32'

        message = "Parametrizing covariance with {} kernel and using Cholesky Solver".format(covtype)
        print(message)


    def lnlikelihood(self, wave, res, flux_err, fsig, tau, fw):
        """
        Return the log likelihood of the Gaussian process

        Conditions the Gaussian process specified by the functional form of the
        stationary kernel and the current values of the hyperparameters on the
        data, and computes the log likelihood. Wraps
        :py:meth:`celerite.GP.log_likelihood`.

        Parameters
        ----------
        wave : array-like, optional
            Wavelengths at which to condition the Gaussian process
        res : array-like
            Flux residual array on which to condition the Gaussian process.
            The kernel parametrization assumes that the mean model has been
            subtracted off.
        flux_err : array-like
            Flux uncertaintyarray on which to condition the Gaussian process
        fsig : float
            The fractional amplitude of the non-trivial stationary kernel. The
            true amplitude is scaled by
            :py:attr:`WDmodel.covariance.WDmodel_CovModel._errscale`
        tau : float
            The characteristic length scale of the non-trivial stationary
            kernel.
        fw : float
            The fractional amplitude of the white noise component of the
            kernel. The true amplitude is scaled by
            :py:attr:`WDmodel.covariance.WDmodel_CovModel._errscale`

        Returns
        -------
        lnlike : float
            The log likelihood of the Gaussian process conditioned on the data.
        """
        gp = self.getgp(wave, flux_err, fsig, tau, fw)
        return gp.log_likelihood(res)


    def predict(self, wave, res, flux_err, fsig, tau, fw, mean_only=False):
        """
        Return the prediction for the Gaussian process

        Conditions the Gaussian process specified by the parametrized with the
        functional form of the stationary kernel and the current values of the
        hyperparameters on the data, and computes returns the prediction at the
        same location as the data. Wraps :py:meth:`celerite.GP.predict`.

        Parameters
        ----------
        wave : array-like, optional
            Wavelengths at which to condition the Gaussian process
        res : array-like
            Flux residual array on which to condition the Gaussian process.
            The kernel parametrization assumes that the mean model has been
            subtracted off.
        flux_err : array-like
            Flux uncertaintyarray on which to condition the Gaussian process
        fsig : float
            The fractional amplitude of the non-trivial stationary kernel. The
            true amplitude is scaled by
            :py:attr:`WDmodel.covariance.WDmodel_CovModel._errscale`
        tau : float
            The characteristic length scale of the non-trivial stationary
            kernel.
        fw : float
            The fractional amplitude of the white noise component of the
            kernel. The true amplitude is scaled by
            :py:attr:`WDmodel.covariance.WDmodel_CovModel._errscale`
        mean_only : bool, optional
            Return only the predicted mean, not the covariance matrix

        Returns
        -------
        wres : array-like
            The prediction of the Gaussian process conditioned on the data at
            the same location i.e. the model.
        cov : array-like, optional
            The computed covariance matrix of the Gaussian process using the
            parametrized stationary kernel evaluated at the locations of the
            data.
        """
        gp = self.getgp(wave, flux_err, fsig, tau, fw)
        return_cov = not(mean_only)
        return gp.predict(res, wave, return_cov)


    def getgp(self, wave, flux_err, fsig, tau, fw):
        """
        Return the celerite.GP instance

        Precomputes the covariance matrix of the Gaussian process specified by
        the functional form of the stationary kernel and the current values of
        the hyperparameters. Wraps :py:class:`celerite.GP`.

        Parameters
        ----------
        wave : array-like, optional
            Wavelengths at which to condition the Gaussian process
        flux_err : array-like
            Flux uncertaintyarray on which to condition the Gaussian process
        fsig : float
            The fractional amplitude of the non-trivial stationary kernel. The
            true amplitude is scaled by
            :py:attr:`WDmodel.covariance.WDmodel_CovModel._errscale`
        tau : float
            The characteristic length scale of the non-trivial stationary
            kernel.
        fw : float
            The fractional amplitude of the white noise component of the
            kernel. The true amplitude is scaled by
            :py:attr:`WDmodel.covariance.WDmodel_CovModel._errscale`

        Returns
        -------
        gp : :py:class:`celerite.GP` instance
            The Gaussian process with covariance matrix precomputed at the
            location of the data

        Notes
        -----
            `fsig`, `tau` and `fw` all must be > 0. This constraint is not
            checked here, but is instead imposed by the samplers/optimizers
            used in the :py:mod:`WDmodel.fit` methods, and by bounds used to
            construct the :py:class:`WDmodel.likelihood.WDmodel_Likelihood`
            instance using the :py:func:`WDmodel.likelihood.setup_likelihood`
            method.
        """
        log_sigma_fw = np.log(fw*self._errscale)
        kw = self._k2(log_sigma_fw)
        if self._ndim != 1:
            log_sigma_fsig = np.log(fsig*self._errscale)
            if self._covtype == 'Matern32':
                log_rho = np.log(tau)
                ku = self._k1(log_sigma_fsig, log_rho, eps=self._coveps)
            elif self._covtype == 'SHO':
                log_omega0 = np.log((2.*np.pi)/tau)
                ku = self._k1(log_sigma_fsig, self._logQ, log_omega0)
            else:
                log_c = 1./np.log(tau)
                ku = self._k1(log_sigma_fsig, log_c)
            kernel = ku + kw
        else:
            kernel = kw

        gp = celerite.GP(kernel, mean=0.)
        gp.compute(wave, flux_err, check_sorted=False)
        return gp
