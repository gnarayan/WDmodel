# -*- coding: UTF-8 -*-
"""
Routines to visualize the DA White Dwarf model atmosphere fit
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from scipy.stats import norm
from itertools import cycle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties as FM
from astropy.visualization import hist
from . import io
from . import passband
import corner
from six.moves import range


def plot_minuit_spectrum_fit(spec, objname, outdir, specfile, scale_factor, model, result, save=True):
    """
    Plot the MLE fit of the spectrum with the model, assuming uncorrelated
    noise.

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum. Must have
        ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    objname : str
        object name - used to title plots
    outdir : str
        controls where the plot is written out if ``save=True``
    specfile : str
        Used in the title, and to set the name of the ``outfile`` if ``save=True``
    scale_factor : float
        factor by which the flux was scaled for y-axis label
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator
    result : dict
        dictionary of parameters with keywords ``value``, ``fixed``, ``scale``,
        ``bounds`` for each. Same format as returned from
        :py:func:`WDmodel.io.read_params`
    save : bool
        if True, save the file

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure` instance

    Notes
    -----
        The MLE fit uses :py:meth:`iminuit.Minuit.migrad` to fit the spectrum
        with the model.  This fit doesn't try to account for the covariance in
        the data, and is not expected to be great - just fast, and capable of
        setting a reasonable initial guess. If it is apparent from the plot
        that this fit is very far off, refine the initial guess to the fitter.
    """
    font_s  = FM(size='small')
    font_m  = FM(size='medium')
    font_l  = FM(size='large')

    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    ax_spec  = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1])

    ax_spec.fill_between(spec.wave, spec.flux+spec.flux_err, spec.flux-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_spec.plot(spec.wave, spec.flux, color='black', linestyle='-', marker='None', label=specfile)

    print_params = ('teff', 'logg', 'av', 'dl')
    outlabel = 'Model\n'
    for param in print_params:
        val = result[param]['value']
        err = result[param]['scale']
        fixed = result[param]['fixed']
        if val is None:
            thislabel = '{} = {} '.format(param, val)
        else:
            thislabel = '{} = {:.3f} '.format(param, val)

        if not fixed:
            thislabel += ' +/- {:.3f}'.format(err)
        else:
            thislabel = '[{} FIXED]'.format(thislabel)
        thislabel +='\n'
        outlabel += thislabel

    fix_labels = list(set(result.keys()) - set(print_params))
    for param in fix_labels:
        val = result[param]['value']
        if val is None:
            thislabel = '{} = {} '.format(param, val)
        else:
            thislabel = '{} = {:.3f} '.format(param, val)
        thislabel = '[{} FIXED]'.format(thislabel)
        thislabel +='\n'
        outlabel += thislabel

    teff = result['teff']['value']
    logg = result['logg']['value']
    av   = result['av']['value']
    dl   = result['dl']['value']
    rv   = result['rv']['value']
    fwhm = result['fwhm']['value']

    pixel_scale = 1./np.median(np.gradient(spec.wave))

    mod = model._get_obs_model(teff, logg, av, fwhm, spec.wave, rv=rv, pixel_scale=pixel_scale)
    smoothedmod = mod* (1./(4.*np.pi*(dl)**2.))

    ax_spec.plot(spec.wave, smoothedmod, color='red', linestyle='-',marker='None', label=outlabel)

    ax_resid.fill_between(spec.wave, spec.flux-smoothedmod+spec.flux_err, spec.flux-smoothedmod-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_resid.plot(spec.wave, spec.flux-smoothedmod,  linestyle='-', marker=None,  color='black')

    ax_resid.set_xlabel('Wavelength~(\AA)',fontproperties=font_m, ha='center')
    ax_spec.set_ylabel('Normalized Flux (Scale factor = {})'.format(1./scale_factor), fontproperties=font_m)
    ax_resid.set_ylabel('Fit Residual Flux', fontproperties=font_m)
    ax_spec.legend(frameon=False, prop=font_s)
    fig.suptitle('Quick Fit for Initial Guess: %s (%s)'%(objname, specfile), fontproperties=font_l)

    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    if save:
        outfile = io.get_outfile(outdir, specfile, '_minuit.pdf')
        fig.savefig(outfile)
    return fig


def plot_mcmc_spectrum_fit(spec, other_spec, objname, specfile, scale_factor, model, covmodel, result, param_names, samples,\
        ndraws=21, everyn=1):
    """
    Plot the spectrum of the DA White Dwarf and the "best fit" model

    The full fit parametrizes the covariance model using a stationary Gaussian
    process as defined by :py:class:`WDmodel.covariance.WDmodel_CovModel`. The
    posterior function constructed in
    :py:class:`WDmodel.likelihood.WDmodel_Posterior` is evaluated by the
    sampler in the :py:func:`WDmodel.fit.fit_model` method. The median value is
    reported as the best-fit value for each of the fit parameters in
    :py:attr:`WDmodel.likelihood.WDmodel_Likelihood.parameter_names`.

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum. Must have
        ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    objname : str
        object name - used to title plots
    outdir : str
        controls where the plot is written out if ``save=True``
    specfile : str
        Used in the title, and to set the name of the ``outfile`` if ``save=True``
    scale_factor : float
        factor by which the flux was scaled for y-axis label
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator
    covmodel : :py:class:`WDmodel.covariance.WDmodel_CovModel` instance
        The parametrized model for the covariance of the spectrum ``spec``
    result : dict
        dictionary of parameters with keywords ``value``, ``fixed``, ``scale``,
        ``bounds`` for each. Same format as returned from
        :py:func:`WDmodel.io.read_params`
    param_names : array-like
        Ordered list of free parameter names
    samples : array-like
        Samples from the flattened Markov Chain with shape ``(N, len(param_names))``
    ndraws : int, optional
        Number of draws to make from the Markov Chain to overplot. Higher
        numbers provide a better sense of the uncertainty in the model at the
        cost of speed and a larger, slower to render output plot.
    everyn : int, optional
        If the posterior function was evaluated using only every nth
        observation from the data, this should be specified to visually
        indicate the observations used.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure` instance
        The output figure
    draws : array-like
        The actual draws from the Markov Chain used in ``fig``

    Notes
    -----
        It's faster to draw samples from the posterior in one location, and
        pass along the same samples to all the methods in :py:mod:`WDmodel.viz`.

        Consequently, most require ``draws`` as an input. This makes all the
        plots connected, and none will return if an error is thrown here, but
        this is the correct behavior as all of them are visualizing one aspect
        of the same fit.

        Each element of ``draws`` contains
            * ``smoothedmod`` - the model spectrum
            * ``wres`` - the prediction from the Gaussian process
            * ``wres_err`` - the diagonal of the covariance matrix for the prediction from the Gaussian process
            * ``full_mod`` - the full model SED, in order to compute the synthetic photometry
            * ``out_draw`` - the dictionary of model parameters from this draw. Same format as ``result``.
    """
    font_s  = FM(size='small')
    font_m  = FM(size='medium')
    font_l  = FM(size='large')

    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    ax_spec  = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1])

    ax_spec.fill_between(spec.wave, spec.flux+spec.flux_err, spec.flux-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_spec.plot(spec.wave, spec.flux, color='black', linestyle='-', marker='None', label=specfile)

    this_draw = io.copy_params(result)
    draws  = samples[np.random.randint(0, len(samples), ndraws),:]

    pixel_scale = 1./np.median(np.gradient(spec.wave))

    # plot one draw of the sample, bundled into a dict
    def plot_one(this_draw, other_spec, color='red', alpha=1., label=None, i=1):
        if other_spec:
            teff = this_draw['teff']['value']
            logg = this_draw['logg']['value']
            av   = this_draw['av']['value']
            rv   = this_draw['rv']['value']
            dl   = this_draw['dl']['value']
            fwhm = this_draw['fwhm2']['value']
            fsig = this_draw['fsig2']['value']
            tau  = this_draw['tau2']['value']
            fw   = this_draw['fw2']['value']
        else:
            teff = this_draw['teff']['value']
            logg = this_draw['logg']['value']
            av   = this_draw['av']['value']
            rv   = this_draw['rv']['value']
            dl   = this_draw['dl']['value']
            fwhm = this_draw['fwhm']['value']
            fsig = this_draw['fsig']['value']
            tau  = this_draw['tau']['value']
            fw   = this_draw['fw']['value']

        mod, full_mod = model._get_full_obs_model(teff, logg, av, fwhm, spec.wave,\
                rv=rv, pixel_scale=pixel_scale)
        smoothedmod = mod* (1./(4.*np.pi*(dl)**2.))

        res = spec.flux - smoothedmod
        wres, cov = covmodel.predict(spec.wave, res, spec.flux_err, fsig, tau, fw)
        ax_spec.plot(spec.wave, smoothedmod+wres,\
                color=color, linestyle='-',marker='None', alpha=alpha, label=label)
        out_draw = io.copy_params(this_draw)
        return smoothedmod, wres, cov, full_mod, out_draw

    # for each draw, update the dict, and plot it
    out = []
    for i in range(ndraws):
        for j, param in enumerate(param_names):
            this_draw[param]['value'] = draws[i,j]
        smoothedmod, wres, cov, full_mod, out_draw = plot_one(this_draw, other_spec, color='orange', alpha=0.3, i=i)
        wres_err = np.diag(cov)**0.5
        out.append((smoothedmod, wres, wres_err, full_mod, out_draw))

    outlabel = 'Model\n'
    for param in result:
        val = result[param]['value']
        errp, errm = result[param]['errors_pm']
        fixed = result[param]['fixed']
        thislabel = '{} = {:.3f} '.format(param, val)
        if not fixed:
            thislabel += ' +{:.3f}/-{:.3f}'.format(errp, errm)
        else:
            thislabel = '[{} FIXED]'.format(thislabel)
        thislabel +='\n'
        outlabel += thislabel

    # finally, overplot the best result draw as solid
    smoothedmod, wres, cov, full_mod, out_draw = plot_one(result, other_spec, color='red', alpha=1., label=outlabel)
    wres_err = np.diag(cov)**0.5
    out.append((smoothedmod, wres, wres_err, full_mod, out_draw))

    # plot the residuals
    ax_resid.fill_between(spec.wave, spec.flux-smoothedmod-wres+spec.flux_err, spec.flux-smoothedmod-wres-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_resid.plot(spec.wave, spec.flux-smoothedmod-wres,  linestyle='-', marker=None,  color='black')

    for draw in out[:-1]:
        ax_resid.plot(spec.wave, draw[0]+draw[1]-smoothedmod-wres, linestyle='-',\
                marker=None, alpha=0.3, color='orange')

    if everyn != 1:
        ax_spec.plot(spec.wave[::everyn], spec.flux[::everyn], color='blue', marker='o', ls='None',\
            alpha=0.5, label='everyn:{:n}'.format(everyn))
        ax_resid.plot(spec.wave[::everyn], (spec.flux-smoothedmod-wres)[::everyn], color='blue', marker='o',\
                alpha=0.5, ls='None')

    ax_resid.axhline(0., color='red', linestyle='--')
    ax_resid.fill_between(spec.wave, +wres_err, -wres_err,\
        facecolor='red', alpha=0.3, interpolate=True)

    # label the axes
    ax_resid.set_xlabel('Wavelength~(\AA)',fontproperties=font_m, ha='center')
    ax_spec.set_ylabel('Normalized Flux (Scale factor = {})'.format(1./scale_factor), fontproperties=font_m)
    ax_resid.set_ylabel('Fit Residual Flux', fontproperties=font_m)
    ax_spec.legend(frameon=False, prop=font_s)
    fig.suptitle('MCMC Fit: %s (%s)'%(objname, specfile), fontproperties=font_l)

    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    return fig, out


def plot_mcmc_photometry_res(objname, phot, phot_dispersion, model, pbs, draws):
    """
    Plot the observed DA white dwarf photometry as well as the "best-fit" model
    magnitudes

    Parameters
    ----------
    objname : str
        object name - used to title plots
    phot : None or :py:class:`numpy.recarray`
        The photometry. Must have
        ``dtype=[('pb', 'str'), ('mag', '<f8'), ('mag_err', '<f8')]``
    phot_dispersion : float, optional
        Excess photometric dispersion to add in quadrature with the
        photometric uncertainties ``phot.mag_err``. Use if the errors are
        grossly underestimated. Default is ``0.``
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator
    pbs : dict
        Passband dictionary containing the passbands corresponding to
        ``phot.pb`` and generated by :py:func:`WDmodel.passband.get_pbmodel`.
    draws : array-like
        produced by :py:func:`plot_mcmc_spectrum_fit` - see notes for content.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure` instance
        The output figure
    mag_draws : array-like
        The magnitudes corresponding to the parameters ``draws`` from the Markov
        Chain used in ``fig``

    Notes
    -----
        Each element of ``mag_draws`` contains
            * ``wres`` - the difference between the observed and synthetic magnitudes
            * ``model_mags`` - the model magnitudes corresponding to the current model parameters
            * ``mu`` - the flux normalization parameter that must be added to the ``model_mags``

    See Also
    --------
    :py:func:`WDmodel.viz.plot_mcmc_spectrum_fit`
    """
    font_s  = FM(size='small')
    font_m  = FM(size='medium')
    font_l  = FM(size='large')

    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    ax_phot   = fig.add_subplot(gs[0])
    ax_resid  = fig.add_subplot(gs[1])

    refwave = np.array([x[4] for x in pbs.values()])
    npb = len(pbs)
    pbind   = np.arange(npb)

    # plot one draw of the sample
    def plot_draw(draw, color='red', alpha=1.0, label=None, linestyle='None'):
        _, _, _, model_spec, params = draw
        mu = params['mu']['value']
        model_mags = passband.get_model_synmags(model_spec, pbs, mu=mu)
        ax_phot.plot(refwave, model_mags.mag, color=color, alpha=alpha, marker='o', label=label, linestyle=linestyle)
        res = phot.mag - model_mags.mag
        return res, model_mags, mu

    out = []
    mag_draws = []
    # plot the draws
    for draw in draws[:-1]:
        res, model_mags, mu = plot_draw(draw, color='orange', alpha=0.3)
        out.append(res)
        mag_draws.append((res, model_mags, mu))

    # plot the magnitudes
    ax_phot.errorbar(refwave, phot.mag, yerr=phot.mag_err, color='k', marker='o',\
            linestyle='None', label='Observed Magnitudes')
    res, model_mags, mu = plot_draw(draws[-1], color='red', alpha=1.0, label='Model Magnitudes', linestyle='--')

    mag_draws.append((res, model_mags, mu))

    # the draws are already samples from the posterior distribution - just take the median
    out = np.array(out)
    errs = np.median(np.abs(out), axis=0)
    scaling  = norm.ppf(3/4.)
    errs/=scaling

    # plot the residuals
    ax_resid.fill_between(pbind, -errs, errs, interpolate=True, facecolor='orange', alpha=0.3)
    ax_resid.errorbar(pbind, res, yerr=phot.mag_err, color='black',  marker='o')
    ax_resid.axhline(0., color='red', linestyle='--')

    # flip the y axis since mags
    ax_phot.invert_yaxis()
    ax_resid.invert_yaxis()

    # label the axes
    ax_resid.set_xlim(-0.5,npb-0.5)
    ax_resid.set_xticks(pbind)
    ax_resid.set_xticklabels(list(pbs.keys()))
    ax_resid.set_xlabel('Passband',fontproperties=font_m, ha='center')
    ax_phot.set_xlabel('Wavelength',fontproperties=font_m, ha='center')
    ax_phot.set_ylabel('Magnitude (Photometric dispersion = {})'.format(phot_dispersion), fontproperties=font_m)
    ax_resid.set_ylabel('Residual (mag)', fontproperties=font_m)
    ax_phot.legend(frameon=False, prop=font_s)
    fig.suptitle('Photometry for {}'.format(objname), fontproperties=font_l)

    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    return fig, mag_draws


def plot_mcmc_spectrum_nogp_fit(spec, objname, specfile, scale_factor,\
        cont_model, draws, covtype='Matern32', everyn=1):
    """
    Plot the spectrum of the DA White Dwarf and the "best fit" model without
    the Gaussian process

    Unlike :py:func:`plot_mcmc_spectrum_fit` this version does not apply the
    prediction from the Gaussian process to the spectrum model to match the
    observed spectrum. This visualization is useful to indicate if the Gaussian
    process - i.e. the kernel choice ``covtype`` used to parametrize the
    covariance is - is appropriate.

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum. Must have
        ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    objname : str
        object name - used to title plots
    outdir : str
        controls where the plot is written out if ``save=True``
    specfile : str
        Used in the title, and to set the name of the outfile if ``save=True``
    scale_factor : float
        factor by which the flux was scaled for y-axis label
    cont_model : :py:class:`numpy.recarray`
        The continuuum model. Must have the same structure as ``spec``
        Produced by :py:func:`WDmodel.fit.pre_process_spectrum`
    draws : array-like
        produced by :py:func:`plot_mcmc_spectrum_fit` - see notes for content.
    covtype : ``{'Matern32', 'SHO', 'Exp', 'White'}``
        stationary kernel type used to parametrize the covariance in
        :py:class:`WDmodel.covariance.WDmodel_CovModel`
    everyn : int, optional
        If the posterior function was evaluated using only every nth
        observation from the data, this should be specified to visually
        indicate the observations used.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure` instance
        The output figure

    See Also
    --------
    :py:func:`WDmodel.viz.plot_mcmc_spectrum_fit`
    """

    font_s  = FM(size='small')
    font_m  = FM(size='medium')
    font_l  = FM(size='large')

    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    ax_spec  = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1])

    # plot the spectrum
    ax_spec.fill_between(spec.wave, spec.flux+spec.flux_err, spec.flux-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_spec.plot(spec.wave, spec.flux, color='black', linestyle='-', marker='None', label=specfile)

    # plot the continuum model
    ax_spec.plot(cont_model.wave, cont_model.flux, color='blue', linestyle='--', marker='None', label='Continuum')

    # plot the residual without the covariance term
    smoothedmod, wres, wres_err, _ , _ = draws[-1]
    ax_resid.fill_between(spec.wave, wres+wres_err, wres-wres_err, facecolor='red', alpha=0.3, interpolate=True)
    ax_resid.fill_between(spec.wave, spec.flux-smoothedmod+spec.flux_err, spec.flux-smoothedmod-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_resid.plot(spec.wave, spec.flux - smoothedmod, color='black', linestyle='-', marker='None')

    bestfit, bestres, _, _, _ = draws[-1]
    def plot_draw(draw, color='red', alpha=1.0, label=None):
        smoothedmod, wres, _, _, _ = draw
        ax_resid.plot(spec.wave, wres+smoothedmod - bestfit,  linestyle='-', marker=None,  color=color, alpha=alpha)
        ax_spec.plot(spec.wave, smoothedmod, color=color, linestyle='-', marker='None', alpha=alpha, label=label)

    # plot each of the draws - we want to get a sense of the range of the covariance to plot wres
    for draw in draws[:-1]:
        plot_draw(draw, color='orange', alpha=0.3)
    plot_draw(draws[-1], color='red', alpha=1.0, label='Model - no Covariance')

    if everyn != 1:
        smoothedmod, wres, _, _, _ = draws[-1]
        ax_spec.plot(spec.wave[::everyn], spec.flux[::everyn], color='blue', marker='o', ls='None',\
                alpha=0.5, label='everyn:{:n}'.format(everyn))
        ax_resid.plot(spec.wave[::everyn], wres[::everyn], marker='o',  color='blue', ls='None', alpha=0.5)

    # label the axes
    ax_resid.set_xlabel('Wavelength~(\AA)',fontproperties=font_m, ha='center')
    ax_spec.set_ylabel('Normalized Flux (Scale factor = {})'.format(1./scale_factor), fontproperties=font_m)
    ax_resid.set_ylabel('Fit Residual Flux', fontproperties=font_m)
    ax_spec.legend(frameon=False, prop=font_s)
    fig.suptitle('MCMC Fit - No {} Covariance: {} ({})'.format(covtype, objname, specfile), fontproperties=font_l)

    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    return fig


def plot_mcmc_line_fit(spec, linedata, model, cont_model, draws, balmer=None):
    """
    Plot a comparison of the normalized hydrogen Balmer lines of the spectrum
    and model

    Note that we fit the full spectrum, not just the lines. The lines are
    extracted using a coarse continuum fit in
    :py:func:`WDmodel.fit.pre_process_spectrum`. This fit is purely cosmetic
    and in no way contributes to the likelihood. It's particularly useful to
    detect small velocity offsets or wavelength calibration errors.

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum. Must have
        ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    linedata : :py:class:`numpy.recarray`
        The observations of the spectrum corresponding to the hydrogen Balmer
        lines. Must have
        ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8'), ('line_mask', 'i4'), ('line_ind', 'i4')]``
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator
    cont_model : :py:class:`numpy.recarray`
        The continuuum model. Must have the same structure as ``spec``
        Produced by :py:func:`WDmodel.fit.pre_process_spectrum`
    draws : array-like
        produced by :py:func:`plot_mcmc_spectrum_fit` - see notes for content.
    balmer : array-like, optional
        list of Balmer lines to plot - elements must be in range ``[1, 6]``
        These correspond to the lines defined in
        :py:attr:`WDmodel.WDmodel.WDmodel._lines`. Default is ``range(1, 7)``

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure` instance
        The output figure containing the line profile plot
    fig2 : :py:class:`matplotlib.figure.Figure` instance
        The output figure containing histograms of the line residuals

    See Also
    --------
    :py:func:`WDmodel.viz.plot_mcmc_spectrum_fit`
    """

    font_xs = FM(size='x-small')
    font_s  = FM(size='small')
    font_m  = FM(size='medium')
    font_l  = FM(size='large')

    # create a figure for the line profiles
    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(1, 1)
    ax_lines  = fig.add_subplot(gs[0])

    if balmer is None:
        balmer = list(model._lines.keys())

    # create another figure with separate axes for each of the lines
    uselines = set(np.unique(linedata.line_mask)) & set(balmer)
    nlines = len(uselines)
    Tot = nlines + 1
    Cols = 3
    Rows = Tot // Cols
    Rows += Tot % Cols
    fig2 = plt.figure(figsize=(10,8))
    gs2 = gridspec.GridSpec(Rows, Cols )

    # get the default color cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = cycle(colors)

    # plot the distribution of residuals for the entire spectrum
    ax_resid  = fig2.add_subplot(gs2[0])
    smoothedmod, wres, _, _, _ = draws[-1]
    res = spec.flux - smoothedmod - wres
    hist(res, bins='knuth', density=True, histtype='stepfilled', color='grey', alpha=0.5, label='Residuals',ax=ax_resid)
    ax_resid.axvline(0., color='red', linestyle='--')

    # label the axes, rotate the tick labels, and get the xlim
    ax_resid.set_xlabel('Fit Residual Flux', fontproperties=font_m)
    ax_resid.set_ylabel('Norm', fontproperties=font_m)
    ax_resid.legend(loc='upper left', frameon=False, prop=font_s)
    plt.setp(ax_resid.get_xticklabels(), rotation=30, horizontalalignment='right')
    (res_xmin, res_xmax) = ax_resid.get_xlim()
    k = 1

    for i, line in enumerate(np.unique(linedata.line_mask)):

        if not line in balmer:
            continue

        # select this line
        mask = (linedata.line_mask == line)
        wave = linedata.wave[mask]

        # restore the line properties
        linename, W0, D, eps = model._lines[line]

        # find the matching indices in the spectrum/continuum model that match the line
        ind  = np.searchsorted(cont_model.wave, wave)
        this_line_cont = cont_model.flux[ind]

        # shift the wavelength so the centroids are 0
        shifted_wave = wave - W0
        shifted_flux = linedata.flux[mask]/this_line_cont
        shifted_ferr  = linedata.flux_err[mask]/this_line_cont

        # plot the lines, adding a small vertical offset between each
        voff = 0.2*i
        ax_lines.fill_between(shifted_wave, shifted_flux + voff + shifted_ferr, shifted_flux + voff - shifted_ferr,\
                facecolor='grey', alpha=0.5, interpolate=True)
        ax_lines.plot(shifted_wave, shifted_flux + voff, linestyle='-', marker='None', color='black')

        # add a text label for each line
        label = '{} ({:.2f})'.format(linename, W0)
        ax_lines.text(shifted_wave[-1]+10 , shifted_flux[-1] + voff, label, fontproperties=font_xs,\
                color='blue', va='top', ha='center', rotation=90)

        # plot one of the draws
        def plot_draw(draw, color='red', alpha=1.0):
            smoothedmod, wres, _, _, _ = draw
            line_model = (smoothedmod + wres)[ind]
            line_model /= this_line_cont
            line_model += voff
            ax_lines.plot(shifted_wave, line_model, linestyle='-', marker='None', color=color, alpha=alpha)

        # overplot the model
        for draw in draws[:-1]:
            plot_draw(draw, color='orange', alpha=0.3)
        plot_draw(draws[-1], color='red', alpha=1.0)

        # overplot the best model err as the bottom layer
        bestmod, bestres, bestres_err, _, _ = draws[-1]
        besthi = (bestmod + bestres + bestres_err)[ind]
        bestlo = (bestmod + bestres - bestres_err)[ind]
        besthi /= this_line_cont
        bestlo /= this_line_cont
        besthi += voff
        bestlo += voff
        ax_lines.fill_between(shifted_wave, besthi, bestlo,\
                              facecolor='red', alpha=0.3, interpolate=True, zorder=-1)

        # plot the residuals of this line
        ax_resid  = fig2.add_subplot(gs2[k])
        hist(linedata.flux[mask] - (smoothedmod + wres)[ind] , bins='knuth', density=True, ax=ax_resid,\
                histtype='stepfilled', label=label, alpha=0.3, color=next(colors))
        ax_resid.axvline(0., color='red', linestyle='--')

        # label the axis and match the limits for the overall residuals
        ax_resid.set_xlabel('Fit Residual Flux', fontproperties=font_m)
        ax_resid.set_ylabel('Norm', fontproperties=font_m)
        ax_resid.set_xlim((res_xmin, res_xmax))
        ax_resid.legend(frameon=False, prop=font_s)
        plt.setp(ax_resid.get_xticklabels(), rotation=30, horizontalalignment='right')
        k+=1

    # label the axes
    ax_lines.set_xlabel('Delta Wavelength~(\AA)',fontproperties=font_m, ha='center')
    ax_lines.set_ylabel('Normalized Flux', fontproperties=font_m)

    fig.suptitle('Line Profiles', fontproperties=font_l)
    fig2.suptitle('Residual Distributions', fontproperties=font_l)

    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    gs2.tight_layout(fig2, rect=[0, 0.03, 1, 0.95])
    return fig, fig2


def plot_mcmc_model(spec, spec2, phot, linedata, scale_factor, scale_factor2, phot_dispersion,\
        objname, objname2, outdir, specfile, specfile2,\
        model, covmodel, covmodel2, cont_model, cont_model2, pbs,\
        params, param_names, samples, samples_lnprob,\
        covtype='Matern32', balmer=None, ndraws=21, everyn=1, savefig=False):
    """
    Make all the plots to visualize the full fit of the DA White Dwarf data

    Wraps :py:func:`plot_mcmc_spectrum_fit`,
    :py:func:`plot_mcmc_photometry_res`,
    :py:func:`plot_mcmc_spectrum_nogp_fit`, :py:func:`plot_mcmc_line_fit` and
    :py:func:`corner.corner` and saves all the plots to a combined PDF, and
    optionally individual PDFs.

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum. Must have
        ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    phot : None or :py:class:`numpy.recarray`
        The photometry. Must have
        ``dtype=[('pb', 'str'), ('mag', '<f8'), ('mag_err', '<f8')]``
    linedata : :py:class:`numpy.recarray`
        The observations of the spectrum corresponding to the hydrogen Balmer
        lines. Must have
        ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8'), ('line_mask', 'i4'), ('line_ind', 'i4')]``
    scale_factor : float
        factor by which the flux was scaled for y-axis label
    phot_dispersion : float, optional
        Excess photometric dispersion to add in quadrature with the
        photometric uncertainties ``phot.mag_err``. Use if the errors are
        grossly underestimated. Default is ``0.``
    objname : str
        object name - used to title plots
    outdir : str
        controls where the plot is written out if ``savefig=True``
    specfile : str
        Used in the title, and to set the name of the ``outfile`` if ``savefig=True``
    model : :py:class:`WDmodel.WDmodel.WDmodel` instance
        The DA White Dwarf SED model generator
    covmodel : :py:class:`WDmodel.covariance.WDmodel_CovModel` instance
        The parametrized model for the covariance of the spectrum ``spec``
    cont_model : :py:class:`numpy.recarray`
        The continuuum model. Must have the same structure as ``spec``
        Produced by :py:func:`WDmodel.fit.pre_process_spectrum`
    pbs : dict
        Passband dictionary containing the passbands corresponding to
        ``phot.pb`` and generated by :py:func:`WDmodel.passband.get_pbmodel`.
    params : dict
        dictionary of parameters with keywords ``value``, ``fixed``, ``scale``,
        ``bounds`` for each. Same format as returned from
        :py:func:`WDmodel.io.read_params`
    param_names : array-like
        Ordered list of free parameter names
    samples : array-like
        Samples from the flattened Markov Chain with shape ``(N, len(param_names))``
    samples_lnprob : array-like
        Log Posterior corresponding to ``samples`` from the flattened Markov
        Chain with shape ``(N,)``
    covtype : ``{'Matern32', 'SHO', 'Exp', 'White'}``
        stationary kernel type used to parametrize the covariance in
        :py:class:`WDmodel.covariance.WDmodel_CovModel`
    balmer : array-like, optional
        list of Balmer lines to plot - elements must be in range ``[1, 6]``
        These correspond to the lines defined in
        :py:attr:`WDmodel.WDmodel.WDmodel._lines`. Default is ``range(1, 7)``
    ndraws : int, optional
        Number of draws to make from the Markov Chain to overplot. Higher
        numbers provide a better sense of the uncertainty in the model at the
        cost of speed and a larger, slower to render output plot.
    everyn : int, optional
        If the posterior function was evaluated using only every nth
        observation from the data, this should be specified to visually
        indicate the observations used.
    savefig : bool
        if True, save the individual figures

    Returns
    -------
    model_spec : :py:class:`numpy.recarray`
        The model spectrum. Has
        ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8'), ('norm_flux', '<f8')]``
        and same shape as input ``spec``. The ``norm_flux`` attribute has the
        model flux without the Gaussian process prediction applied.
    SED_model : :py:class:`numpy.recarray`
        The SED model spectrum. Has
        ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
    model_mags : None or :py:class:`numpy.recarray`
        If there is observed photometry, this contains the model magnitudes.
        Has ``dtype=[('pb', 'str'), ('mag', '<f8')]``
    """
    draws     = None
    mag_draws = None

    outfilename = io.get_outfile(outdir, specfile, '_mcmc.pdf')
    with PdfPages(outfilename) as pdf:
        # plot spectrum and model
        fig, draws  =  plot_mcmc_spectrum_fit(spec,False, objname, specfile, scale_factor,\
                model, covmodel, params, param_names, samples,\
                ndraws=ndraws, everyn=everyn)
        if savefig:
            outfile = io.get_outfile(outdir, specfile, '_mcmc_spectrum.pdf')
            fig.savefig(outfile)
        pdf.savefig(fig)

        # plot spectrum and model
        fig, draws2  =  plot_mcmc_spectrum_fit(spec2,True, objname2, specfile2, scale_factor2,\
                model, covmodel2, params, param_names, samples,\
                ndraws=ndraws, everyn=everyn)
        if savefig:
            outfile = io.get_outfile(outdir, specfile, '_mcmc_spectrum2.pdf')
            fig.savefig(outfile)
        pdf.savefig(fig)

        # plot the photometry and residuals if we actually fit it, else skip
        if phot is not None:
            fig, mag_draws = plot_mcmc_photometry_res(objname, phot, phot_dispersion, model, pbs, draws)
            if savefig:
                outfile = io.get_outfile(outdir, specfile, '_mcmc_phot.pdf')
                fig.savefig(outfile)
            pdf.savefig(fig)

        # plot continuum, model and draws without gp
        fig = plot_mcmc_spectrum_nogp_fit(spec, objname, specfile, scale_factor,\
                cont_model, draws, covtype=covtype, everyn=everyn)
        if savefig:
            outfile = io.get_outfile(outdir, specfile, '_mcmc_nogp.pdf')
            fig.savefig(outfile)
        pdf.savefig(fig)

        fig = plot_mcmc_spectrum_nogp_fit(spec2, objname2, specfile2, scale_factor2,\
                cont_model2, draws2, covtype=covtype, everyn=everyn)
        if savefig:
            outfile = io.get_outfile(outdir, specfile, '_mcmc_nogp2.pdf')
            fig.savefig(outfile)
        pdf.savefig(fig)

        # plot lines
        fig, fig2 = plot_mcmc_line_fit(spec, linedata, model, cont_model, draws, balmer=balmer)
        if savefig:
            outfile = io.get_outfile(outdir, specfile, '_mcmc_lines.pdf')
            fig.savefig(outfile)
            outfile = io.get_outfile(outdir, specfile, '_mcmc_resids.pdf')
            fig2.savefig(outfile)
        pdf.savefig(fig)
        pdf.savefig(fig2)

        # plot corner plot
        fig = corner.corner(samples, bins=51, labels=param_names, show_titles=True,quantiles=(0.16,0.84), smooth=1.)
        if savefig:
            outfile = io.get_outfile(outdir, specfile, '_mcmc_corner.pdf')
            fig.savefig(outfile)
        pdf.savefig(fig)
        message = "Wrote output plot file {}".format(outfilename)
        print(message)
        #endwith

    smoothedmod, wres, wres_err, full_mod, best_params = draws[-1]
    res_spec = []
    res_mod  = []
    bestmu = best_params['mu']['value']
    full_mod.flux*=(10**(-0.4*bestmu))
    for draw in draws[:-1]:
        ts, tr, trerr, tm, params = draw
        mu = params['mu']['value']
        tm.flux*=(10**(-0.4*mu))
        res_spec.append((ts+tr-smoothedmod-wres))
        res_mod.append((tm.flux - full_mod.flux))
    res_spec = np.vstack(res_spec)
    res_mod  = np.vstack(res_mod)
    mad_spec = np.median(np.abs(res_spec), axis=0)
    mad_mod  = np.median(np.abs(res_mod), axis=0)
    scaling  = norm.ppf(3/4.)
    sigma_spec = mad_spec/scaling
    sigma_mod  = mad_mod/scaling

    names=str('wave,flux,flux_err,norm_flux')
    model_spec = np.rec.fromarrays((spec.wave, smoothedmod+wres, sigma_spec, smoothedmod), names=names)
    names=str('wave,flux,flux_err')
    SED_model  = np.rec.fromarrays((full_mod.wave, full_mod.flux, sigma_mod), names=names)

    if mag_draws is not None:
        _, model_mags, _ = mag_draws[-1]
    else:
        model_mags = None
    return model_spec, SED_model, model_mags
