# -*- coding: UTF-8 -*-
"""
The WDmodel package is designed to infer the SED of DA white dwarfs given
spectra and photometry. This main module wraps all the other modules, and their
classes and methods to implement the alogrithm.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import sys
import mpi4py
import numpy as np
from . import io
from . import WDmodel
from . import passband
from . import covariance
from . import fit
from . import viz


sys_excepthook = sys.excepthook
def mpi_excepthook(excepttype, exceptvalue, traceback):
    """
    Overload :py:func:`sys.excepthook` when using :py:class:`mpi4py.MPI` to
    terminate all MPI processes when an Exception is raised.
    """
    sys_excepthook(excepttype, exceptvalue, traceback)
    mpi4py.MPI.COMM_WORLD.Abort(1)


def main(inargs=None):
    """
    Entry point for the :py:mod:`WDmodel` fitter package.

    Parameters
    ----------
    inargs : dict, optional
        Input arguments to configure the fit. If not specified
        :py:data:`sys.argv` is used. inargs must be parseable by
        :py:func:`WDmodel.io.get_options`.

    Raises
    ------
    RuntimeError
        If user attempts to resume the fit without having run it first

    Notes
    -----
    The package is structured into several modules and classes

    ================================================= ===================
                         Module                         Model Component
    ================================================= ===================
    :py:mod:`WDmodel.io`                              I/O methods
    :py:class:`WDmodel.WDmodel.WDmodel`               SED generator
    :py:mod:`WDmodel.passband`                        Throughput model
    :py:class:`WDmodel.covariance.WDmodel_CovModel`   Noise model
    :py:class:`WDmodel.likelihood.WDmodel_Likelihood` Likelihood function
    :py:class:`WDmodel.likelihood.WDmodel_Posterior`  Posterior function
    :py:mod:`WDmodel.fit`                             "Fitting" methods
    :py:mod:`WDmodel.viz`                             Viz methods
    ================================================= ===================

    This method implements our algorithm to infer the DA White Dwarf properties
    and construct the SED model given the data using the methods and classes
    listed above. Once the data is read, the model is configured, and the
    liklihood and posterior functions constructed, the fitter methods evaluate
    the model parameters given the data, using the samplers in :py:mod:`emcee`.
    :py:mod:`WDmodel.mossampler` provides an overloaded
    :py:class:`emcee.PTSampler` with a more reliable auto-correlation estimate.
    Finally, the result is output along with various plots.
    """
    comm = mpi4py.MPI.COMM_WORLD
    size = comm.Get_size()
    if size > 1:
        # force all MPI processes to terminate if we are running with --mpi and an exception is raised
        sys.excepthook = mpi_excepthook

    if inargs is None:
        inargs = sys.argv[1:]

    # parse the arguments
    args, pool= io.get_options(inargs, comm)

    specfile  = args.specfile
    spectable = args.spectable
    lamshift  = args.lamshift
    vel       = args.vel
    bluelim, redlim   = args.trimspec
    rebin     = args.rebin
    rescale   = args.rescale
    blotch    = args.blotch

    specfile2  = args.specfile2
    spectable2 = args.spectable2
    lamshift2  = args.lamshift2
    vel2       = args.vel2
    bluelim2, redlim2   = args.trimspec2
    rebin2     = args.rebin2
    rescale2   = args.rescale2
    blotch2    = args.blotch2

    outdir    = args.outdir
    outroot   = args.outroot

    photfile  = args.photfile
    rvmodel   = args.reddeningmodel
    phot_dispersion = args.phot_dispersion
    pbfile    = args.pbfile
    excludepb = args.excludepb
    ignorephot= args.ignorephot

    covtype   = args.covtype
    coveps    = args.coveps

    samptype  = args.samptype
    ascale    = args.ascale
    ntemps    = args.ntemps
    nwalkers  = args.nwalkers
    nburnin   = args.nburnin
    nprod     = args.nprod
    everyn    = args.everyn
    thin      = args.thin
    redo      = args.redo
    resume    = args.resume

    discard   = args.discard

    balmer    = args.balmerlines
    ndraws    = args.ndraws
    savefig   = args.savefig


    ##### SETUP #####


    # set the object name and create output directories
    objname, outdir1, bs1 = io.set_objname_outdir_for_specfile(specfile, outdir=outdir, outroot=outroot,\
                        redo=redo, resume=resume)
    objname2, outdir2, bs2 = io.set_objname_outdir_for_specfile(specfile2, outdir=outdir, outroot=outroot,\
                        redo=redo, resume=resume)
    # outdir = outdir1+'_'+objname2
    outdir=outdir1
    message = "Writing to outdir {}".format(outdir)
    print(message)

    # init the model
    model = WDmodel.WDmodel(rvmodel=rvmodel)

    if not resume:
        # parse the parameter keywords in the argparse Namespace into a dictionary
        params = io.get_params_from_argparse(args)

        # get resolution - by default, this is None, since it depends on instrument settings for each spectra
        # we can look it up from a lookup table provided by Tom Matheson for our spectra
        # a custom argument from the command line overrides the lookup
        fwhm = params['fwhm']['value']
        fwhm, lamshift = io.get_spectrum_resolution(specfile, spectable, fwhm=fwhm, lamshift=lamshift)
        params['fwhm']['value'] = fwhm

        fwhm2 = params['fwhm2']['value']
        fwhm2, lamshift2 = io.get_spectrum_resolution(specfile2, spectable2, fwhm=fwhm2, lamshift=lamshift2)
        params['fwhm2']['value'] = fwhm2

        # read spectrum
        spec = io.read_spec(specfile)
        spec2 = io.read_spec(specfile2)


        # pre-process spectrum
        out = fit.pre_process_spectrum(spec, bluelim, redlim, model, params,\
                rebin=rebin, lamshift=lamshift, vel=vel, blotch=blotch, rescale=rescale)
        spec, cont_model, linedata, continuumdata, scale_factor, params  = out

        out2 = fit.pre_process_spectrum(spec2, bluelim2, redlim2, model, params,\
                rebin=rebin2, lamshift=lamshift2, vel=vel2, blotch=blotch2, rescale=rescale2)
        spec2, cont_model2, linedata2, continuumdata2, scale_factor2, not_params  = out2

        # get photometry
        if not ignorephot:
            phot = io.get_phot_for_obj(objname, photfile)
        else:
            params['mu']['value'] = 0.
            params['mu']['fixed'] = True
            phot = None

        # exclude passbands that we want excluded
        pbnames = []
        if phot is not None:
            pbnames = np.unique(phot.pb)
            if excludepb is not None:
                pbnames = list(set(pbnames) - set(excludepb))

            # filter the photometry recarray to use only the passbands we want
            useind = [x for x, pb in enumerate(phot.pb) if pb in pbnames]
            useind = np.array(useind)
            phot = phot.take(useind)

            # set the pbnames from the trimmed photometry recarray to preserve order
            pbnames = list(phot.pb)

        # if we cut out out all the passbands, force mu to be fixed
        if len(pbnames) == 0:
            params['mu']['value'] = 0.
            params['mu']['fixed'] = True
            phot = None

        # save the inputs to the fitter
        outfile = io.get_outfile(outdir, specfile, '_inputs.hdf5', check=True, redo=redo, resume=resume)
        io.write_fit_inputs(spec, phot, cont_model, linedata, continuumdata,\
               rvmodel, covtype, coveps, phot_dispersion, scale_factor, outfile)
    else:
        outfile = io.get_outfile(outdir, specfile, '_inputs.hdf5', check=False, redo=redo, resume=resume)
        try:
            spec, cont_model, linedata, continuumdata, phot, fit_config = io.read_fit_inputs(outfile)
        except IOError as e:
            message = '{}\nMust run fit to generate inputs before attempting to resume'.format(e)
            raise RuntimeError(message)
        rvmodel  = fit_config['rvmodel']
        covtype  = fit_config['covtype']
        coveps   = fit_config['coveps']
        scale_factor    = fit_config['scale_factor']
        phot_dispersion = fit_config['phot_dispersion']
        if phot is not None:
            pbnames = list(phot.pb)
        else:
            pbnames = []

    # get the throughput model
    pbs = passband.get_pbmodel(pbnames, model, pbfile=pbfile)


    ##### MINUIT #####


    outfile = io.get_outfile(outdir, specfile, '_params.json', check=True, redo=redo, resume=resume)
    if not resume:
        # to avoid minuit messing up inputs, it can be skipped entirely to force the MCMC to start at a specific position
        if not args.skipminuit:
            # do a quick fit to refine the input params
            migrad_params  = fit.quick_fit_spec_model(spec, model, params)

            # save the minuit fit result - this will not be perfect, but if it's bad, refine starting position
            viz.plot_minuit_spectrum_fit(spec, objname, outdir, specfile, scale_factor,\
                model, migrad_params, save=True)
        else:
            # we didn't run minuit, so we'll assume the user intended to start us at some specific position
            migrad_params = io.copy_params(params)

        if covtype == 'White':
            migrad_params['fsig']['value'] = 0.
            migrad_params['fsig']['fixed'] = True
            migrad_params['tau']['fixed']  = True

        # If we don't have a user supplied initial guess of mu, get a guess
        migrad_params = fit.hyper_param_guess(spec, phot, model, pbs, migrad_params)

        # write out the migrad params - note that if you skipminuit, you are expected to provide the dl value
        # if skipmcmc is set, you can now run the code with MPI
        io.write_params(migrad_params, outfile)
    else:
        try:
            migrad_params = io.read_params(outfile)
        except (OSError,IOError) as e:
            message = '{}\nMust run fit to generate inputs before attempting to resume'.format(e)
            raise RuntimeError(message)

    # init a covariance model instance that's used to model the residuals
    # between the systematic residuals between data and model
    errscale = np.median(spec.flux_err)
    covmodel = covariance.WDmodel_CovModel(errscale, covtype, coveps)
    errscale2 = np.median(spec2.flux_err)
    covmodel2 = covariance.WDmodel_CovModel(errscale2, covtype, coveps)

    ##### MCMC #####


    # skipmcmc can be run to just prepare the inputs
    if not args.skipmcmc:

        # do the fit
        result = fit.fit_model(spec,spec2, phot, model, covmodel,covmodel2 ,pbs, migrad_params,\
                    objname, outdir, specfile,\
                    phot_dispersion=phot_dispersion,\
                    samptype=samptype, ascale=ascale,\
                    ntemps=ntemps, nwalkers=nwalkers, nburnin=nburnin, nprod=nprod,\
                    thin=thin, everyn=everyn,\
                    redo=redo, resume=resume,\
                    pool=pool)

        param_names, samples, samples_lnprob, everyn, shape = result
        ntemps, nwalkers, nprod, nparam = shape
        mcmc_params = io.copy_params(migrad_params)

        # parse the samples in the chain and get the result
        result = fit.get_fit_params_from_samples(param_names, samples, samples_lnprob, mcmc_params,\
                        ntemps=ntemps, nwalkers=nwalkers, nprod=nprod, discard=discard)
        mcmc_params, in_samp, in_lnprob = result

        # write the result to a file
        outfile = io.get_outfile(outdir, specfile, '_result.json')
        io.write_params(mcmc_params, outfile)

        # plot the MCMC output
        plot_out = viz.plot_mcmc_model(spec, spec2, phot, linedata,\
                    scale_factor, scale_factor2, phot_dispersion,\
                    objname, objname2, outdir, specfile, specfile2,\
                    model, covmodel, covmodel2, cont_model, cont_model2, pbs,\
                    mcmc_params, param_names, in_samp, in_lnprob,\
                    covtype=covtype, balmer=balmer,\
                    ndraws=ndraws, everyn=everyn, savefig=savefig)
        model_spec, full_mod, model_mags = plot_out

        spec_model_file = io.get_outfile(outdir, specfile, '_spec_model.dat')
        io.write_spectrum_model(spec, model_spec, spec_model_file)

        full_model_file = io.get_outfile(outdir, specfile, '_full_model.hdf5')
        io.write_full_model(full_mod, full_model_file)

        if phot is not None:
            phot_model_file = io.get_outfile(outdir, specfile, '_phot_model.dat')
            io.write_phot_model(phot, model_mags, phot_model_file)

    return
