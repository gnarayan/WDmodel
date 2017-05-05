import sys
import numpy as np
from . import io
from . import WDmodel
from . import passband
from . import covariance
from . import fit
from . import viz


def main(inargs=None):

    if inargs is None:
        inargs = sys.argv[1:]

    # parse the arguments
    args, pool= io.get_options(inargs)

    specfile  = args.specfile
    spectable = args.spectable
    lamshift  = args.lamshift
    vel       = args.vel
    bluelim, redlim   = args.trimspec
    rebin     = args.rebin
    rescale   = args.rescale
    blotch    = args.blotch

    outdir    = args.outdir
    outroot   = args.outroot

    photfile  = args.photfile
    rvmodel   = args.reddeningmodel
    phot_dispersion = args.phot_dispersion
    excludepb = args.excludepb
    ignorephot= args.ignorephot

    covtype   = args.covtype
    usehodlr  = args.usehodlr
    tol       = args.hodlr_tol
    nleaf     = args.hodlr_nleaf

    samptype  = args.samptype
    ascale    = args.ascale
    ntemps    = args.ntemps
    nwalkers  = args.nwalkers
    nburnin   = args.nburnin
    nprod     = args.nprod
    everyn    = args.everyn
    thin      = args.thin
    redo      = args.redo

    discard   = args.discard

    balmer    = args.balmerlines
    ndraws    = args.ndraws
    savefig   = args.savefig


    ##### SETUP #####


    # set the object name and create output directories
    objname, outdir = io.set_objname_outdir_for_specfile(specfile, outdir=outdir, outroot=outroot, redo=redo)
    print "Writing to outdir {}".format(outdir)

    # parse the parameter keywords in the argparse Namespace into a dictionary
    params = io.get_params_from_argparse(args)

    # get resolution - by default, this is None, since it depends on instrument settings for each spectra
    # we can look it up from a lookup table provided by Tom Matheson for our spectra
    # a custom argument from the command line overrides the lookup
    fwhm = params['fwhm']['value']
    fwhm = io.get_spectrum_resolution(specfile, spectable, fwhm=fwhm)
    params['fwhm']['value'] = fwhm

    # read spectrum
    spec = io.read_spec(specfile)

    # init the model
    model = WDmodel.WDmodel()

    # pre-process spectrum
    out = fit.pre_process_spectrum(spec, bluelim, redlim, model, params,\
            rebin=rebin, lamshift=lamshift, vel=vel, blotch=blotch, rescale=rescale)
    spec, cont_model, linedata, continuumdata, scale_factor, params  = out

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
    outfile = io.get_outfile(outdir, specfile, '_inputs.hdf5', check=True, redo=redo)
    io.write_fit_inputs(spec, phot, cont_model, linedata, continuumdata,\
           rvmodel, covtype, usehodlr, nleaf, tol, phot_dispersion, scale_factor, outfile)

    # get the throughput model
    pbs = passband.get_pbmodel(pbnames, model)


    ##### MINUIT #####


    # to avoid minuit messing up inputs, it can be skipped entirely to force the MCMC to start at a specific position
    if not args.skipminuit:
        # do a quick fit to refine the input params
        migrad_params  = fit.quick_fit_spec_model(spec, model, params, rvmodel=rvmodel)

        # save the minuit fit result - this will not be perfect, but if it's bad, refine starting position
        viz.plot_minuit_spectrum_fit(spec, objname, outdir, specfile, scale_factor,\
            model, migrad_params, rvmodel=rvmodel, save=True)
    else:
        # we didn't run minuit, so we'll assume the user intended to start us at some specific position
        migrad_params = io.copy_params(params)

    # init a covariance model instance that's used to model the residuals
    # between the systematic residuals between data and model
    errscale = np.median(spec.flux_err)
    covmodel = covariance.WDmodel_CovModel(errscale, covtype, nleaf, tol, usehodlr)
    if covtype == 'White':
        migrad_params['fsig']['value'] = 0.
        migrad_params['fsig']['fixed'] = True
        migrad_params['tau']['fixed']  = True

    # If we don't have a user supplied initial guess of mu, get a guess
    migrad_params = fit.hyper_param_guess(spec, phot, model, pbs, migrad_params, rvmodel=rvmodel)

    # write out the migrad params - note that if you skipminuit, you are expected to provide the dl value
    # if skipmcmc is set, you can now run the code with MPI
    outfile = io.get_outfile(outdir, specfile, '_params.json')
    io.write_params(migrad_params, outfile)


    ##### MCMC #####


    # skipmcmc can be run to just prepare the inputs
    if not args.skipmcmc:

        # do the fit
        result = fit.fit_model(spec, phot, model, covmodel, pbs, migrad_params,\
                    objname, outdir, specfile,\
                    rvmodel=rvmodel, phot_dispersion=phot_dispersion,\
                    samptype=samptype, ascale=ascale,\
                    ntemps=ntemps, nwalkers=nwalkers, nburnin=nburnin, nprod=nprod,\
                    thin=thin, everyn=everyn,\
                    redo=redo,\
                    pool=pool)

        param_names, samples, samples_lnprob = result
        mcmc_params = io.copy_params(migrad_params)

        # parse the samples in the chain and get the result
        result = fit.get_fit_params_from_samples(param_names, samples, samples_lnprob, mcmc_params,\
                        ntemps=ntemps, nwalkers=nwalkers, nprod=nprod, discard=discard)
        mcmc_params, in_samp, in_lnprob = result

        # write the result to a file
        outfile = io.get_outfile(outdir, specfile, '_result.json')
        io.write_params(mcmc_params, outfile)

        # plot the MCMC output
        plot_out = viz.plot_mcmc_model(spec, phot, linedata,\
                    scale_factor, phot_dispersion,\
                    objname, outdir, specfile,\
                    model, covmodel, cont_model, pbs,\
                    mcmc_params, param_names, in_samp, in_lnprob,\
                    covtype=covtype, rvmodel=rvmodel, balmer=balmer,\
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
