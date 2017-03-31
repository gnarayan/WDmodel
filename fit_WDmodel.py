#!/usr/bin/env python
import sys
import argparse
from emcee.utils import MPIPool
import numpy as np
import WDmodel
import WDmodel.io
import WDmodel.fit
import WDmodel.viz
import WDmodel.pbmodel
import WDmodel.covmodel


def get_options(args=None):
    """
    Get command line options for the WDmodel fitter
    """

    # create a config parser that will take a single option - param file
    conf_parser = argparse.ArgumentParser(add_help=False)

    # config options - this lets you specify a parameter configuration file,
    # set the default parameters values from it, and override them later as needed
    # if not supplied, it'll use the default parameter file included in the package
    conf_parser.add_argument("--param_file", required=False, default=None,\
            help="Specify parameter config JSON file")

    args, remaining_argv = conf_parser.parse_known_args(args)
    params = WDmodel.io.read_params(param_file=args.param_file)

    # now that we've gotten the param_file and the params (either custom, or default), create the parse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                    parents=[conf_parser],\
                    description=__doc__,\
                    epilog="If running fit_WDmodel.py with MPI using mpirun,\
                    mpi must be the first argument, and -np must be at least 2.")

    # create a couple of custom types to use with the parser
    # this type exists to make a quasi bool type instead of store_false/store_true
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    # this type exists for parameters where we can't or don't really want the
    # user to guess a default value - better to make a guess internally than
    # have a bad  starting point
    def NoneOrFloat(v):
        if v.lower() in ("none", "null", "nan"):
            return None
        else:
            return float(v)

    parser.register('type','bool',str2bool)
    parser.register('type','NoneOrFloat',NoneOrFloat)

    # spectrum options
    spectrum = parser.add_argument_group('spectrum', 'Spectrum options')
    spectrum.add_argument('--specfile', required=True, \
            help="Specify spectrum to fit")
    spectrum.add_argument('--spectable', required=False,  default="data/spectroscopy/spectable_resolution.dat",\
            help="Specify file containing a fwhm lookup table for specfile")
    spectrum.add_argument('--lamshift', required=False, type=float, default=0.,\
            help="Specify a flat wavelength shift in Angstrom to fix  slit centering errors")
    spectrum.add_argument('--vel', required=False, type=float, default=0.,\
            help="Specify a velocity shift in kmps to apply to the spectrum")
    spectrum.add_argument('--trimspec', required=False, nargs=2, default=(None,None),
                type='NoneOrFloat', metavar=("BLUELIM", "REDLIM"), help="Trim spectrum to wavelength range")
    spectrum.add_argument('--rebin',  required=False, type=int, default=1,\
            help="Rebin the spectrum by an integer factor. Output wavelengths remain uncorrelated.")
    spectrum.add_argument('--rescale',  required=False, action="store_true", default=False,\
            help="Rescale the spectrum to make the noise ~1. Changes the value, bounds, scale on dl also")
    spectrum.add_argument('--blotch', required=False, action='store_true',\
            default=False, help="Blotch the spectrum to remove gaps/cosmic rays before fitting?")

    # photometry options
    reddeninglaws = ('od94', 'ccm89', 'gcc09', 'f99', 'fm07', 'wd01', 'd03')
    phot = parser.add_argument_group('photometry', 'Photometry options')
    phot.add_argument('--photfile', required=False,  default="data/photometry/WDphot_C22.dat",\
            help="Specify file containing photometry lookup table for objects")
    phot.add_argument('--reddeningmodel', required=False, choices=reddeninglaws, default='od94',\
            help="Specify functional form of reddening law" )
    phot.add_argument('--phot_dispersion', required=False, type=float, default=0.003,\
            help="Specify a flat photometric dispersion error in mag to add in quadrature to the measurement errors")
    phot.add_argument('--excludepb', nargs='+',\
            help="Specify passbands to exclude" )
    phot.add_argument('--ignorephot',  required=False, action="store_true", default=False,\
            help="Ignores missing photometry and does the fit with just the spectrum")

    # fitting options
    model = parser.add_argument_group('model',\
            'Model options. Modify using --param_file or CL. CL overrides. Caveat emptor.')
    for param in params:
        # we can't reasonably expect a user supplied guess or a static value to
        # work for some parameters. Allow None for these, and we'll determine a
        # good starting guess from the data. Note that we can actually just get
        # a starting guess for FWHM, but it's easier to use a lookup table.
        if param in ('fwhm','dl','mu'):
            dtype = 'NoneOrFloat'
        else:
            dtype = float

        model.add_argument('--{}'.format(param), required=False, type=dtype, default=params[param]['value'],\
                help="Specify param {} value".format(param))
        model.add_argument('--{}_fix'.format(param), required=False, default=params[param]['fixed'], type="bool",\
                help="Specify if param {} is fixed".format(param))
        model.add_argument('--{}_scale'.format(param), required=False, type=float, default=params[param]['scale'],\
                help="Specify param {} scale/step size".format(param))
        model.add_argument('--{}_bounds'.format(param), required=False, nargs=2, default=params[param]["bounds"],
                type=float, metavar=("LOWERLIM", "UPPERLIM"), help="Specify param {} bounds".format(param))

    # covariance model options
    covmodel = parser.add_argument_group('covariance model', 'Covariance model options - changes what fsig and tau mean')
    covmodel.add_argument('--covtype', required=False, choices=('White','ExpSquared','Matern32','Matern52','Exp'),\
                default='ExpSquared', help='Specify a parametric form for the covariance function to model the spectrum')
    covmodel.add_argument('--usebasic',  required=False, action="store_true", default=False,\
            help="Use the BasicSolver over the HODLR solver i.e. you want to support global warming.")
    covmodel.add_argument('--solver_tol', required=False, type=float, default=1e-12,\
            help="Specify tolerance for HODLR solver")
    covmodel.add_argument('--solver_nleaf',  required=False, type=int, default=100,\
            help="Specify size of smallest matrix blocks before HODLR solves system directly")

    # MCMC config options
    mcmc = parser.add_argument_group('mcmc', 'MCMC options')
    mcmc.add_argument('--skipminuit',  required=False, action="store_true", default=False,\
            help="Skip Minuit fit - make sure to specify dl guess")
    mcmc.add_argument('--skipmcmc',  required=False, action="store_true", default=False,\
            help="Skip MCMC - if you skip both minuit and MCMC, simply prepares files")
    mcmc.add_argument('--ascale', required=False, type=float, default=2.0,\
            help="Specify proposal scale for MCMC")
    mcmc.add_argument('--nwalkers',  required=False, type=int, default=300,\
            help="Specify number of walkers to use (0 disables MCMC)")
    mcmc.add_argument('--nburnin',  required=False, type=int, default=200,\
            help="Specify number of steps for burn-in")
    mcmc.add_argument('--nprod',  required=False, type=int, default=2000,\
            help="Specify number of steps for production")
    mcmc.add_argument('--everyn',  required=False, type=int, default=1,\
            help="Use only every nth point in data for computing likelihood - useful for testing.")
    mcmc.add_argument('--discard',  required=False, type=float, default=5,\
            help="Specify percentage of steps to be discarded")

    # visualization options
    viz = parser.add_argument_group('viz', 'Visualization options')
    viz.add_argument('-b', '--balmerlines', nargs='+', type=int, default=range(1,7,1),\
            help="Specify Balmer lines to visualize [1:7]")
    viz.add_argument('--ndraws', required=False, type=int, default=21,\
            help="Specify number of draws from posterior to overplot for model")
    viz.add_argument('--savefig',  required=False, action="store_true", default=False,\
            help="Save individual plots")

    # output options
    output = parser.add_argument_group('output', 'Output options')
    output.add_argument('--outroot', required=False,
            help="Specify a custom output root directory. Directories go under outroot/objname/subdir.")
    output.add_argument('-o', '--outdir', required=False,\
            help="Specify a custom output directory. Overrides outroot.")
    output.add_argument('--redo',  required=False, action="store_true", default=False,\
            help="Clobber existing fits")

    args = parser.parse_args(args=remaining_argv)

    # some sanity checking for option values
    balmer = args.balmerlines
    try:
        balmer = np.atleast_1d(balmer).astype('int')
        if np.any((balmer < 1) | (balmer > 6)):
            raise ValueError
    except (TypeError, ValueError):
        message = 'Invalid balmer line value - must be in range [1,6]'
        raise ValueError(message)

    if args.rebin < 1:
        message = 'Rebin must be integer GE 1. Note that 1 does nothing. ({:g})'.format(args.rebin)
        raise ValueError(message)

    if args.phot_dispersion < 0.:
        message = 'Photometric dispersion must be GE 0. ({:g})'.format(args.phot_dispersion)
        raise ValueError(message)

    if args.solver_tol <= 0:
        message = 'HODLR Solver tolerance must be greater than 0. ({:g})'.format(args.solver_tol)
        raise ValueError(message)

    if args.solver_nleaf <= 0:
        message = 'HODLR Solver min matrix block size must be greater than zero ({})'.format(args.solver_nleaf)
        raise ValueError(message)

    if args.nwalkers <= 0:
        message = 'Number of walkers must be greater than zero for MCMC ({})'.format(args.nwalkers)
        raise ValueError(message)

    if args.nwalkers%2 != 0:
        message = 'Number of walkers must be even ({})'.format(args.nwalkers)
        raise ValueError(message)

    if args.nburnin <= 0:
        message = 'Number of burnin steps must be greater than zero ({})'.format(args.nburnin)
        raise ValueError(message)

    if args.nprod <= 0:
        message = 'Number of production steps must be greater than zero ({})'.format(args.nprod)
        raise ValueError(message)

    if not (0 <= args.discard < 100):
        message = 'Discard must be a percentage (0-100) ({})'.format(args.discard)
        raise ValueError(message)

    if args.everyn < 1:
        message = 'EveryN must be integer GE 1. Note that 1 does nothing. ({:g})'.format(args.everyn)
        raise ValueError(message)

    return args


#**************************************************************************************************************


def main(inargs=None, pool=None):

    # Wait for instructions from the master process if we are running MPI
    if pool is not None:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    if inargs is None:
        inargs = sys.argv[1:]

    # parse the arguments
    args   = get_options(inargs)

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
    usebasic  = args.usebasic
    tol       = args.solver_tol
    nleaf     = args.solver_nleaf

    ascale    = args.ascale
    nwalkers  = args.nwalkers
    nburnin   = args.nburnin
    nprod     = args.nprod
    everyn    = args.everyn
    redo      = args.redo

    discard   = args.discard

    balmer    = args.balmerlines
    ndraws    = args.ndraws
    savefig   = args.savefig


    ##### SETUP #####


    # set the object name and create output directories
    objname, outdir = WDmodel.io.set_objname_outdir_for_specfile(specfile, outdir=outdir, outroot=outroot)
    print "Writing to outdir {}".format(outdir)

    # parse the parameter keywords in the argparse Namespace into a dictionary
    params = WDmodel.io.get_params_from_argparse(args)

    # get resolution - by default, this is None, since it depends on instrument settings for each spectra
    # we can look it up from a lookup table provided by Tom Matheson for our spectra
    # a custom argument from the command line overrides the lookup
    fwhm = params['fwhm']['value']
    fwhm = WDmodel.io.get_spectrum_resolution(specfile, spectable, fwhm=fwhm)
    params['fwhm']['value'] = fwhm

    # read spectrum
    spec = WDmodel.io.read_spec(specfile)

    # init the model
    model = WDmodel.WDmodel()

    # pre-process spectrum
    out = WDmodel.fit.pre_process_spectrum(spec, bluelim, redlim, model, params,\
            rebin=rebin, lamshift=lamshift, vel=vel, blotch=blotch, rescale=rescale)
    spec, cont_model, linedata, continuumdata, scale_factor, params  = out

    # get photometry
    if not ignorephot:
        phot = WDmodel.io.get_phot_for_obj(objname, photfile)
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
    outfile = WDmodel.io.get_outfile(outdir, specfile, '_inputs.hdf5', check=True, redo=redo)
    WDmodel.io.write_fit_inputs(spec, phot, cont_model, linedata, continuumdata,\
           rvmodel, covtype, usebasic, nleaf, tol, phot_dispersion, scale_factor, outfile)

    # get the throughput model
    pbs = WDmodel.pbmodel.get_pbmodel(pbnames, model)


    ##### MINUIT #####


    # to avoid minuit messing up inputs, it can be skipped entirely to force the MCMC to start at a specific position
    if not args.skipminuit:
        # do a quick fit to refine the input params
        migrad_params  = WDmodel.fit.quick_fit_spec_model(spec, model, params, rvmodel=rvmodel)

        # save the minuit fit result - this will not be perfect, but if it's bad, refine starting position
        WDmodel.viz.plot_minuit_spectrum_fit(spec, objname, outdir, specfile, scale_factor,\
            model, migrad_params, rvmodel=rvmodel, save=True)
    else:
        # we didn't run minuit, so we'll assume the user intended to start us at some specific position
        migrad_params = WDmodel.io.copy_params(params)

    # init a covariance model instance that's used to model the residuals
    # between the systematic residuals between data and model
    errscale = np.median(spec.flux_err)
    covmodel = WDmodel.covmodel.WDmodel_CovModel(errscale, covtype, nleaf, tol, usebasic)
    if covtype == 'White':
        migrad_params['tau']['fixed'] = True

    # If we don't have a user supplied initial guess of mu, get a guess
    migrad_params = WDmodel.fit.hyper_param_guess(spec, phot, model, pbs, migrad_params, rvmodel=rvmodel)

    # write out the migrad params - note that if you skipminuit, you are expected to provide the dl value
    # if skipmcmc is set, you can now run the code with MPI
    outfile = WDmodel.io.get_outfile(outdir, specfile, '_params.json')
    WDmodel.io.write_params(migrad_params, outfile)


    ##### MCMC #####


    # skipmcmc can be run to just prepare the inputs
    if not args.skipmcmc:

        # do the fit
        result = WDmodel.fit.fit_model(spec, phot, model, covmodel, pbs, migrad_params,\
                    objname, outdir, specfile,\
                    rvmodel=rvmodel, phot_dispersion=phot_dispersion,\
                    ascale=ascale, nwalkers=nwalkers, nburnin=nburnin, nprod=nprod, everyn=everyn,\
                    redo=redo,\
                    pool=pool)

        param_names, samples, samples_lnprob = result
        mcmc_params = WDmodel.io.copy_params(migrad_params)

        # parse the samples in the chain and get the result
        result = WDmodel.fit.get_fit_params_from_samples(param_names, samples, samples_lnprob, mcmc_params,\
                        nwalkers=nwalkers, nprod=nprod, discard=discard)
        mcmc_params, in_samp, in_lnprob = result

        # write the result to a file
        outfile = WDmodel.io.get_outfile(outdir, specfile, '_result.json')
        WDmodel.io.write_params(mcmc_params, outfile)

        # plot the MCMC output
        plot_out = WDmodel.viz.plot_mcmc_model(spec, phot, linedata,\
                    scale_factor, phot_dispersion,\
                    objname, outdir, specfile,\
                    model, covmodel, cont_model, pbs,\
                    mcmc_params, param_names, in_samp, in_lnprob,\
                    covtype=covtype, rvmodel=rvmodel, balmer=balmer, ndraws=ndraws, savefig=savefig)
        model_spec, full_mod, model_mags = plot_out

        spec_model_file = WDmodel.io.get_outfile(outdir, specfile, '_spec_model.dat')
        WDmodel.io.write_spectrum_model(spec, model_spec, spec_model_file)

        full_model_file = WDmodel.io.get_outfile(outdir, specfile, '_full_model.hdf5')
        WDmodel.io.write_full_model(full_mod, mcmc_params['mu']['value'], full_model_file)

        if phot is not None:
            phot_model_file = WDmodel.io.get_outfile(outdir, specfile, '_phot_model.dat')
            WDmodel.io.write_phot_model(phot, model_mags, phot_model_file)

    return


#**************************************************************************************************************


if __name__=='__main__':
    mpi = False
    startmpi =  str(sys.argv[1]).lower()
    if startmpi.startswith('mpi'):
        # if the first argument is "mpi" then start a pool
        # if it is mpil, additionally enable load balancing
        mpi = True
        loadbalance = False
        if startmpi == 'mpil':
            loadbalance = True
        pool = MPIPool(loadbalance=loadbalance)
        inargs = sys.argv[2:]
    else:
        # run single threaded
        pool = None
        inargs = sys.argv[1:]

    main(inargs, pool)
