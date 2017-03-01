#!/usr/bin/env python
import sys
import warnings
warnings.simplefilter('once')
import argparse
from emcee.utils import MPIPool
import numpy as np
import WDmodel
import WDmodel.io
import WDmodel.fit
import WDmodel.viz


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
    spectrum.add_argument('--trimspec', required=False, nargs=2, default=(None,None), 
                type='NoneOrFloat', metavar=("BLUELIM", "REDLIM"), help="Trim spectrum to wavelength range")
    spectrum.add_argument('--blotch', required=False, action='store_true',\
            default=False, help="Blotch the spectrum to remove gaps/cosmic rays before fitting?")

    # photometry options
    phot = parser.add_argument_group('photometry', 'Photometry options')
    phot.add_argument('--photfile', required=False,  default="data/WDphot_C22.dat",\
            help="Specify file containing photometry lookup table for objects")
    phot.add_argument('--reddeningmodel', required=False, default='od94',\
            help="Specify functional form of reddening law" )
    phot.add_argument('--excludepb', nargs='+',\
            help="Specify passbands to exclude" )
    phot.add_argument('--ignorephot',  required=False, action="store_true", default=False,\
            help="Ignores missing photometry and does the fit with just the spectrum")

    # fitting options
    model = parser.add_argument_group('model',\
            'Model options. Modify using --param_file or CL. CL overrides. Caveat emptor.')
    for param in params:
        if param in ('fwhm','dl'):
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

    # MCMC config options
    mcmc = parser.add_argument_group('mcmc', 'MCMC options')
    mcmc.add_argument('--skipminuit',  required=False, action="store_true", default=False,\
            help="Skip Minuit fit - make sure to specify dl guess")
    mcmc.add_argument('--skipmcmc',  required=False, action="store_true", default=False,\
            help="Skip MCMC - if you skip both minuit and MCMC, simply prepares files")
    mcmc.add_argument('--ascale', required=False, type=float, default=2.0,\
            help="Specify proposal scale for MCMC") 
    mcmc.add_argument('--nwalkers',  required=False, type=int, default=200,\
            help="Specify number of walkers to use (0 disables MCMC)")
    mcmc.add_argument('--nburnin',  required=False, type=int, default=50,\
            help="Specify number of steps for burn-in")
    mcmc.add_argument('--nprod',  required=False, type=int, default=1000,\
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
    output.add_argument('-o', '--outdir', required=False,\
            help="Specify a custom output directory. Default is CWD+objname/ subdir")
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

    reddeninglaws = ('od94', 'ccm89', 'gcc09', 'f99', 'fm07', 'wd01', 'd03')
    if not args.reddeningmodel in reddeninglaws:
        message = 'That reddening law is not known (%s)'%' '.join(reddeninglaws) 
        raise ValueError(message)

    if args.nwalkers < 0:
        message = 'Number of walkers must be greater than zero for MCMC'
        raise ValueError(message)

    if args.nburnin < 0:
        message = 'Number of burnin steps must be greater than zero'
        raise ValueError(message)

    if args.nprod <= 0:
        message = 'Number of walkers must be greater than zero'
        raise ValueError(message)

    if not (0 <= args.discard < 100):
        message = 'Discard must be a percentage (0-100)'
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
    bluelim, redlim   = args.trimspec
    blotch    = args.blotch

    outdir    = args.outdir

    photfile  = args.photfile
    rvmodel   = args.reddeningmodel
    excludepb = args.excludepb
    ignorephot= args.ignorephot
    
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
    objname, outdir = WDmodel.io.set_objname_outdir_for_specfile(specfile, outdir=outdir)

    # parse the parameter keywords in the argparse Namespace into a dictionary
    params = WDmodel.io.get_params_from_argparse(args)

    # get resolution - by default, this is None, since it depends on instrument settings for each spectra
    # we can look it up from a lookup table provided by Tom Matheson for our spectra
    # a custom argument from the command line overrides the lookup
    fwhm = params['fwhm']['value']
    fwhm = WDmodel.io.get_spectrum_resolution(specfile, fwhm=fwhm)
    params['fwhm']['value'] = fwhm

    # read spectrum
    spec = WDmodel.io.read_spec(specfile)

    # pre-process spectrum
    out = WDmodel.fit.pre_process_spectrum(spec, bluelim, redlim, blotch=blotch)
    spec, cont_model, linedata, continuumdata = out

    # get photometry 
    # TODO - something to fix photometry normalization if we get None
    if not ignorephot:
        phot = WDmodel.io.get_phot_for_obj(objname, photfile)
    else:
        phot = None

    # save the inputs to the fitter
    outfile = WDmodel.io.get_outfile(outdir, specfile, '_inputs.hdf5')
    WDmodel.io.write_fit_inputs(spec, phot, cont_model, linedata, continuumdata,\
            outfile, redo=redo)

    # init the model, and determine the coarse normalization to match the spectrum
    model = WDmodel.WDmodel()


    ##### MINUIT #####


    # to avoid minuit messing up inputs, it can be skipped entirely to force the MCMC to start at a specific position
    if not args.skipminuit:
        # do a quick fit to refine the input params
        migrad_params  = WDmodel.fit.quick_fit_spec_model(spec, model, params, rvmodel=rvmodel)

        # save the minuit fit result - this will not be perfect, but if it's bad, refine starting position
        WDmodel.viz.plot_minuit_spectrum_fit(spec, objname, outdir, specfile,\
            model, migrad_params, rvmodel=rvmodel, save=True)
    else:
        # we didn't run minuit, so we'll assume the user intended to start us at some specific position
        migrad_params = WDmodel.io.copy_params(params)

    # write out the migrad params - note that if you skipminuit, you are expected to provide the dl value
    # if skipmcmc is set, you can now run the code with MPI
    outfile = WDmodel.io.get_outfile(outdir, specfile, '_params.json')
    WDmodel.io.write_params(migrad_params, outfile)


    ##### MCMC #####


    # skipmcmc can be run to just prepare the inputs 
    if not args.skipmcmc:

        # exclude passbands that we want excluded 
        if phot is not None:
            pbnames = np.unique(phot.pb) 
            if excludepb is not None:
                pbnames = list(set(pbnames) - set(excludepb))
        else:
            pbnames = []

        # get the throughput model 
        pbmodel = WDmodel.io.get_pbmodel(pbnames)

        # fit the spectrum
        if pool is None:
            result = WDmodel.fit.fit_model(spec, phot, model, pbmodel, migrad_params,\
                        objname, outdir, specfile,\
                        rvmodel=rvmodel,\
                        ascale=ascale, nwalkers=nwalkers, nburnin=nburnin, nprod=nprod, everyn=everyn,\
                        redo=redo)
        else:
            result = WDmodel.fit.mpi_fit_model(spec, phot, model, pbmodel, migrad_params,\
                        objname, outdir, specfile,\
                        rvmodel=rvmodel,\
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
        WDmodel.viz.plot_mcmc_model(spec, phot, linedata,\
                    objname, outdir, specfile,\
                    model, cont_model,\
                    mcmc_params, param_names, in_samp, in_lnprob,\
                    rvmodel=rvmodel, balmer=balmer, ndraws=ndraws, savefig=savefig)

    return


#**************************************************************************************************************

if __name__=='__main__':
    mpi = False
    if str(sys.argv[1]).lower() == 'mpi':
        pool = MPIPool()
        mpi = True
        inargs = sys.argv[2:]
    else:
        pool = None
        inargs = sys.argv[1:]

    main(inargs, pool)

    if mpi:
        # Close the processes.
        pool.close()


