#!/usr/bin/env python
import sys
import warnings
warnings.simplefilter('once')
import argparse
import numpy as np
import WDmodel
import WDmodel.io
import WDmodel.fit
import WDmodel.viz


def get_options():
    """
    Get command line options for the WDmodel fitter
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # create a couple of custom types to use with the parser 
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

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
    phot.add_argument('--ignorephot',  required=False, action="store_true", default=False,\
            help="Ignores missing photometry and does the fit with just the spectrum")

    # fitting options
    model = parser.add_argument_group('model', 'Model options')
    params = WDmodel.io.read_param_defaults()
    for param in params:
        model.add_argument('--{}'.format(param), required=False, type=float, default=params[param]['default'],\
                help="Specify parameter {} value".format(param))
        model.add_argument('--{}_fixs'.format(param), required=False, default=params[param]['fixed'], type="bool",\
                help="Specify if parameter {} is fixed or not".format(param))
        model.add_argument('--{}_bounds'.format(param), required=False, nargs=2, default=params[param]["bounds"], 
                type='NoneOrFloat', metavar=("LOWERLIM", "UPPERLIM"), help="Specify parameter {} bounds".format(param))

    # MCMC config options
    mcmc = parser.add_argument_group('mcmc', 'MCMC options')
    mcmc.add_argument('--nwalkers',  required=False, type=int, default=200,\
            help="Specify number of walkers to use (0 disables MCMC)")
    mcmc.add_argument('--nburnin',  required=False, type=int, default=200,\
            help="Specify number of steps for burn-in")
    mcmc.add_argument('--nprod',  required=False, type=int, default=1000,\
            help="Specify number of steps for production")
    mcmc.add_argument('--discard',  required=False, type=float, default=5,\
            help="Specify percentage of steps to be discarded")

    # visualization options
    viz = parser.add_argument_group('viz', 'Visualization options')
    viz.add_argument('-b', '--balmerlines', nargs='+', type=int, default=range(1,7,1),\
            help="Specify Balmer lines to visualize [1:7]")

    # output options
    output = parser.add_argument_group('output', 'Output options')
    output.add_argument('-o', '--outdir', required=False,\
            help="Specify a custom output directory. Default is CWD+objname/ subdir")
    output.add_argument('--redo',  required=False, action="store_true", default=False,\
            help="Clobber existing fits")

    args = parser.parse_args()

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

    if args.nburnin <= 0:
        message = 'Number of walkers must be greater than zero'
        raise ValueError(message)

    if args.nprod <= 0:
        message = 'Number of walkers must be greater than zero'
        raise ValueError(message)

    if not (0 <= args.discard < 100):
        message = 'Discard must be a percentage (0-100)'
        raise ValueError(message)

    return args

#**************************************************************************************************************

def main():
    args   = get_options() 

    specfile  = args.specfile
    bluelim, redlim   = args.trimspec
    blotch    = args.blotch
    fwhm      = args.fwhm

    outdir    = args.outdir

    photfile  = args.photfile
    rv        = args.rv
    rvmodel   = args.reddeningmodel
    ignorephot= args.ignorephot
    
    nwalkers  = args.nwalkers
    nburnin   = args.nburnin
    nprod     = args.nprod
    discard   = args.discard
    redo      = args.redo

    balmer    = args.balmerlines

    # read spectrum
    spec = WDmodel.io.read_spec(specfile)

    # get resolution
    fwhm = WDmodel.io.get_spectrum_resolution(specfile, fwhm=fwhm)

    # set the object name and create output directories
    objname, outdir = WDmodel.io.set_objname_outdir_for_specfile(specfile, outdir=outdir)

    # pre-process spectrum
    out = WDmodel.fit.pre_process_spectrum(spec, bluelim, redlim, blotch=blotch)
    spec, cont_model, linedata, continuumdata = out

    # get photometry 
    phot = WDmodel.io.get_phot_for_obj(objname, photfile, ignore=ignorephot)

    # save the inputs to the fitter
    WDmodel.io.save_fit_inputs(spec, phot,\
            cont_model, linedata, continuumdata,\
            outdir, specfile, redo=redo)

    # fit the spectrum
    model, result = WDmodel.fit.fit_model(spec, phot,\
                objname, outdir, specfile,\
                rv=rv, rvmodel=rvmodel, fwhm=fwhm,\
                nwalkers=nwalkers, nburnin=nburnin, nprod=nprod,\
                redo=redo)

    # plot output
    WDmodel.viz.plot_model(spec, phot,\
            objname, outdir, specfile,\
            model, result,\
            balmer=balmer, discard=discard)
    return


#**************************************************************************************************************

if __name__=='__main__':
    main()
    #cProfile.run('main()', 'profile.dat')
    #import pstats
    #p = pstats.Stats('profile.dat')
    #p.sort_stats('cumulative','time').print_stats(20)


