#!/usr/bin/env python

from __future__ import print_function

import sys
from fit_WDmodel import get_options
import numpy as np
import WDmodel
import WDmodel.fit
import WDmodel.pbmodel
import WDmodel.viz
import WDmodel.io


def main():
    args = get_options(sys.argv[1:])

    specfile  = args.specfile
    outdir    = args.outdir
    outroot   = args.outroot
    discard   = args.discard
    balmer    = args.balmerlines
    ndraws    = args.ndraws
    savefig   = args.savefig

    # set the object name and create output directories
    objname, outdir = WDmodel.io.set_objname_outdir_for_specfile(specfile, outdir=outdir, outroot=outroot)

    # restore data
    input_file = WDmodel.io.get_outfile(outdir, specfile, '_inputs.hdf5')
    res = WDmodel.io.read_fit_inputs(input_file)
    spec, cont_model, linedata, continuumdata, phot, fit_config = res
    errscale = np.median(spec.flux_err)
    covtype  = fit_config['covtype']
    usehodlr = fit_config['usehodlr']
    nleaf    = fit_config['nleaf']
    tol      = fit_config['tol']
    scale_factor = fit_config['scale_factor']
    rvmodel  = fit_config['rvmodel']
    phot_dispersion = fit_config.get('phot_dispersion',0.)

    # init model
    model = WDmodel.WDmodel()

    # init a covariance model instance that's used to model the residuals
    # between the systematic residuals between data and model
    covmodel = WDmodel.covmodel.WDmodel_CovModel(errscale, covtype, nleaf=nleaf, tol=tol, usehodlr=usehodlr)

    # restore params
    param_file = WDmodel.io.get_outfile(outdir, specfile, '_result.json')
    mcmc_params = WDmodel.io.read_params(param_file)

    # exclude passbands that we want excluded
    pbnames = []
    if phot is not None:
        pbnames = np.unique(phot.pb)

        # filter the photometry recarray to use only the passbands we want
        useind = [x for x, pb in enumerate(phot.pb) if pb in pbnames]
        useind = np.array(useind)
        phot = phot.take(useind)

        # set the pbnames from the trimmed photometry recarray to preserve order
        pbnames = list(phot.pb)

    # if we cut out out all the passbands, force mu to be fixed
    if len(pbnames) == 0:
        mcmc_params['mu']['value'] = 0.
        mcmc_params['mu']['fixed'] = True
        phot = None

    # get the throughput model
    pbs = WDmodel.pbmodel.get_pbmodel(pbnames, model)

    # restore samples and prob
    chain_file = WDmodel.io.get_outfile(outdir, specfile, '_mcmc.hdf5')
    samples, samples_lnprob, chain_params = WDmodel.io.read_mcmc(chain_file)
    param_names = chain_params['param_names']
    nwalkers    = chain_params['nwalkers']
    nprod       = chain_params['nprod']

    # parse chain
    result = WDmodel.fit.get_fit_params_from_samples(param_names, samples, samples_lnprob, mcmc_params,\
                    nwalkers=nwalkers, nprod=nprod, discard=discard)
    mcmc_params, in_samp, in_lnprob = result

    # plot the MCMC output
    model_spec, full_mod, model_mags = WDmodel.viz.plot_mcmc_model(spec, phot, linedata,\
                scale_factor, phot_dispersion,\
                objname, outdir, specfile,\
                model, covmodel, cont_model, pbs,\
                mcmc_params, param_names, in_samp, in_lnprob,\
                rvmodel=rvmodel, balmer=balmer, ndraws=ndraws, savefig=savefig)

    spec_model_file = WDmodel.io.get_outfile(outdir, specfile, '_spec_model.dat')
    WDmodel.io.write_spectrum_model(spec, model_spec, spec_model_file)

    full_model_file = WDmodel.io.get_outfile(outdir, specfile, '_full_model.hdf5')
    WDmodel.io.write_full_model(full_mod, full_model_file)

    if phot is not None:
        phot_model_file = WDmodel.io.get_outfile(outdir, specfile, '_phot_model.dat')
        WDmodel.io.write_phot_model(phot, model_mags, phot_model_file)

    return

if __name__ =='__main__':
    main()
