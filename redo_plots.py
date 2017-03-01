#!/usr/bin/env python

from __future__ import print_function

import sys
from fit_WDmodel import get_options
from numpy import unique
import WDmodel
import WDmodel.fit
import WDmodel.viz
import WDmodel.io


def main():
    args = get_options(sys.argv[1:])

    specfile  = args.specfile
    outdir    = args.outdir
    rvmodel   = args.reddeningmodel
    nwalkers  = args.nwalkers
    nburnin   = args.nburnin
    nprod     = args.nprod
    discard   = args.discard
    balmer    = args.balmerlines
    ndraws    = args.ndraws
    savefig   = args.savefig
    excludepb = args.excludepb

    # set the object name and create output directories
    objname, outdir = WDmodel.io.set_objname_outdir_for_specfile(specfile, outdir=outdir)

    # restore data
    input_file = WDmodel.io.get_outfile(outdir, specfile, '_inputs.hdf5')
    res = WDmodel.io.read_fit_inputs(input_file)
    spec, cont_model, linedata, continuumdata, phot = res


    # init model
    model = WDmodel.WDmodel()

    # exclude passbands that we want excluded 
    if phot is not None:
        pbnames = unique(phot.pb) 
        if excludepb is not None:
            pbnames = list(set(pbnames) - set(excludepb))
    else:
        pbnames = []

    # get the throughput model 
    pbmodel = WDmodel.io.get_pbmodel(pbnames)

    # restore params
    param_file = WDmodel.io.get_outfile(outdir, specfile, '_result.json')
    mcmc_params = WDmodel.io.read_params(param_file)

    # restore samples and prob
    chain_file = WDmodel.io.get_outfile(outdir, specfile, '_mcmc.hdf5')
    param_names, samples, samples_lnprob = WDmodel.io.read_mcmc(chain_file)

    # parse chain 
    result = WDmodel.fit.get_fit_params_from_samples(param_names, samples, samples_lnprob, mcmc_params,\
                    nwalkers=nwalkers, nprod=nprod, discard=discard)
    mcmc_params, in_samp, in_lnprob = result

    # plot the MCMC output
    WDmodel.viz.plot_mcmc_model(spec, phot, linedata,\
                objname, outdir, specfile,\
                model, cont_model,\
                mcmc_params, param_names, in_samp, in_lnprob,\
                rvmodel=rvmodel, balmer=balmer, ndraws=ndraws, savefig=savefig)

    return

if __name__ =='__main__':
    main()




