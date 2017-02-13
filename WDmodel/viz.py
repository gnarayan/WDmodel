import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties as FM
#import corner
#from matplotlib import rc
#rc('text', usetex=True)
#rc('font', family='serif')
#rc('ps', usedistiller='xpdf')
#rc('text.latex', preamble = ','.join('''\usepackage{amsmath}'''.split()))

def plot_spectrum_fit(spec, objname, outdir, specfile, model, result, rv=3.1, rvmodel='od94', save=False):
    
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

    # todo get this from likelihood wrapper 
    teff, logg, av, c, fwhm = result
    mod = model._get_obs_model(teff, logg, av, fwhm, spec.wave)
    smoothedmod = mod*c
    outlabel = 'Model\nTeff = %.2f K\nlog(g) = %.3f\nAv = %.3f mag\nc = %f\nFWHM = %.3f Angstrom\n' %result

    ax_spec.plot(spec.wave, smoothedmod, color='red', linestyle='-',marker='None', label=outlabel)

    ax_resid.fill_between(spec.wave, spec.flux-smoothedmod+spec.flux_err, spec.flux-smoothedmod-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_resid.plot(spec.wave, spec.flux-smoothedmod,  linestyle='-', marker=None,  color='black')

    ax_resid.set_xlabel('Wavelength~(\AA)',fontproperties=font_m, ha='center')
    ax_spec.set_ylabel('Normalized Flux', fontproperties=font_m)
    ax_resid.set_ylabel('Fit Residual Flux', fontproperties=font_m)
    ax_spec.legend(frameon=False, prop=font_s)
    fig.suptitle('%s (%s)'%(objname, specfile), fontproperties=font_l)
    
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    if save:
        outfile = os.path.join(outdir, os.path.basename(specfile).replace('.flm','_minuit.pdf'))
        fig.savefig(outfile)
    return fig 


def plot_model(spec, phot, objname, outdir, specfile, model, result, rv=3.1, rvmodel='od94', discard=5):

    outfilename = os.path.join(outdir, os.path.basename(specfile.replace('.flm','.pdf')))
    with PdfPages(outfilename) as pdf:
        fig =  plot_spectrum_fit(spec, objname, specfile, model, result)
        pdf.savefig(fig)

        #labels = [r"Teff" , r"log(g)", r"A_V"]
        #samples = samples[int(round(discard*samples.shape[0]/100)):]
        #fig = corner.corner(samples, bins=41, labels=labels, show_titles=True,quantiles=(0.16,0.84),\
        #     use_math_text=True)
        #pdf.savefig(fig)
        #endwith
    

