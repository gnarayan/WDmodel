#!/usr/bin/env python
import sys
import os
import numpy as np
import scipy.interpolate as scinterp 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import rec2txt

def main():
    #
    # m_app = m_ins -2.5log10(EE_r10/EE_r) + zp 
    #

    uvis_mag_sys = 'VEGAMAG'
    uvis_pb = 'F275W,F336W,F475W,F625W,F775W'
    uvis_pb = uvis_pb.split(',')
    uvis_rap = 7.5 # pixels
    uvis_zp = np.recfromtxt('UVIS2_zptmag_r10pix.ascii', names=True)
    uvis_ee = np.recfromtxt('UVIS2_EE_curves.ascii', names=True)

    out = []

    
    for pb in uvis_pb:
        ind_pb = (uvis_zp.pb == pb)
        ee = uvis_ee[pb]
        pix = uvis_ee.pixel
        f = scinterp.interp1d(pix, ee, kind='slinear')

        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(pix, ee, 'k.')
        pixrange = np.linspace(pix.min(), pix.max(), 100, endpoint=True)
        ax.plot(pixrange, f(pixrange), 'r-', alpha=0.5)
        ax.set_xlabel('pixel')
        ax.set_ylabel('EE fraction')
        fig.suptitle(pb)
        fig.savefig('figures/{}_ee_curve.pdf'.format(pb))

        EE_r10 = f(10.)
        EE_r   = f(uvis_rap)
        corr   = 2.5*np.log10(EE_r10/EE_r)
        ZP_r10 = uvis_zp[uvis_mag_sys][ind_pb][0]
        
        print pb, ZP_r10, corr, ZP_r10-corr
        out.append((pb, ZP_r10, corr, ZP_r10-corr))


    ir_mag_sys = 'VEGAmag'
    ir_pb  = 'F160W'
    ir_rap_pix = 5. # pix
    ir_scale = 0.13 # arcsec/pix
    ir_rap = ir_rap_pix * ir_scale
    ir_zp = np.recfromtxt('IR_zptmag_r0.4arcsec.ascii', names=True)
    ir_ee = np.recfromtxt('IR_EEarcsec_curves.ascii', names=True)
    pb = ir_pb

    ind_pb = (ir_zp.pb == pb)
    ZP_r0p4 = ir_zp[ir_mag_sys][ind_pb][0]

    # the IR EE curves are a 2D surface in wavelength and radius
    arcsec  = ir_ee.arcsec
    wave    = np.arange(0.7, 1.8, 0.1)
    surf = ir_ee.view(np.float).reshape(ir_ee.shape + (-1,))[:,1:]
    f = scinterp.interp2d(wave, arcsec, surf, kind='linear')

    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    WAVE, ARCSEC = np.meshgrid(wave, arcsec)
    ax = Axes3D(fig)
    ax.plot_wireframe(WAVE, ARCSEC, surf)
    ax.set_xlabel('wavelength (micron)')
    ax.set_ylabel('aperture (arcsec)')
    ax.set_zlabel('EE fraction')
    fig.suptitle(pb)
    fig.savefig('figures/{}_ee_curve.pdf'.format(pb))

    ir_wave = ir_zp['PHOTPLAM'][ind_pb][0]
    ir_wave /= 10000.
    
    EE_r0p4 = f(ir_wave, 0.4)[0]
    EE_r    = f(ir_wave, ir_rap)[0]
    corr = 2.5*np.log10(EE_r0p4/EE_r)

    print pb, ZP_r0p4, corr, ZP_r0p4-corr
    out.append((pb, ZP_r0p4, corr, ZP_r0p4-corr))

    out = np.rec.fromrecords(out, names='pb,MAST_zp,EE_corr,WDcalib_zp')
    with open('MAST_zp_to_WD_zp.txt','w') as f:
        f.write(rec2txt(out, precision=5)+'\n')



if __name__=='__main__':
    sys.exit(main())
