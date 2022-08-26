import numpy as np
from gammaALPs.core import Source, ALP, ModuleList
from gammaALPs.base import environs, transfer
import healpy as hp
import sys, os
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import Planck15
from scipy.interpolate import interp1d
from scipy.special import zeta
from scipy.interpolate import RectBivariateSpline as RBSpline
import yaml

data_path = "data_store/"
gamma_gamma_disp = np.loadtxt(data_path + "gamma_gamma_dispersion_g2_Dobrynina_et_al.dat")
g2 = interp1d(gamma_gamma_disp[:, 0], gamma_gamma_disp[:, 1], fill_value = 'extrapolate')
omega_B = lambda T: m_e_GeV**2 /( np.pi**4/(30.0 * zeta(3)) * (T * u.K * c.k_B).to('GeV'))

m_e_GeV = (c.m_e * c.c**2.).to("GeV")
chiCMB = ((Planck15.Tcmb0 * c.k_B).to("GeV") ** 4. * np.pi **2. / 15. / m_e_GeV ** 4. *
          44. * c.alpha**2. / 135.).value
chiIR = chiCMB / (Planck15.Tcmb0 * c.k_B).to("GeV")**4 * (41 * u.K * c.k_B).to("GeV")**4 * 1.3e-5

def rho_dust_warm(r, z):
    hw = 3.3 ## kpc
    zw = 0.09  ## kpc
    return np.exp(-r/hw - np.abs(z)/zw)

def line_of_sight_POV_EARTH(s, l, b):
    r_dot = 8.5 ## kpc
    ctheta = np.cos(np.radians(l)) * np.cos(np.radians(b))
    r_spherical = np.sqrt(s**2 + r_dot**2 - 2.0 * r_dot * s * ctheta)
    z_Earth = s * np.sin(np.radians(b))
    R_cylindrical_GC = np.sqrt(r_spherical**2 - z_Earth**2)
    return R_cylindrical_GC, z_Earth

def run_GMF_calc(m_a, g_agg, N_Ebin, model = 'jansson12c', fancy_chi = False, first_half = True, Emin = 20, Emax = 5e5, NSIDE = 64):

    pix = np.arange(hp.nside2npix(NSIDE))  # get the pixels
    ll, bb = hp.pixelfunc.pix2ang(NSIDE, pix, lonlat=True)  #  get the galactic coordinates for each pixel


    E_range = np.power(10, np.linspace(np.log10(Emin), np.log10(Emax), N_Ebin + 1))
    EGeV = E_range * 1e-3  # energy
    pgg2 = np.zeros((pix.shape[0],EGeV.shape[0]))  # array to store the results
    src = Source(z=0.1, ra=0., dec=0.)  # some random source for initialization

    # coupling and mass at which we want to calculate the conversion probability:
    g = g_agg ### units of 1e-11 GeV^-1
    m = m_a  ### units of neV
    pa_in = np.diag([0., 0., 1.])  # the inital polarization matrix; a pure ALP state

    for i, l in enumerate(ll):
        #### For NSIDE = 64 to reduce the amount of memory used during the calculation
        if first_half:
            if i > 25001:
                continue
        else:
            if i <= 25001:
                continue
        src.l = l
        src.b = bb[i]
        test_s = np.logspace(-3, 2, 1000) ### kpc

        if fancy_chi:
            los = np.array(line_of_sight_POV_EARTH(test_s, src.l, src.b))
            ISRfield = rho_dust_warm(*los)
            chi_ = ISRfield * test_s[np.newaxis,:]**0 * chiIR * g2(EGeV[:,np.newaxis]/omega_B(41.0)) + g2(EGeV[:,np.newaxis]/omega_B(Planck15.Tcmb0.value)) * chiCMB * test_s[np.newaxis,:]**0
            chispl = RBSpline(EGeV, test_s, chi_, kx = 1, ky = 1, s = 0)
        ml = ModuleList(ALP(m=m,g=g),
                   src,
                   pin=pa_in,  # pure ALP beam
                   EGeV = EGeV,
                   log_level = 'warning')
        ml.add_propagation("GMF", 0, model=model, chi = chispl if fancy_chi else None)  # add the propagation module
        px, py, pa = ml.run()  # run the code
        pgg2[i] = px + py  # save the result

        if i < ll.size - 1:
            del ml

    hp.write_map('gammaALPs_{}_gagg_{}GeV-1_ma_{}neV_100GeV_1PeV_60Ebins_{}.fits'.format(model, g_agg, m_a, int(first_half)), pgg2.T)
    
def create_bash_run_script(yaml_):
    with open(yaml_, 'r') as f:
        inputs = yaml.load(f)
    path = inputs['GMF_conversion']['path']
    with open("{}calculate_GMF_maps.sh".format(path), '+w') as out_file:
        out_file.write("#!/bin/bash\n")
        out_file.write("\n")
        out_file.write("masses='{}'\n".format(" ".join(list(map(str, inputs['GMF_conversion']['m_a'])))))
        out_file.write("GMF='{}'\n".format(inputs['GMF_conversion']['model']))
        out_file.write("g_agg='{}'\n".format(" ".join(list(map(str, inputs['GMF_conversion']['g_agg'])))))
        out_file.write("\n")
        out_file.write("for M in $masses\n")
        out_file.write("do  \n")
        out_file.write("    for g in $g_agg\n")
        out_file.write("    do\n")
        out_file.write("        for B in $GMF\n")
        out_file.write("        do\n")
        out_file.write("            nice -n 5 python3 calculate_GMF_conv_prob.py $M 1 0 $B $g\n")
        out_file.write("            wait\n")
        out_file.write("            nice -n 5 python3 calculate_GMF_conv_prob.py $M 0 0 $B $g\n")
        out_file.write("            wait\n")
        out_file.write("            nice -n 5 python3 calculate_GMF_conv_prob.py $M 0 1 $B $g\n")
        out_file.write("            rm -rf *_1.fits *_0.fits\n")
        out_file.write("        done\n")
        out_file.write("    done\n")
        out_file.write("done \n")
    os.system("chmod +x {}calculate_GMF_maps.sh".format(path))

def main():
    N_Ebin = 60
    
    m_a = float(sys.argv[1])
    is_first_half = bool(int(sys.argv[2]))
    is_sum = bool(int(sys.argv[3]))
    model_gmf = sys.argv[4]
    gagg = float(sys.argv[5])

    if is_sum:
        map_1 = hp.read_map('gammaALPs_{}_gagg_{}GeV-1_ma_{}neV_100GeV_1PeV_60Ebins_0.fits'.format(model_gmf, gagg, m_a), field = None)
        map_2 = hp.read_map('gammaALPs_{}_gagg_{}GeV-1_ma_{}neV_100GeV_1PeV_60Ebins_1.fits'.format(model_gmf, gagg, m_a), field = None)
        hp.write_map('gammaALPs_{}_gagg_{}GeV-1_ma_{}neV_100GeV_1PeV_60Ebins.fits'.format(model_gmf.upper(), gagg, m_a), map_1 + map_2)
    else:
        run_GMF_calc(m_a, gagg, N_Ebin, fancy_chi = True, Emin = 1e5, Emax = 1e9, model = model_gmf, first_half = is_first_half)

if __name__ == "__main__":
    main()
