import numpy as np
import sys, os
from os import path
from multiprocessing import Pool
import multiprocessing as mp
from functools import reduce
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import Planck18
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.special import zeta, spence
import matplotlib.pyplot as plt
from functools import partial
import argparse
from gammaALPs.core import Source, ALP, ModuleList
from scipy.interpolate import RectBivariateSpline as RBSpline
import yaml

source_path = "./"

# --- Unit conversions factors as astropy quantities --- #
kgGeV = (1. * u.kg * c.c**2.).to("J").to("GeV")
sGeV = u.s / c.hbar.to("GeV s")
gaussGeV = c.e.value  * 1e-4 / np.sqrt(c.alpha * 4. * np.pi) * kgGeV / sGeV
GeV2kpc = (c.hbar * c.c).to("GeV kpc")
cm2GeV = (u.cm / (c.hbar * c.c)).to("GeV-1")
m_e_GeV = (c.m_e * c.c**2.).to("GeV")
Bcrit_muGauss = m_e_GeV ** 2. / np.sqrt(4. * np.pi * c.alpha) / gaussGeV * u.G.to("1e-6 G") * u.Unit("1e-6 G")

# --- chi CMB from Dobryina et al 2015
chiCMB = ((Planck18.Tcmb0 * c.k_B).to("GeV") ** 4. * np.pi **2. / 15. / m_e_GeV ** 4. *
          44. * c.alpha**2. / 135.).value

data_path = "data_store/"
gamma_gamma_disp = np.loadtxt(data_path + "gamma_gamma_dispersion_g2_Dobrynina_et_al.dat")
g2 = interp1d(gamma_gamma_disp[:, 0], gamma_gamma_disp[:, 1], fill_value = 'extrapolate')
omega_B = lambda T: m_e_GeV**2 /( np.pi**4/(30.0 * zeta(3)) * (T * u.K * c.k_B).to('GeV'))

## --- chi IR (cold), optical, UV (normal, MW-like galaxies) from 1404.2578, Schober et al.:
rho0_MW = 2.0 / (np.pi * 1.5e4**2 * 500) #(SFR at z = 0, MW: 6 Msol/y, volume: pi*R0^2*H0 (pc))
chiIR = chiCMB / (Planck18.Tcmb0 * c.k_B).to("GeV")**4 * (41 * u.K * c.k_B).to("GeV")**4 * 1.3e-5
chiOPT = chiCMB / (Planck18.Tcmb0 * c.k_B).to("GeV")**4 * (3500 * u.K * c.k_B).to("GeV")**4 * 8.9e-13
chiUV = chiCMB / (Planck18.Tcmb0 * c.k_B).to("GeV")**4 * (1.8e4 * u.K * c.k_B).to("GeV")**4 * 8.4e-17

#### Scaling relations for gas density and magnetic field strength (Schober et al.: doi:10.3847/0004-637X/827/2/109)
H = lambda z: 500/(1+z)
R = lambda z: 15000/(1+z)
rho_star = lambda z: 2.0 / (np.pi * R(z)**2 * H(z))
N_z = lambda n0, z: (rho_star(z)/rho_star(0.))**(1./1.4) * (1 + z)**3 * n0
B_z = lambda n0, z: np.power(N_z(n0, z), 1/6.) * np.power(rho_star(z) * H(z), 1/3.)
B_scale = lambda B0, n0, z: B0*1e-6 * B_z(n0, z) / B_z(n0, 0.)

def SFR_evolution(z):
    eta = -10
    a = 3.4
    b = -0.3
    c = -3.5
    rho0 = 1.0
    z1 = 1.
    z2 = 4.
    B = np.power(1.+z1, 1 - a/b)
    C = np.power(1. + z1, (b-a)/c) * np.power(1.+z2, 1 - b/c)
    return rho0 * np.power(np.power(1 + z, a * eta) + np.power((1. + z)/B, b*eta) + np.power((1. + z)/C, c*eta), 1/eta)

def chi_SFG_z(zred):
    ### MW-like galaxy
    SFR__z_evolution_atz0 = 0.10839579747624511
    SFR_z_evolution = lambda zred, kappa1 = 3.0/5.0, kappa2 = 14./15., zm= 5.4: kappa2 * np.exp(kappa1 * (zred - zm))/(kappa2 - kappa1 + kappa1 * np.exp(kappa2 * (zred - zm))) * (1. + zred)**3 / SFR__z_evolution_atz0 

    test_s = np.logspace(-3, 2, 1000) ### kpc
    EGeV = np.logspace(0.,7.,5000)
    UV_contribution = SFR_z_evolution(zred) * chiUV * g2(EGeV[:,np.newaxis]/omega_B(1.8e4))
    OPT_contribution = SFR_z_evolution(zred) * chiOPT * g2(EGeV[:,np.newaxis]/omega_B(3500.0))
    IR_contribution = SFR_z_evolution(zred) * chiIR * g2(EGeV[:,np.newaxis]/omega_B(41.0))
    CMB_contribution = chiCMB * g2(EGeV[:,np.newaxis]/omega_B(Planck18.Tcmb0.value * (1. + zred))) * (1. + zred)**4
    
    chi_ = test_s[np.newaxis,:]**0 * UV_contribution + test_s[np.newaxis,:]**0 * OPT_contribution + test_s[np.newaxis,:]**0 * IR_contribution + CMB_contribution * test_s[np.newaxis,:]**0
    chispl = RBSpline(EGeV, test_s, chi_, kx = 1, ky = 1, s = 0)
    return chispl

def xi_integrand_dilogarithm(E, eps):
    me = (c.m_e * c.c**2).to('eV').value
    sigma_0 = 1.25e-25
    beta_m = np.sqrt(1 - me**2/E/eps)
    Li2 = lambda x: -1.0 * spence(x)
    integrand_factor = 4 * sigma_0 * (1 - beta_m * beta_m) * (1 - beta_m * beta_m)
    term1 = Li2((1. - beta_m) / 2)
    term2 = Li2((1. + beta_m) / 2)
    term3 = beta_m * (1. + beta_m * beta_m) / (1. - beta_m * beta_m)
    term4 = 0.5 * ((1. + beta_m**2 * beta_m**2) / (1. - beta_m * beta_m) - np.log((1. - beta_m * beta_m) / 4.)) * np.log((1 + beta_m)/(1-beta_m))
    return integrand_factor * (term1 - term2 - term3 + term4)

def diff_number_density_blackbody(norm, E, T):
    n_IR = lambda E: (8.0 * np.pi / (c.h*c.c)**3 * (E * u.eV)**2 / (np.exp(E / ((T * u.K * c.k_B).to('eV').value)) - 1)).to('1/(eV cm3)').value
    return norm * n_IR(E)

def eps_integrand(E, z):
    me = (c.m_e * c.c**2).to('eV').value
    T_CMB = 2.73
    samples_ = np.logspace(np.log10(me**2/(E * (1-1e-10))), np.log10(1e5), 5000)
    diff_CMB_density = diff_number_density_blackbody((1 + z)**3, samples_, T_CMB * (1 + z))
    integrand_values = xi_integrand_dilogarithm(E, samples_) * diff_CMB_density
    res = trapz(integrand_values, samples_)
    return (res / u.cm).to('1/kpc').value
    
def photon_alp_conversion_probability_at_z(z, Ldom, R_prop, B_T, B_T_sig, n0, m_a, g_agg, EGeV, Nsample = 5000, is_B_const = False):
    transition_prob = np.zeros((Nsample, len(EGeV)))
    alp = ALP(m_a, g_agg)
    arbitrary_source = Source(z = z, l=0., b=8.)
    pin = np.diag((1.,1.,0.)) * 0.5
    chispl = chi_SFG_z(z)
    r_kpc = np.logspace(-3, np.log10(R_prop), 500)
    gamma_ = [eps_integrand(E * 1e9, z) for E in EGeV]
    disp = np.array(gamma_)[:, np.newaxis] * np.ones((len(gamma_), len(r_kpc)))
    
    
    for k in range(Nsample):
        BT_list = np.random.normal(loc = B_T, scale = B_T_sig, size = 1)
        ml = ModuleList(alp, arbitrary_source, pin = pin, EGeV = EGeV, log_level = 'error')
        
        ml.add_propagation(environ='ICMCell',
                    order=0,   # order of the module
                    B0 = BT_list if is_B_const else B_scale(BT_list, n0, z) / 1e-6,  # B field strength
                    L0= Ldom / (1. + z),  # cell size
                    nsim=1,  # one single realization
                    n0 = N_z(z, n0),  # electron density
                    r_abell = R_prop / (1. + z),  # full path, chosen that we only have a single cell
                    beta=0.0, #exponent of electron density profile (default: 1.)
                    eta=0.0,  #exponent for scaling of B field with electron density (default = 2./3.)
                    chi = chispl,
                    
        )
        #### Add absoprtion on CMB background light, other radiation fields are suppressed by orders of magnitude
        ml.add_disp_abs(EGeV, r_kpc, disp, 0, type_matrix='absorption')
        px, py, pa = ml.run()
        transition_prob[k] += pa[0]
        del ml
    return np.mean(transition_prob, axis = 0)

def create_Pga_bash_run_script(yaml_):
    with open(yaml_, 'r') as f:
        inputs = yaml.load(f)
    path = inputs['SBG_ALP_conversion']['path']
    with open("{}run_Pga_cellmodel_scan.sh".format(path), '+w') as out_file:
        out_file.write("#!/bin/bash\n")
        out_file.write("\n")
        out_file.write("ma_vals='{}'\n".format(" ".join(list(map(str, inputs['SBG_ALP_conversion']['m_a'])))))
        out_file.write("gagg_vals='{}'\n".format(" ".join(list(map(str, inputs['SBG_ALP_conversion']['g_agg'])))))
        out_file.write("\n")
        out_file.write("for m in $ma_vals\n")
        out_file.write("do  \n")
        out_file.write("    for g in $gagg_vals\n")
        out_file.write("    do\n")
        out_file.write("            nice -n 5 python3 run_ALP_cellmodel_propagation_gammaALPs.py -f 'photon_alp_conv_probability_ma_'$m'neV_gagg_'$g'e-11_upto_z_10_1GeV_10PeVtest_file_new_gammaALPs.dat' -m $m -g $g -ne {} -B {} -sB {} -L {} -R {}\n".format(inputs['SBG_ALP_conversion']['ne0'], inputs['SBG_ALP_conversion']['B0'], inputs['SBG_ALP_conversion']['sigmaB'], inputs['SBG_ALP_conversion']['Ldom'], inputs['SBG_ALP_conversion']['R_prop']))
        out_file.write("            wait\n")
        out_file.write("    done\n")
        out_file.write("done \n")
    os.system("chmod +x {}run_Pga_cellmodel_scan.sh".format(path))

def main():    
    z_space = np.linspace(0., 10., 51)
    EGeV = np.logspace(0.,8, 1000)

    parser = argparse.ArgumentParser(description = "Return the conversion probability photon -> ALP for a collection of galaxies at redshift z and energy E!")

    parser.add_argument('--outfile', '-f',
                        type=str,
                        dest='outfile',
                        help='outfile tag',
                        default="./test.dat"
    )
    parser.add_argument('--M_axion', '-m',
                        type=float,
                        dest='M_a',
                        help='Axion mass in units of neV',
                        default=1.
    )
    parser.add_argument('--G_agg', '-g',
                        type=float,
                        dest='g_a',
                        help='Value of the ALP_photon coupling constant: in units of 1e-11 GeV^-1',
                        default=10.
    )
    parser.add_argument('--electron_density', '-ne',
                        type=float,
                        dest='n_e',
                        help='Electron density of typical galaxy at z = 0.',
                        default=0.05
    )
    parser.add_argument('--B_transversal', '-B',
                        type=float,
                        dest='BT',
                        help='Field strength of transverse magnetic field (of coherent galactic magnetic field) at z = 0 in micro Gauss.',
                        default=5.0
    )
    parser.add_argument('--sig_B_transversal', '-sB',
                        type=float,
                        dest='sig_BT',
                        help='Uncertainty of field strength of transverse magnetic field (of coherent galactic magnetic field) at z = 0 in micro Gauss.',
                        default=3.0
    )
    parser.add_argument('--Ldom', '-L',
                        type=float,
                        dest='Ldom',
                        help='Mean domain length of propagation cell at z = 0 in kpc.',
                        default=1.0
    )
    parser.add_argument('--Rscale', '-R',
                        type=float,
                        dest='R',
                        help='Scaling radius of typical galaxy at z = 0 in kpc (boundary to which ALP conversion is computed).',
                        default=10.0
    )
    args = parser.parse_args()
    outfile = source_path + args.outfile 

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
            
    z_mappable = chunks(z_space, 5)
    P_ga_list = []
    for z_ in z_mappable:
        ncpu = min(15, mp.cpu_count()-2)
        p=Pool(ncpu)
        mappable_func = partial(photon_alp_conversion_probability_at_z, 
                            Ldom = args.Ldom,
                            R_prop = args.R,
                            B_T = args.BT,
                            B_T_sig = args.sig_BT,
                            n0 = args.n_e,
                            m_a = args.M_a, 
                            g_agg = args.g_a,
                            EGeV = EGeV,
                            Nsample = 5000,
        )
        P_ga_list_tmp = p.map(mappable_func, z_)
        P_ga_list += P_ga_list_tmp
        del p
    z_ = np.repeat(z_space, len(EGeV))
    E_ = np.repeat(np.array([EGeV]), len(z_space), axis = 0).flatten()
    P_ga_list = np.array(P_ga_list).flatten()
    res_ = np.vstack((z_, E_, P_ga_list)).T
    np.savetxt(outfile, res_)

if __name__ == "__main__":
    main()
