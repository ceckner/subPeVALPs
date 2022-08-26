import astropy.units as u
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
from scipy.integrate import trapz
import sys, os
import argparse
import yaml

phi_crab = 2.26e-11 # 1 / cm2 / s in range 1 TeV to 100 TeV

def radial_Lorimer(r, A):
    r_sun = 8.5
    B = 1.9 #### Lorimer model C (preferred), Mon. Not. R. Astron. Soc. 372, 777–800 (2006)
    C = 5.0 #### Lorimer model C (preferred), Mon. Not. R. Astron. Soc. 372, 777–800 (2006)
    return A * (r/r_sun)**B * np.exp(-C * (r - r_sun)/r_sun)

def height_Lorimer(z):
    H = 0.2
    return np.exp(-np.abs(z)/H)

def three_dim_source_distribution(r, z):
    r_ = np.concatenate((np.linspace(0, 1e-6, 1000), np.logspace(-6, 3, 5000)))
    z_ = np.linspace(0, 10, 100000)
    int_r = trapz(radial_Lorimer(r_, 1.) * r_, r_)
    int_z = trapz(height_Lorimer(z_), z_)
    full_int = int_r * int_z * 4.0 * np.pi
    return radial_Lorimer(r, 1./full_int) * height_Lorimer(z)

def source_luminosity_fct(L):
    R = 0.019  #1/yr, supernova rate
    alpha = 1.5
    tau = 1.8e3 # yr
    L_max = (4.9e35 * u.erg / u.s).to('TeV/s').value # 'erg/s'
    return R * (alpha - 1.) * tau / L_max * np.power(L/L_max, -alpha)

def average_source_spectrum(E, beta, Ecut):
    f_free = lambda e: np.exp(-e/Ecut) * np.power(e / 1.0, -beta)
    sample_E = np.logspace(0, 2, 10000)
    K = trapz(f_free(sample_E), sample_E)
    K2 = trapz(f_free(sample_E) * sample_E, sample_E)
    return f_free(E) / K, K2 / K 

def r_c_gal(s, b, l):
    r_dot = 8.5 ## kpc
    return np.sqrt(r_dot**2 + s**2 - 2.0 * r_dot * s * np.cos(np.radians(l))*np.cos(np.radians(b)))

def los_integral_source_distribution_old(phi, b, l, beta, Ecut):
    r_dot = 8.5
    r_ = np.logspace(-10, 3, 50000) 
    spec, avg_E = average_source_spectrum(1e3, beta, Ecut)
    f_rho = lambda r: np.cos(np.radians(b)) * 4.0 * np.pi * r * r * r * r * avg_E * three_dim_source_distribution(r, r * np.sin(np.radians(b))) * source_luminosity_fct(4.0 * np.pi * (r * r * u.kpc**2).to('cm2').value * avg_E * phi)
    rho_ = f_rho(r_c_gal(r_, b, l))
    return trapz(rho_, r_, axis = 0)

def los_integral_source_distribution(phi, b, l, beta, Ecut):
    r_dot = 8.5
    r_ = np.logspace(-10, 3, 50000)
    spec, avg_E = average_source_spectrum(1e3, beta, Ecut)
    f_rho = lambda r: np.cos(np.radians(b)) * 4.0 * np.pi * r * r * r * r * avg_E * three_dim_source_distribution(r_c_gal(r, b, l), r * np.sin(np.radians(b))) * source_luminosity_fct(4.0 * np.pi * (r * r * u.kpc**2).to('cm2').value * avg_E * phi)
    rho_ = f_rho(r_)
    return trapz(rho_, r_, axis = 0)

def angular_integral_source_distribution(phi, beta, Ecut, lmin = 25.0, lmax = 100., bmin = -5., bmax = 5.):
    sample_l = np.linspace(lmin, lmax, int(lmax - lmin))
    sample_b = np.linspace(bmin, bmax, int(bmax - bmin)*5)
    f_l_integrand = lambda l: trapz(np.array(list(map(lambda b: los_integral_source_distribution(phi, b, l, beta, Ecut), sample_b))), np.radians(sample_b))
    full_integral = np.array(list(map(f_l_integrand, sample_l)))
    return trapz(full_integral, np.radians(sample_l))

def final_flux_integral_source_distribution(E, flux_threshold, beta, Ecut, n_jobs = 6, lmin = 25.0, lmax = 100., bmin = -5., bmax = 5.):
    sample_phi = np.logspace(-25, np.log10(flux_threshold), 50)
    avg_spec, avg_E = average_source_spectrum(E, beta, Ecut)
    full_integral = np.array(Parallel(n_jobs = n_jobs)(delayed(angular_integral_source_distribution)(phi, beta, Ecut, lmin = 25.0, lmax = 100., bmin = -5., bmax = 5.) for phi in sample_phi))
    return trapz(full_integral * sample_phi, sample_phi) * avg_spec * (u.kpc**2).to('cm2') / ((np.sin(np.radians(bmax)) - np.sin(np.radians(bmin))) * (np.radians(lmax) - np.radians(lmin)))
   
def final_flux_integral_source_distribution_detected(E, flux_threshold, beta, Ecut, n_jobs = 6, lmin = 25.0, lmax = 100., bmin = -5., bmax = 5.):
    sample_phi = np.logspace(np.log10(flux_threshold), np.log10(phi_crab),  50)
    avg_spec, avg_E = average_source_spectrum(E, beta, Ecut)
    full_integral = np.array(Parallel(n_jobs = n_jobs)(delayed(angular_integral_source_distribution)(phi, beta, Ecut, lmin = 25.0, lmax = 100., bmin = -5., bmax = 5.) for phi in sample_phi))
    return trapz(full_integral * sample_phi, sample_phi) * avg_spec * (u.kpc**2).to('cm2') / ((np.sin(np.radians(bmax)) - np.sin(np.radians(bmin))) * (np.radians(lmax) - np.radians(lmin)))
 
def run_subthreshold_flux_calculation(yaml_, experiment, path = './'):
    with open(yaml_, 'r') as f:
        inputs = yaml.load(f)
        
    flux_threshold = inputs["Subthreshold_flux"][experiment]["S_det"]
    beta = inputs["Subthreshold_flux"][experiment]["beta"]
    Ecutoff = inputs["Subthreshold_flux"][experiment]["E_c"]
    lmin, lmax = inputs["Subthreshold_flux"][experiment]["GLON"]
    bmin, bmax = inputs["Subthreshold_flux"][experiment]["GLAT"]
    E_samples = np.logspace(0, 3, 30)
    
    subthresh_flux = np.array([final_flux_integral_source_distribution(E, flux_threshold * phi_crab, beta, Ecutoff, n_jobs = 15, lmin = lmin, lmax = lmax, bmin = bmin, bmax = bmax) for E in E_samples])
    np.savetxt(path + 'subthreshold_population_geq1TeV_fluxTH_{}_beta_{}_Ecutoff_{}TeV_lmin_{}deg_lmax_{}deg_bmin_{}deg_bmax_{}deg.txt'.format(flux_threshold, beta, Ecutoff, lmin, lmax, bmin, bmax), np.vstack((E_samples, subthresh_flux)).T)

def main():
    source_path = os.getcwd() + "/"
    
    parser = argparse.ArgumentParser(description = "Calculate the gamma-ray flux of a TeV sub-threshold population model according to [V. Vecchiotti+, arXiv:2107.14584].")

    parser.add_argument('--exp', '-e',
                        type=str,
                        dest='experiment',
                        help='Experiment for which the flux contribution is computed.',
                        default='HAWC'
    )
    parser.add_argument('--model_def', '-file',
                        type=str,
                        dest='yaml',
                        help='YAML file containing the parameter definitions for the source populations to be calculated.',
                        default='analysis_definition_file.yaml'
    )
    parser.add_argument('--root',
                        type=str,
                        dest='path',
                        help='Path to working directory.',
                        default='./'
    )
    options = parser.parse_args()
    
    run_subthreshold_flux_calculation(options.yaml, 
                         options.experiment, 
                         path = options.path)
    
if __name__ == "__main__":
    main()
