import numpy as np
import warnings
import logging
import sys
from os import path
from multiprocessing import Pool
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import reduce
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import Planck15
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.optimize import bisect
from scipy.special import zeta
import matplotlib.pyplot as plt
from functools import partial
import argparse
from scipy.interpolate import RectBivariateSpline as RBSpline
from scipy.integrate import quad, trapz
from gammaALPs.core import Source, ALP, ModuleList
import healpy as hp
from gammapy.maps import Map, MapAxis, WcsNDMap, WcsGeom, HpxMap
from astropy.coordinates import SkyCoord

data_path = "data_store/"

def line_of_sight_POV_EARTH(s, l, b):
    r_dot = 8.5 ## kpc
    ctheta = np.cos(np.radians(l)) * np.cos(np.radians(b))
    r_spherical = np.sqrt(s**2 + r_dot**2 - 2.0 * r_dot * s * ctheta)
    z_Earth = s * np.sin(np.radians(b))
    R_cylindrical_GC = np.sqrt(r_spherical**2 - z_Earth**2)
    return R_cylindrical_GC, z_Earth

def read_EGAL_conv_propb_file(data_file):
    z, E, Pag = np.loadtxt(data_file, unpack = True)
    z = np.array(sorted(list(set(z))))
    E = np.array(sorted(list(set(E))))
    EGAL_Pagg_spl = RBSpline(z,E,Pag.reshape((len(z), len(E))), kx = 1, ky = 1, s = 0)
    return EGAL_Pagg_spl

def read_MW_ALPgamma_conv_propb_file(data_file, mask = None, EGeV = np.logspace(2, 6, 61)):
    if mask is None:
        alp_map = hp.read_map(data_file, field = None)
    else:
        alp_map = hp.read_map(data_file, field = None) * mask
    f_prob_ga = interp1d(EGeV, alp_map, axis = 0, fill_value = 'extrapolate')
    return f_prob_ga

def SFR_evolution(z, kind = 'fiducial'):
    parameters = {
        'fiducial': (3.4, -0.3, -3.5),
        'high': (3.6, -0.1, -2.5),
        'low': (3.2, -0.5, -4.5),
    }
    eta = -10
    a = parameters[kind][0]
    b = parameters[kind][1]
    c = parameters[kind][2]
    rho0 = 1.0
    z1 = 1.
    z2 = 4.
    B = np.power(1.+z1, 1 - a/b)
    C = np.power(1. + z1, (b-a)/c) * np.power(1.+z2, 1 - b/c)
    return rho0 * np.power(np.power(1 + z, a * eta) + np.power((1. + z)/B, b*eta) + np.power((1. + z)/C, c*eta), 1/eta)

def dNdE_gamma(E, N0, Eb, alpha):
    E /= 2.0  ### due to production mechanism
    gamma_to_nu = 2.0 / 3.0  ### two photons per three neutrinos
    Eb *= 1e3 ###GeV
    return N0 * np.power((E / Eb)**2 + (E/Eb)**(2 * alpha), -0.5) * gamma_to_nu

def dNdE_nu(E, N0, Eb, alpha):
    Eb *= 1e3 ###GeV
    return N0 * np.power((E / Eb)**2 + (E/Eb)**(2 * alpha), -0.5)

def cosmological_measure(z):
    H0 = 70.
    Lambda_L = 0.7
    Lambda_m = 0.3
    return H0 * (1. + z) * np.sqrt(Lambda_L + Lambda_m * (1. + z)**3)

def gamma_flux_integrand(E, z, N0, Eb, alpha, Pga_EGAL, Pag_GAL, kind = 'fiducial'):
    c = 3e5
    return c/4./np.pi * dNdE_gamma(E * (1. + z), N0, Eb, alpha) * (1. + z) * SFR_evolution(z, kind = kind) / cosmological_measure(z) * Pga_EGAL(z, E) * Pag_GAL(E)

def diffuse_ALP_flux(N0, Eb, alpha, f_Pga_EGAL, kind = 'fiducial'):
    dE_ = np.logspace(2, 6, 5000)
    f_Pag_GAL = interp1d(dE_, np.ones_like(dE_), fill_value = 'extrapolate')
    dNdE_ALP = np.array([quad(lambda z: gamma_flux_integrand(E, z, N0, Eb, alpha, f_Pga_EGAL, f_Pag_GAL, kind = kind), 0., 10.)[0] for E in dE_])
    f_dNdE_ALP = interp1d(dE_, dNdE_ALP, fill_value = 'extrapolate')
    return dE_, f_dNdE_ALP

def get_lb_mask(NSIDE, lmin, lmax, bmin, bmax, keep_360 = False):
    NPIX = 12 * NSIDE * NSIDE
    lb_mask = np.zeros(NPIX)
    coordinates = hp.pixelfunc.pix2ang(NSIDE, np.arange(0,NPIX), lonlat = True)
    for k, (l, b) in enumerate(zip(coordinates[0], coordinates[1])):
        if l > 180 and not keep_360:
            l -= 360.
        if lmin <= l and lmax >= l and bmin <= b and bmax >= b:
            lb_mask[k] = 1.0
    return lb_mask

def return_flux_at_Earth_in_ROI(f_dNdE_ALP, EGeV, f_Pag_GAL, lon_1, lon_2, lat_1, lat_2, NSIDE = 64, keep_360 = False):
    target_roi = get_lb_mask(NSIDE, lon_1, lon_2, lat_1, lat_2, keep_360 = keep_360)
    flux_output = np.zeros_like(EGeV)
    conversion_data = f_Pag_GAL(EGeV)
    for k, x in enumerate(target_roi):
        if x != 0.0:
            flux_output += conversion_data[:, k] * f_dNdE_ALP(EGeV) * (4.0 * np.pi / (12 * NSIDE**2))
    return flux_output / (target_roi.sum() / (12 * NSIDE*NSIDE))

def get_Tibet_ASMD_data_points(f_dNdE_ALP, additional_components = None):
    energy_bins = [(1.19378e+2, 1.00000e+2, 1.61730e+2), (2.19109e+2, 1.61730e+2, 4.02157e+2), (5.31220e+2, 4.02157e+2, 1.00000e+3)]
    data_points = []
    for ecentre, emin, emax in energy_bins:
        E_ = np.logspace(np.log10(emin * 1e3), np.log10(emax * 1e3), 5000)
        integrated_flux = trapz(f_dNdE_ALP(E_), E_) / trapz(np.ones_like(E_), E_)
        data_points.append(integrated_flux * (ecentre * 1e3)**2.7)
        if not additional_components is None:
            for comp in additional_components:
                tmp_comp = trapz(comp(E_), E_) / trapz(np.ones_like(E_), E_)
                tmp_comp *= (ecentre * 1e3)**2.7
                data_points.append(tmp_comp)
    try:
        data_points = np.array(data_points).reshape((3, len(additional_components)+1)).T
    except:
        data_points = np.array(data_points)
    return data_points

def get_HAWC_integrated_data_points(f_gamma_flux):
    energy_bins = np.logspace(4, 5, 6)
    energy_centers = np.array([np.sqrt(E1 * E2) for E1, E2 in zip(energy_bins, energy_bins[1:])])
    data_points = []
    for ecentre, emin, emax in zip(energy_centers, energy_bins, energy_bins[1:]):
        E_ = np.logspace(np.log10(emin * 1e3), np.log10(emax * 1e3), 5000)
        integrated_flux = trapz(f_gamma_flux(E_), E_) / trapz(np.ones_like(E_), E_)
        data_points.append(integrated_flux * (ecentre * 1e3)**2.7)
    data_points = np.array(data_points)
    return data_points

def derive_final_data_points_tibet_ASMD(MW_conv, EGAL_conv, N0, Eb, alpha, lon1, lon2, lat1, lat2, keep_360 = True):
    sfrd_parameters = {
        'fiducial': (3.4, -0.3, -3.5),
        'high':  (3.6, -0.1, -2.5),
        'low':  (3.2, -0.5, -4.5),
    }
    
    norm_ = {}
    for k, v in sfrd_parameters.items():
        norm_[k] = get_source_spectrum_normalisation(lambda z: SFR_evolution(z, kind = k), lambda E: dNdE_nu(E, 1., Eb, alpha), 1e5, 2.1233333333333335e-8)

    f_Pga_EGAL = read_EGAL_conv_propb_file(EGAL_conv)
    f_Pag_GAL = read_MW_ALPgamma_conv_propb_file(MW_conv)
    dE_, f_dNdE_ALP = diffuse_ALP_flux(norm_[N0], Eb, alpha, f_Pga_EGAL, kind = N0)
    tibet_array_flux = return_flux_at_Earth_in_ROI(f_dNdE_ALP, dE_, f_Pag_GAL, lon1, lon2, lat1, lat2, keep_360 = keep_360)
    f_dNdE_ALP = interp1d(dE_, tibet_array_flux, fill_value = 'extrapolate')
    components = get_Tibet_ASMD_data_points(f_dNdE_ALP)
    return components

def get_source_spectrum_normalisation(sfrd_function, nu_spectrum, E_ref, flux_ref, cosmo = FlatLambdaCDM(Om0 = 0.3, H0 = 70.0)):
    cosmological_measure = lambda z: cosmo.H0.value * (1. + z) * np.sqrt((1. - cosmo.Om0) + cosmo.Om0 * (1. + z)**3)
    flux_integrand = lambda E, z: c.c.to('km/s').value/4./np.pi * nu_spectrum(E * (1. + z)) * (1. + z) * sfrd_function(z) / cosmological_measure(z)
    value_Eref = quad(lambda z: flux_integrand(E_ref, z), 0, 100)[0] * E_ref**2
    return flux_ref/value_Eref

if __name__ == "__main__":
    pass
    
