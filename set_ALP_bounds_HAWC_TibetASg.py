from calculate_diffuse_EGAL_ALP_spectrum import *
from gammapy.maps import Map
from matplotlib import rc
import os
from iminuit import Minuit
from astropy.io import fits
import yaml 
import argparse

rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['cmr']})
rc('font',**{'family':'serif','serif':['cmr']})
rc('text.latex', preamble=r'\usepackage{soul}')

source_path = os.getcwd() + "/"

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

def get_HAWC_integrated_data_points(f_gamma_flux):
    energy_bins = np.logspace(4, 5, 6)
    energy_centers = np.array([np.sqrt(E1 * E2) for E1, E2 in zip(energy_bins, energy_bins[1:])])
    data_points = []
    for ecentre, emin, emax in zip(energy_centers, energy_bins, energy_bins[1:]):
        E_ = np.logspace(np.log10(emin), np.log10(emax), 5000)
        integrated_flux = trapz(f_gamma_flux(E_), E_) / trapz(np.ones_like(E_), E_)
        data_points.append(integrated_flux * (ecentre)**2.7)
    data_points = np.array(data_points)
    return data_points, energy_centers

def PowerLaw(E, alpha, E0, N0):
    return np.power(E/E0, alpha) * N0

def get_gammaray_emission_from_IEM_file(file_name, lmin, lmax, bmin, bmax, E_index):
    iem_file = Map.read(file_name)
    E_unit = iem_file.geom.axes[0].center.unit
    sr_pix = np.mean(iem_file.geom.solid_angle().value[0])
    lon_roi = list(filter(lambda x: x > lmin and x < lmax, set(iem_file.geom.get_coord().lon.value[0][0])))
    lat_roi = list(filter(lambda x: x < bmax and x > bmin, set(iem_file.geom.get_coord().lat.value.flatten())))
    L, B = np.meshgrid(sorted(lon_roi), sorted(lat_roi))
    if E_unit == u.MeV:
        gamma_dNdE = [iem_file.get_by_coord((L.flatten(), B.flatten(), [E])).sum() * sr_pix * 1e3 / (sr_pix * float(len(L.flatten()))) for E in iem_file.geom.axes[0].center.value]
        return interp1d(np.log10(iem_file.geom.axes[0].center.value * 1e-3), np.log10(gamma_dNdE), fill_value = 'extrapolate')
    elif E_unit == u.GeV:
        gamma_dNdE = [iem_file.get_by_coord((L.flatten(), B.flatten(), [E])).sum() * sr_pix / (sr_pix * float(len(L.flatten()))) for E in iem_file.geom.axes[0].center.value]
        return interp1d(np.log10(iem_file.geom.axes[0].center.value), np.log10(gamma_dNdE), fill_value = 'extrapolate')
    elif E_unit == u.TeV:
        gamma_dNdE = [iem_file.get_by_coord((L.flatten(), B.flatten(), [E])).sum() * sr_pix * 1e-3 / (sr_pix * float(len(L.flatten()))) for E in iem_file.geom.axes[0].center.value]
        return interp1d(np.log10(iem_file.geom.axes[0].center.value * 1e3), np.log10(gamma_dNdE), fill_value = 'extrapolate')

def get_gammaray_emission_from_IEM_hpx_file(file_name, lmin, lmax, bmin, bmax, E_index, keep_360 = True):
    iem_file = Map.read(file_name)
    E_unit = iem_file.geom.axes[0].center.unit
    sr_pix = np.mean(iem_file.geom.solid_angle().value[0])
    NSIDE = iem_file.geom.nside[0]
    roi_mask = get_lb_mask(NSIDE, lmin, lmax, bmin, bmax, keep_360 = keep_360)
    gamma_dNdE = (iem_file.data * roi_mask).sum(axis = 1) * sr_pix / (np.sum(roi_mask) * 4.0 * np.pi/ (12 * NSIDE**2))
    if E_unit == u.MeV:
        return interp1d(np.log10(iem_file.geom.axes[0].center.value * 1e-3), np.log10(gamma_dNdE), fill_value = 'extrapolate')
    elif E_unit == u.GeV:
        return interp1d(np.log10(iem_file.geom.axes[0].center.value), np.log10(gamma_dNdE), fill_value = 'extrapolate')
    elif E_unit == u.TeV:
        return interp1d(np.log10(iem_file.geom.axes[0].center.value * 1e3), np.log10(gamma_dNdE), fill_value = 'extrapolate')
  
def get_ALP_tibet_interpolation(alp_files, lmin, lmax, bmin, bmax, N0 = 2.418524953009968e-19, Eb = 25.0, alpha = 2.87):
    full_alp_data = np.zeros((len(alp_files), 3))
    couplings = []
    for k, (ma, g, egal_file, mw_file) in enumerate(alp_files):
        alp_data = derive_final_data_points_tibet_ASMD(mw_file, egal_file, N0, Eb, alpha, lmin, lmax, bmin, bmax)
        full_alp_data[k] += alp_data
        couplings.append(g)
    f_egal_SFG_alp_data = interp1d(np.log10(couplings), full_alp_data, axis = 0, fill_value = 'extrapolate')
    return f_egal_SFG_alp_data

def get_ALP_IEM_tibet_interpolation(alp_files, f_iem, f_subPS, lmin, lmax, bmin, bmax, NSIDE = 512, N0 = 'fiducial', Eb = 25.0, alpha = 2.87):
    full_alp_data = np.zeros((len(alp_files), 3))
    full_iem_data = np.zeros((len(alp_files), 3))
    full_subPS_data = np.zeros((len(alp_files), 3))
    couplings = []
    roi_mask = get_lb_mask(NSIDE, lmin, lmax, bmin, bmax, keep_360 = True)
    sr_pix = 4.0 * np.pi/ (12 * NSIDE**2)
    for k, (ma, g, egal_file, mw_file) in enumerate(alp_files):
        alp_data = derive_final_data_points_tibet_ASMD(mw_file, egal_file, N0, Eb, alpha, lmin, lmax, bmin, bmax)
        full_alp_data[k] += alp_data
        f_Pag_GAL = read_MW_ALPgamma_conv_propb_file(mw_file)
        sample_E = np.logspace(4.9, 6.1, 100)
        modulated_IE_data = (np.array([hp.ud_grade((1. - f_Pag_GAL(ee)), NSIDE) * 10**f_iem(np.log10(ee)) for ee in sample_E]) * roi_mask).sum(axis = 1) * sr_pix / (np.sum(roi_mask) * 4.0 * np.pi/ (12 * NSIDE**2))
        f_modulated_iem_flux = interp1d(np.log10(sample_E), np.log10(modulated_IE_data), axis = 0, fill_value = 'extrapolate')
        data_points = get_Tibet_ASMD_data_points(lambda E: 10**f_modulated_iem_flux(np.log10(E)))
        full_iem_data[k] += data_points
        f_Pag_avg = get_lb_astro_ALP_average_modulation(f_Pag_GAL, lmin, lmax, bmin, bmax, 4.9, 6.1, NSIDE = 64)
        f_modulated_subPS_flux = lambda E: f_Pag_avg(E) * 10**f_subPS(np.log10(E*1e-3)) * 1e-3
        data_points = get_Tibet_ASMD_data_points(f_modulated_subPS_flux)
        full_subPS_data[k] += data_points
        couplings.append(g)
    f_egal_SFG_alp_data = interp1d(np.log10(couplings), full_alp_data, axis = 0, fill_value = 'extrapolate')
    f_IEM_data = interp1d(np.log10(couplings), full_iem_data, axis = 0, fill_value = 'extrapolate')
    f_subPS_data = interp1d(np.log10(couplings), full_subPS_data, axis = 0, fill_value = 'extrapolate')
    return f_egal_SFG_alp_data, f_IEM_data, f_subPS_data


def get_ALP_HAWC_interpolation(alp_files, bmin, bmax, lmin = 43.0, lmax = 73.0, N0 = 2.418524953009968e-19):
    full_alp_data = []
    couplings = []
    for k, (ma, g, egal_file, mw_file) in enumerate(alp_files):
        f_Pga_EGAL = read_EGAL_conv_propb_file(egal_file)
        f_Pag_GAL = read_MW_ALPgamma_conv_propb_file(mw_file)
        dE_, f_dNdE_ALP = diffuse_ALP_flux(N0, 25., 2.87, f_Pga_EGAL) 
        out_f = return_flux_at_Earth_in_ROI(f_dNdE_ALP, dE_, f_Pag_GAL, lmin, lmax, bmin, bmax)
        full_alp_data.append(out_f)
        couplings.append(g)
    f_egal_SFG_alp_data = RBSpline(np.log10(couplings), dE_, np.array(full_alp_data), kx = 1, ky = 1, s = 0)
    return f_egal_SFG_alp_data

def get_ALP_HAWC_binned_interpolation(alp_files, bmin, bmax, N0 = 'fiducial', Eb = 25.0, alpha = 2.87):
    sfrd_parameters = {
        'fiducial': (3.4, -0.3, -3.5),
        'high':  (3.6, -0.1, -2.5),
        'low':  (3.2, -0.5, -4.5),
    }
    
    norm_ = {}
    for k, v in sfrd_parameters.items():
        norm_[k] = get_source_spectrum_normalisation(lambda z: SFR_evolution(z, kind = k), lambda E: dNdE_nu(E, 1., Eb, alpha), 1e5, 2.1233333333333335e-8)
    full_alp_data = []
    couplings = []
    for k, (ma, g, egal_file, mw_file) in enumerate(alp_files):
        f_Pga_EGAL = read_EGAL_conv_propb_file(egal_file)
        f_Pag_GAL = read_MW_ALPgamma_conv_propb_file(mw_file)
        dE_, f_dNdE_ALP = diffuse_ALP_flux(norm_[N0], Eb, alpha, f_Pga_EGAL, kind = N0) 
        out_f = return_flux_at_Earth_in_ROI(f_dNdE_ALP, dE_, f_Pag_GAL, 43, 73, bmin, bmax)
        f_gamma_flux = interp1d(dE_, out_f, fill_value = 'extrapolate')
        data_points, energy = get_HAWC_integrated_data_points(f_gamma_flux)
        full_alp_data.append(data_points)
        couplings.append(g)
    f_egal_SFG_alp_data = interp1d(np.log10(couplings), np.array(full_alp_data), axis = 0, fill_value = 'extrapolate')
    return f_egal_SFG_alp_data

def get_ALP_IEM_HAWC_binned_interpolation(alp_files, f_iem, f_subPS, bmin, bmax, NSIDE = 512, N0 = 2.418524953009968e-19, Eb = 25.0, alpha = 2.87):
    
    sfrd_parameters = {
        'fiducial': (3.4, -0.3, -3.5),
        'high':  (3.6, -0.1, -2.5),
        'low':  (3.2, -0.5, -4.5),
    }
    norm_ = {}
    for k, v in sfrd_parameters.items():
        norm_[k] = get_source_spectrum_normalisation(lambda z: SFR_evolution(z, kind = k), lambda E: dNdE_nu(E, 1., Eb, alpha), 1e5, 2.1233333333333335e-8)
    full_alp_data = []
    full_iem_data = []
    full_subPS_data = []
    couplings = []
    roi_mask = get_lb_mask(NSIDE, 43, 73, bmin, bmax, keep_360 = True)
    sr_pix = 4.0 * np.pi/ (12 * NSIDE**2)
    for k, (ma, g, egal_file, mw_file) in enumerate(alp_files):
        f_Pga_EGAL = read_EGAL_conv_propb_file(egal_file)
        f_Pag_GAL = read_MW_ALPgamma_conv_propb_file(mw_file)
        dE_, f_dNdE_ALP = diffuse_ALP_flux(norm_[N0], Eb, alpha, f_Pga_EGAL, kind = N0) 
        out_f = return_flux_at_Earth_in_ROI(f_dNdE_ALP, dE_, f_Pag_GAL, 43, 73, bmin, bmax)
        f_gamma_flux = interp1d(dE_, out_f, fill_value = 'extrapolate')
        data_points, energy = get_HAWC_integrated_data_points(f_gamma_flux)
        full_alp_data.append(data_points)
        sample_E = np.logspace(3.9, 5.1, 100)
        modulated_IE_data = (np.array([hp.ud_grade((1. - f_Pag_GAL(ee)), NSIDE) * 10**f_iem(np.log10(ee)) for ee in sample_E]) * roi_mask).sum(axis = 1) * sr_pix / (np.sum(roi_mask) * 4.0 * np.pi/ (12 * NSIDE**2))
        f_modulated_iem_flux = interp1d(np.log10(sample_E), np.log10(modulated_IE_data), axis = 0, fill_value = 'extrapolate')
        data_points, energy = get_HAWC_integrated_data_points(lambda E: 10**f_modulated_iem_flux(np.log10(E)))
        full_iem_data.append(data_points)
        f_Pag_avg = get_lb_astro_ALP_average_modulation(f_Pag_GAL, 43, 73, bmin, bmax, 3.9, 5.1, NSIDE = 64)
        f_modulated_subPS_flux = lambda E: f_Pag_avg(E) * 10**f_subPS(np.log10(E*1e-3)) * 1e-3
        data_points, energy = get_HAWC_integrated_data_points(f_modulated_subPS_flux)
        full_subPS_data.append(data_points)
        couplings.append(g)
    f_egal_SFG_alp_data = interp1d(np.log10(couplings), np.array(full_alp_data), axis = 0, fill_value = 'extrapolate')
    f_modulated_iem = interp1d(np.log10(couplings), np.array(full_iem_data), axis = 0, fill_value = 'extrapolate')
    f_modulated_subPS = interp1d(np.log10(couplings), np.array(full_subPS_data), axis = 0, fill_value = 'extrapolate')
    return f_egal_SFG_alp_data, f_modulated_iem, f_modulated_subPS

def get_lb_astro_ALP_average_modulation(f_Pag_GAL, lmin, lmax, bmin, bmax, Emin, Emax, NSIDE = 64):
    sample_E = np.logspace(Emin, Emax, 500)
    res_ = np.zeros(len(sample_E))
    for k, energ in enumerate(sample_E):
        reduced_map = get_lb_mask(NSIDE, lmin, lmax, bmin, bmax, keep_360 = True) * (1. - f_Pag_GAL(energ))
        average_Pag = reduced_map[reduced_map > 0.].mean()
        res_[k] += average_Pag 
    return interp1d(sample_E, res_, fill_value = 'extrapolate')

def tibet_chi2_modulated(g_agg, f_ALPflux, f_iem, f_PSsub = None, is_small = True):
    g_agg = np.log10(g_agg)
    tibet_asMD_large = np.array([8.89751e-05, 6.02785e-05, 8.66951e-05])
    tibet_asMD_high_1sigma = np.array([1.09511e-4, 7.61423e-5, 1.27968e-4])
    tibet_asMD_small = np.array([0.000165897, 0.00010263, 0.000198954])
    tibet_asMD_high_1sigma_small = np.array([1.98954e-4, 1.29639e-4, 2.93670e-4])
    iem_data = f_iem(g_agg)
    alp_data = f_ALPflux(g_agg)
    
    total_chi2 = 0.
    if is_small:
        if not f_PSsub is None:
            subPS_data = f_PSsub(g_agg)
            for tibet, sig_tibet, cr_contrib, alp in zip(tibet_asMD_small, tibet_asMD_high_1sigma_small, iem_data + subPS_data, alp_data):
                total_chi2 += ((tibet - alp - cr_contrib) * (tibet - alp - cr_contrib)) / ((sig_tibet - tibet) * (sig_tibet - tibet))
            return total_chi2
        else:
            for tibet, sig_tibet, cr_contrib, alp in zip(tibet_asMD_small, tibet_asMD_high_1sigma_small, iem_data, alp_data):
                total_chi2 += ((tibet - alp - cr_contrib) * (tibet - alp - cr_contrib)) / ((sig_tibet - tibet) * (sig_tibet - tibet))
            return total_chi2
    else:
        if not f_PSsub is None:
            subPS_data = f_PSsub(g_agg)
            for tibet, sig_tibet, cr_contrib, alp in zip(tibet_asMD_large, tibet_asMD_high_1sigma, iem_data + subPS_data, alp_data):
                total_chi2 += ((tibet - alp - cr_contrib) * (tibet - alp - cr_contrib)) / ((sig_tibet - tibet) * (sig_tibet - tibet))
            return total_chi2
        else:
            for tibet, sig_tibet, cr_contrib, alp in zip(tibet_asMD_large, tibet_asMD_high_1sigma, iem_data, alp_data):
                total_chi2 += ((tibet - alp - cr_contrib) * (tibet - alp - cr_contrib)) / ((sig_tibet - tibet) * (sig_tibet - tibet))
            return total_chi2

def hawc_chi2_modulated(g_agg, f_ALPflux, f_iem, f_PSsub = None):
    g_agg = np.log10(g_agg)
    hawc_N0_large = np.array([0.00014081, 0.00014717, 0.00015383, 0.00016078, 0.00016805])
    hawc_N0_large_error = np.array([0.00015709, 0.00016419, 0.00017161, 0.00017936, 0.00018747])
    hawc_N0_small = np.array([0.00022866, 0.00023812, 0.00024797, 0.00025822, 0.0002689 ])
    hawc_N0_small_error = np.array([0.00025052, 0.00026089, 0.00027168, 0.00028291, 0.00029461])
    iem_data = f_iem(g_agg)
    alp_data = f_ALPflux(g_agg)
    
    total_chi2 = 0.
    if not f_PSsub is None:
        subPS_data = f_PSsub(g_agg)
        for tibet, sig_tibet, cr_contrib, alp in zip(hawc_N0_large, hawc_N0_large_error, iem_data + subPS_data, alp_data):
            total_chi2 += ((tibet - alp - cr_contrib) * (tibet - alp - cr_contrib)) / ((sig_tibet - tibet) * (sig_tibet - tibet))
        return total_chi2
    else:
        for tibet, sig_tibet, cr_contrib, alp in zip(hawc_N0_large, hawc_N0_large_error, iem_data, alp_data):
            total_chi2 += ((tibet - alp - cr_contrib) * (tibet - alp - cr_contrib)) / ((sig_tibet - tibet) * (sig_tibet - tibet))
        return total_chi2

def chi2_tibet_binned_modulated(g_agg, f_alp_tibet, f_iem_tibet, f_PSsub_tibet = None, is_small_tibet = True):
    chi2_tibet = tibet_chi2_modulated(g_agg, f_ALPflux = f_alp_tibet, f_iem = f_iem_tibet, f_PSsub = f_PSsub_tibet, is_small = is_small_tibet)
    total_chi2 = chi2_tibet
    return total_chi2

def chi2_hawc_binned_modulated(g_agg, f_alp_hawc, f_iem_hawc, f_PSsub_hawc = None):
    chi2_hawc = hawc_chi2_modulated(g_agg, f_ALPflux = f_alp_hawc, f_iem = f_iem_hawc, f_PSsub = f_PSsub_hawc)
    total_chi2 = chi2_hawc 
    return total_chi2

def joint_chi2_tibet_hawc_binned_modulated(g_agg, f_alp_hawc, f_alp_tibet, f_iem_hawc, f_iem_tibet, f_PSsub_tibet = None, f_PSsub_hawc = None, is_small_tibet = True):
    chi2_hawc = hawc_chi2_modulated(g_agg, f_ALPflux = f_alp_hawc, f_iem = f_iem_hawc, f_PSsub = f_PSsub_hawc)
    chi2_tibet = tibet_chi2_modulated(g_agg, f_ALPflux = f_alp_tibet, f_iem = f_iem_tibet, f_PSsub = f_PSsub_tibet, is_small = is_small_tibet)

    total_chi2 = chi2_hawc + chi2_tibet
    return total_chi2
    
def derive_ULIM_chi2_tibet_modulated(iem_, alp_data = None, is_small_tibet = True, use_sources = None, f_tibet = None, N0 = 'fiducial', dlnL = 2.71):
    if not alp_data is None:
        f_subPS_tibet = use_sources
        f_iem_tibet = iem_
        if is_small_tibet:
            f_tibet = get_ALP_IEM_tibet_interpolation(alp_data, f_iem_tibet, f_subPS_tibet, 25., 100., -5., 5., N0 = N0)
        else:
            f_tibet = get_ALP_IEM_tibet_interpolation(alp_data, f_iem_tibet, f_subPS_tibet, 50., 200., -5., 5., N0 = N0)
        f_alp_tibet, f_iem_tibet, f_PSsub_tibet = f_tibet
    else:
        f_alp_tibet, f_iem_tibet, f_PSsub_tibet = f_tibet
    minimise_me = Minuit(lambda g: chi2_tibet_binned_modulated(g, f_alp_tibet = f_alp_tibet, f_iem_tibet = f_iem_tibet, f_PSsub_tibet = f_PSsub_tibet, is_small_tibet = is_small_tibet), g = 5.0e-11, error_g = 0.01, limit_g = (1e-15, 1e-8))
    minimise_me.migrad()
    ulim = bisect(lambda g: chi2_tibet_binned_modulated(g, f_alp_tibet = f_alp_tibet, f_iem_tibet = f_iem_tibet, f_PSsub_tibet = f_PSsub_tibet, is_small_tibet = is_small_tibet) - minimise_me.fval - dlnL, minimise_me.fitarg["g"], 1e-8)
    if alp_data is None:
        return ulim
    else:
        return ulim, f_tibet
    
def derive_ULIM_chi2_hawc_modulated(iem_, alp_data = None, use_sources = None, f_hawc = None, N0 = 'fiducial', dlnL = 2.71):
    if not alp_data is None:
        f_subPS_hawc = use_sources
        f_iem_hawc = iem_
        f_hawc = get_ALP_IEM_HAWC_binned_interpolation(alp_data, f_iem_hawc, f_subPS_hawc, -4, 4, N0 = N0)
        f_alp_hawc, f_iem_hawc, f_PSsub_hawc = f_hawc
    else:
        f_alp_hawc, f_iem_hawc, f_PSsub_hawc = f_hawc
    minimise_me = Minuit(lambda g: chi2_hawc_binned_modulated(g, f_alp_hawc = f_alp_hawc, f_iem_hawc = f_iem_hawc, f_PSsub_hawc = f_PSsub_hawc), g = 5.0e-11, error_g = 0.01, limit_g = (1e-15, 1e-8))
    minimise_me.migrad()
    ulim = bisect(lambda g: chi2_hawc_binned_modulated(g, f_alp_hawc = f_alp_hawc, f_iem_hawc = f_iem_hawc, f_PSsub_hawc = f_PSsub_hawc) - minimise_me.fval - dlnL, minimise_me.fitarg["g"], 1e-8)
    if alp_data is None:
        return ulim
    else:
        return ulim, f_hawc
    
def derive_ULIM_chi2_joint_tibet_hawc_modulated(iem_, alp_data = None, is_small_tibet = True, use_sources = None, f_hawc = None, f_tibet = None, N0 = 'fiducial', dlnL = 2.71):
    if not alp_data is None:
        f_subPS_tibet, f_subPS_hawc = use_sources
        f_iem_tibet, f_iem_hawc = iem_
        f_hawc = get_ALP_IEM_HAWC_binned_interpolation(alp_data, f_iem_hawc, f_subPS_hawc, -4, 4, N0 = N0)
        if is_small_tibet:
            f_tibet = get_ALP_IEM_tibet_interpolation(alp_data, f_iem_tibet, f_subPS_tibet, 25., 100., -5., 5., N0 = N0)
        else:
            f_tibet = get_ALP_IEM_tibet_interpolation(alp_data, f_iem_tibet, f_subPS_tibet, 50., 200., -5., 5., N0 = N0)
        f_alp_tibet, f_iem_tibet, f_PSsub_tibet = f_tibet
        f_alp_hawc, f_iem_hawc, f_PSsub_hawc = f_hawc
    else:
        f_alp_tibet, f_iem_tibet, f_PSsub_tibet = f_tibet
        f_alp_hawc, f_iem_hawc, f_PSsub_hawc = f_hawc
    minimise_me = Minuit(lambda g: joint_chi2_tibet_hawc_binned_modulated(g, f_alp_hawc = f_alp_hawc, f_alp_tibet = f_alp_tibet, f_iem_hawc = f_iem_hawc, f_iem_tibet = f_iem_tibet, f_PSsub_tibet = f_PSsub_tibet, f_PSsub_hawc = f_PSsub_hawc, is_small_tibet = is_small_tibet), g = 5.0e-11, error_g = 0.01, limit_g = (1e-15, 1e-8))
    minimise_me.migrad()
    ulim = bisect(lambda g: joint_chi2_tibet_hawc_binned_modulated(g, f_alp_hawc = f_alp_hawc, f_alp_tibet = f_alp_tibet, f_iem_hawc = f_iem_hawc, f_iem_tibet = f_iem_tibet, f_PSsub_tibet = f_PSsub_tibet, f_PSsub_hawc = f_PSsub_hawc, is_small_tibet = is_small_tibet) - minimise_me.fval - dlnL, minimise_me.fitarg["g"], 1e-8)
    if alp_data is None:
        return ulim
    else:
        return ulim, f_tibet, f_hawc
    
def prepare_astro_background_components(IE_model_MAX, IE_model_MIN, sub_threshold_flux_HAWC, subthreshhold_flux_TibetASg, path = './'):
    iem_file = Map.read(IE_model_MAX)
    f_IEM_MAX = interp1d(np.log10(iem_file.geom.axes[0].center.value), np.log10(iem_file.data), axis = 0, fill_value = 'extrapolate')

    iem_file = Map.read(IE_model_MIN)
    f_IEM_MIN = interp1d(np.log10(iem_file.geom.axes[0].center.value), np.log10(iem_file.data), axis = 0, fill_value = 'extrapolate')
    
    energy_list, sub_thresh_tibet = np.loadtxt(subthreshhold_flux_TibetASg, unpack = True)
    f_subPS_tibet = interp1d(np.log10(energy_list), np.log10(sub_thresh_tibet), fill_value ='extrapolate')

    energy_list, sub_thresh_hawc = np.loadtxt(sub_threshold_flux_HAWC, unpack = True)
    f_subPS_hawc = interp1d(np.log10(energy_list), np.log10(sub_thresh_hawc), fill_value ='extrapolate')
    
    return (f_IEM_MIN, f_IEM_MAX), (f_subPS_tibet, f_subPS_hawc)

def create_ALP_data_input(m_a, g_agg, Pga_EGAL, Pga_GAL, path = './'):
    ALP_files = []
    for m in m_a:
        current_mass = []
        current_egal = list(filter(lambda x: str(m)+"neV" in x, Pga_EGAL))
        current_gal = list(filter(lambda x: str(m)+"neV" in x, Pga_GAL))
        for g, egal, gal in zip(g_agg, current_egal, current_gal):
            current_mass.append((m, g * 1e-11, path + egal, path + gal))
        ALP_files.append(current_mass)
    return ALP_files
    
def derive_ALP_bounds(ALP_component, IE_component, subPS_component, is_IEM_uncertainty = True, N0 = 'fiducial', dlnL = 3.84):
    limits_95percent_joint = {}
    f_subPS_tibet_small, f_subPS_hawc_large = subPS_component
    if is_IEM_uncertainty:
        f_IEM_MIN, f_IEM_MAX = IE_component
        for current_mass in ALP_component:
            m_a = current_mass[0][0]
            ulim_max, f_tibet_, f_hawc = derive_ULIM_chi2_joint_tibet_hawc_modulated((f_IEM_MAX, f_IEM_MAX), alp_data = current_mass, is_small_tibet = True, use_sources = (f_subPS_tibet_small, f_subPS_hawc_large), f_hawc = None, f_tibet = None, N0 = N0, dlnL = dlnL)
            ulim_min, f_tibet, f_hawc = derive_ULIM_chi2_joint_tibet_hawc_modulated((f_IEM_MIN, f_IEM_MIN), alp_data = current_mass, is_small_tibet = True, use_sources = (f_subPS_tibet_small, f_subPS_hawc_large), f_hawc = None, f_tibet = None, N0 = N0, dlnL = dlnL)
            limits_95percent_joint["M_{}neV".format(m_a)] = (ulim_min, ulim_max)
        
        m_a = [1e-2]
        lim_min = [limits_95percent_joint["M_1.0neV"][0]]
        lim_max = [limits_95percent_joint["M_1.0neV"][1]]
        for k, v in limits_95percent_joint.items():
            m_a.append(float(k.replace("M_", "").replace('neV', '')))
            lim_min.append(v[0])
            lim_max.append(v[1])
        f_alp_lims_min = interp1d(np.log10(m_a), np.log10(lim_min), kind = 'slinear', fill_value = 'extrapolate')
        f_alp_lims_max = interp1d(np.log10(m_a), np.log10(lim_max), kind = 'slinear', fill_value = 'extrapolate')
        
        ma_test = np.linspace(-4, 5, 5000)
        np.savetxt("joint_limits_95CL_HAWC_TibetASg_gammaIEM_MIN_subthresh_modulated_full_couplings.dat", np.vstack((10**ma_test * 1e-9, 10**f_alp_lims_min(ma_test))).T)
        np.savetxt("joint_limits_95CL_HAWC_TibetASg_gammaIEM_MAX_subthresh_modulated_full_couplings.dat", np.vstack((10**ma_test * 1e-9, 10**f_alp_lims_max(ma_test))).T)
    else:
        #### This corresponds to the B-field uncertainty study
        f_IEM = IE_component
        for current_mass in ALP_component:
            m_a = current_mass[0][0]
            ulim_Bconst, f_tibet_, f_hawc = derive_ULIM_chi2_joint_tibet_hawc_modulated((f_IEM, f_IEM), alp_data = current_mass, is_small_tibet = True, use_sources = (f_subPS_tibet_small, f_subPS_hawc_large), f_hawc = None, f_tibet = None, N0 = N0, dlnL = dlnL)
            limits_95percent_joint["M_{}neV".format(m_a)] = ulim_max
            
        m_a = [1e-2]
        lim_Bconst = [limits_95percent_joint["M_1.0neV"][0]]
        for k, v in limits_95percent_joint.items():
            m_a.append(float(k.replace("M_", "").replace('neV', '')))
            lim_tmp.append(v)
        f_alp_lims = interp1d(np.log10(m_a), np.log10(lim_tmp), kind = 'slinear', fill_value = 'extrapolate')
        
        ma_test = np.linspace(-4, 5, 5000)
        np.savetxt("joint_limits_95CL_HAWC_TibetASg_gammaIEM_MIN_subthresh_modulated_Bconst.dat", np.vstack((10**ma_test * 1e-9, 10**f_alp_lims(ma_test))).T)
        
def main():
    
    parser = argparse.ArgumentParser(description = "Derive ALP bounds from the cumulative emission of star-forming galaxies according to [Eckner, Calore, arXiv:2204.12487].")

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
    
    with open(options.yaml, 'r') as f:
        inputs = yaml.load(f)
        
    ALP_data = create_ALP_data_input(inputs["ALP_bounds"]["m_a"], 
                                     inputs["ALP_bounds"]["g_agg"], 
                                     inputs["ALP_bounds"]["ALP_data_EGAL"], 
                                     inputs["ALP_bounds"]["ALP_data_GAL"],
                                     path = inputs["ALP_bounds"]["path"])
    
    IE_comp, subPS_comp = prepare_astro_background_components(
        IE_model_MAX = inputs["ALP_bounds"]["IE_model_max"], 
        IE_model_MIN = inputs["ALP_bounds"]["IE_model_min"], 
        sub_threshold_flux_HAWC = inputs["ALP_bounds"]["subPS_HAWC"], 
        subthreshhold_flux_TibetASg = inputs["ALP_bounds"]["subPS_TibetASg"],
    )
    derive_ALP_bounds(
        ALP_component = ALP_data,
        IE_component = IE_comp,
        subPS_component = subPS_comp,
    )
    
if __name__ == "__main__":
    main()
        
    
