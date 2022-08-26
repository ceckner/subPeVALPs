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

def PowerLaw(E, alpha, E0, N0):
    return np.power(E/E0, alpha) * N0

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

def get_ALP_flux_interpolation_in_ROI(alp_files, bmin, bmax, lmin = 43.0, lmax = 73.0, N0 = 'fiducial', Eb = 25.0, alpha =2.87):
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
        dE_, f_dNdE_ALP = diffuse_ALP_flux(norm_[N0], Eb, alpha, f_Pga_EGAL) 
        out_f = return_flux_at_Earth_in_ROI(f_dNdE_ALP, dE_, f_Pag_GAL, lmin, lmax, bmin, bmax)
        full_alp_data.append(out_f)
        couplings.append(g)
    f_egal_SFG_alp_data = RBSpline(np.log10(couplings), dE_, np.array(full_alp_data), kx = 1, ky = 1, s = 0)
    return f_egal_SFG_alp_data

def plotting_routine(yaml_, N0 = 'fiducial', kind = 'TibetASg', source_path = './'):
    with open(yaml_, 'r') as f:
        inputs = yaml.load(f)
        
    IEM_MIN = inputs["ALP_bounds"]["IE_model_min"]  
    IEM_MAX = inputs["ALP_bounds"]["IE_model_max"]  
    
    subPS_data = inputs["ALP_bounds"]["subPS_TibetASg"]  
    
    ALP_data = create_ALP_data_input(inputs["plotting"]["ALP_data"]["m_a"], 
                                     inputs["plotting"]["ALP_data"]["g_agg"], 
                                     inputs["plotting"]["ALP_data"]["ALP_data_EGAL"], 
                                     inputs["plotting"]["ALP_data"]["ALP_data_GAL"],
                                     path = inputs["plotting"]["ALP_data"]["path"],
    )[0]
    g_agg_ref = inputs["plotting"]["ALP_data"]["g_agg_plot"]
    
    if kind == 'TibetASg':
        tibet_ASMD_points_small = [
            (1.19378e+2, 1.65897e-4, 1.98954e-4, 1.31333e-4, 1.00000e+2, 1.61730e+2), 
            (2.19109e+2, 1.02630e-4, 1.29639e-4, 7.71370e-5, 1.57689e+2, 4.02157e+2),
            (5.31220e+2, 1.98954e-4, 2.93670e-4, 1.29639e-4, 4.02157e+2, 1.00000e+3),
        ]
        
        lmin, lmax = inputs["plotting"]["TibetASg"]["GLON"]
        bmin, bmax = inputs["plotting"]["TibetASg"]["GLAT"]
    
        f_ALP_Tibet = get_ALP_flux_interpolation_in_ROI(ALP_data, lmin = lmin, lmax = lmax, bmin = bmin, bmax = bmax, N0 = N0)
        f_IEM_MAX_tibet = get_gammaray_emission_from_IEM_hpx_file(IEM_MAX, lmin, lmax, bmin, bmax, 0)
        f_IEM_MIN_tibet = get_gammaray_emission_from_IEM_hpx_file(IEM_MIN, lmin, lmax, bmin, bmax, 0)
        
        energy_list, sub_thresh_pop_PL = np.loadtxt(subPS_data, unpack = True)
        f_sub_PL = interp1d(np.log10(energy_list), np.log10(sub_thresh_pop_PL), fill_value ='extrapolate')

        x_size = 20
        y_size = 20 / ((1. + np.sqrt(5))/2.0)

        fig = plt.figure(figsize = (x_size, y_size))
        ax = fig.gca()

        plt.rc('xtick', labelsize=40)    
        plt.rc('ytick', labelsize=40)
        ax.tick_params(labelsize = 40)
        
        cr_E = np.logspace(2, 5, 500)
        E_alp = np.logspace(3, 6, 5000)
        E_astro = E_alp * 1e-3

        plt.errorbar(tibet_ASMD_points_small[0][0], tibet_ASMD_points_small[0][1], yerr = [[tibet_ASMD_points_small[0][1] - tibet_ASMD_points_small[0][3]], [tibet_ASMD_points_small[0][2]-tibet_ASMD_points_small[0][1]]], xerr = [[tibet_ASMD_points_small[0][0] - tibet_ASMD_points_small[0][4]], [tibet_ASMD_points_small[0][5] - tibet_ASMD_points_small[0][0]]], capsize = 3, lw = 5, color = 'crimson', label = 'Tibet AS$\gamma$\n($25^{\\circ}< \\ell < 100^{\\circ}$)')
        plt.errorbar(tibet_ASMD_points_small[1][0], tibet_ASMD_points_small[1][1], yerr = [[tibet_ASMD_points_small[1][1] - tibet_ASMD_points_small[1][3]], [tibet_ASMD_points_small[1][2]-tibet_ASMD_points_small[1][1]]], xerr = [[tibet_ASMD_points_small[1][0] - tibet_ASMD_points_small[1][4]], [tibet_ASMD_points_small[1][5] - tibet_ASMD_points_small[1][0]]], capsize = 3, lw = 5, color = 'crimson')
        plt.errorbar(tibet_ASMD_points_small[2][0], tibet_ASMD_points_small[2][1], yerr = [[tibet_ASMD_points_small[2][1] - tibet_ASMD_points_small[2][3]], [tibet_ASMD_points_small[2][2]-tibet_ASMD_points_small[2][1]]], xerr = [[tibet_ASMD_points_small[2][0] - tibet_ASMD_points_small[2][4]], [tibet_ASMD_points_small[2][5] - tibet_ASMD_points_small[2][0]]], capsize = 3, lw = 5, color = 'crimson')

        plt.loglog(E_astro, 10**f_IEM_MAX_tibet(np.log10(E_astro*1e3))  * (E_astro * 1e3)**2.7, lw = 5,color = '#c2a5cf', label = '$\Phi^{\mathrm{IE}}$ (\\texttt{MAX})')
        plt.loglog(E_astro, 10**f_IEM_MIN_tibet(np.log10(E_astro*1e3))  * (E_astro * 1e3)**2.7, lw = 5, ls = '--', color = '#c2a5cf', label = '$\Phi^{\mathrm{IE}}$ (\\texttt{MIN})')

        plt.loglog(E_astro, 10**f_sub_PL(np.log10(E_astro)) * (E_astro)**2.7 * 1e3**1.7, lw = 5., ls = '-', color = '#1b7837', label = '$\Phi^{\mathrm{sTH}}$')

        plt.loglog(E_astro, 10**f_sub_PL(np.log10(E_astro)) * (E_astro)**2.7 * 1e3**1.7 + 10**f_IEM_MAX_tibet(np.log10(E_astro*1e3))  * (E_astro * 1e3)**2.7, lw = 5., ls = '-', color = '#762A83', label = '$\Phi^{\mathrm{astro}}$ (\\texttt{MAX})')
        plt.loglog(E_astro, 10**f_sub_PL(np.log10(E_astro)) * (E_astro)**2.7 * 1e3**1.7 + 10**f_IEM_MIN_tibet(np.log10(E_astro*1e3))  * (E_astro * 1e3)**2.7, lw = 5., ls = '--', color = '#762A83', label = '$\Phi^{\mathrm{astro}}$ (\\texttt{MIN})')


        E_alp = np.logspace(3, 6, 5000)
        plt.loglog(E_alp * 1e-3, f_ALP_Tibet(np.log10(g_agg_ref * 1e-11), E_alp)[0] * E_alp**2.7, lw = 5, color = '#e66101', label = '$\Phi^{\mathrm{ALP}}$')
        
        plt.text(1.2, 4e-6, '$m_a = ' + str(inputs["plotting"]["ALP_data"]["m_a"]) + '$ neV\n$g_{a\gamma\gamma} \sim ' + str(g_agg_ref) + '\\times10^{-11}\;\mathrm{GeV}^{-1}$', fontsize = 35)

        plt.ylim([3e-6, 1e-3])
        plt.xlim([1e-0, 1e3])

        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_tick_params(width=2, length=10, which = 'major')
        ax.yaxis.set_tick_params(width=1, length=7, which = 'minor')
        ax.xaxis.set_tick_params(width=2, length=10, which = 'major')
        ax.xaxis.set_tick_params(width=1, length=7, which = 'minor')

        plt.xlabel('$E$ [TeV]', fontsize = 50)
        plt.ylabel(r'$E^{2.7}\frac{\mathrm{d}\Phi}{\mathrm{d}E}\;\left[\mathrm{GeV}^{1.7}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}\right]$', fontsize = 50)

        ax.legend(loc = 2, fontsize  = 35, ncol = 3, facecolor = 'None', edgecolor = 'None')
        plt.savefig(source_path + "spectral_comparison_astro_ALP_TibetASg_ROI_GLON_{}deg_{}deg_GLAT_{}deg_{}deg.pdf".format(lmin, lmax, bmin, bmax), dpi = 600, bbox_inches = 'tight')
    if kind == 'HAWC':
        tibet_ASMD_points_small = [
            (1.19378e+2, 1.65897e-4, 1.98954e-4, 1.31333e-4, 1.00000e+2, 1.61730e+2), 
            (2.19109e+2, 1.02630e-4, 1.29639e-4, 7.71370e-5, 1.57689e+2, 4.02157e+2),
            (5.31220e+2, 1.98954e-4, 2.93670e-4, 1.29639e-4, 4.02157e+2, 1.00000e+3),
        ]
        
        lmin, lmax = inputs["plotting"]["HAWC"]["GLON"]
        bmin, bmax = inputs["plotting"]["HAWC"]["GLAT"]
    
        f_ALP_Tibet = get_ALP_flux_interpolation_in_ROI(ALP_data, lmin = lmin, lmax = lmax, bmin = bmin, bmax = bmax, N0 = N0)
        f_IEM_MAX_tibet = get_gammaray_emission_from_IEM_hpx_file(IEM_MAX, lmin, lmax, bmin, bmax, 0)
        f_IEM_MIN_tibet = get_gammaray_emission_from_IEM_hpx_file(IEM_MIN, lmin, lmax, bmin, bmax, 0)
        
        energy_list, sub_thresh_pop_PL = np.loadtxt(subPS_data, unpack = True)
        f_sub_PL = interp1d(np.log10(energy_list), np.log10(sub_thresh_pop_PL), fill_value ='extrapolate')

        x_size = 20
        y_size = 20 / ((1. + np.sqrt(5))/2.0)

        fig = plt.figure(figsize = (x_size, y_size))
        ax = fig.gca()

        plt.rc('xtick', labelsize=40)    
        plt.rc('ytick', labelsize=40)
        ax.tick_params(labelsize = 40)

        cr_E = np.logspace(2, 5, 500)
        E_alp = np.logspace(3, 6, 5000)
        E_astro = E_alp * 1e-3

        plt.fill_between(cr_E*1e-3, PowerLaw(cr_E, -2.604, 7000., (5.45 + 0.25 + 0.38) * 1e-15) * cr_E**2.7, PowerLaw(cr_E, -2.604, 7000., (5.45 - 0.25 - 0.44) * 1e-15) * cr_E**2.7,  color = 'crimson', lw = 1, alpha = 0.5, label = 'HAWC: Galactic diffuse')
        plt.loglog(cr_E*1e-3, PowerLaw(cr_E, -2.604, 7000., 5.45e-15) * cr_E**2.7, color = 'crimson', lw = 5)

        plt.loglog(E_astro, 10**f_IEM_MAX_tibet(np.log10(E_astro*1e3))  * (E_astro * 1e3)**2.7, lw = 5,color = '#c2a5cf', label = '$\Phi^{\mathrm{IE}}$ (\\texttt{MAX})')
        plt.loglog(E_astro, 10**f_IEM_MIN_tibet(np.log10(E_astro*1e3))  * (E_astro * 1e3)**2.7, lw = 5, ls = '--', color = '#c2a5cf', label = '$\Phi^{\mathrm{IE}}$ (\\texttt{MIN})')

        plt.loglog(E_astro, 10**f_sub_PL(np.log10(E_astro)) * (E_astro)**2.7 * 1e3**1.7, lw = 5., ls = '-', color = '#1b7837', label = '$\Phi^{\mathrm{sTH}}$')

        plt.loglog(E_astro, 10**f_sub_PL(np.log10(E_astro)) * (E_astro)**2.7 * 1e3**1.7 + 10**f_IEM_MAX_tibet(np.log10(E_astro*1e3))  * (E_astro * 1e3)**2.7, lw = 5., ls = '-', color = '#762A83', label = '$\Phi^{\mathrm{astro}}$ (\\texttt{MAX})')
        plt.loglog(E_astro, 10**f_sub_PL(np.log10(E_astro)) * (E_astro)**2.7 * 1e3**1.7 + 10**f_IEM_MIN_tibet(np.log10(E_astro*1e3))  * (E_astro * 1e3)**2.7, lw = 5., ls = '--', color = '#762A83', label = '$\Phi^{\mathrm{astro}}$ (\\texttt{MIN})')


        E_alp = np.logspace(3, 6, 5000)
        plt.loglog(E_alp * 1e-3, f_ALP_Tibet(np.log10(g_agg_ref * 1e-11), E_alp)[0] * E_alp**2.7, lw = 5, color = '#e66101', label = '$\Phi^{\mathrm{ALP}}$')
        
        plt.text(1.2, 4e-6, '$m_a = ' + str(inputs["plotting"]["ALP_data"]["m_a"]) + '$ neV\n$g_{a\gamma\gamma} \sim ' + str(g_agg_ref) + '\\times10^{-11}\;\mathrm{GeV}^{-1}$', fontsize = 35)

        plt.ylim([3e-6, 1e-3])
        plt.xlim([1e-0, 1e3])

        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_tick_params(width=2, length=10, which = 'major')
        ax.yaxis.set_tick_params(width=1, length=7, which = 'minor')
        ax.xaxis.set_tick_params(width=2, length=10, which = 'major')
        ax.xaxis.set_tick_params(width=1, length=7, which = 'minor')

        plt.xlabel('$E$ [TeV]', fontsize = 50)
        plt.ylabel(r'$E^{2.7}\frac{\mathrm{d}\Phi}{\mathrm{d}E}\;\left[\mathrm{GeV}^{1.7}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}\right]$', fontsize = 50)

        ax.legend(loc = 2, fontsize  = 35, ncol = 3, facecolor = 'None', edgecolor = 'None')
        plt.savefig(source_path + "spectral_comparison_astro_ALP_HAWC_ROI_GLON_{}deg_{}deg_GLAT_{}deg_{}deg.pdf".format(lmin, lmax, bmin, bmax), dpi = 600, bbox_inches = 'tight')

def main():
    parser = argparse.ArgumentParser(description = "Plot the flux contribution of all emission components (incl. ALPs) according to Fig. 4 in [Eckner, Calore, arXiv:2204.12487].")

    parser.add_argument('--model_def', '-file',
                        type=str,
                        dest='yaml',
                        help='YAML file containing the parameter definitions for the plotting script.',
                        default='analysis_definition_file.yaml'
    )
    parser.add_argument('--kind', '-exp',
                        type=str,
                        dest='kind',
                        help='Define for which experiment to plot the emission components. Accepted values: "TibetASg", "HAWC"',
                        default='TibetASg',
    )
    parser.add_argument('--root',
                        type=str,
                        dest='path',
                        help='Path to working directory.',
                        default='./'
    )
    options = parser.parse_args()
    
    plotting_routine(yaml_ = options.yaml,
                     kind = options.kind,
                     source_path = options.path,
    )
    
if __name__ == "__main__":
    main()
