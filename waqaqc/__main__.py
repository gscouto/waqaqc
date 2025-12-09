from . import run_waqaqc
import argparse
import sys

from waqaqc.config import defaults


def main():
    parser = argparse.ArgumentParser(prog='waqaqc - WEAVE-Apertif Quality Assurance (WAQA) Quality Control plots')

    parser.add_argument("data_path", type=str,
                        help="Full path to the main data folder")
    parser.add_argument("ob_list", type=str,
                        help="List of OBs to be processed in a list format, e.g. ['11111','11112','11113']")
    parser.add_argument("temp_path", type=str,
                        help="Stellar templates path for PyParadise fitting.")
    parser.add_argument("temp_file", type=str,
                        help="Stellar templates file to be used (located in temp_path).")
    parser.add_argument("-ac", "--aps_cube", type=int, default=defaults['aps_cube'],
                        help="Flag to run the APS cube creator. 1 = yes / 0 = no.")
    parser.add_argument("-qp", "--qc_plots", type=int, default=defaults['qc_plots'],
                        help="Flag to run the QC plots. 1 = yes / 0 = no.")
    parser.add_argument("-pp", "--pyp_params", type=int, default=defaults['pyp_params'],
                        help="Flag to run the creation of PyParadise parameters files. 1 = yes / 0 = no.")
    parser.add_argument("-sf", "--spec_fit", type=int, default=defaults['spec_fit'],
                        help="Flag to run PyParadise (spectral fitting). 1 = yes / 0 = no.")
    parser.add_argument("-tc", "--table_creator", type=int, default=defaults['table_creator'],
                        help="Flag to run the table creator. 1 = yes / 0 = no.")
    parser.add_argument("-n", "--nproc", type=int, default=defaults['nproc'],
                        help="Processing cores number to be used in parallel when fitting the spectra.")
    parser.add_argument("-af", "--aps_flag", type=int, default=defaults['aps_flag'],
                        help="Flag to include APS data in the QC plots. 1 = yes / 0 = no.")
    parser.add_argument("-spf", "--sky_plot_flag", type=int, default=defaults['sky_plot_flag'],
                        help="Flag to save fit plots of the warc and sky lines. 1 = yes / 0 = no.")
    parser.add_argument("-t_snr", "--target_snr", type=float, default=defaults['target_snr'],
                        help="Target signal to noise ratio for the Voronoi binning")
    parser.add_argument("-lvls", "--levels", type=float, default=defaults['levels'],
                        help="Contour levels list on SNR maps, e.g. [5, 30]")
    parser.add_argument("-bw", "--blue_wav", type=float, default=defaults['blue_wav'],
                        help="Central wavelength of the window used to measure SNR (corrected by redshift) "
                             "for the blue arm.")
    parser.add_argument("-rw", "--red_wav", type=float, default=defaults['red_wav'],
                        help="Central wavelength of the window used to measure SNR (corrected by redshift) "
                             "for the red arm.")
    parser.add_argument("-aw", "--aps_wav", type=float, default=defaults['aps_wav'],
                        help="Central wavelength of the window used to measure SNR for the APS spectra.")
    parser.add_argument("-cf", "--cov_flag", type=int, default=defaults['cov_flag'],
                        help="Flag to apply covariance correction to Voroni binning. 1 = yes / 0 = no.")
    parser.add_argument("-r", "--redshift", type=float, default=defaults['redshift'],
                        help="Redshift list for each OB galaxy, with same size as the 'ob_list'. If no value is given, "
                             "default is zero. Note: if an APS file is found within the OB directory, the redshift "
                             "value found in this file is used instead, e.g. [0.023, 0.001].")
    parser.add_argument("-bff", "--blue_fit_flag", type=int, default=defaults['blue_fit_flag'],
                        help="Flag for fitting the blue arm datacube with PyParadise. 1 = yes / 0 = no.")
    parser.add_argument("-rff", "--red_fit_flag", type=int, default=defaults['red_fit_flag'],
                        help="Flag for fitting the red arm datacube with PyParadise. 1 = yes / 0 = no.")
    parser.add_argument("-aff", "--aps_fit_flag", type=int, default=defaults['aps_fit_flag'],
                        help="Flag for fitting the APS datacube with PyParadise. 1 = yes / 0 = no.")
    parser.add_argument("-vr", "--vel_range", type=float, default=defaults['vel_range'],
                        help="Velocity range where PyParadise is allowed to fit centered around the given redshift "
                             "systemic velocity (in km/s).")
    parser.add_argument("-dm", "--disp_max", type=float, default=defaults['disp_max'],
                        help="Maximum velocity dispersion allowed to the stellar continuum fit (in km/s).")
    parser.add_argument("-lf", "--line_flux", type=float, default=defaults['line_flux'],
                        help="Initial guess for the emission line fluxes (order of magnitude values).")
    parser.add_argument("-lv", "--line_vel", type=float, default=defaults['line_vel'],
                        help="Velocity initial guess (in km/s). If redshift values were obtained (either from input or "
                             "APS file), -the systemic velocity is added within the code.")
    parser.add_argument("-amin", "--aps_lam_min", type=float, default=defaults['aps_lam_min'],
                        help="Lower wavelength limit for spectral fitting the APS spectra.")
    parser.add_argument("-amax", "--aps_lam_max", type=float, default=defaults['aps_lam_max'],
                        help="Upper wavelength limit for spectral fitting the APS spectra.")
    parser.add_argument("-bmin", "--blue_lam_min", type=float, default=defaults['blue_lam_min'],
                        help="Lower wavelength limit for spectral fitting the blue arm spectra.")
    parser.add_argument("-bmax", "--blue_lam_max", type=float, default=defaults['blue_lam_max'],
                        help="Upper wavelength limit for spectral fitting the blue arm spectra.")
    parser.add_argument("-rmin", "--red_lam_min", type=float, default=defaults['red_lam_min'],
                        help="Lower wavelength limit for spectral fitting the red arm spectra.")
    parser.add_argument("-rmax", "--red_lam_max", type=float, default=defaults['red_lam_max'],
                        help="Upper wavelength limit for spectral fitting the red arm spectra.")
    parser.add_argument("-vf", "--vorbin_flag", type=int, default=defaults['vorbin_flag'],
                        help="Flag for doing the spectral fit in the Voronoi binned datacube (1), or use the created "
                             "SNR filtered datacube (0).")
    parser.add_argument("-rf", "--resol_flag", type=int, default=defaults['resol_flag'],
                        help="Flag for instrumental FWHM used. 1 is for using the derived from the QC plots, 0 "
                             "is for using given values.")
    parser.add_argument("-fwhmb", "--fwhm_blue", type=float, default=defaults['fwhm_blue'],
                        help="Constant (both spatially and with wavelength) instrumental FWHM for the blue arm "
                             "spectral fitting (in Ang).")
    parser.add_argument("-fwhmr", "--fwhm_red", type=float, default=defaults['fwhm_red'],
                        help="Constant (both spatially and with wavelength) instrumental FWHM for the red arm "
                             "spectral fitting (in Ang).")
    parser.add_argument("-fwhma", "--fwhm_aps", type=float, default=defaults['fwhm_aps'],
                        help="Constant (both spatially and with wavelength) instrumental FWHM for the APS "
                             "spectral fitting (in Ang).")
    parser.add_argument("-ef", "--el_flag", type=int, default=defaults['el_flag'],
                        help="Flag for doing emission-line fitting within PyParadise. 1 = yes / 0 = no.")
    parser.add_argument("-bf", "--boot_flag", type=int, default=defaults['boot_flag'],
                        help="Flag for doing bootstrap in PyParadise. 1 = yes / 0 = no.")
    parser.add_argument("-scf", "--sigmaclip_flag", type=int, default=defaults['sigmaclip_flag'],
                        help="Flag for doing sigma clipping before running PyParadise. 1 = yes / 0 = no.")
    parser.add_argument("-scl", "--sigmaclip_limit", type=float, default=defaults['sigmaclip_limit'],
                        help="Arbitrary value to be used as a limit to the sigma clipping. It is used as a multiple "
                             "factor to derived standard deviation.")

    args = parser.parse_args()

    if args.redshift is None:
        args.redshift = [defaults['redshift']] * len(args.ob_list)
        # validate size
    elif len(args.redshift) != len(args.ob_list):
        print(
            f"Error: redshift list has {len(args.redshift)} entries but ob_list has {len(args.ob_list)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_waqaqc.run(args=args)


if __name__ == '__main__':
    main()
