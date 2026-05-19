import os
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, FK5
from astropy.stats import sigma_clip
import astropy.units as u
from astroquery.ipac.ned import Ned
from vorbin.voronoi_2d_binning import voronoi_2d_binning
import warnings
import multiprocessing as mp
from scipy.optimize import curve_fit
import re
from waqaqc import QC_plots_L0

matplotlib.use("Agg")


def vorbin_loop(args):
    i, vorbin_m, cam = args

    nb_pixs = np.where(vorbin_m == i)
    if cam == 'WEAVEBLUE':
        nbc = np.nanmean(blue_cube_data[:, nb_pixs[1], nb_pixs[0]], axis=1)
        nbc_err = np.nanmean(blue_cube_err[:, nb_pixs[1], nb_pixs[0]], axis=1)
    elif cam == 'WEAVERED':
        nbc = np.nanmean(red_cube_data[:, nb_pixs[1], nb_pixs[0]], axis=1)
        nbc_err = np.nanmean(red_cube_err[:, nb_pixs[1], nb_pixs[0]], axis=1)
    elif cam == 'APS':
        nbc = np.nanmean(aps_cube_data[:, nb_pixs[1], nb_pixs[0]], axis=1)
        nbc_err = np.nanmean(aps_cube_err[:, nb_pixs[1], nb_pixs[0]], axis=1)
    else:
        nbc = np.zeros(np.shape(blue_cube_data)[0])
        nbc_err = np.zeros(np.shape(blue_cube_data)[0])

    return nbc, nbc_err, nb_pixs


def gauss_hermite(x, a, b, amp, x0, sigma, h3=0.0, h4=0.0):
    """
    Gauss–Hermite function for fitting asymmetric spectral lines.

    Parameters
    ----------
    x : array-like
        Input wavelength or pixel array.
    a, b : float
        Linear continuum parameters (baseline = a + b*x).
    amp : float
        Amplitude of the Gaussian component.
    x0 : float
        Central position (mean).
    sigma : float
        Standard deviation (dispersion).
    h3, h4 : float, optional
        Gauss–Hermite coefficients for skewness (h3) and kurtosis (h4).

    Returns
    -------
    model : array
        Gauss–Hermite line profile.
    """
    y = (x - x0) / sigma
    gauss_f = np.exp(-0.5 * y ** 2)

    # Normalized Hermite polynomials H3 and H4 (physicists' definition)
    H3 = (2 * y ** 3 - 3 * y) / np.sqrt(6)
    H4 = (4 * y ** 4 - 12 * y ** 2 + 3) / np.sqrt(24)

    hermite = (1 + h3 * H3 + h4 * H4)
    return a + b * x + amp * gauss_f * hermite


def get_xy_peak_positions(
        cube, wave, bin_size=20, use_centroid_if_fail=True,
        clip_sigma=5.0, clip_iters=3, preserve_radius=20
):
    nwave, ny, nx = cube.shape
    n_bins = nwave // bin_size
    cy, cx = ny // 2, nx // 2  # image center

    y_grid, x_grid = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    r_grid = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    preserve_mask = r_grid <= preserve_radius

    x_axis = np.arange(nx)
    y_axis = np.arange(ny)
    x_peaks = np.full(n_bins, np.nan, dtype=np.float32)
    y_peaks = np.full(n_bins, np.nan, dtype=np.float32)
    central_waves = np.full(n_bins, np.nan, dtype=np.float32)

    for b in range(n_bins):
        i_start = b * bin_size
        i_end = i_start + bin_size

        subcube = cube[i_start:i_end]
        collapsed = np.nansum(subcube, axis=0)  # (ny, nx)

        # Mask outside the core for sigma clipping
        clipped = sigma_clip(collapsed, sigma=clip_sigma, maxiters=clip_iters, masked=True)
        clipped_mask = clipped.mask

        # Ensure center is *never* masked
        clipped_mask[preserve_mask] = False

        clean = np.where(clipped_mask, 0.0, collapsed)

        # Get 1D profiles
        profile_x = np.sum(clean, axis=0)
        profile_y = np.sum(clean, axis=1)
        central_waves[b] = np.mean(wave[i_start:i_end])

        try:
            popt_x, _ = curve_fit(
                gaussian, x_axis, profile_x,
                p0=[np.max(profile_x), np.argmax(profile_x), 3.0, np.min(profile_x)]
            )
            x_peaks[b] = popt_x[1]
        except:
            if use_centroid_if_fail:
                x_peaks[b] = np.sum(x_axis * profile_x) / np.sum(profile_x)

        try:
            popt_y, _ = curve_fit(
                gaussian, y_axis, profile_y,
                p0=[np.max(profile_y), np.argmax(profile_y), 3.0, np.min(profile_y)]
            )
            y_peaks[b] = popt_y[1]
        except:
            if use_centroid_if_fail:
                y_peaks[b] = np.sum(y_axis * profile_y) / np.sum(profile_y)

    return central_waves, x_peaks, y_peaks


def html_plots(ob, redshift, args):
    warnings.filterwarnings("ignore")

    file_dir = args.data_path + ob + '/'

    if len([x for x in os.listdir(file_dir) if ('LWVE' in x)]) > 0:
        aps_files = [x for x in os.listdir(file_dir) if 'LWVE' in x]
        largest_file = max(aps_files, key=lambda fl: os.path.getsize(os.path.join(file_dir, fl)))
        aps_file = fits.open(os.path.join(file_dir, largest_file))
        redshift = round(aps_file['PATCH_TABLE'].data['Z'][0], 6)

    global blue_cube_data, blue_cube_err, red_cube_data, red_cube_err, aps_cube_data, aps_cube_err

    blue_cube = fits.open(file_dir + np.sort([x for x in os.listdir(file_dir) if ('stackcube' in x)])[1])
    red_cube = fits.open(file_dir + np.sort([x for x in os.listdir(file_dir) if ('stackcube' in x)])[0])

    blue_cube_data = blue_cube[1].data
    blue_cube_err = blue_cube[2].data

    red_cube_data = red_cube[1].data
    red_cube_err = red_cube[2].data

    gal_name = blue_cube[0].header['CCNAME1']
    date = blue_cube[0].header['DATE-OBS']

    gal_dir = str(blue_cube[0].header['OBID']) + '_' + gal_name + '_' + blue_cube[0].header['MODE'] + '/'
    os.makedirs(gal_dir, exist_ok=True)

    output_str = str(blue_cube[0].header['OBID']) + '_' + date + '_' + gal_name + '_' + blue_cube[0].header['MODE']

    # wa_id_kw = blue_cube[0].header.get('INFILE')
    #
    # if wa_id_kw is not None:
    #     wa_id = wa_id_kw.split('_')[3]
    # else:
    #     wa_id = '_'

    wa_id_kw = blue_cube[0].header.get('INFILE')

    if wa_id_kw is not None:
        match = re.search(r'J\d+\.\d+[+-]\d+\.\d+', wa_id_kw)
        if match:
            wa_id = match.group(0)
        else:
            wa_id = '_'
    else:
        wa_id = '_'

    targetSN = args.target_snr
    levels = args.levels  # SNR levels to display

    colap_b_map = np.sum(blue_cube[1].data[:], axis=0)
    colap_r_map = np.sum(red_cube[1].data[:], axis=0)

    mean_b_map = np.nanmean(blue_cube[1].data[:], axis=0)
    mean_r_map = np.nanmean(red_cube[1].data[:], axis=0)

    median_b_map = np.nanmedian(blue_cube[1].data[:], axis=0)
    median_r_map = np.nanmedian(red_cube[1].data[:], axis=0)

    blue_sky_cube = blue_cube[3].data - blue_cube[1].data
    red_sky_cube = red_cube[3].data - red_cube[1].data

    mean_b_sky_map = np.nanmean(blue_sky_cube, axis=0)
    mean_r_sky_map = np.nanmean(red_sky_cube, axis=0)

    median_b_sky_map = np.nanmedian(blue_sky_cube, axis=0)
    median_r_sky_map = np.nanmedian(red_sky_cube, axis=0)

    del blue_sky_cube, red_sky_cube

    mask_bright_b = mean_b_map > mean_b_sky_map
    mask_medium_b = (median_b_map > median_b_sky_map) & (mean_b_map <= mean_b_sky_map)
    mask_faint_b = (mean_b_map > 0) & (median_b_map <= median_b_sky_map)

    mask_bright_r = mean_r_map > mean_r_sky_map
    mask_medium_r = (median_r_map > median_r_sky_map) & (mean_r_map <= mean_r_sky_map)
    mask_faint_r = (mean_r_map > 0) & (median_r_map <= median_r_sky_map)

    int_b_spec_bright = np.sum(blue_cube[1].data * mask_bright_b[np.newaxis, :, :], axis=(1, 2)) * \
                        blue_cube[5].data[:] / np.sum(mask_bright_b)
    int_b_spec_medium = np.sum(blue_cube[1].data * mask_medium_b[np.newaxis, :, :], axis=(1, 2)) * \
                        blue_cube[5].data[:] / np.sum(mask_medium_b)
    int_b_spec_faint = np.sum(blue_cube[1].data * mask_faint_b[np.newaxis, :, :], axis=(1, 2)) * \
                       blue_cube[5].data[:] / np.sum(mask_faint_b)
    int_r_spec_bright = np.sum(red_cube[1].data * mask_bright_r[np.newaxis, :, :], axis=(1, 2)) * \
                        red_cube[5].data[:] / np.sum(mask_bright_r)
    int_r_spec_medium = np.sum(red_cube[1].data * mask_medium_r[np.newaxis, :, :], axis=(1, 2)) * \
                        red_cube[5].data[:] / np.sum(mask_medium_r)
    int_r_spec_faint = np.sum(red_cube[1].data * mask_faint_r[np.newaxis, :, :], axis=(1, 2)) * \
                       red_cube[5].data[:] / np.sum(mask_faint_r)

    int_b_sky_spec = np.sum(blue_cube[3].data - blue_cube[1].data, axis=(1, 2)) * \
                     blue_cube[5].data[:] / np.sum(mask_faint_b)
    int_r_sky_spec = np.sum(red_cube[3].data - red_cube[1].data, axis=(1, 2)) * \
                     red_cube[5].data[:] / np.sum(mask_faint_r)

    lam_r = red_cube[1].header['CRVAL3'] + (np.arange(red_cube[1].header['NAXIS3']) * red_cube[1].header['CD3_3'])
    lam_b = blue_cube[1].header['CRVAL3'] + (np.arange(blue_cube[1].header['NAXIS3']) * blue_cube[1].header['CD3_3'])

    blue_cen_wave = lam_b[(np.abs(lam_b - args.blue_wav)).argmin()]
    red_cen_wave = lam_r[(np.abs(lam_r - args.red_wav)).argmin()]

    if blue_cube[0].header['MODE'] == 'LOWRES':
        sgn_wind = 50
        spec_pix = 0.5
    elif blue_cube[0].header['MODE'] == 'HIGHRES':
        sgn_wind = 250
        spec_pix = 0.1
    else:
        sgn_wind = 50
        spec_pix = 0.5

    med_b = np.median(blue_cube[1].data[np.where(lam_b == blue_cen_wave)[0][0] - sgn_wind:
                                        np.where(lam_b == blue_cen_wave)[0][0] + sgn_wind], axis=0)
    sgn_b = np.mean(blue_cube[1].data[np.where(lam_b == blue_cen_wave)[0][0] - sgn_wind:
                                      np.where(lam_b == blue_cen_wave)[0][0] + sgn_wind], axis=0)
    rms_b = np.sqrt(1 / np.mean(blue_cube[2].data[np.where(lam_b == blue_cen_wave)[0][0] - sgn_wind:
                                                  np.where(lam_b == blue_cen_wave)[0][0] + sgn_wind], axis=0))
    snr_b = sgn_b / rms_b
    snr_b = snr_b * np.sqrt(spec_pix)

    med_r = np.median(red_cube[1].data[np.where(lam_r == red_cen_wave)[0][0] - sgn_wind:
                                       np.where(lam_r == red_cen_wave)[0][0] + sgn_wind], axis=0)
    sgn_r = np.mean(red_cube[1].data[np.where(lam_r == red_cen_wave)[0][0] - sgn_wind:
                                     np.where(lam_r == red_cen_wave)[0][0] + sgn_wind], axis=0)
    rms_r = np.sqrt(1 / np.mean(red_cube[2].data[np.where(lam_r == red_cen_wave)[0][0] - sgn_wind:
                                                 np.where(lam_r == red_cen_wave)[0][0] + sgn_wind], axis=0))
    snr_r = sgn_r / rms_r
    snr_r = snr_r * np.sqrt(spec_pix)

    axis_header = fits.Header()
    axis_header['NAXIS1'] = blue_cube[1].header['NAXIS1']
    axis_header['NAXIS2'] = blue_cube[1].header['NAXIS2']
    axis_header['CD1_1'] = blue_cube[1].header['CD1_1']
    axis_header['CD2_2'] = blue_cube[1].header['CD2_2']
    axis_header['CRPIX1'] = blue_cube[1].header['CRPIX1']
    axis_header['CRPIX2'] = blue_cube[1].header['CRPIX2']
    axis_header['CRVAL1'] = blue_cube[1].header['CRVAL1']
    axis_header['CRVAL2'] = blue_cube[1].header['CRVAL2']
    axis_header['CTYPE1'] = blue_cube[1].header['CTYPE1']
    axis_header['CTYPE2'] = blue_cube[1].header['CTYPE2']
    axis_header['CUNIT1'] = blue_cube[1].header['CUNIT1']
    axis_header['CUNIT2'] = blue_cube[1].header['CUNIT2']

    file_list = np.sort([x for x in os.listdir(file_dir) if ("APS" not in x) & ('single' in x)])  # single files list
    mode = blue_cube[0].header['MODE']
    warc_list = np.sort([x for x in os.listdir(file_dir) if ('warc' in x) and
                         fits.getheader(os.path.join(file_dir, x), 0).get('MODE') == mode])  # WARC files list

    # ==================================================================

    # Start doing the L0 plots
    print('Doing L0 raw data plots')

    L0_results = QC_plots_L0.plots(blue_cube, file_dir, gal_dir, file_list, warc_list,
                                   output_str, redshift, spec_pix, args)

    fig_l0, blue_spec_resol, red_spec_resol, blue_fiber_through, red_fiber_through, \
        blue_wave_calib, red_wave_calib = L0_results

    # ==================================================================

    # creating plots for L1 datacubes

    print('Doing L1 datacubes plots')
    print('')

    # fig = plt.figure(figsize=(14, 90))
    fig = plt.figure(figsize=(14, 66))

    fig.suptitle('L1 QC plots / CASUVERS = ' + blue_cube[0].header['CASUVERS'], size=22, weight='bold')

    # gs = gridspec.GridSpec(20, 2, height_ratios=[1, 0.6, 0.6, 1, 1, 1, 1, 0.6, 0.6, 1, 0.6, 0.6, 1, 1, 1, 1,
    #                                              1, 1, 0.6, 0.6], width_ratios=[0.5, 0.5])
    gs = gridspec.GridSpec(14, 2, height_ratios=[1, 0.6, 0.6, 1, 1, 1, 1, 0.6, 0.6, 1, 0.6, 0.6, 1, 0.6],
                           width_ratios=[0.5, 0.5])
    gs.update(left=0.07, right=0.95, bottom=0.02, top=0.97, wspace=0.2, hspace=0.25)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # ------

    wcs = WCS(axis_header)
    ax = plt.subplot(gs[0, 0], projection=wcs)

    im = ax.imshow(np.log10(colap_b_map), origin='lower')

    cube_central_waves, cube_x_peaks, cube_y_peaks = get_xy_peak_positions(blue_cube[1].data, lam_b,
                                                                           bin_size=blue_cube[1].data.shape[0])

    ypmax_b = round(cube_x_peaks[0])
    xpmax_b = round(cube_y_peaks[0])

    ax.plot(xpmax_b, ypmax_b, 'x', color='red', markersize=4, label=str(xpmax_b) + ', ' + str(ypmax_b))
    ax.set_title('Collapsed Blue Arm Datacube')
    ax.set_xlabel('Right Ascension (J2000)')
    ax.set_ylabel('Declination (J2000)')
    ax.legend()
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'log scale')

    ax = plt.subplot(gs[0, 1])
    im = ax.imshow(np.log10(colap_r_map), origin='lower')

    cube_central_waves, cube_x_peaks, cube_y_peaks = get_xy_peak_positions(red_cube[1].data, lam_r,
                                                                           bin_size=red_cube[1].data.shape[0])

    ypmax_r = round(cube_x_peaks[0])
    xpmax_r = round(cube_y_peaks[0])

    ax.plot(xpmax_r, ypmax_r, 'x', color='red', markersize=4, label=str(xpmax_r) + ', ' + str(ypmax_r))
    ax.set_title('Collapsed Red Arm Datacube')
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    ax.legend()
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'log scale')

    # ------

    ax = plt.subplot(gs[1, 0])
    ax.plot(lam_b, blue_cube[1].data[:, ypmax_b, xpmax_b] * blue_cube[5].data[:])
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel('Flux')
    ax.set_title('Blue spectrum at (' + str(xpmax_b) + ', ' + str(ypmax_b) + ') [flux peak]')

    ax = plt.subplot(gs[1, 1])
    ax.plot(lam_r, red_cube[1].data[:, ypmax_r, xpmax_r] * red_cube[5].data[:])
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel('Flux')
    ax.set_title('Red spectrum at (' + str(xpmax_r) + ', ' + str(ypmax_r) + ') [flux peak]')

    # ------

    # plot sensitivity function

    ax = plt.subplot(gs[2, 0])
    ax.plot(lam_b, blue_cube[5].data)
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel('Flux')
    ax.set_yscale('log')
    ax.set_title('Blue sensitivity function - mean')

    ax = plt.subplot(gs[2, 1])
    ax.plot(lam_r, red_cube[5].data)
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel('Flux')
    ax.set_yscale('log')
    ax.set_title('Red sensitivity function - mean')

    # ------

    # mean and median maps

    ax = plt.subplot(gs[3, 0])
    im = ax.imshow(np.log10(median_b_map), origin='lower')
    ax.contour(mask_faint_b, levels=[0.5], colors=['r'])
    ax.contour(mask_medium_b, levels=[0.5], colors=['b'])
    ax.contour(mask_bright_b, levels=[0.5], colors=['k'])
    ax.set_title(r'Median Map (blue cube)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'counts log scale')

    ax = plt.subplot(gs[3, 1])
    im = ax.imshow(np.log10(median_r_map), origin='lower')
    ax.contour(mask_faint_r, levels=[0.5], colors=['r'])
    ax.contour(mask_medium_r, levels=[0.5], colors=['b'])
    ax.contour(mask_bright_r, levels=[0.5], colors=['k'])
    ax.set_title(r'Median Map (red cube)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'counts log scale')

    ax = plt.subplot(gs[4, 0])
    im = ax.imshow(np.log10(median_b_sky_map), origin='lower')
    ax.set_title(r'Median Sky Map (blue cube)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'counts log scale')

    ax = plt.subplot(gs[4, 1])
    im = ax.imshow(np.log10(median_r_sky_map), origin='lower')
    ax.set_title(r'Median Sky Map (red cube)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'counts log scale')

    ax = plt.subplot(gs[5, 0])
    im = ax.imshow(np.log10(mean_b_map), origin='lower')
    ax.set_title(r'Mean Map (blue cube)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'counts log scale')

    ax = plt.subplot(gs[5, 1])
    im = ax.imshow(np.log10(mean_r_map), origin='lower')
    ax.set_title(r'Mean Map (red cube)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'counts log scale')

    ax = plt.subplot(gs[6, 0])
    im = ax.imshow(np.log10(mean_b_sky_map), origin='lower')
    ax.set_title(r'Mean Sky Map (blue cube)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'counts log scale')

    ax = plt.subplot(gs[6, 1])
    im = ax.imshow(np.log10(mean_r_sky_map), origin='lower')
    ax.set_title(r'Mean Sky Map (red cube)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'counts log scale')

    # ------

    # sky spectra

    ax = plt.subplot(gs[7, :])
    ax.plot(lam_b, int_b_sky_spec, color='gray', alpha=0.2, label='sky')
    ax.plot(lam_b, int_b_spec_faint, color='red', label='faint')
    ax.plot(lam_b, int_b_spec_medium, color='blue', label='medium')
    ax.plot(lam_b, int_b_spec_bright, color='black', label='bright')
    ax.set_yscale('log')
    ax.set_ylim(3e-18, 1e-15)
    ax.set_title(r'Sky spectra (blue arm)', fontsize=10)
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r'Mean intensity [erg/s/cm$^2$/$\AA$/px]')
    ax.grid()
    ax.legend()

    ax = plt.subplot(gs[8, :])
    ax.plot(lam_r, int_r_sky_spec, color='gray', alpha=0.2, label='sky')
    ax.plot(lam_r, int_r_spec_faint, color='red', label='faint')
    ax.plot(lam_r, int_r_spec_medium, color='blue', label='medium')
    ax.plot(lam_r, int_r_spec_bright, color='black', label='bright')
    ax.set_yscale('log')
    ax.set_ylim(3e-18, 1e-15)
    ax.set_title(r'Sky spectra (red arm)', fontsize=10)
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r'Mean intensity [erg/s/cm$^2$/$\AA$/px]')
    ax.grid()
    ax.legend()

    # ------

    ax = plt.subplot(gs[9, 0])
    im = ax.imshow(snr_b, origin='lower')
    cs = ax.contour(snr_b, levels, linestyles=np.array([':', '-']), colors='white')
    m1 = mlines.Line2D([], [], color='black', linestyle=':', markersize=5, label='SNR = ' + str(levels[0]))
    m2 = mlines.Line2D([], [], color='black', linestyle='-', markersize=5, label='SNR = ' + str(levels[1]))
    ax.legend(handles=[m1, m2], framealpha=1, fontsize=8, loc='lower left')

    ax.set_title(r'SNR @' + str(blue_cen_wave) + '$\AA$')
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'SNR')

    ims_xlims = ax.get_xlim()
    ims_ylims = ax.get_ylim()

    ax = plt.subplot(gs[9, 1])
    im = ax.imshow(snr_r, origin='lower')
    cs = ax.contour(snr_r, levels, linestyles=np.array([':', '-']), colors='white')
    m1 = mlines.Line2D([], [], color='black', linestyle=':', markersize=5, label='SNR = ' + str(levels[0]))
    m2 = mlines.Line2D([], [], color='black', linestyle='-', markersize=5, label='SNR = ' + str(levels[1]))
    ax.legend(handles=[m1, m2], framealpha=1, fontsize=8, loc='lower left')

    ax.set_title(r'SNR @' + str(red_cen_wave) + '$\AA$')
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'SNR')

    fits.writeto(gal_dir + 'SNR_map_blue.fits', snr_b, overwrite=True)
    fits.writeto(gal_dir + 'SNR_map_red.fits', snr_r, overwrite=True)

    # ------

    ax = plt.subplot(gs[10, 0])
    ax.plot(med_b, snr_b, 'o', color='blue', alpha=0.3, markeredgecolor='white')
    ax.set_ylabel(r'SNR [@' + str(blue_cen_wave) + '$\AA$]')
    ax.set_xlabel(r'Median Flux [@' + str(blue_cen_wave - (sgn_wind * blue_cube[1].header['CD3_3'])) + '-' +
                  str(blue_cen_wave + (sgn_wind * blue_cube[1].header['CD3_3'])) + '$\AA$]')
    ax.grid(True, alpha=0.3, zorder=-1)

    ax = plt.subplot(gs[10, 1])
    ax.plot(med_r, snr_r, 'o', color='red', alpha=0.3, markeredgecolor='black')
    ax.set_ylabel(r'SNR [@' + str(red_cen_wave) + '$\AA$]')
    ax.set_xlabel(r'Median Flux [@' + str(red_cen_wave - (sgn_wind * red_cube[1].header['CD3_3'])) + '-' +
                  str(red_cen_wave + (sgn_wind * red_cube[1].header['CD3_3'])) + '$\AA$]')
    ax.grid(True, alpha=0.3, zorder=-1)

    # ------

    ax = plt.subplot(gs[11, 0])
    ax.hist(snr_b[snr_b >= 3], 30, histtype='step', lw=2)
    ax.set_yscale('log')
    ax.set_ylabel(r'N pixels [SNR $\geq$ 3]')
    ax.set_xlabel(r'SNR [@' + str(blue_cen_wave) + '$\AA$]')

    int_spec_b = np.sum(blue_cube[1].data * ((snr_b >= 3)[np.newaxis, :, :]), axis=(1, 2)) * \
                 blue_cube[5].data[:]
    in_ax = ax.inset_axes([0.6, 0.55, 0.35, 0.3])
    in_ax.set_title(r'integrated spec [SNR$\geq$3]', fontsize=10, pad=14)
    in_ax.plot(lam_b, int_spec_b)
    in_ax.axvline(blue_cen_wave - (sgn_wind * blue_cube[1].header['CD3_3']), linestyle='--', color='black')
    in_ax.axvline(blue_cen_wave + (sgn_wind * blue_cube[1].header['CD3_3']), linestyle='--', color='black')

    ax = plt.subplot(gs[11, 1])
    ax.hist(snr_r[snr_r >= 3], 30, histtype='step', lw=2)
    ax.set_yscale('log')
    ax.set_ylabel(r'N pixels [SNR $\geq$ 3]')
    ax.set_xlabel(r'SNR [@' + str(red_cen_wave) + '$\AA$]')

    int_spec_r = np.sum(red_cube[1].data * ((snr_r >= 3)[np.newaxis, :, :]), axis=(1, 2)) * \
                 red_cube[5].data[:]
    in_ax = ax.inset_axes([0.6, 0.55, 0.35, 0.3])
    in_ax.set_title(r'integrated spec [SNR$\geq$3]', fontsize=10, pad=14)
    in_ax.plot(lam_r, int_spec_r)
    in_ax.axvline(red_cen_wave - (sgn_wind * red_cube[1].header['CD3_3']), linestyle='--', color='black')
    in_ax.axvline(red_cen_wave + (sgn_wind * red_cube[1].header['CD3_3']), linestyle='--', color='black')

    # ------ flux calibration: maps variation along a spectral window

    # step = int(len(lam_b) / 5)  # blue
    #
    # for k in np.arange(5):
    #     step_i = step * k
    #     if k == 4:
    #         step_f = len(lam_b) - 1
    #     else:
    #         step_f = step * (k + 1)
    #     std_map = np.std(blue_cube[1].data[step_i:step_f, :, :], axis=0)
    #     mean_map = np.mean(blue_cube[1].data[step_i:step_f, :, :], axis=0)
    #     diff_map = abs(mean_map - std_map)
    #     diff_map[diff_map == 0] = np.nan
    #     median_diff = np.nanmedian(abs(mean_map - std_map))
    #
    #     ax = plt.subplot(gs[12 + k, 0])
    #     im = ax.imshow(diff_map, origin='lower', norm=LogNorm())
    #     plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'abs (mean - std) [counts]')
    #     ax.set_title(r'Blue cube flux calibration (part ' + str(k + 1) + '/5) / ' + str(lam_b[step_i]) + '$\AA$ - ' +
    #                  str(lam_b[step_f]) + '$\AA$', fontsize=10)
    #     plt.contour(diff_map, np.array([median_diff]), linestyles=np.array([':']), colors='white', alpha=0.5)
    #     m1 = mlines.Line2D([], [], color='black', linestyle=':', markersize=5, label='median( abs (mean - std)) = ' +
    #                                                                                  str(round(median_diff, 1)))
    #     ax.legend(handles=[m1], framealpha=1, fontsize=8, loc='lower left')
    #     ax.set_xlabel('X [px]')
    #     ax.set_ylabel('Y [px]')
    #
    # step = int(len(lam_r) / 5)  # red
    #
    # for k in np.arange(5):
    #     step_i = step * k
    #     if k == 4:
    #         step_f = len(lam_r) - 1
    #     else:
    #         step_f = step * (k + 1)
    #     std_map = np.std(red_cube[1].data[step_i:step_f, :, :], axis=0)
    #     mean_map = np.mean(red_cube[1].data[step_i:step_f, :, :], axis=0)
    #     diff_map = abs(mean_map - std_map)
    #     diff_map[diff_map == 0] = np.nan
    #     median_diff = np.nanmedian(abs(mean_map - std_map))
    #
    #     ax = plt.subplot(gs[12 + k, 1])
    #     im = ax.imshow(diff_map, origin='lower', norm=LogNorm())
    #     plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'abs (mean - std) [counts]')
    #     ax.set_title(r'Red cube flux calibration (part ' + str(k + 1) + '/5) / ' + str(lam_r[step_i]) + '$\AA$ - ' +
    #                  str(lam_r[step_f]) + '$\AA$', fontsize=10)
    #     plt.contour(diff_map, np.array([median_diff]), linestyles=np.array([':']), colors='white', alpha=0.5)
    #     m1 = mlines.Line2D([], [], color='black', linestyle=':', markersize=5, label='median( abs (mean - std)) = ' +
    #                                                                                  str(round(median_diff, 1)))
    #     ax.legend(handles=[m1], framealpha=1, fontsize=8, loc='lower left')
    #     ax.set_xlabel('X [px]')
    #     ax.set_ylabel('Y [px]')

    # ------

    # doing voronoi binning

    print('Doing L1 blue datacube voronoi')
    print('')

    pixelsize = 1

    yy, xx = np.indices(snr_b.shape)

    x_t = np.ravel(xx)
    y_t = np.ravel(yy)

    sgn_t_r = np.ravel(sgn_r)
    sgn_t_b = np.ravel(sgn_b)
    rms_t_r = np.ravel(rms_r)
    rms_t_b = np.ravel(rms_b)

    x_t_b = x_t[sgn_t_b / rms_t_b > 1]
    y_t_b = y_t[sgn_t_b / rms_t_b > 1]
    sgn_tt_b = sgn_t_b[sgn_t_b / rms_t_b > 1]
    rms_tt_b = rms_t_b[sgn_t_b / rms_t_b > 1]

    def sn_func_blue(index, signal, noise):

        factor = 1 + 1.53 * np.log10(index.size) ** 1.19

        sn_cov = np.sum(signal[index]) / np.sqrt(np.sum((noise[index] * factor) ** 2))

        return sn_cov

    if args.cov_flag == 1:
        try:
            binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_b, y_t_b, sgn_tt_b, rms_tt_b,
                                                                                      targetSN, pixelsize=pixelsize,
                                                                                      plot=0, quiet=1,
                                                                                      sn_func=sn_func_blue, cvt=False)
            vorbin_sn = targetSN
        except:
            binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_b, y_t_b, sgn_tt_b, rms_tt_b,
                                                                                      10, pixelsize=pixelsize, plot=0,
                                                                                      quiet=1, sn_func=sn_func_blue,
                                                                                      cvt=False)
            vorbin_sn = 10.
    else:
        try:
            binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_b, y_t_b, sgn_tt_b, rms_tt_b,
                                                                                      targetSN, pixelsize=pixelsize,
                                                                                      plot=0, quiet=1, cvt=False)
            vorbin_sn = targetSN
        except:
            binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_b, y_t_b, sgn_tt_b, rms_tt_b,
                                                                                      10, pixelsize=pixelsize, plot=0,
                                                                                      quiet=1, cvt=False)
            vorbin_sn = 10.

    # ax = plt.subplot(gs[17, 0])
    ax = plt.subplot(gs[12, 0])

    xmin, xmax = 0, sgn_b.shape[1] - 1
    ymin, ymax = 0, sgn_b.shape[0] - 1
    img = np.full((blue_cube[1].data.shape[2], blue_cube[1].data.shape[1]), np.nan)  # use nan for missing data
    j = np.round(x_t_b / pixelsize).astype(int)
    k = np.round(y_t_b / pixelsize).astype(int)
    img[j, k] = binNum

    ax.imshow(np.rot90(img), interpolation='nearest', cmap='prism',
              extent=[xmin - pixelsize / 2, xmax + pixelsize / 2,
                      ymin - pixelsize / 2, ymax + pixelsize / 2])
    ax.plot(xNode, yNode, '+w', scalex=False, scaley=False)  # do not rescale after imshow()
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    ax.set_xlim(ims_xlims)
    ax.set_ylim(ims_ylims)
    ax.imshow(snr_b * 0., zorder=-1, cmap='Greys', interpolation='nearest')
    ax.set_title(r'Voronoi binning / Target SNR = ' + str(vorbin_sn))

    # ax = plt.subplot(gs[18, 0])
    ax = plt.subplot(gs[13, 0])

    rad = np.sqrt((xNode - xpmax_b) ** 2 + (yNode - ypmax_b) ** 2)  # Use centroids, NOT generators
    ax.plot(np.sqrt((x_t_b - xpmax_b) ** 2 + (y_t_b - ypmax_b) ** 2), sgn_tt_b / rms_tt_b, ',k')
    ax.plot(rad[nPixels < 2], sn[nPixels < 2], 'xb', label='Not binned')
    ax.plot(rad[nPixels > 1], sn[nPixels > 1], 'or', label='Voronoi bins')
    ax.set_xlabel('R [pixels]')
    ax.set_ylabel('Bin S/N')
    ax.axis([np.min(rad), np.max(rad), 0, np.max(sn) * 1.05])  # x0, x1, y0, y1
    ax.axhline(vorbin_sn)
    ax.legend()

    fits.writeto(gal_dir + 'vorbin_map_blue.fits', np.flip(np.rot90(img), axis=0), overwrite=True)

    # saving voronoi datacube
    vorbin_map = img

    nb_cube_data = np.zeros((int(np.nanmax(vorbin_map) + 1), blue_cube[1].data.shape[0]))
    nb_cube_err = np.zeros((int(np.nanmax(vorbin_map) + 1), blue_cube[1].data.shape[0]))

    with mp.Pool(args.nproc) as pool:
        nb_cube = pool.starmap(vorbin_loop, zip((i, vorbin_map, blue_cube[0].header['CAMERA'])
                                                for i in np.arange(np.nanmax(vorbin_map) + 1)))

    for i in np.arange(int(np.nanmax(vorbin_map)) + 1):
        nb_cube_data[i] = nb_cube[i][0] * blue_cube[5].data[:]
        nb_cube_err[i] = nb_cube[i][1] * blue_cube[5].data[:]

    cube_head = fits.Header()

    cube_head['SIMPLE'] = True
    cube_head['BITPIX'] = -32
    cube_head['NAXIS'] = 2
    cube_head['NAXIS1'] = nb_cube_data.shape[0]
    cube_head['NAXIS2'] = nb_cube_data.shape[1]
    cube_head['CTYPE2'] = 'WAVELENGTH'
    cube_head['CUNIT2'] = 'Angstrom'
    cube_head['CDELT2'] = blue_cube[1].header['CD3_3']
    cube_head['CRVAL2'] = blue_cube[1].header['CRVAL3']
    cube_head['CRPIX2'] = blue_cube[1].header['CRPIX3']
    cube_head['DISPAXIS'] = 1
    cube_head['CNAME'] = gal_name
    cube_head['W_Z'] = redshift
    cube_head['N_FLUX'] = ('1e-19', 'normalized spectra flux')

    n_cube = fits.HDUList([fits.PrimaryHDU(),
                           fits.ImageHDU(data=nb_cube_data / 1e-19, header=cube_head, name='DATA'),
                           fits.ImageHDU(data=nb_cube_err / 1e-19, header=cube_head, name='ERROR')])

    n_cube.writeto(gal_dir + 'blue_vorbin_RSS.fits', overwrite=True)

    #

    print('Doing L1 red datacube voronoi')
    print('')

    x_t_r = x_t[sgn_t_r / rms_t_r > 3]
    y_t_r = y_t[sgn_t_r / rms_t_r > 3]
    sgn_tt_r = sgn_t_r[sgn_t_r / rms_t_r > 3]
    rms_tt_r = rms_t_r[sgn_t_r / rms_t_r > 3]

    def sn_func_red(index, signal, noise):

        factor = 1 + 1.53 * np.log10(index.size) ** 1.19

        sn_cov = np.sum(signal[index]) / np.sqrt(np.sum((noise[index] * factor) ** 2))

        return sn_cov

    if args.cov_flag == 1:
        try:
            binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_r, y_t_r, sgn_tt_r, rms_tt_r,
                                                                                      targetSN, pixelsize=pixelsize,
                                                                                      plot=0, quiet=1,
                                                                                      sn_func=sn_func_red, cvt=False)
            vorbin_sn = targetSN
        except:
            binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_r, y_t_r, sgn_tt_r, rms_tt_r,
                                                                                      10, pixelsize=pixelsize, plot=0,
                                                                                      quiet=1, sn_func=sn_func_red,
                                                                                      cvt=False)
            vorbin_sn = 10.
    else:
        try:
            binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_r, y_t_r, sgn_tt_r, rms_tt_r,
                                                                                      targetSN, pixelsize=pixelsize,
                                                                                      plot=0, quiet=1, cvt=False)
            vorbin_sn = targetSN
        except:
            binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_r, y_t_r, sgn_tt_r, rms_tt_r,
                                                                                      10, pixelsize=pixelsize, plot=0,
                                                                                      quiet=1, cvt=False)
            vorbin_sn = 10.

    # ax = plt.subplot(gs[17, 1])
    ax = plt.subplot(gs[12, 1])

    xmin, xmax = 0, sgn_r.shape[1] - 1
    ymin, ymax = 0, sgn_r.shape[0] - 1
    img = np.full((red_cube[1].data.shape[2], red_cube[1].data.shape[1]), np.nan)  # use nan for missing data
    j = np.round(x_t_r / pixelsize).astype(int)
    k = np.round(y_t_r / pixelsize).astype(int)
    img[j, k] = binNum

    ax.imshow(np.rot90(img), interpolation='nearest', cmap='prism',
              extent=[xmin - pixelsize / 2, xmax + pixelsize / 2,
                      ymin - pixelsize / 2, ymax + pixelsize / 2])
    ax.plot(xNode, yNode, '+w', scalex=False, scaley=False)  # do not rescale after imshow()
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    ax.set_xlim(ims_xlims)
    ax.set_ylim(ims_ylims)
    ax.imshow(snr_r * 0., zorder=-1, cmap='Greys', interpolation='nearest')
    ax.set_title(r'Voronoi binning / Target SNR = ' + str(vorbin_sn))

    # ax = plt.subplot(gs[18, 1])
    ax = plt.subplot(gs[13, 1])

    rad = np.sqrt((xNode - xpmax_b) ** 2 + (yNode - ypmax_b) ** 2)  # Use centroids, NOT generators
    ax.plot(np.sqrt((x_t_b - xpmax_r) ** 2 + (y_t_b - ypmax_r) ** 2), sgn_tt_b / rms_tt_b, ',k')
    ax.plot(rad[nPixels < 2], sn[nPixels < 2], 'xb', label='Not binned')
    ax.plot(rad[nPixels > 1], sn[nPixels > 1], 'or', label='Voronoi bins')
    ax.set_xlabel('R [pixels]')
    ax.set_ylabel('Bin S/N')
    ax.axis([np.min(rad), np.max(rad), 0, np.max(sn) * 1.05])  # x0, x1, y0, y1
    ax.axhline(vorbin_sn)
    ax.legend()

    fits.writeto(gal_dir + 'vorbin_map_red.fits', np.flip(np.rot90(img), axis=0), overwrite=True)

    # saving voronoi datacube
    vorbin_map = img

    nb_cube_data = np.zeros((int(np.nanmax(vorbin_map) + 1), red_cube[1].data.shape[0]))
    nb_cube_err = np.zeros((int(np.nanmax(vorbin_map) + 1), red_cube[1].data.shape[0]))

    with mp.Pool(args.nproc) as pool:
        nb_cube = pool.starmap(vorbin_loop, zip((i, vorbin_map, red_cube[0].header['CAMERA'])
                                                for i in np.arange(np.nanmax(vorbin_map) + 1)))

    for i in np.arange(int(np.nanmax(vorbin_map)) + 1):
        nb_cube_data[i] = nb_cube[i][0] * red_cube[5].data[:]
        nb_cube_err[i] = nb_cube[i][1] * red_cube[5].data[:]

    cube_head = fits.Header()
    cube_head['SIMPLE'] = True
    cube_head['BITPIX'] = -32
    cube_head['NAXIS'] = 2
    cube_head['NAXIS1'] = nb_cube_data.shape[0]
    cube_head['NAXIS2'] = nb_cube_data.shape[1]
    cube_head['CTYPE2'] = 'WAVELENGTH'
    cube_head['CUNIT2'] = 'Angstrom'
    cube_head['CDELT2'] = red_cube[1].header['CD3_3']
    cube_head['CRVAL2'] = red_cube[1].header['CRVAL3']
    cube_head['CRPIX2'] = red_cube[1].header['CRPIX3']
    cube_head['DISPAXIS'] = 1
    cube_head['CNAME'] = gal_name
    cube_head['W_Z'] = redshift
    cube_head['N_FLUX'] = ('1e-19', 'normalized spectra flux')

    n_cube = fits.HDUList([fits.PrimaryHDU(),
                           fits.ImageHDU(data=nb_cube_data / 1e-19, header=cube_head, name='DATA'),
                           fits.ImageHDU(data=nb_cube_err / 1e-19, header=cube_head, name='ERROR')])

    n_cube.writeto(gal_dir + 'red_vorbin_RSS.fits', overwrite=True)

    # ------

    # central_waves, x_peaks, y_peaks = get_xy_peak_positions(blue_cube[1].data, lam_b)
    #
    # cube_central_waves, cube_x_peaks, cube_y_peaks = get_xy_peak_positions(blue_cube[1].data, lam_b,
    #                                                                        bin_size=blue_cube[1].data.shape[0])
    #
    # # ax = plt.subplot(gs[19, 0])
    # ax = plt.subplot(gs[14, 0])
    # ax.plot(central_waves, x_peaks - cube_x_peaks[0], '+', color='blue', ms=2,
    #         label='X center mean = ' + str(round(cube_x_peaks[0], 1)))
    # ax.plot(central_waves, y_peaks - cube_y_peaks[0], '+', color='red', ms=2,
    #         label='Y center mean = ' + str(round(cube_y_peaks[0], 1)))
    # ax.set_xlabel(r'$\lambda$ [$\AA$]')
    # ax.set_ylabel(r'X and Y center')
    # ax.set_ylim([-5, 5])
    # ax.set_title('Peak flux spaxel (Blue)')
    # ax.legend(markerscale=5)
    #
    # central_waves, x_peaks, y_peaks = get_xy_peak_positions(red_cube[1].data, lam_r)
    #
    # cube_central_waves, cube_x_peaks, cube_y_peaks = get_xy_peak_positions(red_cube[1].data, lam_r,
    #                                                                        bin_size=red_cube[1].data.shape[0])

    # # ax = plt.subplot(gs[19, 1])
    # ax = plt.subplot(gs[14, 1])
    # ax.plot(central_waves, x_peaks - cube_x_peaks[0], '+', color='blue', ms=2,
    #         label='X center mean = ' + str(round(cube_x_peaks[0], 1)))
    # ax.plot(central_waves, y_peaks - cube_y_peaks[0], '+', color='red', ms=2,
    #         label='Y center mean = ' + str(round(cube_y_peaks[0], 1)))
    # ax.set_xlabel(r'$\lambda$ [$\AA$]')
    # ax.set_ylabel(r'X and Y center')
    # ax.set_ylim([-5, 5])
    # ax.set_title('Peak flux spaxel (Red)')
    # ax.legend(markerscale=5)

    fig_l1 = output_str + '_L1.png'

    fig.savefig(fig_l1)

    # ==================================================================

    # creating plots for APS

    if args.aps_flag == 1:

        print('Doing L2 APS plots')
        print('')

        aps_maps = fits.open(gal_dir + '/' + gal_name + '_APS_maps.fits')

        if os.path.exists(gal_dir + '/' + gal_name + '_cube.fits'):
            aps_cube = fits.open(gal_dir + '/' + gal_name + '_cube.fits')

        else:
            aps_cube = fits.open(gal_dir + '/' + gal_name + '_vorbin_cube.fits')

        aps_cube_data = aps_cube[1].data
        aps_cube_err = aps_cube[2].data

        aps_cen_wave = args.aps_wav

        colap_a_map = np.nansum(aps_cube[1].data[:], axis=0)

        lam_a = aps_cube[1].header['CRVAL3'] + (np.arange(aps_cube[1].header['NAXIS3']) * aps_cube[1].header['CDELT3'])

        sgn_lam = min(lam_a, key=lambda x: abs(x - aps_cen_wave))
        sgn_a = np.mean(aps_cube[1].data[np.where(lam_a == sgn_lam)[0][0] - sgn_wind:
                                         np.where(lam_a == sgn_lam)[0][0] + sgn_wind], axis=0)
        rms_a = np.sqrt(sgn_a)
        snr_a = sgn_a / rms_a

        # doing the plots

        axis_header = fits.Header()
        axis_header['NAXIS1'] = aps_cube[1].header['NAXIS1']
        axis_header['NAXIS2'] = aps_cube[1].header['NAXIS2']
        axis_header['CD1_1'] = aps_cube[1].header['CDELT1']
        axis_header['CD2_2'] = aps_cube[1].header['CDELT2']
        axis_header['CRPIX1'] = aps_cube[1].header['CRPIX1']
        axis_header['CRPIX2'] = aps_cube[1].header['CRPIX2']
        axis_header['CRVAL1'] = aps_cube[1].header['CRVAL1']
        axis_header['CRVAL2'] = aps_cube[1].header['CRVAL2']
        axis_header['CTYPE1'] = aps_cube[1].header['CTYPE1']
        axis_header['CTYPE2'] = aps_cube[1].header['CTYPE2']
        axis_header['CUNIT1'] = aps_cube[1].header['CUNIT1']
        axis_header['CUNIT2'] = aps_cube[1].header['CUNIT2']

        fig = plt.figure(figsize=(14, 26))

        fig.suptitle('L2/APS QC plots / APSVERS = ' + aps_cube[1].header['APSVERS'], size=22, weight='bold')

        if os.path.exists(gal_dir + '/' + gal_name + '_cube.fits'):
            subplots_rows = np.array([3, 4, 5, 6])
            gs = gridspec.GridSpec(7, 3, height_ratios=[1, 1, 1, 0.5, 1, 1, 1], width_ratios=[1, 1, 1])
        else:
            subplots_rows = np.array([2, 3, 4, 5])
            gs = gridspec.GridSpec(6, 3, height_ratios=[1, 1, 0.5, 1, 1, 1], width_ratios=[1, 1, 1])
        gs.update(left=0.09, right=0.95, bottom=0.02, top=0.95, wspace=0.3, hspace=0.25)

        wcs = WCS(axis_header)

        ax = plt.subplot(gs[0, 0], projection=wcs)
        im = ax.imshow(np.log10(colap_a_map), origin='lower')

        cube_central_waves, cube_x_peaks, cube_y_peaks = get_xy_peak_positions(aps_cube[1].data, lam_a,
                                                                               bin_size=aps_cube[1].data.shape[0])

        ypmax_a = round(cube_x_peaks[0])
        xpmax_a = round(cube_y_peaks[0])

        if xpmax_a > aps_cube[1].data.shape[2] - 1:
            xpmax_a = aps_cube[1].data.shape[2] - 1
        if ypmax_a > aps_cube[1].data.shape[1] - 1:
            ypmax_a = aps_cube[1].data.shape[1] - 1

        ax.plot(xpmax_a, ypmax_a, 'x', color='red', markersize=4, label=str(xpmax_a) + ', ' + str(ypmax_a))
        ax.set_title('Collapsed APS Datacube')
        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')
        ax.legend()
        plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'log scale')

        # ------

        ax = plt.subplot(gs[0, 1:])
        ax.plot(lam_a, aps_cube[1].data[:, ypmax_a, xpmax_a])
        ax.set_xlabel(r'$\lambda$ [$\AA$]')
        ax.set_ylabel('Counts')
        ax.set_title('APS spectrum at (' + str(xpmax_a) + ', ' + str(ypmax_a) + ') [flux peak]')

        # ------

        ax = plt.subplot(gs[1, 0])
        im = ax.imshow(snr_a, origin='lower')
        cs = ax.contour(snr_a, levels, linestyles=np.array([':', '-']), colors='white')
        m1 = mlines.Line2D([], [], color='black', linestyle=':', markersize=5, label='SNR = ' + str(levels[0]))
        m2 = mlines.Line2D([], [], color='black', linestyle='-', markersize=5, label='SNR = ' + str(levels[1]))
        ax.legend(handles=[m1, m2], framealpha=1, fontsize=8, loc='lower left')

        ax.set_title(r'SNR @' + str(aps_cen_wave) + '$\AA$')
        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')
        plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'SNR')

        fits.writeto(gal_dir + 'SNR_map_aps.fits', snr_a, overwrite=True)

        # ------

        ax = plt.subplot(gs[1, 1:])
        ax.hist(snr_a[snr_a >= 3], 30, histtype='step', lw=2)
        ax.set_yscale('log')
        ax.set_ylabel(r'N pixels [SNR $\geq$ 3]')
        ax.set_xlabel(r'SNR [@' + str(aps_cen_wave) + '$\AA$]')

        int_spec_a = np.sum(aps_cube[1].data * ((snr_a >= 3)[np.newaxis, :, :]), axis=(1, 2))
        in_ax = ax.inset_axes([0.55, 0.5, 0.4, 0.3])
        in_ax.set_title(r'integrated spec [SNR$\geq$3]', fontsize=10)
        in_ax.plot(lam_a, int_spec_a)
        in_ax.axvline(aps_cen_wave - sgn_wind * aps_cube[1].header['CDELT3'], linestyle='--', color='black')
        in_ax.axvline(aps_cen_wave + sgn_wind * aps_cube[1].header['CDELT3'], linestyle='--', color='black')

        # doing voronoi binning

        if os.path.exists(gal_dir + '/' + gal_name + '_cube.fits'):

            print('Doing L2 APS voronoi binning')
            print('')

            pixelsize = 1

            yy_a, xx_a = np.indices(snr_a.shape)

            x_ta = np.ravel(xx_a)
            y_ta = np.ravel(yy_a)

            sgn_t_a = np.ravel(sgn_a)
            rms_t_a = np.ravel(rms_a)

            x_t_a = x_ta[sgn_t_a / rms_t_a > 3]
            y_t_a = y_ta[sgn_t_a / rms_t_a > 3]
            sgn_tt_a = sgn_t_a[sgn_t_a / rms_t_a > 3]
            rms_tt_a = rms_t_a[sgn_t_a / rms_t_a > 3]

            def sn_func_aps(index, signal, noise):
                factor = 1 + 1.53 * np.log10(index.size) ** 1.19

                sn_cov = np.sum(signal[index]) / np.sqrt(np.sum((noise[index] * factor) ** 2))

                return sn_cov

            if args.cov_flag == 1:
                try:
                    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_a, y_t_a, sgn_tt_a,
                                                                                              rms_tt_a,
                                                                                              targetSN,
                                                                                              pixelsize=pixelsize,
                                                                                              plot=0,
                                                                                              quiet=1,
                                                                                              sn_func=sn_func_aps)
                    vorbin_sn = targetSN
                except:
                    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_a, y_t_a, sgn_tt_a,
                                                                                              rms_tt_a,
                                                                                              10, pixelsize=pixelsize,
                                                                                              plot=0,
                                                                                              quiet=1,
                                                                                              sn_func=sn_func_aps)
                    vorbin_sn = 10.
            else:
                try:
                    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_a, y_t_a, sgn_tt_a,
                                                                                              rms_tt_a,
                                                                                              targetSN,
                                                                                              pixelsize=pixelsize,
                                                                                              plot=0,
                                                                                              quiet=1)
                    vorbin_sn = targetSN
                except:
                    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_a, y_t_a, sgn_tt_a,
                                                                                              rms_tt_a,
                                                                                              10, pixelsize=pixelsize,
                                                                                              plot=0,
                                                                                              quiet=1)
                    vorbin_sn = 10.

            ax = plt.subplot(gs[2, 0])

            xmin, xmax = 0, sgn_a.shape[1] - 1
            ymin, ymax = 0, sgn_a.shape[0] - 1
            nx = sgn_a.shape[1]
            ny = sgn_a.shape[0]
            img = np.full((nx, ny), np.nan)  # use nan for missing data
            j = np.round((x_t_a - xmin) / pixelsize).astype(int)
            k = np.round((y_t_a - ymin) / pixelsize).astype(int)
            img[j, k] = binNum

            ax.imshow(np.rot90(img), interpolation='nearest', cmap='prism',
                      extent=[xmin - pixelsize / 2, xmax + pixelsize / 2,
                              ymin - pixelsize / 2, ymax + pixelsize / 2])
            ax.plot(xNode, yNode, '+w', scalex=False, scaley=False)  # do not rescale after imshow()
            ax.set_xlabel('X [px]')
            ax.set_ylabel('Y [px]')
            ax.set_title(r'Voronoi binning / Target SNR = ' + str(vorbin_sn))

            fits.writeto(gal_dir + 'vorbin_map_aps.fits', np.flip(np.rot90(img), axis=0), overwrite=True)

            ax = plt.subplot(gs[2, 1:])

            rad = np.sqrt((xNode - xpmax_b) ** 2 + (yNode - ypmax_b) ** 2)  # Use centroids, NOT generators
            ax.plot(np.sqrt((x_t_a - xpmax_a) ** 2 + (y_t_a - ypmax_a) ** 2), sgn_tt_a / rms_tt_a, ',k')
            ax.plot(rad[nPixels < 2], sn[nPixels < 2], 'xb', label='Not binned')
            ax.plot(rad[nPixels > 1], sn[nPixels > 1], 'or', label='Voronoi bins')
            ax.set_xlabel('R [pixels]')
            ax.set_ylabel('Bin S/N')
            ax.axis([np.min(rad), np.max(rad), 0, np.max(sn) * 1.05])  # x0, x1, y0, y1
            ax.axhline(vorbin_sn)
            ax.legend()

            # saving voronoi datacube
            vorbin_map = img

            na_cube_data = np.zeros((int(np.nanmax(vorbin_map) + 1), aps_cube[1].data.shape[0]))
            na_cube_err = np.zeros((int(np.nanmax(vorbin_map) + 1), aps_cube[2].data.shape[0]))

            with mp.Pool(args.nproc) as pool:
                nb_cube = pool.starmap(vorbin_loop, zip((i, vorbin_map, 'APS')
                                                        for i in np.arange(np.nanmax(vorbin_map) + 1)))

            for i in np.arange(int(np.nanmax(vorbin_map)) + 1):
                na_cube_data[i] = nb_cube[i][0]
                na_cube_err[i] = nb_cube[i][1]

            cube_head = aps_cube[1].header.copy()
            cube_head['W_Z'] = redshift

            cube_head = fits.Header()
            cube_head['SIMPLE'] = True
            cube_head['BITPIX'] = -32
            cube_head['NAXIS'] = 2
            cube_head['NAXIS1'] = na_cube_data.shape[0]
            cube_head['NAXIS2'] = na_cube_data.shape[1]
            cube_head['CTYPE2'] = 'WAVELENGTH'
            cube_head['CUNIT2'] = 'Angstrom'
            cube_head['CDELT2'] = aps_cube[1].header['CDELT3']
            cube_head['CRVAL2'] = aps_cube[1].header['CRVAL3']
            cube_head['CRPIX2'] = aps_cube[1].header['CRPIX3']
            cube_head['DISPAXIS'] = 1
            cube_head['CNAME'] = gal_name
            cube_head['W_Z'] = redshift
            cube_head['N_FLUX'] = ('1e-19', 'normalized spectra flux')

            n_cube = fits.HDUList([fits.PrimaryHDU(),
                                   fits.ImageHDU(data=na_cube_data, header=cube_head, name='DATA'),
                                   fits.ImageHDU(data=na_cube_err, header=cube_head, name='ERROR')])

            n_cube.writeto(gal_dir + 'aps_vorbin_RSS.fits', overwrite=True)

        # ------

        central_waves, x_peaks, y_peaks = get_xy_peak_positions(aps_cube[1].data, lam_a)

        cube_central_waves, cube_x_peaks, cube_y_peaks = get_xy_peak_positions(aps_cube[1].data, lam_a,
                                                                               bin_size=aps_cube[1].data.shape[0])

        ax = plt.subplot(gs[subplots_rows[0], :])
        ax.plot(central_waves, x_peaks - cube_x_peaks[0], '+', color='blue', ms=2,
                label='X center mean = ' + str(round(cube_x_peaks[0], 1)))
        ax.plot(central_waves, y_peaks - cube_y_peaks[0], '+', color='red', ms=2,
                label='Y center mean = ' + str(round(cube_y_peaks[0], 1)))
        ax.set_xlabel(r'$\lambda$ [$\AA$]')
        ax.set_ylabel(r'X and Y center')
        ax.set_ylim([-5, 5])
        ax.set_title('Peak flux spaxel (L2)')
        ax.legend(markerscale=5)

        # ------

        ext_names = [ext.name for ext in aps_maps]

        if 'V' in ext_names:
            ax = plt.subplot(gs[subplots_rows[1], 0])
            im = ax.imshow(aps_maps['V'].data, origin='lower', cmap='bwr',
                           vmin=np.nanpercentile(aps_maps['V'].data, 10),
                           vmax=np.nanpercentile(aps_maps['V'].data, 90))
            ax.set_title(r'APS V')
            ax.set_xlabel('X [px]')
            ax.set_ylabel('Y [px]')
            ax.contour(aps_maps['V'].data, levels=[-300, -200, -100, 0, 100, 200, 300], colors=['black'], alpha=0.3)
            plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'V')

            ax = plt.subplot(gs[subplots_rows[1], 1])
            im = ax.imshow(aps_maps['sigma'].data, origin='lower', vmin=0,
                           vmax=np.nanpercentile(aps_maps['sigma'].data, 90))
            ax.set_title(r'APS sigma')
            ax.set_xlabel('X [px]')
            ax.set_ylabel('Y [px]')
            ax.contour(aps_maps['sigma'].data, levels=[0, 100, 200, 300], colors=['black'], alpha=0.3)
            plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'sigma')

        # ------

        if 'FLUX_[OIII]_5006.77' in ext_names:
            if np.nanmax(aps_maps['FLUX_[OIII]_5006.77'].data) > 0:
                ax = plt.subplot(gs[subplots_rows[2], 0])
                cmax = np.nanpercentile(aps_maps['FLUX_[OIII]_5006.77'].data, 90)
                if cmax < 0.1:
                    cmax = 0.1
                im = ax.imshow(aps_maps['FLUX_[OIII]_5006.77'].data,
                               norm=LogNorm(vmin=0.1, vmax=cmax),
                               origin='lower')
                ax.set_title(r'APS F([OIII]5007)')
                ax.set_xlabel('X [px]')
                ax.set_ylabel('Y [px]')
                plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'flux')

            ax = plt.subplot(gs[subplots_rows[2], 1])
            im = ax.imshow(aps_maps['V_[OIII]_5006.77'].data, origin='lower', cmap='bwr',
                           vmin=np.nanpercentile(aps_maps['V_[OIII]_5006.77'].data, 10),
                           vmax=np.nanpercentile(aps_maps['V_[OIII]_5006.77'].data, 90))
            ax.set_title(r'APS V([OIII]5007)')
            ax.set_xlabel('X [px]')
            ax.set_ylabel('Y [px]')
            ax.contour(aps_maps['V_[OIII]_5006.77'].data, levels=[-300, -200, -100, 0, 100, 200, 300], colors=['black'],
                       alpha=0.3)
            plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'V')

            ax = plt.subplot(gs[subplots_rows[2], 2])
            im = ax.imshow(aps_maps['SIGMA_[OIII]_5006.77'].data, origin='lower', vmin=0,
                           vmax=np.nanpercentile(aps_maps['SIGMA_[OIII]_5006.77'].data, 90))
            ax.set_title(r'APS SIGMA([OIII]5007)')
            ax.set_xlabel('X [px]')
            ax.set_ylabel('Y [px]')
            ax.contour(aps_maps['SIGMA_[OIII]_5006.77'].data, levels=[0, 100, 200, 300], colors=['black'], alpha=0.3)
            plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'sigma')

        # ------

        if 'FLUX_HA_6562.80' in ext_names:
            if np.nanmax(aps_maps['FLUX_HA_6562.80'].data) > 0:
                ax = plt.subplot(gs[subplots_rows[3], 0])
                cmax = np.nanpercentile(aps_maps['FLUX_HA_6562.80'].data, 90)
                if cmax < 0.1:
                    cmax = 0.1
                im = ax.imshow(aps_maps['FLUX_HA_6562.80'].data,
                               norm=LogNorm(vmin=0.1, vmax=cmax),
                               origin='lower')
                ax.set_title(r'APS F(HA)')
                ax.set_xlabel('X [px]')
                ax.set_ylabel('Y [px]')
                plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'flux')

            ax = plt.subplot(gs[subplots_rows[3], 1])
            im = ax.imshow(aps_maps['V_HA_6562.80'].data, origin='lower', cmap='bwr',
                           vmin=np.nanpercentile(aps_maps['V_HA_6562.80'].data, 10),
                           vmax=np.nanpercentile(aps_maps['V_HA_6562.80'].data, 90))
            ax.set_title(r'APS V(HA)')
            ax.set_xlabel('X [px]')
            ax.set_ylabel('Y [px]')
            ax.contour(aps_maps['V_HA_6562.80'].data, levels=[-300, -200, -100, 0, 100, 200, 300], colors=['black'],
                       alpha=0.3)
            plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'V')

            ax = plt.subplot(gs[subplots_rows[3], 2])
            im = ax.imshow(aps_maps['SIGMA_HA_6562.80'].data, origin='lower', vmin=0,
                           vmax=np.nanpercentile(aps_maps['SIGMA_HA_6562.80'].data, 90))
            ax.set_title(r'APS SIGMA(HA)')
            ax.set_xlabel('X [px]')
            ax.set_ylabel('Y [px]')
            ax.contour(aps_maps['SIGMA_HA_6562.80'].data, levels=[0, 100, 200, 300], colors=['black'],
                       alpha=0.3)
            plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04, label=r'sigma')

        # ------

        fig_l2 = output_str + '_L2.png'

        fig.savefig(fig_l2)

        text = f'''
        <html>
          <body style="background-color:white;">
            <div style="text-align: center;">
              <h1>Data report {date}</h1>
              <h1>CNAME {blue_cube[0].header['CCNAME1']}</h1>
              <h1>IFUNAME {blue_cube[0].header['IFUNAME']}</h1>
              <h1>OBID {blue_cube[0].header['OBID']}</h1>
              <h1>LIFU MODE {blue_cube[0].header['MODE']}</h1>

              <!-- Navigation buttons -->
              <div style="margin: 20px;">
                <a href="#fig_l0" style="margin:10px; text-decoration:none; padding:8px 16px; background-color:#007BFF; 
                color:white; border-radius:8px;">L0 plots</a>
                <a href="#fig_l1" style="margin:10px; text-decoration:none; padding:8px 16px; background-color:#007BFF; 
                color:white; border-radius:8px;">L1 plots</a>
                <a href="#fig_l2" style="margin:10px; text-decoration:none; padding:8px 16px; background-color:#007BFF; 
                color:white; border-radius:8px;">L2 plots</a>
              </div>

              <!-- Figures with anchors -->
              <div id="fig_l0">
                <img src="{fig_l0}" class="center">
              </div>

              <div id="fig_l1">
                <img src="{fig_l1}" class="center">
              </div>

              <div id="fig_l2">
                <img src="{fig_l2}" class="center">
              </div>

              <!-- Back to top button -->
              <div style="margin: 20px;">
                <a href="#top" style="text-decoration:none; padding:8px 16px; background-color:#28a745; color:white; 
                border-radius:8px;">Back to Top</a>
              </div>
            </div>
          </body>
        </html>
        '''

    else:

        text = f'''
        <html>
          <body style="background-color:white;">
            <div style="text-align: center;">
              <h1>Data report {date}</h1>
              <h1>CNAME {blue_cube[0].header['CCNAME1']}</h1>
              <h1>IFUNAME {blue_cube[0].header['IFUNAME']}</h1>
              <h1>OBID {blue_cube[0].header['OBID']}</h1>
              <h1>LIFU MODE {blue_cube[0].header['MODE']}</h1>

              <!-- Navigation buttons -->
              <div style="margin: 20px;">
                <a href="#fig_l0" style="margin:10px; text-decoration:none; padding:8px 16px; background-color:#007BFF; 
                color:white; border-radius:8px;">L0 plots</a>
                <a href="#fig_l1" style="margin:10px; text-decoration:none; padding:8px 16px; background-color:#007BFF; 
                color:white; border-radius:8px;">L1 plots</a>
              </div>

              <!-- Figures with anchors -->
              <div id="fig_l0">
                <img src="{fig_l0}" class="center">
              </div>

              <div id="fig_l1">
                <img src="{fig_l1}" class="center">
              </div>

              <!-- Back to top button -->
              <div style="margin: 20px;">
                <a href="#top" style="text-decoration:none; padding:8px 16px; background-color:#28a745; color:white; 
                border-radius:8px;">Back to Top</a>
              </div>
            </div>
          </body>
        </html>
        '''

    f = open(output_str + ".html", "w")

    f.write(text)
    f.close()

    if args.aps_flag == 1:
        qc_plot_dir = 'CPSv' + blue_cube[0].header['CASUVERS'] + '_APSv' + aps_cube[1].header['APSVERS']
    else:
        qc_plot_dir = 'CPSv' + blue_cube[0].header['CASUVERS']

    # ---- creating table for index.html

    coord = SkyCoord(
        ra=blue_cube[1].header['CRVAL1'],
        dec=blue_cube[1].header['CRVAL2'],
        unit=(u.deg, u.deg),
        frame="icrs"
    )

    res = Ned.query_region(coord, radius=10 * u.arcsec)

    gal = res[res["Type"].astype(str) == "G"]

    if len(gal) == 0:
        print("No galaxy found in NED within the search radius.")
        obj_nme = ''
    else:
        closest_gal = gal[gal["Separation"] == gal["Separation"].min()]
        obj_nme = closest_gal['Object Name'][0]
        obj_nme = re.sub(r"\s+NED\d+$", "", obj_nme)

    with open(output_str + ".txt", "w") as f:
        f.write(gal_name + '\n')
        f.write(obj_nme + '\n')
        f.write(str(blue_cube[0].header['OBID']) + '\n')
        f.write(blue_cube[0].header['MODE'] + '\n')
        f.write(date + '\n')
        f.write(blue_cube[0].header['TRIMESTE'] + '\n')
        f.write(str(blue_spec_resol) + '\n')
        f.write(str(red_spec_resol) + '\n')
        f.write(str(blue_fiber_through) + '\n')
        f.write(str(red_fiber_through) + '\n')
        f.write(str(blue_wave_calib) + '\n')
        f.write(str(red_wave_calib) + '\n')
        f.write(wa_id + '\n')

    os.makedirs(qc_plot_dir, exist_ok=True)
    os.system('mv ' + str(blue_cube[0].header['OBID']) + '*.png ' + qc_plot_dir + '/.')
    os.system('mv ' + str(blue_cube[0].header['OBID']) + '*.txt ' + qc_plot_dir + '/.')
    os.system('mv ' + str(blue_cube[0].header['OBID']) + '*.html ' + qc_plot_dir + '/.')
