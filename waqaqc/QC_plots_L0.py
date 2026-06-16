import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord, FK5
import requests
from PIL import Image
from io import BytesIO
import multiprocessing as mp
from scipy.optimize import curve_fit
import tqdm
from waqaqc.signalWEAVE import signalWEAVE
from scipy.interpolate import interp1d
from speclite import filters
import astropy.units as u
from astroquery.sdss import SDSS

try:
    from importlib.resources import files  # Python 3.9+
except ImportError:
    from importlib_resources import files  # Backport for Python 3.8


def fiber_lines(args):
    # Function to measure WARC and sky spectra lines and return the measured parameters
    fiber, cen_lam, lamp_spec, lamp_lam, lam_wind, sky_plot_flag, sky_plot_dir, file_cam = args

    fib_flux = []
    fib_cen = []
    fib_sigma = []

    if file_cam == 'WEAVEBLUE':
        fiber_dir = sky_plot_dir + 'BLUE/' + str(fiber) + '/'
    else:
        fiber_dir = sky_plot_dir + 'RED/' + str(fiber) + '/'

    if sky_plot_flag == 1 and fiber % 25 == 0:
        os.makedirs(fiber_dir, exist_ok=True)

    for i in np.arange(len(cen_lam)):
        lam_wind_c = np.where(lamp_lam == min(lamp_lam, key=lambda x: abs(x - cen_lam[i])))[0][0]
        w_lam = lamp_lam[lam_wind_c - lam_wind: lam_wind_c + lam_wind]
        w_spec = lamp_spec[fiber][lam_wind_c - lam_wind: lam_wind_c + lam_wind]

        popt, pcov = curve_fit(gauss, w_lam, w_spec, p0=[0, 0, max(w_spec), cen_lam[i], 1],
                               bounds=([-np.inf, -np.inf, 0, 0, 0],
                                       [np.inf, np.inf, np.inf, np.inf, np.inf]))

        f_fit = np.sum(gauss(w_lam, *popt)) - np.nanmedian([gauss(w_lam, *popt)[0],
                                                            gauss(w_lam, *popt)[-1]])

        fib_flux.append(f_fit)
        fib_cen.append(popt[3])
        fib_sigma.append(popt[4] * 2.355)

        if sky_plot_flag == 1 and fiber % 25 == 0:
            fig_skyline = plt.figure(figsize=(5, 4))
            plt.plot(w_lam, w_spec, color='black')
            plt.plot(w_lam, gauss(w_lam, *popt), color='red')
            plt.xlabel(r'$\lambda$ [$\AA$]')
            plt.ylabel(r'flux')
            plt.annotate('cenlam = ' + str(round(popt[3], 2)), (0.01, 0.9), xycoords='axes fraction',
                         fontsize=10)
            plt.annotate('FWHM = ' + str(round(popt[4] * 2.355, 2)), (0.01, 0.85), xycoords='axes fraction',
                         fontsize=10)
            plt.annotate('flux = ' + str(round(f_fit, 1)), (0.01, 0.8), xycoords='axes fraction',
                         fontsize=10)
            fig_skyline.savefig(fiber_dir + str(round(popt[3])) + '.pdf')
            plt.close(fig_skyline)

    warc_flux = np.ravel(fib_flux)
    warc_flux_med = np.nanmedian(np.ravel(fib_flux))
    warc_cen = np.ravel(fib_cen)
    warc_cen_med = np.nanmedian(np.ravel(fib_cen))
    warc_sigma = np.ravel(fib_sigma)
    warc_sigma_med = np.nanmedian(np.ravel(fib_sigma))

    return warc_flux, warc_flux_med, warc_cen, warc_cen_med, warc_sigma, warc_sigma_med


def get_images(ra, dec, filters="grizy"):
    """Query ps1filenames.py service to get a list of images

    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table


def geturl(ra, dec, size=240, output_size=None, filters="grizy", im_format="jpg", color=False):
    """Get URL for images in the table

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    im_format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """

    if color and im_format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if im_format not in ("jpg", "png", "fits"):
        raise ValueError("im_format must be one of jpg, png, fits")
    table = get_images(ra, dec, filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={im_format}")
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0, len(table) // 2, len(table) - 1]]
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + "&{}={}".format(param, table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase + filename)
    return url


def gauss(x, a, b, amp, x0, sigma):
    return a + b * x + amp * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def plots(blue_cube, file_dir, gal_dir, file_list, warc_list, output_str, redshift, spec_pix, args):
    # setting parameters to be passed as QC parameters
    red_spec_resol = 0
    blue_spec_resol = 0
    red_fiber_through = 0
    blue_fiber_through = 0
    red_wave_calib = 0
    blue_wave_calib = 0

    mode = blue_cube[0].header['MODE']

    rows = 11 + len(file_list)

    fig = plt.figure(figsize=(14, 3.5 * rows))

    fig.suptitle('L0 QC plots', size=22, weight='bold')

    gs = gridspec.GridSpec(rows, 3, height_ratios=np.concatenate((np.array([1]), np.zeros(rows - 1) + 0.5)),
                           width_ratios=[1, 1, 1])
    gs.update(left=0.07, right=0.95, bottom=0.02, top=0.95, wspace=0.3, hspace=0.3)

    # Creating PanSTARRS composite image
    nsc = SkyCoord(ra=blue_cube[1].header['CRVAL1'], dec=blue_cube[1].header['CRVAL2'], unit='deg', frame=FK5)

    url = geturl(nsc.ra.value, nsc.dec.value, size=480, filters="grizy", output_size=None, im_format="jpg", color=True)
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))

    wcs_pan = WCS(naxis=2)
    wcs_pan.wcs.crval = [nsc.ra.deg, nsc.dec.deg]
    wcs_pan.wcs.crpix = [im.size[0] / 2., im.size[1] / 2.]
    wcs_pan.wcs.cdelt = np.array([-0.25 / 3600, 0.25 / 3600])  # arcsec/pixel to deg/pixel
    wcs_pan.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    ny, nx = blue_cube[1].data.shape[-2:]
    cdelt_arcsec = np.abs(blue_cube[1].header.get('CD1_1', 0.25)) * 3600

    fov_x = nx * cdelt_arcsec
    fov_y = ny * cdelt_arcsec
    radius = max(fov_x, fov_y) / 2

    theta = np.linspace(0, 2 * np.pi, 7)[:-1] + np.radians(30)  # rotate to flat top
    hex_x = radius * np.cos(theta)
    hex_y = radius * np.sin(theta)

    # Convert to sky coordinates
    center = SkyCoord(ra=blue_cube[1].header['CRVAL1'], dec=blue_cube[1].header['CRVAL2'], unit='deg')
    hex_ra = center.ra.deg + (hex_x / 3600) / np.cos(np.radians(center.dec.deg))
    hex_dec = center.dec.deg + (hex_y / 3600)
    hex_pix = wcs_pan.world_to_pixel(SkyCoord(hex_ra, hex_dec, unit='deg'))
    hex_coords = np.array(hex_pix).T  # shape (6, 2)

    ax = plt.subplot(gs[0, :], projection=wcs_pan)
    ax.imshow(im, origin='lower')
    patch = Polygon(hex_coords, closed=True, edgecolor='cyan', facecolor='none', lw=2)
    ax.add_patch(patch)
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title('WEAVE FoV on PanSTARRS')
    ax.grid(color='white', ls='dotted')

    # ----- setting warc and sky lines

    WARC_LINES = {
        "LOWRES": {
            "WEAVEBLUE": np.array([3606., 3738., 3850., 3995., 4104., 4132., 4290., 4400., 4511., 4545., 4579., 4609.,
                                   4765., 4806., 4965., 5187., 5410.]),
            "WEAVERED": np.array([7788., 7979., 8046., 8159., 8384., 8606., 8748., 8850., 9008., 9180.]),
        },
        "HIGHRES": {
            "WEAVEBLUE": np.array([4727., 4765., 4806., 4848., 4880., 4965., 5017., 5091., 5159., 5231.]),
            "WEAVERED": np.array([6457., 6531., 6584., 6644., 6677., 6684., 6753., 6767.]),
        }
    }

    SKY_LINES = {
        "LOWRES": {
            "WEAVEBLUE": np.array([5577.]),
            "WEAVERED": np.array([6864., 6923., 6949., 6978., 7316., 7341., 7370., 7402., 7750., 7794., 7821., 7890.,
                                  7931., 7993., 8062., 8399., 8430., 8465., 8505., 8886., 8920., 8959., 9002., 9376.,
                                  9440.]),
        },
        "HIGHRES": {
            "WEAVEBLUE": np.array([5198., 5239., 5256.]),
            "WEAVERED": np.array([6170., 6258., 6287., 6300., 6330., 6363., 6533., 6553., 6577.]),
        }
    }

    # ------

    # LSF plots

    sky_plot_flag = args.sky_plot_flag

    sky_plot_dir = gal_dir + 'sky_fit_plots/'
    warc_plot_dir = gal_dir + 'warc_fit_plots/'
    if sky_plot_flag == 1:
        try:
            os.system('rm -r ' + sky_plot_dir)
        except:
            pass
        os.makedirs(sky_plot_dir, exist_ok=True)
        try:
            os.system('rm -r ' + warc_plot_dir)
        except:
            pass
        os.makedirs(warc_plot_dir, exist_ok=True)

    lam_wind = 10
    if mode == 'HIGHRES':
        lam_wind = 50

    warc_cen_blue = []
    warc_sigma_blue = []
    warc_cen_med_blue = []
    warc_sigma_med_blue = []

    warc_cen_red = []
    warc_sigma_red = []
    warc_cen_med_red = []
    warc_sigma_med_red = []

    warc_cen_blue_ext = []
    warc_cen_red_ext = []

    # Measuring WARC files lines
    for j in np.arange(len(warc_list)):

        print('     LSF plots: Measuring WARC fibers (WARC file ' + str(j + 1) + '/' + str(len(warc_list)) + '):')

        warc_name = warc_list[j][:-4]
        warc_file = fits.open(file_dir + warc_name + '.fit')

        file_cam = warc_file[0].header['CAMERA']

        lamp_lam = (np.arange(warc_file[1].header['NAXIS1']) * warc_file[1].header['CD1_1']) + warc_file[1].header[
            'CRVAL1']
        lamp_spec = warc_file[1].data

        cen_lam = WARC_LINES[mode][file_cam]

        with mp.Pool(args.nproc) as pool:
            warc_stats = pool.starmap(fiber_lines,
                                      tqdm.tqdm(zip((fiber, cen_lam, lamp_spec, lamp_lam, lam_wind, sky_plot_flag,
                                                     warc_plot_dir, file_cam)
                                                    for fiber in np.arange(len(lamp_spec))), total=len(lamp_spec)))
        print('')

        if file_cam == 'WEAVEBLUE':
            warc_cen_blue_ext = np.zeros((len(warc_stats), len(cen_lam)))
            for i in np.arange(len(warc_stats)):
                warc_cen_blue.extend(warc_stats[i][2])
                warc_cen_med_blue.append(warc_stats[i][3])
                warc_sigma_blue.extend(warc_stats[i][4])
                warc_sigma_med_blue.append(warc_stats[i][5])
                for l in np.arange(len(cen_lam)):
                    warc_cen_blue_ext[i, l] = warc_stats[i][2][l]

        if file_cam == 'WEAVERED':
            warc_cen_red_ext = np.zeros((len(warc_stats), len(cen_lam)))
            for i in np.arange(len(warc_stats)):
                warc_cen_red.extend(warc_stats[i][2])
                warc_cen_med_red.append(warc_stats[i][3])
                warc_sigma_red.extend(warc_stats[i][4])
                warc_sigma_med_red.append(warc_stats[i][5])
                for l in np.arange(len(cen_lam)):
                    warc_cen_red_ext[i, l] = warc_stats[i][2][l]

    warc_cen_blue = np.ravel(warc_cen_blue)
    warc_sigma_blue = np.ravel(warc_sigma_blue)

    warc_cen_red = np.ravel(warc_cen_red)
    warc_sigma_red = np.ravel(warc_sigma_red)

    # Measuring skylines
    b_cont = 0
    r_cont = 0
    single_file_list = []
    for k in np.arange(len(file_list)):
        single_file = fits.open(file_dir + file_list[k])
        if (single_file[0].header['CAMERA'] == 'WEAVEBLUE') & (b_cont == 0):
            single_file_list.append(k)
            b_cont += 1
        if (single_file[0].header['CAMERA'] == 'WEAVERED') & (r_cont == 0):
            single_file_list.append(k)
            r_cont += 1

    for k in np.arange(len(single_file_list)):
        print(
            '     LSF plots: Measuring sky fibers (Single file ' + str(k + 1) + '/' + str(len(single_file_list)) + '):')

        single_file = fits.open(file_dir + file_list[k])
        single_name = single_file[1].name[:-5] + file_list[k][6:-4]

        file_cam = single_file[0].header['CAMERA']

        sky_lam = (np.arange(single_file[1].header['NAXIS1']) * single_file[1].header['CD1_1']) + \
                  single_file[1].header['CRVAL1']
        sky_spec = (single_file[3].data - single_file[1].data) * single_file[5].data * 1e15

        resol = single_file[6].data['RESOL']

        sky_flux = []
        sky_flux_med = []
        sky_sigma_med = []

        cen_lam = SKY_LINES[mode][file_cam]

        with mp.Pool(args.nproc) as pool:
            warc_stats = pool.starmap(fiber_lines,
                                      tqdm.tqdm(zip((fiber, cen_lam, sky_spec, sky_lam, lam_wind, sky_plot_flag,
                                                     sky_plot_dir, file_cam)
                                                    for fiber in np.arange(len(sky_spec))), total=len(sky_spec)))
        print('')

        sky_cen = np.zeros((len(warc_stats), len(cen_lam)))
        sky_sigma = np.zeros((len(warc_stats), len(cen_lam)))

        for i in np.arange(len(warc_stats)):
            sky_flux.extend(warc_stats[i][0])
            sky_flux_med.append(warc_stats[i][1])
            sky_sigma_med.append(warc_stats[i][5])
            for l in np.arange(len(cen_lam)):
                sky_cen[i, l] = warc_stats[i][2][l]
                sky_sigma[i, l] = warc_stats[i][4][l]

        if (single_file[1].name[:-5] == 'RED') & (mode == 'LOWRES'):
            fit_sky_cen = np.ravel(sky_cen[np.isfinite(sky_cen)])
            fit_sky_sigma = np.ravel(sky_sigma[np.isfinite(sky_sigma)])
            popt, pcov = curve_fit(polynom, fit_sky_cen, fit_sky_sigma, maxfev=5000)

        sky_diff_m = []
        sky_cen_diff_m = []

        for i in np.arange(6):
            sky_diff_m.append(np.median((np.ravel(sky_sigma_med) - resol)[i * 100:(i + 1) * 100]))
            sky_cen_diff_m.append(np.mean([i * 100, (i + 1) * 100]))

        # ------- plotting the sky resolution

        ax = plt.subplot(gs[1 + (5 * k), 0])
        ax.plot(np.arange(len(resol)), np.ravel(sky_sigma_med), '.', color=single_file[1].name[:-5], alpha=0.5,
                zorder=-1,
                label='sky fits')
        if len(warc_list) > 0:
            if single_file[0].header['CAMERA'] == 'WEAVEBLUE' and len(warc_sigma_med_blue) > 0:
                ax.plot(np.arange(len(resol)), np.ravel(warc_sigma_med_blue), '.', color='orange', alpha=0.5, zorder=-1,
                        label='warc fits')
            if single_file[0].header['CAMERA'] == 'WEAVERED' and len(warc_sigma_med_red) > 0:
                ax.plot(np.arange(len(resol)), np.ravel(warc_sigma_med_red), '.', color='orange', alpha=0.5, zorder=-1,
                        label='warc fits')
        ax.plot(np.arange(len(resol)), resol, 's', color='dimgray', markersize=3, alpha=0.5, zorder=-1,
                label='reduc info')
        ax.set_xlabel('fiber #')
        ax.set_ylabel('FWHM [A]')
        ax.legend()

        y_all = sky_sigma.ravel()

        if single_file[0].header['CAMERA'] == 'WEAVEBLUE' and len(warc_sigma_med_blue) > 0:
            y_all = np.concatenate([y_all, warc_sigma_blue])

        if single_file[0].header['CAMERA'] == 'WEAVERED' and len(warc_sigma_med_red) > 0:
            y_all = np.concatenate([y_all, warc_sigma_red])

        ymin, ymax = np.nanpercentile(y_all, [0.1, 99.9])
        ymin = 0.9 * ymin
        ymax = 1.1 * ymax

        ax_t = plt.subplot(gs[1 + (5 * k), 1])
        ax_t.plot(sky_cen, sky_sigma, '.', color=single_file[1].name[:-5], alpha=0.1, zorder=-1)
        title = single_name
        if len(warc_list) > 0:
            if single_file[0].header['CAMERA'] == 'WEAVEBLUE' and len(warc_sigma_med_blue) > 0:
                ax_t.plot(warc_cen_blue, warc_sigma_blue, '.', color='orange', alpha=0.1, zorder=-2)
                if len(warc_list) > 1:
                    title = title + '  ' + warc_list[1][:-4]
                else:
                    title = title + '  ' + warc_list[0][:-4]
            if single_file[0].header['CAMERA'] == 'WEAVERED' and len(warc_sigma_med_red) > 0:
                ax_t.plot(warc_cen_red, warc_sigma_red, '.', color='orange', alpha=0.1, zorder=-2)
                title = title + '  ' + warc_list[0][:-4]
        title = title + ' / spectral resolution'
        t = ax_t.set_title(title)

        if (single_file[1].name[:-5] == 'RED') & (mode == 'LOWRES'):
            ax_t.plot(sky_lam, polynom(sky_lam, *popt), linestyle='--', color='gray')
            ax_t.annotate(r'FWHM = ' + ('%.2g' % popt[0]) + ' + ' + ('%.2g' % popt[1]) + '$\lambda$ + ' + (
                    '%.2g' % popt[2]) + '$\lambda^2$', (0.02, 0.95), xycoords='axes fraction')
        ax_t.set_xlim([min(sky_lam), max(sky_lam)])
        ax_t.set_ylim(ymin, ymax)
        ax_t.set_xlabel(r'$\lambda$ [$\AA$]')
        ax_t.set_ylabel('FWHM [A]')

        if mode == 'LOWRES':
            exp_res = 2500
        else:
            exp_res = 10000

        y_all = (sky_cen / sky_sigma).ravel()

        if single_file[0].header['CAMERA'] == 'WEAVEBLUE' and len(warc_sigma_med_blue) > 0:
            y_all = np.concatenate([y_all, warc_cen_blue / warc_sigma_blue])

        if single_file[0].header['CAMERA'] == 'WEAVERED' and len(warc_sigma_med_red) > 0:
            y_all = np.concatenate([y_all, warc_cen_red / warc_sigma_red])

        ymin, ymax = np.nanpercentile(y_all, [0.1, 99.9])
        ymin = 0.9 * ymin
        ymax = 1.1 * ymax

        ax = plt.subplot(gs[1 + (5 * k), 2])
        ax.plot(sky_cen, sky_cen / sky_sigma, '.', color=single_file[1].name[:-5], alpha=0.1, zorder=-1)
        if len(warc_list) > 0:
            if single_file[0].header['CAMERA'] == 'WEAVEBLUE' and len(warc_sigma_med_blue) > 0:
                ax.plot(warc_cen_blue, warc_cen_blue / warc_sigma_blue, '.', color='orange', alpha=0.1, zorder=-2)
            if single_file[0].header['CAMERA'] == 'WEAVERED' and len(warc_sigma_med_red) > 0:
                ax.plot(warc_cen_red, warc_cen_red / warc_sigma_red, '.', color='orange', alpha=0.1, zorder=-2)
        ax.axhline(exp_res, linestyle='--', color='black', zorder=-3)
        ax.axhline(np.mean(sky_lam) / np.nanmedian(resol), linestyle='-.', color='gray', zorder=-3)
        ax.axhline(np.mean(sky_lam) / (np.nanmedian(resol) + np.std(resol)), linestyle='-.', color='gray', zorder=-3,
                   alpha=0.5)
        ax.axhline(np.mean(sky_lam) / (np.nanmedian(resol) - np.std(resol)), linestyle='-.', color='gray', zorder=-3,
                   alpha=0.5)
        ax.set_xlim([min(sky_lam), max(sky_lam)])
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(r'R [$\lambda$ / FWHM]')
        ax.set_xlabel(r'$\lambda$ [$\AA$]')

        # analyzing spectral resolution parameter
        if single_file[0].header['CAMERA'] == 'WEAVEBLUE':
            blue_spec_resol = np.round(100 * np.sum(sky_cen / sky_sigma > (exp_res - (0.1 * exp_res)))
                                       / len(np.ravel(sky_cen)))
        else:
            red_spec_resol = np.round(100 * np.sum(sky_cen / sky_sigma > (exp_res - (0.1 * exp_res)))
                                      / len(np.ravel(sky_cen)))

        # saving spectral resolution text file

        if (single_file[1].name[:-5] == 'RED') & (mode == 'LOWRES'):
            np.savetxt(gal_dir + '/resol_table_' + single_name + '.txt',
                       np.column_stack([sky_lam, polynom(sky_lam, *popt)]),
                       fmt=['%.1f', '%.2f'])
        else:
            np.savetxt(gal_dir + '/resol_table_' + single_name + '.txt',
                       np.column_stack([sky_lam, (sky_lam * 0) + np.median(sky_sigma)]), fmt=['%.1f', '%.2f'])

        if single_file[0].header['CAMERA'] == 'WEAVEBLUE':
            resol_fibinfo_blue = np.nanmedian(resol)
        if single_file[0].header['CAMERA'] == 'WEAVERED':
            resol_fibinfo_red = np.nanmedian(resol)

        # ------- plotting the fiber throughput

        sky_flux_func = sky_flux_med / np.median(sky_flux_med)

        ax = plt.subplot(gs[2 + (5 * k), :])
        ax.plot(sky_flux_func, color=single_file[1].name[:-5], alpha=0.5)
        ax.set_xlabel('fiber #')
        ax.set_ylabel('relative median sky lines flux')
        ax.set_ylim([0.7, 1.3])
        ax.set_title('fiber throughput')
        ft_m = np.median(sky_flux_func)
        ft_p1_l, ft_p1_h = np.percentile(sky_flux_func, [15.87, 84.13])
        ft_p1 = (ft_p1_h - ft_p1_l) / 2
        ft_p3_l, ft_p3_h = np.percentile(sky_flux_func, [0.135, 99.865])
        ft_p3 = (ft_p3_h - ft_p3_l) / 2
        ax.annotate(r'median = ' + f"{ft_m:.2f}", (0.9, 0.9), xycoords='axes fraction')
        ax.annotate(r'84 perc = ' + f"{ft_p1:.2f}", (0.9, 0.8), xycoords='axes fraction')
        ax.annotate(r'99 perc = ' + f"{ft_p3:.2f}", (0.9, 0.7), xycoords='axes fraction')
        ax.grid()

        if single_file[0].header['CAMERA'] == 'WEAVEBLUE':
            blue_fiber_through = np.round(100 * np.sum((sky_flux_func > 0.99) & (sky_flux_func < 1.01))
                                          / len(sky_flux_func))
        else:
            red_fiber_through = np.round(100 * np.sum((sky_flux_func > 0.99) & (sky_flux_func < 1.01))
                                         / len(sky_flux_func))

        # ------- plotting the wavelength solution

        sky_cal_func = np.nanmedian(sky_cen - np.nanmedian(sky_cen, axis=0), axis=1)

        ax = plt.subplot(gs[3 + (5 * k), :])
        ax.plot(sky_cal_func, color=single_file[1].name[:-5],
                alpha=0.5, label='sky lines')
        if single_file[0].header['CAMERA'] == 'WEAVEBLUE' and len(warc_cen_blue_ext) > 0:
            ax.plot(np.nanmedian(warc_cen_blue_ext - np.nanmedian(warc_cen_blue_ext, axis=0), axis=1), color='orange',
                    alpha=0.5, label='warc lines')
        if single_file[0].header['CAMERA'] == 'WEAVERED' and len(warc_cen_red_ext) > 0:
            ax.plot(np.nanmedian(warc_cen_red_ext - np.nanmedian(warc_cen_red_ext, axis=0), axis=1), color='orange',
                    alpha=0.5, label='warc lines')
        ax.set_xlabel('fiber #')
        ax.set_ylabel(r'relative sky line offsets [$\AA$]')
        ax.set_ylim([-0.5, 0.5])
        ax.set_title('wavelength calibration')
        ws_sky_m = np.median(sky_cal_func)
        ws_warc_m = np.median(np.nanmedian(warc_cen_blue_ext - np.nanmedian(warc_cen_blue_ext, axis=0)))
        ws_sky_p1_l, ws_sky_p1_h = np.percentile(sky_cal_func, [15.87, 84.13])
        ws_sky_p1 = (ws_sky_p1_h - ws_sky_p1_l) / 2
        ws_warc_p1_l, ws_warc_p1_h = np.percentile(np.nanmedian(warc_cen_blue_ext -
                                                                np.nanmedian(warc_cen_blue_ext, axis=0)),
                                                   [15.87, 84.13])
        ws_warc_p1 = (ws_warc_p1_h - ws_warc_p1_l) / 2
        ax.annotate(r'sky median = ' + f"{ws_sky_m:.3f}" + r' $\pm$ ' + f"{ws_sky_p1:.3f}"
                    + r'$\AA$', (0.1, 0.2), xycoords='axes fraction')
        ax.annotate(r'warc median = ' + f"{ws_warc_m:.3f}" + r' $\pm$ ' + f"{ws_warc_p1:.3f}"
                    + r'$\AA$', (0.1, 0.1), xycoords='axes fraction')
        ax.grid()
        ax.legend()

        if single_file[0].header['CAMERA'] == 'WEAVEBLUE':
            blue_wave_calib = np.round(100 * np.sum(abs(sky_cal_func) < 0.2 * spec_pix) / len(sky_cal_func))
        else:
            red_wave_calib = np.round(100 * np.sum(abs(sky_cal_func) < 0.2 * spec_pix) / len(sky_cal_func))

        # ------ estimate SNR using the ETC

        if (single_file[1].name[:-5] == 'BLUE') & (mode == 'LOWRES'):
            sgn_band = np.mean(single_file[1].data[:, (sky_lam > 5000) & (sky_lam < 6000)], axis=1)
            rms_band = np.sqrt(1 / np.mean(single_file[2].data[:, (sky_lam > 5000) & (sky_lam < 6000)], axis=1))
            snr_band = sgn_band / rms_band
            snr_band = snr_band * np.sqrt(spec_pix)

            data_path = files("waqaqc.data").joinpath("johnsonV.dat")
            with data_path.open("r") as f:
                band_data = np.loadtxt(f)
            f_vega = 3.39e-9
            band = 'V'
            ins_mode = 'blueLR'

        if (single_file[1].name[:-5] == 'RED') & (mode == 'LOWRES'):
            sgn_band = np.mean(single_file[1].data[:, (sky_lam > 8000) & (sky_lam < 9000)], axis=1)
            rms_band = np.sqrt(1 / np.mean(single_file[2].data[:, (sky_lam > 8000) & (sky_lam < 9000)], axis=1))
            snr_band = sgn_band / rms_band
            snr_band = snr_band * np.sqrt(spec_pix)

            data_path = files("waqaqc.data").joinpath("johnsonI.dat")
            with data_path.open("r") as f:
                band_data = np.loadtxt(f)
            f_vega = 9.24e-10
            band = 'I'
            ins_mode = 'redLR'

        if (single_file[1].name[:-5] == 'BLUE') & (mode == 'HIGHRES'):
            sgn_band = np.mean(single_file[1].data[:, (sky_lam > 5000) & (sky_lam < 6000)], axis=1)
            rms_band = np.sqrt(1 / np.mean(single_file[2].data[:, (sky_lam > 5000) & (sky_lam < 6000)], axis=1))
            snr_band = sgn_band / rms_band
            snr_band = snr_band * np.sqrt(spec_pix)

            data_path = files("waqaqc.data").joinpath("johnsonV.dat")
            with data_path.open("r") as f:
                band_data = np.loadtxt(f)
            f_vega = 3.39e-9
            band = 'V'
            ins_mode = 'greenHR'

        if (single_file[1].name[:-5] == 'RED') & (mode == 'HIGHRES'):
            sgn_band = np.mean(single_file[1].data[:, (sky_lam > 6550) & (sky_lam < 7550)], axis=1)
            rms_band = np.sqrt(1 / np.mean(single_file[2].data[:, (sky_lam > 6550) & (sky_lam < 7550)], axis=1))
            snr_band = sgn_band / rms_band
            snr_band = snr_band * np.sqrt(spec_pix)

            data_path = files("waqaqc.data").joinpath("johnsonR.dat")
            with data_path.open("r") as f:
                band_data = np.loadtxt(f)
            f_vega = 3.08e-9
            band = 'R'
            ins_mode = 'redHR'

        band_wave = band_data[:, 0]
        band_trans = band_data[:, 1]
        interp_T = interp1d(band_wave, band_trans, bounds_error=False, fill_value=0.0)
        T_lambda = interp_T(sky_lam)
        T_norm = np.trapz(T_lambda, sky_lam)
        single_band = single_file[1].data * np.mean(single_file[5].data[:], axis=0) * T_lambda[np.newaxis, :]
        flux_band = np.zeros((single_band.shape[0]))
        for i in range(single_band.shape[1] - 1):
            delta_lambda = sky_lam[i + 1] - sky_lam[i]
            avg_flux = 0.5 * (single_band[:, i] + single_band[:, i + 1])
            flux_band += avg_flux * delta_lambda
        flux_band /= T_norm
        mag_band = -2.5 * np.log10(flux_band / f_vega)
        mag_band[~np.isfinite(mag_band)] = np.nan

        etc_mag = np.linspace(13, np.nanmax(mag_band) + 1)
        etc_snr = []

        seeing = np.round((single_file[0].header['SEEINGB'] + single_file[0].header['SEEINGE']) / 2, 2)
        exp_time = np.round(single_file[0].header['EXPTIME'], 2)
        if (single_file[0].header['SKYBRTEL'] == -99) & (single_file[0].header['SKYBRZEN'] == -99):
            try:
                sky_bright = np.round(single_file[0].header['SKBRMODB'], 2)
            except:
                sky_bright = 21.0
        elif single_file[0].header['SKYBRTEL'] == -99:
            try:
                sky_bright = np.round(single_file[0].header['SKYBRZEN'], 2)
            except:
                sky_bright = 21.0
        else:
            try:
                sky_bright = np.round(single_file[0].header['SKYBRTEL'], 2)
            except:
                sky_bright = 21.0
        if (sky_bright == -99) | (sky_bright == 0):
            sky_bright = 21.0
        air_mass = np.round(single_file[0].header['AIRMASS'], 2)

        for i in etc_mag:
            result = signalWEAVE(mag=i, time=exp_time, band=band, seeing_input=seeing, instrument_mode=ins_mode,
                                 skysb=sky_bright, airmass=air_mass, LIFU=True, verbose=False)
            etc_snr.append(np.round(result['SNR'], 2))
        etc_snr = np.array(etc_snr)

        ax = plt.subplot(gs[4 + (5 * k), :])
        ax.plot(etc_mag, etc_snr, color='gray', linestyle='--', label='ETC')
        ax.scatter(mag_band.flatten(), snr_band.flatten(), s=20, marker='o', alpha=0.3, color=single_file[1].name[:-5],
                   edgecolor='black', label='fiber')
        ax.set_xlim([13, 26])
        ax.set_ylim([0.05, 110])
        ax.set_yscale('log')
        ax.set_xlabel(band + ' band mag (Vega)')
        ax.set_ylabel(r'S/N ratio [per $\AA$]')
        ax.set_title('fiber SNR vs ETC estimate')
        ax.annotate(r'mode = ' + ins_mode, (0.02, 0.3), xycoords='axes fraction')
        ax.annotate(r'airmass = ' + f"{air_mass:.2f}", (0.02, 0.25), xycoords='axes fraction')
        ax.annotate(r'seeing = ' + f"{seeing:.2f}", (0.02, 0.20), xycoords='axes fraction')
        ax.annotate(r'sky brightness = ' + f"{sky_bright:.2f}", (0.02, 0.15), xycoords='axes fraction')
        ax.grid()
        ax.legend()

        mag_obs = mag_band.flatten()[np.isfinite(mag_band.flatten())]
        snr_obs = snr_band.flatten()[np.isfinite(mag_band.flatten())]

        snr_band_etc = np.interp(mag_obs, etc_mag, etc_snr, left=None, right=None)

        ax = plt.subplot(gs[5 + (5 * k), :])
        # ax.scatter(mag_obs, (abs(snr_obs-snr_band_etc)/snr_band_etc), s=20, marker='o', alpha=0.3,
        #            color=single_file[1].name[:-5], edgecolor='black')
        ax.scatter(mag_obs, snr_obs / snr_band_etc, s=20, marker='o', alpha=0.3,
                   color=single_file[1].name[:-5], edgecolor='black')
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        ax.set_xlim([13, 26])
        # ax.set_ylim([0., 1.5])
        ax.set_ylim([0., 2.0])
        ax.set_xlabel(band + ' band mag (Vega)')
        # ax.set_ylabel(r'S/N ratio (|$\Delta$ SNR| / SNR_ETC) [per $\AA$]')
        ax.set_ylabel(r'SNR$_{\mathrm{obs}}$ / SNR$_{\mathrm{ETC}}$ [per $\AA$]')
        ax.grid()

    # ------ flux calibration plots

    for k in np.arange(len(file_list)):
        single_file = fits.open(file_dir + file_list[k])
        lam = (np.arange(single_file[1].header['NAXIS1']) * single_file[1].header['CD1_1']) + \
              single_file[1].header['CRVAL1']
        ax = plt.subplot(gs[11 + k, :])
        ax.plot(lam, np.mean(single_file[1].data, axis=0) * np.median(single_file[5].data, axis=0), color='black',
                label='mean flux spec')
        if single_file[1].name[:-5] == 'RED':
            color = 'red'
        elif single_file[1].name[:-5] == 'BLUE':
            color = 'blue'
        else:
            color = 'orange'
        ax.plot(lam, np.std(single_file[1].data * np.median(single_file[5].data, axis=0), axis=0), color=color,
                label='std spec')
        ax.set_xlabel(r'$\lambda$ [$\AA$]')
        ax.set_ylabel(r'flux [erg s$^{-1}$ cm$^{-2}$ $\AA$]')
        ax.set_title('flux calibration / ' + file_list[k])
        ax.legend()

    # ------ SDSS flux calibration plots

    for k in np.arange(len(single_file_list)):
        single_file = fits.open(file_dir + file_list[k])
        if (single_file[1].name[:-5] == 'BLUE') & (mode == 'LOWRES'):
            sdss = filters.load_filters('sdss2010-g')

            flux = (single_file[1].data * single_file[5].data * u.erg / u.s / u.cm ** 2 / u.AA)
            lam = (np.arange(single_file[1].header['NAXIS1']) *
                   single_file[1].header['CD1_1'] +single_file[1].header['CRVAL1']) * u.AA

            g_mags = sdss.get_ab_magnitudes(flux, lam)['sdss2010-g'].value

            g_cat = single_file[6].data['MAG_G']

        if (single_file[1].name[:-5] == 'RED') & (mode == 'LOWRES'):
            sdss = filters.load_filters('sdss2010-i')

            flux = single_file[1].data * single_file[5].data * u.erg / u.s / u.cm ** 2 / u.AA
            lam = ((np.arange(single_file[1].header['NAXIS1']) * single_file[1].header['CD1_1']) +
                   single_file[1].header['CRVAL1']) * u.AA

            i_mags = sdss.get_ab_magnitudes(flux, lam)['sdss2010-i'].value

            i_cat = single_file[6].data['MAG_I']

    delta_g = g_mags - g_cat
    delta_i = i_mags - i_cat

    delta_gi = ((g_mags - i_mags) -(g_cat - i_cat))
    delta_gi_median = np.nanmedian(delta_gi)

    breakpoint()

    plt.figure()

    plt.subplot(311)
    plt.plot(delta_g, '.', ms=2)
    plt.ylabel('Δg')

    plt.subplot(312)
    plt.plot(delta_i, '.', ms=2)
    plt.ylabel('Δi')

    plt.subplot(313)
    plt.plot(delta_gi, '.', ms=2)
    plt.ylabel('Δ(g-i)')
    plt.xlabel('Fiber')

    plt.tight_layout()
    plt.savefig('teste.png')

    # ------

    fig_l0 = output_str + '_L0.png'

    fig.savefig(fig_l0)

    print('')

    # ------ create master resolution file

    file_list_b = [x for x in os.listdir(gal_dir) if ("BLUE" in x) & ('resol' in x)]
    file_list_r = [x for x in os.listdir(gal_dir) if ("RED" in x) & ('resol' in x)]

    lam_b_res = Table.read(gal_dir + file_list_b[0], format='ascii')['col1'].data
    lam_b_res = np.around(np.arange(2500, max(lam_b_res), lam_b_res[1] - lam_b_res[0]),
                          1)  # workaround # a weird interpolation thing from pyparadise
    lam_r_res = np.around(Table.read(gal_dir + file_list_r[0], format='ascii')['col1'].data, 1)
    lam_aps_res = np.around(np.arange(min(lam_b_res), max(lam_r_res), lam_b_res[1] - lam_b_res[0]), 1)

    resol_blue = lam_b_res * 0
    resol_red = lam_r_res * 0

    for i in file_list_b:
        resol_blue = resol_blue + Table.read(gal_dir + i, format='ascii')['col2'].data[0]
    for i in file_list_r:
        resol_red = resol_red + Table.read(gal_dir + i, format='ascii')['col2'].data

    resol_blue = resol_blue / len(file_list_b)
    resol_red = resol_red / len(file_list_b)
    resol_blue[~np.isfinite(resol_blue)] = resol_fibinfo_blue
    resol_red[~np.isfinite(resol_red)] = resol_fibinfo_red
    resol_aps = lam_aps_res * 0

    for i in lam_aps_res:
        if (i >= min(lam_b_res)) & (i <= max(lam_b_res)):
            resol_aps[lam_aps_res == i] = resol_blue[lam_b_res == i]
        if (i >= min(lam_r_res)) & (i <= max(lam_r_res)):
            resol_aps[lam_aps_res == i] = resol_red[lam_r_res == i]
        if (i >= min(lam_r_res)) and (i <= max(lam_b_res)):
            resol_aps[lam_aps_res == i] = (resol_blue[lam_b_res == i] + resol_red[lam_r_res == i]) / 2
        if (i >= max(lam_b_res)) & (i <= min(lam_r_res)):
            resol_aps[lam_aps_res == i] = (resol_blue[-1] + resol_red[0]) / 2

    lam_aps_res = lam_aps_res / (1 + redshift)

    np.savetxt(gal_dir + '/resol_table_blue.txt', np.column_stack([lam_b_res, np.around(resol_blue, 2)]),
               fmt=['%.1f', '%.2f'])
    np.savetxt(gal_dir + '/resol_table_red.txt', np.column_stack([lam_r_res, np.around(resol_red, 2)]),
               fmt=['%.1f', '%.2f'])
    np.savetxt(gal_dir + '/resol_table_aps.txt', np.column_stack([lam_aps_res, np.around(resol_aps, 2)]),
               fmt=['%.1f', '%.2f'])
    modes = np.array(['blue', 'red', 'aps'])
    avs = np.array([np.round(np.mean(resol_blue), 2), np.round(np.mean(resol_red), 2), np.round(np.mean(resol_aps), 2)])
    np.savetxt(gal_dir + '/resol_table_mean.txt', np.column_stack((modes, avs)), fmt='%s')

    return fig_l0, blue_spec_resol, red_spec_resol, blue_fiber_through, red_fiber_through, blue_wave_calib, \
           red_wave_calib


def polynom(x, a, b, c):
    return a + b * x + c * (x ** 2)
