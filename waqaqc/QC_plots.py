import os
import configparser
import json

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colorbar import Colorbar
from vorbin.voronoi_2d_binning import voronoi_2d_binning
import warnings
from astropy.table import Table
import requests
from PIL import Image
from io import BytesIO
from paramiko import SSHClient
from scp import SCPClient
from astropy.coordinates import SkyCoord, FK5
import multiprocessing as mp
from scipy.optimize import curve_fit


def getimages(ra, dec, filters="grizy"):
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


def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    """Get URL for images in the table

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """

    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra, dec, filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={format}")
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

    return nbc, nbc_err, nb_pixs


def html_plots(self):
    warnings.filterwarnings("ignore")

    config = configparser.ConfigParser()
    config.read(self)

    file_dir = config.get('APS_cube', 'file_dir')

    global blue_cube_data
    global blue_cube_err

    global red_cube_data
    global red_cube_err

    global aps_cube_data
    global aps_cube_err

    blue_cube = fits.open(file_dir + config.get('QC_plots', 'blue_cube'))
    red_cube = fits.open(file_dir + config.get('QC_plots', 'red_cube'))

    blue_cube_data = blue_cube[1].data
    blue_cube_err = blue_cube[2].data

    red_cube_data = red_cube[1].data
    red_cube_err = red_cube[2].data

    gal_name = blue_cube[0].header['CCNAME1']
    date = blue_cube[0].header['DATE-OBS']

    gal_dir = gal_name + '_' + blue_cube[0].header['MODE'] + '_' + str(blue_cube[0].header['OBID']) + '/'
    os.makedirs(gal_dir, exist_ok=True)

    targetSN = np.float(config.get('QC_plots', 'target_SN'))
    levels = np.array(json.loads(config.get('QC_plots', 'levels'))).astype(np.float)  # SNR levels to display

    colap_b_map = np.sum(blue_cube[1].data[:], axis=0)
    colap_r_map = np.sum(red_cube[1].data[:], axis=0)

    mean_b_map = np.nanmean(blue_cube[1].data[:], axis=0)
    mean_r_map = np.nanmean(red_cube[1].data[:], axis=0)

    median_b_map = np.nanmedian(blue_cube[1].data[:], axis=0)
    median_r_map = np.nanmedian(red_cube[1].data[:], axis=0)

    blue_sky_cube = blue_cube[3].data - blue_cube[1].data
    red_sky_cube = red_cube[3].data - red_cube[1].data
    fits.writeto(gal_dir + 'blue_sky_cube.fits', blue_sky_cube, overwrite=True)
    fits.writeto(gal_dir + 'red_sky_cube.fits', red_sky_cube, overwrite=True)

    mean_b_sky_map = np.nanmean(blue_sky_cube, axis=0)
    mean_r_sky_map = np.nanmean(red_sky_cube, axis=0)

    median_b_sky_map = np.nanmedian(blue_sky_cube, axis=0)
    median_r_sky_map = np.nanmedian(red_sky_cube, axis=0)

    mask_bright_b = mean_b_map > mean_b_sky_map
    mask_medium_b = (median_b_map > median_b_sky_map) & (mean_b_map <= mean_b_sky_map)
    mask_faint_b = (mean_b_map > 0) & (median_b_map <= median_b_sky_map)
    mask_all_b = mean_b_map > 0

    mask_bright_r = mean_r_map > mean_r_sky_map
    mask_medium_r = (median_r_map > median_r_sky_map) & (mean_r_map <= mean_r_sky_map)
    mask_faint_r = (mean_r_map > 0) & (median_r_map <= median_r_sky_map)
    mask_all_r = mean_r_map > 0

    int_b_spec_bright = np.sum(blue_cube[1].data * mask_bright_b[np.newaxis, :, :], axis=(1, 2)) * \
                        np.mean(blue_cube[5].data[:],axis=0) / np.sum(mask_bright_b)
    int_b_spec_medium = np.sum(blue_cube[1].data * mask_medium_b[np.newaxis, :, :], axis=(1, 2)) * \
                        np.mean(blue_cube[5].data[:],axis=0) / np.sum(mask_medium_b)
    int_b_spec_faint = np.sum(blue_cube[1].data * mask_faint_b[np.newaxis, :, :], axis=(1, 2)) * \
                       np.mean(blue_cube[5].data[:],axis=0) / np.sum(mask_faint_b)
    int_r_spec_bright = np.sum(red_cube[1].data * mask_bright_r[np.newaxis, :, :], axis=(1, 2)) * \
                        np.mean(red_cube[5].data[:],axis=0) / np.sum(mask_bright_r)
    int_r_spec_medium = np.sum(red_cube[1].data * mask_medium_r[np.newaxis, :, :], axis=(1, 2)) * \
                        np.mean(red_cube[5].data[:],axis=0) / np.sum(mask_medium_r)
    int_r_spec_faint = np.sum(red_cube[1].data * mask_faint_r[np.newaxis, :, :], axis=(1, 2)) * \
                       np.mean(red_cube[5].data[:],axis=0) / np.sum(mask_faint_r)

    int_b_sky_spec = np.sum(blue_cube[3].data - blue_cube[1].data, axis=(1, 2)) * np.mean(blue_cube[5].data[:],axis=0) / np.sum(
        mask_faint_b)
    int_r_sky_spec = np.sum(red_cube[3].data - red_cube[1].data, axis=(1, 2)) * np.mean(red_cube[5].data[:],axis=0) / np.sum(
        mask_faint_r)

    blue_cen_wave = int(config.get('QC_plots', 'blue_wav'))
    red_cen_wave = int(config.get('QC_plots', 'red_wav'))

    lam_r = red_cube[1].header['CRVAL3'] + (np.arange(red_cube[1].header['NAXIS3']) * red_cube[1].header['CD3_3'])
    lam_b = blue_cube[1].header['CRVAL3'] + (np.arange(blue_cube[1].header['NAXIS3']) * blue_cube[1].header['CD3_3'])

    med_b = np.median(blue_cube[1].data[np.where(lam_b == blue_cen_wave)[0][0] -
                                        100:np.where(lam_b == blue_cen_wave)[0][0] + 100], axis=0)
    sgn_b = np.mean(blue_cube[1].data[np.where(lam_b == blue_cen_wave)[0][0] -
                                      100:np.where(lam_b == blue_cen_wave)[0][0] + 100], axis=0)
    rms_b = np.sqrt(1/np.mean(blue_cube[2].data[np.where(lam_b == blue_cen_wave)[0][0] -
                                      100:np.where(lam_b == blue_cen_wave)[0][0] + 100], axis=0))
    # rms_b = np.sqrt(sgn_b)
    snr_b = sgn_b / rms_b

    med_r = np.median(red_cube[1].data[np.where(lam_r == red_cen_wave)[0][0] -
                                       100:np.where(lam_r == red_cen_wave)[0][0] + 100], axis=0)
    sgn_r = np.mean(red_cube[1].data[np.where(lam_r == red_cen_wave)[0][0] -
                                     100:np.where(lam_r == red_cen_wave)[0][0] + 100], axis=0)
    rms_r = np.sqrt(1/np.mean(red_cube[2].data[np.where(lam_r == red_cen_wave)[0][0] -
                                     100:np.where(lam_r == red_cen_wave)[0][0] + 100], axis=0))
    # rms_r = np.sqrt(sgn_r)
    snr_r = sgn_r / rms_r

    nsc = SkyCoord(ra=blue_cube[1].header['CRVAL1'], dec=blue_cube[1].header['CRVAL2'], unit='deg', frame=FK5)

    url = geturl(nsc.ra.value, nsc.dec.value, size=480, filters="grizy", output_size=None, format="jpg", color=True)
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))

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

    # doing the plots

    print('Doing L0 raw data plots')

    file_list = [x for x in os.listdir(file_dir) if ("APS" not in x) & ('single' in x)]
    warc_list = [x for x in os.listdir(file_dir) if ('warc' in x)]

    rows = len(file_list)

    fig = plt.figure(figsize=(14, 5*rows))

    fig.suptitle('L0 QC plots', size=22, weight='bold')

    gs = gridspec.GridSpec(1+rows, 3, height_ratios=np.concatenate((np.array([1]), np.zeros(rows)+0.5)),
                           width_ratios=[1, 1, 1])
    gs.update(left=0.07, right=0.9, bottom=0.05, top=0.95, wspace=0.3, hspace=0.25)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    cdelt1 = 0.25
    cdelt2 = 0.25
    crpix1 = im.size[1] / 2.
    crpix2 = im.size[1] / 2.

    x_ax = (np.arange(im.size[1]) + 1 - crpix1) * cdelt1
    y_ax = (np.arange(im.size[0]) + 1 - crpix2) * cdelt2

    ax = plt.subplot(gs[0, :])
    ax.set_title('PanSTARRS composite image')
    ax.imshow(im, extent=[min(x_ax), max(x_ax), max(y_ax), min(y_ax)])
    ax.plot(0, 0, marker=(6, 0, 0), color='red', markerfacecolor='none', markersize=275)
    ax2 = ax.secondary_xaxis('top')
    ay2 = ax.secondary_yaxis('right')
    ax.tick_params(width=2, size=10)
    ax2.tick_params(width=2, size=10)
    ay2.tick_params(width=2, size=10)
    ax.tick_params(axis="y", direction="inout", color='white')
    ax.tick_params(axis="x", direction="inout", color='white')
    ax2.tick_params(direction="inout", color='white')
    ay2.tick_params(direction="inout", color='white')
    ax.set_ylabel(r'$\Delta$X [arcsec]')
    ax.set_xlabel(r'$\Delta$Y [arcsec]')

    # ------

    # LSF plots

    def gauss(x, a, b, amp, x0, sigma):
        return a + b * x + amp * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def polinom(x, a, b, c):
        return a + b * x + c * (x ** 2)

    if blue_cube[0].header['MODE'] == 'LOWRES':
        lam_wind = 10
    if blue_cube[0].header['MODE'] == 'HIGHRES':
        lam_wind = 30

    if len(warc_list) > 0:
        warc_cen_blue = []
        warc_sigma_blue = []
        warc_cen_med_blue = []
        warc_sigma_med_blue = []

        warc_cen_red = []
        warc_sigma_red = []
        warc_cen_med_red = []
        warc_sigma_med_red = []

        for i in np.arange(2):

            warc_name = warc_list[i][:-4]
            warc_file = fits.open(file_dir + warc_name + '.fit')

            lamp_lam = (np.arange(warc_file[1].header['NAXIS1']) * warc_file[1].header['CD1_1']) + warc_file[1].header[
                'CRVAL1']
            lamp_spec = warc_file[1].data

            for fiber in np.arange(len(lamp_spec)):

                print('     LSF plots: Measuring WARC fibers: ' + str(round(100. * fiber / len(lamp_spec), 1)) + '%',
                      end='\r')

                cen_lam = lamp_lam[lam_wind+1:-(lam_wind+2)][np.diff(lamp_spec[fiber])[lam_wind+1:-(lam_wind+1)] < -1500]

                fib_cen = []
                fib_sigma = []

                old_cen = 0

                for i in np.arange(len(cen_lam) - 1):
                    lam_wind_c = np.where(lamp_lam == min(lamp_lam, key=lambda x: abs(x - cen_lam[i])))[0][0]
                    w_lam = lamp_lam[lam_wind_c - lam_wind: lam_wind_c + lam_wind]
                    w_spec = lamp_spec[fiber][lam_wind_c - lam_wind: lam_wind_c + lam_wind]
                    if (abs(cen_lam[i] - old_cen) > 5) and (all(w_spec < 330000.)) and (
                            w_spec[int(w_spec.size / 2)] / 4 > w_spec[0]) and (
                            w_spec[int(w_spec.size / 2)] / 4 > w_spec[-1]):
                        try:
                            popt, pcov = curve_fit(gauss, w_lam, w_spec, p0=[0, 0, max(w_spec) / 2, cen_lam[i], 3],
                                                   bounds=([-np.inf, -np.inf, 0, 0, 0],
                                                           [np.inf, np.inf, np.inf, np.inf, np.inf]))

                            if (popt[4] * 2.355 > 0.1) & (popt[4] * 2.355 < 5.0):
                                fib_cen.append(popt[3])
                                fib_sigma.append(popt[4] * 2.355)

                                old_cen = cen_lam[i]

                        except:
                            pass

                if warc_file[0].header['CAMERA'] == 'WEAVEBLUE':
                    warc_cen_blue.extend(np.ravel(fib_cen))
                    warc_sigma_blue.extend(np.ravel(fib_sigma))
                    warc_cen_med_blue.append(np.nanmedian(np.ravel(fib_cen)))
                    warc_sigma_med_blue.append(np.nanmedian(np.ravel(fib_sigma)))

                if warc_file[0].header['CAMERA'] == 'WEAVERED':
                    warc_cen_red.extend(np.ravel(fib_cen))
                    warc_sigma_red.extend(np.ravel(fib_sigma))
                    warc_cen_med_red.append(np.nanmedian(np.ravel(fib_cen)))
                    warc_sigma_med_red.append(np.nanmedian(np.ravel(fib_sigma)))

        print('')

        warc_cen_blue = np.ravel(warc_cen_blue)
        warc_sigma_blue = np.ravel(warc_sigma_blue)

        warc_cen_red = np.ravel(warc_cen_red)
        warc_sigma_red = np.ravel(warc_sigma_red)

    for k in np.arange(len(file_list)):

        single_file = fits.open(file_dir+file_list[k])
        single_name = single_file[1].name[:-5] + file_list[k][6:-4]

        sky_lam = (np.arange(single_file[1].header['NAXIS1']) * single_file[1].header['CD1_1']) + \
                  single_file[1].header['CRVAL1']
        sky_spec = (single_file[3].data - single_file[1].data) * single_file[5].data * 1e15

        resol = single_file[6].data['RESOL']

        sky_cen = []
        sky_sigma = []
        sky_cen_med = []
        sky_sigma_med = []

        for fiber in np.arange(len(sky_spec)):

            print('     LSF plots (' + str(k) + '/' + str(len(file_list)) + '): Measuring sky fibers: ' +
                  str(round(100. * fiber / len(sky_spec),1)) + '%', end='\r')

            if blue_cube[0].header['MODE'] == 'LOWRES':
                cen_lam = sky_lam[lam_wind+1:-(lam_wind+2)][np.diff(sky_spec[fiber])[lam_wind+1:-(lam_wind+1)] < -0.2]
            if blue_cube[0].header['MODE'] == 'HIGHRES':
                cen_lam = sky_lam[lam_wind+1:-(lam_wind+2)][np.diff(sky_spec[fiber])[lam_wind+1:-(lam_wind+1)] < -0.03]

            old_cen = 0

            fib_cen = []
            fib_sigma = []

            for i in np.arange(len(cen_lam) - 1):
                lam_wind_c = np.where(sky_lam == min(sky_lam, key=lambda x: abs(x - cen_lam[i])))[0][0]
                w_lam = sky_lam[lam_wind_c - lam_wind: lam_wind_c + lam_wind]
                w_spec = sky_spec[fiber][lam_wind_c - lam_wind: lam_wind_c + lam_wind]
                if (abs(cen_lam[i] - old_cen) > 5) and (w_spec[int(w_spec.size / 2)] / 4 > w_spec[0]) and (
                        w_spec[int(w_spec.size / 2)] / 4 > w_spec[-1]):
                    try:
                        popt, pcov = curve_fit(gauss, w_lam, w_spec, p0=[0, 0, max(w_spec) / 2, cen_lam[i], 3],
                                               bounds=([-np.inf,-np.inf,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf]))

                        if (popt[4] * 2.355 > 0.1) & (popt[4] * 2.355 < 5.0):
                            fib_cen.append(popt[3])
                            fib_sigma.append(popt[4] * 2.355)

                            old_cen = cen_lam[i]

                    except:
                        pass

            sky_cen.extend(np.ravel(fib_cen))
            sky_sigma.extend(np.ravel(fib_sigma))

            sky_cen_med.append(np.median(np.ravel(fib_cen)))
            sky_sigma_med.append(np.median(np.ravel(fib_sigma)))

        sky_cen = np.ravel(sky_cen)
        sky_sigma = np.ravel(sky_sigma)

        print('')

        if (single_file[1].name[:-5] == 'RED') & (blue_cube[0].header['MODE'] == 'LOWRES'):
            popt, pcov = curve_fit(polinom, sky_cen, sky_sigma, maxfev=5000)

        sky_diff_m = []
        sky_cen_diff_m = []

        for i in np.arange(6):
            sky_diff_m.append(np.median((np.ravel(sky_sigma_med) - resol)[i * 100:(i + 1) * 100]))
            sky_cen_diff_m.append(np.mean([i * 100, (i + 1) * 100]))

        if len(warc_list) > 0:
            warc_diff_m_blue = []
            warc_cen_diff_m_blue = []
            warc_diff_m_red = []
            warc_cen_diff_m_red = []

            for i in np.arange(6):
                warc_diff_m_blue.append(np.median((np.ravel(warc_sigma_med_blue) - resol)[i * 100:(i + 1) * 100]))
                warc_cen_diff_m_blue.append(np.mean([i * 100, (i + 1) * 100]))

                warc_diff_m_red.append(np.median((np.ravel(warc_sigma_med_red) - resol)[i * 100:(i + 1) * 100]))
                warc_cen_diff_m_red.append(np.mean([i * 100, (i + 1) * 100]))

            warc_diff_m_blue = np.ravel(warc_diff_m_blue)
            warc_diff_m_red = np.ravel(warc_diff_m_red)

        sky_diff_m = np.ravel(sky_diff_m)

        ax = plt.subplot(gs[1+k, 0])
        ax.plot(np.arange(len(resol)), np.ravel(sky_sigma_med), '.', color = single_file[1].name[:-5], alpha=0.5, zorder=-1,
                label='sky fits')
        if len(warc_list) > 0:
            if single_file[0].header['CAMERA'] == 'WEAVEBLUE':
                ax.plot(np.arange(len(resol)), np.ravel(warc_sigma_med_blue), '.', color = 'orange', alpha=0.5, zorder=-1,
                    label='warc fits')
            if single_file[0].header['CAMERA'] == 'WEAVERED':
                ax.plot(np.arange(len(resol)), np.ravel(warc_sigma_med_red), '.', color = 'orange', alpha=0.5, zorder=-1,
                    label='warc fits')
        ax.plot(np.arange(len(resol)), resol, 's', color='dimgray', markersize=3, alpha=0.5, zorder=-1, label='reduc info')
        ax.set_xlabel('fiber #')
        ax.set_ylabel('FWHM [A]')
        ax.legend()

        ax = plt.subplot(gs[1+k, 1])
        ax.plot(np.arange(len(resol)), np.ravel(sky_sigma_med) - resol, '.', color = single_file[1].name[:-5], alpha=0.5,
                zorder=-1, label='sky fits')
        ax.scatter(sky_cen_diff_m, sky_diff_m, marker='*', color='black', zorder=3)
        if len(warc_list) > 0:
            if single_file[0].header['CAMERA'] == 'WEAVEBLUE':
                ax.plot(np.arange(len(resol)), np.ravel(warc_sigma_med_blue) - resol, '.', color = 'orange', alpha=0.5, zorder=-1, label='warc fits')
                ax.scatter(warc_cen_diff_m_blue, warc_diff_m_blue, marker='*', color='black', zorder=3)
                ax.set_title(single_name + '  ' + warc_list[1][:-4])
            if single_file[0].header['CAMERA'] == 'WEAVERED':
                ax.plot(np.arange(len(resol)), np.ravel(warc_sigma_med_red) - resol, '.', color = 'orange', alpha=0.5, zorder=-1, label='warc fits')
                ax.scatter(warc_cen_diff_m_red, warc_diff_m_red, marker='*', color='black', zorder=3)
                ax.set_title(single_name + '  ' + warc_list[0][:-4])
        else:
            ax.set_title(single_name)
        ax.set_xlabel('fiber #')
        ax.set_ylabel(r'$\Delta$ FWMH [A]')
        ax.axhline(y=0, color='black', linestyle='--')

        ax = plt.subplot(gs[1+k, 2])
        ax.plot(sky_cen, sky_sigma, '.', color = single_file[1].name[:-5], alpha=0.1, zorder=-1)
        if len(warc_list) > 0:
            if single_file[0].header['CAMERA'] == 'WEAVEBLUE':
                ax.plot(warc_cen_blue, warc_sigma_blue, '.', color = 'orange', alpha=0.1, zorder=-2)
            if single_file[0].header['CAMERA'] == 'WEAVERED':
                ax.plot(warc_cen_red, warc_sigma_red, '.', color = 'orange', alpha=0.1, zorder=-2)
        if (single_file[1].name[:-5] == 'RED') & (blue_cube[0].header['MODE'] == 'LOWRES'):
            ax.plot(sky_lam, polinom(sky_lam, *popt), linestyle='--', color='gray')
            ax.annotate(r'FWHM = ' + ('%.2g' % popt[0]) + ' + ' + ('%.2g' % popt[1]) + '$\lambda$ + ' + (
                        '%.2g' % popt[2]) + '$\lambda^2$', (0.02, 0.95), xycoords='axes fraction')
        else:
            ax.axhline(y=np.median(sky_sigma), color = single_file[1].name[:-5])
            ax.annotate(r'median FWHM = ' + str(round(np.median(sky_sigma), 2)), (0.02, 0.95), xycoords='axes fraction')
        ax.set_ylabel('FWHM [A]')
        print('')

        if (single_file[1].name[:-5] == 'RED') & (blue_cube[0].header['MODE'] == 'LOWRES'):
            np.savetxt(gal_dir+'/resol_table_'+single_name+'.txt', np.column_stack([sky_lam, polinom(sky_lam, *popt)]),
                       fmt=['%.1f','%.2f'])
        else:
            np.savetxt(gal_dir+'/resol_table_'+single_name+'.txt',
                       np.column_stack([sky_lam, (sky_lam*0)+ np.median(sky_sigma)]), fmt=['%.1f','%.2f'])

        if single_file[0].header['CAMERA'] == 'WEAVEBLUE':
            resol_fibinfo_blue = np.nanmedian(resol)
        if single_file[0].header['CAMERA'] == 'WEAVERED':
            resol_fibinfo_red = np.nanmedian(resol)

    # ------

    fig_l0 = date + '_' + gal_name + '_' + blue_cube[0].header['MODE'] + '_' + str(blue_cube[0].header['OBID']) \
             + '_L0.png'

    fig.savefig(fig_l0)

    print('')

    # ------ create master resolution file

    file_list_b = [x for x in os.listdir(gal_dir) if ("BLUE" in x) & ('resol' in x)]
    file_list_r = [x for x in os.listdir(gal_dir) if ("RED" in x) & ('resol' in x)]

    lam_b_res = Table.read(gal_dir+file_list_b[0], format='ascii')['col1'].data
    lam_b_res = np.around(np.arange(2500, max(lam_b_res), lam_b_res[1]-lam_b_res[0]),1) # workaround # a weird interpolation thing from pyparadise
    lam_r_res = np.around(Table.read(gal_dir+file_list_r[0], format='ascii')['col1'].data,1)
    lam_aps_res = np.around(np.arange(min(lam_b_res), max(lam_r_res), lam_b_res[1]-lam_b_res[0]),1)

    resol_blue = lam_b_res*0
    resol_red = lam_r_res*0

    for i in file_list_b:
        resol_blue = resol_blue + Table.read(gal_dir+i, format='ascii')['col2'].data[0]
    for i in file_list_r:
        resol_red = resol_red + Table.read(gal_dir+i, format='ascii')['col2'].data

    resol_blue = resol_blue / len(file_list_b)
    resol_red = resol_red / len(file_list_b)
    resol_blue[~np.isfinite(resol_blue)] = resol_fibinfo_blue
    resol_red[~np.isfinite(resol_red)] = resol_fibinfo_red
    resol_aps = lam_aps_res*0

    for i in lam_aps_res:
        if (i >= min(lam_b_res)) & (i <= max(lam_b_res)):
            resol_aps[lam_aps_res == i] = resol_blue[lam_b_res == i]
        if (i >= min(lam_r_res)) & (i <= max(lam_r_res)):
            resol_aps[lam_aps_res == i] = resol_red[lam_r_res == i]
        if (i >= min(lam_r_res)) and (i <= max(lam_b_res)):
            resol_aps[lam_aps_res == i] = (resol_blue[lam_b_res == i] + resol_red[lam_r_res == i]) / 2
        if (i >= max(lam_b_res)) & (i <= min(lam_r_res)):
            resol_aps[lam_aps_res == i] = (resol_blue[-1] + resol_red[0]) / 2

    lam_aps_res = lam_aps_res/(1+np.float(config.get('pyp_params', 'redshift')))

    np.savetxt(gal_dir + '/resol_table_blue.txt', np.column_stack([lam_b_res, np.around(resol_blue, 2)]),
               fmt=['%.1f', '%.2f'])
    np.savetxt(gal_dir + '/resol_table_red.txt', np.column_stack([lam_r_res, np.around(resol_red, 2)]),
               fmt=['%.1f', '%.2f'])
    np.savetxt(gal_dir + '/resol_table_aps.txt', np.column_stack([lam_aps_res, np.around(resol_aps, 2)]),
               fmt=['%.1f', '%.2f'])
    modes = np.array(['blue','red','aps'])
    avs = np.array([np.round(np.mean(resol_blue), 2), np.round(np.mean(resol_red), 2), np.round(np.mean(resol_aps), 2)])
    np.savetxt(gal_dir + '/resol_table_mean.txt', np.column_stack((modes, avs)), fmt='%s')

    # ==================================================================

    # creating plots for L1 datacubes

    print('Doing L1 datacubes plots')
    print('')

    fig = plt.figure(figsize=(14, 68))

    fig.suptitle('L1 QC plots', size=22, weight='bold')

    gs = gridspec.GridSpec(15, 5, height_ratios=[1, 0.6, 0.6, 1, 1, 1, 1, 0.6, 0.6, 1, 0.6, 0.6, 1, 0.6, 0.6],
                           width_ratios=[1, 0.06, 0.3, 1, 0.06])
    gs.update(left=0.07, right=0.9, bottom=0.05, top=0.95, wspace=0.0, hspace=0.25)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # ------

    wcs = WCS(axis_header)
    ax = plt.subplot(gs[0, 0], projection=wcs)

    im = ax.imshow(np.log10(colap_b_map), origin='lower')

    ypmax_b = np.where(colap_b_map == np.nanmax(colap_b_map))[0]
    xpmax_b = np.where(colap_b_map == np.nanmax(colap_b_map))[1]

    ax.plot(xpmax_b, ypmax_b, 'x', color='red', markersize=4, label=str(xpmax_b) + ', ' + str(ypmax_b))
    ax.set_title('Collapsed Blue Arm Datacube')
    ax.set_xlabel('Right Ascension (J2000)')
    ax.set_ylabel('Declination (J2000)')
    ax.legend()

    cbax = plt.subplot(gs[0, 1])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('log scale')

    ax = plt.subplot(gs[0, 3])
    im = ax.imshow(np.log10(colap_r_map), origin='lower')

    ypmax_r = np.where(colap_r_map == np.nanmax(colap_r_map))[0]
    xpmax_r = np.where(colap_r_map == np.nanmax(colap_r_map))[1]

    ax.plot(xpmax_r, ypmax_r, 'x', color='red', markersize=4, label=str(xpmax_r) + ', ' + str(ypmax_r))
    ax.set_title('Collapsed Red Arm Datacube')
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    ax.legend()

    cbax = plt.subplot(gs[0, 4])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('log scale')

    # ------

    ax = plt.subplot(gs[1, 0])
    ax.plot(lam_b, blue_cube[1].data[:, ypmax_b[0], xpmax_b[0]])
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel('Counts')
    ax.set_title('Blue spectrum at (' + str(xpmax_b[0]) + ', ' + str(ypmax_b[0]) + ')')

    ax = plt.subplot(gs[1, 3])
    ax.plot(lam_r, red_cube[1].data[:, ypmax_r[0], xpmax_r[0]])
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel('Counts')
    ax.set_title('Red spectrum at (' + str(xpmax_r[0]) + ', ' + str(ypmax_r[0]) + ')')

    # ------

    # plot sensitivity function

    ax = plt.subplot(gs[2, 0])
    ax.plot(lam_b, blue_cube[5].data)
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel('Flux')
    ax.set_yscale('log')
    ax.set_title('Blue sensitivity function - mean')

    ax = plt.subplot(gs[2, 3])
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

    cbax = plt.subplot(gs[3, 1])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('counts log scale')

    ax = plt.subplot(gs[3, 3])
    im = ax.imshow(np.log10(median_r_map), origin='lower')
    ax.contour(mask_faint_r, levels=[0.5], colors=['r'])
    ax.contour(mask_medium_r, levels=[0.5], colors=['b'])
    ax.contour(mask_bright_r, levels=[0.5], colors=['k'])
    ax.set_title(r'Median Map (red cube)', fontsize=10)

    cbax = plt.subplot(gs[3, 4])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('counts log scale')

    ax = plt.subplot(gs[4, 0])
    im = ax.imshow(np.log10(median_b_sky_map), origin='lower')
    ax.set_title(r'Median Sky Map (blue cube)', fontsize=10)

    cbax = plt.subplot(gs[4, 1])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('counts log scale')

    ax = plt.subplot(gs[4, 3])
    im = ax.imshow(np.log10(median_r_sky_map), origin='lower')
    ax.set_title(r'Median Sky Map (red cube)', fontsize=10)

    cbax = plt.subplot(gs[4, 4])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('counts log scale')

    ax = plt.subplot(gs[5, 0])
    im = ax.imshow(np.log10(mean_b_map), origin='lower')
    ax.set_title(r'Mean Map (blue cube)', fontsize=10)

    cbax = plt.subplot(gs[5, 1])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('counts log scale')

    ax = plt.subplot(gs[5, 3])
    im = ax.imshow(np.log10(mean_r_map), origin='lower')
    ax.set_title(r'Mean Map (red cube)', fontsize=10)

    cbax = plt.subplot(gs[5, 4])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('counts log scale')

    ax = plt.subplot(gs[6, 0])
    im = ax.imshow(np.log10(mean_b_sky_map), origin='lower')
    ax.set_title(r'Mean Sky Map (blue cube)', fontsize=10)

    cbax = plt.subplot(gs[6, 1])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('counts log scale')

    ax = plt.subplot(gs[6, 3])
    im = ax.imshow(np.log10(mean_r_sky_map), origin='lower')
    ax.set_title(r'Mean Sky Map (red cube)', fontsize=10)

    cbax = plt.subplot(gs[6, 4])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('counts log scale')

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

    ims_xlims = ax.get_xlim()
    ims_ylims = ax.get_ylim()

    cbax = plt.subplot(gs[9, 1])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('SNR')

    ax = plt.subplot(gs[9, 3])
    im = ax.imshow(snr_r, origin='lower')
    cs = ax.contour(snr_r, levels, linestyles=np.array([':', '-']), colors='white')
    m1 = mlines.Line2D([], [], color='black', linestyle=':', markersize=5, label='SNR = ' + str(levels[0]))
    m2 = mlines.Line2D([], [], color='black', linestyle='-', markersize=5, label='SNR = ' + str(levels[1]))
    ax.legend(handles=[m1, m2], framealpha=1, fontsize=8, loc='lower left')

    ax.set_title(r'SNR @' + str(red_cen_wave) + '$\AA$')
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')

    cbax = plt.subplot(gs[9, 4])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('SNR')

    # ------

    ax = plt.subplot(gs[10, 0])
    ax.plot(med_b, snr_b, 'o', color='blue', alpha=0.3, markeredgecolor='white')
    ax.set_ylabel(r'SNR [@' + str(blue_cen_wave) + '$\AA$]')
    ax.set_xlabel(r'Median Flux [@' + str(blue_cen_wave - 50) + '-' + str(blue_cen_wave + 50) + '$\AA$]')
    ax.grid(True, alpha=0.3, zorder=-1)

    ax = plt.subplot(gs[10, 3])
    ax.plot(med_r, snr_r, 'o', color='red', alpha=0.3, markeredgecolor='black')
    ax.set_ylabel(r'SNR [@' + str(red_cen_wave) + '$\AA$]')
    ax.set_xlabel(r'Median Flux [@' + str(red_cen_wave - 50) + '-' + str(red_cen_wave + 50) + '$\AA$]')
    ax.grid(True, alpha=0.3, zorder=-1)

    # ------

    ax = plt.subplot(gs[11, 0])
    ax.hist(snr_b[snr_b >= 3], 30, histtype='step', lw=2)
    ax.set_yscale('log')
    ax.set_ylabel(r'N pixels [SNR $\geq$ 3]')
    ax.set_xlabel(r'SNR [@' + str(blue_cen_wave) + '$\AA$]')

    int_spec_b = np.sum(blue_cube[1].data * ((snr_b >= 3)[np.newaxis, :, :]), axis=(1, 2))
    in_ax = ax.inset_axes([0.55, 0.5, 0.4, 0.3])
    in_ax.set_title(r'integrated spec [SNR$\geq$3]', fontsize=10)
    in_ax.plot(lam_b, int_spec_b)
    in_ax.axvline(blue_cen_wave - 50, linestyle='--', color='black')
    in_ax.axvline(blue_cen_wave + 50, linestyle='--', color='black')

    ax = plt.subplot(gs[11, 3])
    ax.hist(snr_r[snr_r >= 3], 30, histtype='step', lw=2)
    ax.set_yscale('log')
    ax.set_ylabel(r'N pixels [SNR $\geq$ 3]')
    ax.set_xlabel(r'SNR [@' + str(red_cen_wave) + '$\AA$]')

    int_spec_r = np.sum(red_cube[1].data * ((snr_r >= 3)[np.newaxis, :, :]), axis=(1, 2))
    in_ax = ax.inset_axes([0.55, 0.5, 0.4, 0.3])
    in_ax.set_title(r'integrated spec [SNR$\geq$3]', fontsize=10)
    in_ax.plot(lam_r, int_spec_r)
    in_ax.axvline(red_cen_wave - 50, linestyle='--', color='black')
    in_ax.axvline(red_cen_wave + 50, linestyle='--', color='black')

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

    #binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_b, y_t_b, sgn_tt_b, rms_tt_b,
                                                                              # targetSN,
                                                                              # pixelsize=pixelsize, plot=0, quiet=1)

    def sn_func_blue(index, signal, noise):

        factor = 1 + 1.53 * np.log10(index.size) ** 1.19

        sn = np.sum(signal[index]) / np.sqrt(np.sum((noise[index] * factor) ** 2))

        return  sn
    
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_b, y_t_b, sgn_tt_b, rms_tt_b,
                                                                              targetSN, pixelsize=pixelsize, plot=0,
                                                                              quiet=0, sn_func=sn_func_blue)

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
    ax.set_title(r'Voronoi binning / Target SNR = ' + str(targetSN))

    ax = plt.subplot(gs[13, 0])

    rad = np.sqrt((xBar - xpmax_b[0]) ** 2 + (yBar - ypmax_b[0]) ** 2)  # Use centroids, NOT generators
    plt.plot(np.sqrt((x_t_b - xpmax_b[0]) ** 2 + (y_t_b - ypmax_b[0]) ** 2), sgn_tt_b / rms_tt_b, ',k')
    plt.plot(rad[nPixels < 2], sn[nPixels < 2], 'xb', label='Not binned')
    plt.plot(rad[nPixels > 1], sn[nPixels > 1], 'or', label='Voronoi bins')
    plt.xlabel('R [pixels]')
    plt.ylabel('Bin S/N')
    plt.axis([np.min(rad), np.max(rad), 0, np.max(sn) * 1.05])  # x0, x1, y0, y1
    plt.axhline(targetSN)
    plt.legend()

    fits.writeto(gal_dir + 'vorbin_map_blue.fits', np.flip(np.rot90(img), axis=0), overwrite=True)

    # saving voronoi datacube
    vorbin_map = img
    nb_cube_data = np.zeros(blue_cube[1].data.shape)
    nb_cube_err = np.zeros(blue_cube[1].data.shape)

    with mp.Pool(int(config.get('APS_cube', 'n_proc'))) as pool:
        nb_cube = pool.starmap(vorbin_loop, zip((i, vorbin_map, blue_cube[0].header['CAMERA'])
                                                for i in np.arange(np.nanmax(vorbin_map) + 1)))

    for i in np.arange(int(np.nanmax(vorbin_map)) + 1):
        for j in np.arange(len(nb_cube[i][2][0])):
            nb_cube_data[:, nb_cube[i][2][1][j], nb_cube[i][2][0][j]] = nb_cube[i][0] * np.mean(blue_cube[5].data[:],axis=0)
            nb_cube_err[:, nb_cube[i][2][1][j], nb_cube[i][2][0][j]] = nb_cube[i][1] * np.mean(blue_cube[5].data[:],axis=0)

    cube_head = fits.Header()
    cube_head['SIMPLE'] = True
    cube_head['BITPIX'] = -32
    cube_head['NAXIS'] = 3
    cube_head['NAXIS1'] = blue_cube[1].data.shape[2]
    cube_head['NAXIS2'] = blue_cube[1].data.shape[1]
    cube_head['NAXIS3'] = blue_cube[1].data.shape[0]
    cube_head['CTYPE3'] = 'WAVELENGTH'
    cube_head['CUNIT3'] = 'Angstrom'
    cube_head['CDELT3'] = blue_cube[1].header['CD3_3']
    cube_head['DISPAXIS'] = 1
    cube_head['CRVAL3'] = blue_cube[1].header['CRVAL3']
    cube_head['CRPIX3'] = blue_cube[1].header['CRPIX3']
    cube_head['CRPIX1'] = blue_cube[1].header['CRPIX1']
    cube_head['CRPIX2'] = blue_cube[1].header['CRPIX2']
    cube_head['CRVAL1'] = blue_cube[1].header['CRVAL1']
    cube_head['CRVAL2'] = blue_cube[1].header['CRVAL2']
    cube_head['CDELT1'] = axis_header['CD1_1']
    cube_head['CDELT2'] = axis_header['CD2_2']
    cube_head['CTYPE1'] = 'RA---TAN'
    cube_head['CTYPE2'] = 'DEC--TAN'
    cube_head['CUNIT1'] = 'deg'
    cube_head['CUNIT2'] = 'deg'
    cube_head['CUNIT2'] = 'deg'
    cube_head['CNAME'] = gal_name
    cube_head['W_Z'] = config.get('pyp_params', 'redshift')
    cube_head['N_FLUX'] = ('1e-19', 'normalized spectra flux')

    n_cube = fits.HDUList([fits.PrimaryHDU(),
                           fits.ImageHDU(data=nb_cube_data / 1e-19, header=cube_head, name='DATA'),
                           fits.ImageHDU(data=nb_cube_err / 1e-19, header=cube_head, name='ERROR')])

    n_cube.writeto(gal_dir + 'blue_cube_vorbin.fits', overwrite=True)

    #

    print('Doing L1 red datacube voronoi')
    print('')

    x_t_r = x_t[sgn_t_r / rms_t_r > 3]
    y_t_r = y_t[sgn_t_r / rms_t_r > 3]
    sgn_tt_r = sgn_t_r[sgn_t_r / rms_t_r > 3]
    rms_tt_r = rms_t_r[sgn_t_r / rms_t_r > 3]

    # binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_r, y_t_r, sgn_tt_r, rms_tt_r,
    #                                                                           targetSN,
    #                                                                           pixelsize=pixelsize, plot=0, quiet=1)

    def sn_func_red(index, signal, noise):

        # factor = 1 + 1.77 * np.log10(index.size) ** 1.54  # this would be the correct function, but it does not work
        factor = 1 + 1.53 * np.log10(index.size) ** 1.19

        sn = np.sum(signal[index]) / np.sqrt(np.sum((noise[index] * factor) ** 2))

        return sn

    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_r, y_t_r, sgn_tt_r, rms_tt_r,
                                                                              targetSN, pixelsize=pixelsize, plot=0,
                                                                              quiet=1, sn_func=sn_func_red)

    ax = plt.subplot(gs[12, 3])

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
    ax.set_title(r'Voronoi binning / Target SNR = ' + str(targetSN))

    ax = plt.subplot(gs[13, 3])

    rad = np.sqrt((xBar - xpmax_r[0]) ** 2 + (yBar - ypmax_r[0]) ** 2)  # Use centroids, NOT generators
    plt.plot(np.sqrt((x_t_b - xpmax_r[0]) ** 2 + (y_t_b - ypmax_r[0]) ** 2), sgn_tt_b / rms_tt_b, ',k')
    plt.plot(rad[nPixels < 2], sn[nPixels < 2], 'xb', label='Not binned')
    plt.plot(rad[nPixels > 1], sn[nPixels > 1], 'or', label='Voronoi bins')
    plt.xlabel('R [pixels]')
    plt.ylabel('Bin S/N')
    plt.axis([np.min(rad), np.max(rad), 0, np.max(sn) * 1.05])  # x0, x1, y0, y1
    plt.axhline(targetSN)
    plt.legend()

    fits.writeto(gal_dir + 'vorbin_map_red.fits', np.flip(np.rot90(img), axis=0), overwrite=True)

    # saving voronoi datacube
    vorbin_map = img
    nb_cube_data = np.zeros(red_cube[1].data.shape)
    nb_cube_err = np.zeros(red_cube[1].data.shape)

    with mp.Pool(int(config.get('APS_cube', 'n_proc'))) as pool:
        nb_cube = pool.starmap(vorbin_loop, zip((i, vorbin_map, red_cube[0].header['CAMERA'])
                                                for i in np.arange(np.nanmax(vorbin_map) + 1)))

    for i in np.arange(int(np.nanmax(vorbin_map)) + 1):
        for j in np.arange(len(nb_cube[i][2][0])):
            nb_cube_data[:, nb_cube[i][2][1][j], nb_cube[i][2][0][j]] = nb_cube[i][0] * np.mean(red_cube[5].data[:],axis=0)
            nb_cube_err[:, nb_cube[i][2][1][j], nb_cube[i][2][0][j]] = nb_cube[i][1] * np.mean(red_cube[5].data[:],axis=0)

    cube_head = fits.Header()
    cube_head['SIMPLE'] = True
    cube_head['BITPIX'] = -32
    cube_head['NAXIS'] = 3
    cube_head['NAXIS1'] = red_cube[1].data.shape[2]
    cube_head['NAXIS2'] = red_cube[1].data.shape[1]
    cube_head['NAXIS3'] = red_cube[1].data.shape[0]
    cube_head['CTYPE3'] = 'WAVELENGTH'
    cube_head['CUNIT3'] = 'Angstrom'
    cube_head['CDELT3'] = red_cube[1].header['CD3_3']
    cube_head['DISPAXIS'] = 1
    cube_head['CRVAL3'] = red_cube[1].header['CRVAL3']
    cube_head['CRPIX3'] = red_cube[1].header['CRPIX3']
    cube_head['CRPIX1'] = red_cube[1].header['CRPIX1']
    cube_head['CRPIX2'] = red_cube[1].header['CRPIX2']
    cube_head['CRVAL1'] = red_cube[1].header['CRVAL1']
    cube_head['CRVAL2'] = red_cube[1].header['CRVAL2']
    cube_head['CDELT1'] = axis_header['CD1_1']
    cube_head['CDELT2'] = axis_header['CD2_2']
    cube_head['CTYPE1'] = 'RA---TAN'
    cube_head['CTYPE2'] = 'DEC--TAN'
    cube_head['CUNIT1'] = 'deg'
    cube_head['CUNIT2'] = 'deg'
    cube_head['CNAME'] = gal_name
    cube_head['W_Z'] = config.get('pyp_params', 'redshift')
    cube_head['N_FLUX'] = ('1e-19', 'normalized spectra flux')

    n_cube = fits.HDUList([fits.PrimaryHDU(),
                           fits.ImageHDU(data=nb_cube_data / 1e-19, header=cube_head, name='DATA'),
                           fits.ImageHDU(data=nb_cube_err / 1e-19, header=cube_head, name='ERROR')])

    n_cube.writeto(gal_dir + 'red_cube_vorbin.fits', overwrite=True)
    #

    # ------

    xc = []
    yc = []
    lamc_b = []

    for i in np.arange(len(blue_cube[1].data[:, 0, 0])):
        xcx = np.where(blue_cube[1].data[i, :, :] == np.nanmax(blue_cube[1].data[i, :, :]))[1]
        ycy = np.where(blue_cube[1].data[i, :, :] == np.nanmax(blue_cube[1].data[i, :, :]))[0]
        if len(xcx) < len(blue_cube[1].data[0, 0, :]):
            for j in np.arange(len(xcx)):
                xc.append(xcx[j])
                yc.append(ycy[j])
                lamc_b.append(lam_b[i])

    xc = np.array(xc)
    yc = np.array(yc)
    lamc_b = np.array(lamc_b)

    ax = plt.subplot(gs[14, 0])
    ax.plot(lamc_b, xc, '+', color='blue', ms=2, label='X center')
    ax.plot(lamc_b, yc, '+', color='red', ms=2, label='Y center')
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r'X and Y center')
    ax.set_title('Peak flux spaxel (Blue)')
    ax.legend(markerscale=5)

    xc = []
    yc = []
    lamc_r = []

    for i in np.arange(len(red_cube[1].data[:, 0, 0])):
        xcx = np.where(red_cube[1].data[i, :, :] == np.nanmax(red_cube[1].data[i, :, :]))[1]
        ycy = np.where(red_cube[1].data[i, :, :] == np.nanmax(red_cube[1].data[i, :, :]))[0]
        if len(xcx) < len(red_cube[1].data[0, 0, :]):
            for j in np.arange(len(xcx)):
                xc.append(xcx[j])
                yc.append(ycy[j])
                lamc_r.append(lam_r[i])

    xc = np.array(xc)
    yc = np.array(yc)
    lamc_r = np.array(lamc_r)

    ax = plt.subplot(gs[14, 3])
    ax.plot(lamc_r, xc, '+', color='blue', ms=2)
    ax.plot(lamc_r, yc, '+', color='red', ms=2)
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r'X and Y center')
    ax.set_title('Peak flux spaxel (Red)')

    fig_l1 = date + '_' + gal_name + '_' + blue_cube[0].header['MODE'] + '_' + str(blue_cube[0].header['OBID'])\
             + '_L1.png'

    fig.savefig(fig_l1)

    # ==================================================================

    # creating plots for APS

    if int(config.get('QC_plots', 'aps_flag')) == 1:

        print('Doing L2 APS plots')
        print('')

        aps_cube = fits.open(gal_dir + '/' + gal_name + '_cube.fits')

        aps_cube_data = aps_cube[0].data
        aps_cube_err = aps_cube[1].data

        targetSN = np.float(config.get('QC_plots', 'target_SN'))
        levels = np.array(json.loads(config.get('QC_plots', 'levels'))).astype(np.float)  # SNR levels to display

        aps_cen_wave = int(config.get('QC_plots', 'aps_wav'))

        colap_a_map = np.nansum(aps_cube[0].data[:], axis=0)

        lam_a = aps_cube[0].header['CRVAL3'] + (np.arange(aps_cube[0].header['NAXIS3']) * aps_cube[0].header['CDELT3'])

        sgn_a = np.mean(aps_cube[0].data[np.where(lam_a == min(lam_a, key=lambda x: abs(x - aps_cen_wave)))[0][0] - 100:
                                         np.where(lam_a == min(lam_a, key=lambda x: abs(x - aps_cen_wave)))[0][0] + 100]
                        , axis=0)
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
        cube_head['CNAME'] = gal_name
        cube_head['W_Z'] = config.get('pyp_params', 'redshift')

        fig = plt.figure(figsize=(16, 20))

        fig.suptitle('L2/APS QC plots', size=22, weight='bold')

        gs = gridspec.GridSpec(4, 5, height_ratios=[1, 1, 1, 1], width_ratios=[1, 0.06, 0.4, 1, 1])
        gs.update(left=0.07, right=0.9, bottom=0.05, top=0.92, wspace=0.0, hspace=0.25)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        wcs = WCS(axis_header)

        ax = plt.subplot(gs[0, 0], projection=wcs)
        im = ax.imshow(np.log10(colap_a_map), origin='lower')

        ypmax_a = np.where(colap_a_map == np.nanmax(colap_a_map))[0]
        xpmax_a = np.where(colap_a_map == np.nanmax(colap_a_map))[1]

        ax.plot(xpmax_a, ypmax_a, 'x', color='red', markersize=4, label=str(xpmax_a) + ', ' + str(ypmax_a))
        ax.set_title('Collapsed APS Datacube')
        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')
        ax.legend()

        cbax = plt.subplot(gs[0, 1])
        cbar = Colorbar(ax=cbax, mappable=im)
        cbar.set_label('log scale')

        # ------

        ax = plt.subplot(gs[0, 3:5])
        ax.plot(lam_a, aps_cube[0].data[:, ypmax_a[0], xpmax_a[0]])
        ax.set_xlabel(r'$\lambda$ [$\AA$]')
        ax.set_ylabel('Counts')
        ax.set_title('APS spectrum at (' + str(xpmax_a[0]) + ', ' + str(ypmax_a[0]) + ')')

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

        cbax = plt.subplot(gs[1, 1])
        cbar = Colorbar(ax=cbax, mappable=im)
        cbar.set_label('SNR')

        # ------

        ax = plt.subplot(gs[1, 3:5])
        ax.hist(snr_a[snr_a >= 3], 30, histtype='step', lw=2)
        ax.set_yscale('log')
        ax.set_ylabel(r'N pixels [SNR $\geq$ 3]')
        ax.set_xlabel(r'SNR [@' + str(aps_cen_wave) + '$\AA$]')

        int_spec_a = np.sum(aps_cube[0].data * ((snr_a >= 3)[np.newaxis, :, :]), axis=(1, 2))
        in_ax = ax.inset_axes([0.55, 0.5, 0.4, 0.3])
        in_ax.set_title(r'integrated spec [SNR$\geq$3]', fontsize=10)
        in_ax.plot(lam_a, int_spec_a)
        in_ax.axvline(aps_cen_wave - 100, linestyle='--', color='black')
        in_ax.axvline(aps_cen_wave + 100, linestyle='--', color='black')

        # doing voronoi binning

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

        binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_a, y_t_a, sgn_tt_a, rms_tt_a,
                                                                                  targetSN,
                                                                                  pixelsize=pixelsize, plot=0, quiet=1)

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
        ax.set_title(r'Voronoi binning / Target SNR = ' + str(targetSN))

        fits.writeto(gal_dir + 'vorbin_map_aps.fits', np.flip(np.rot90(img), axis=0), overwrite=True)

        ax = plt.subplot(gs[2, 3:5])

        rad = np.sqrt((xBar - xpmax_a[0]) ** 2 + (yBar - ypmax_a[0]) ** 2)  # Use centroids, NOT generators
        plt.plot(np.sqrt((x_t_a - xpmax_a[0]) ** 2 + (y_t_a - ypmax_a[0]) ** 2), sgn_tt_a / rms_tt_a, ',k')
        plt.plot(rad[nPixels < 2], sn[nPixels < 2], 'xb', label='Not binned')
        plt.plot(rad[nPixels > 1], sn[nPixels > 1], 'or', label='Voronoi bins')
        plt.xlabel('R [pixels]')
        plt.ylabel('Bin S/N')
        plt.axis([np.min(rad), np.max(rad), 0, np.max(sn) * 1.05])  # x0, x1, y0, y1
        plt.axhline(targetSN)
        plt.legend()

        # saving voronoi datacube
        vorbin_map = img
        na_cube_data = np.zeros(aps_cube[0].data.shape)
        na_cube_err = np.zeros(aps_cube[0].data.shape)

        with mp.Pool(int(config.get('APS_cube', 'n_proc'))) as pool:
            nb_cube = pool.starmap(vorbin_loop, zip((i, vorbin_map, 'APS')
                                                    for i in np.arange(np.nanmax(vorbin_map) + 1)))

        for i in np.arange(int(np.nanmax(vorbin_map)) + 1):
            for j in np.arange(len(nb_cube[i][2][0])):
                na_cube_data[:, nb_cube[i][2][1][j], nb_cube[i][2][0][j]] = nb_cube[i][0]
                na_cube_err[:, nb_cube[i][2][1][j], nb_cube[i][2][0][j]] = nb_cube[i][1]

        cube_head = aps_cube[0].header.copy()
        cube_head['W_Z'] = config.get('pyp_params', 'redshift')

        n_cube = fits.HDUList([fits.PrimaryHDU(),
                               fits.ImageHDU(data=na_cube_data, header=cube_head, name='DATA'),
                               fits.ImageHDU(data=na_cube_err, header=cube_head, name='ERROR')])

        n_cube.writeto(gal_dir + 'aps_cube_vorbin.fits', overwrite=True)

        # ------

        xc = []
        yc = []
        lamc_a = []

        for i in np.arange(len(aps_cube[0].data[:, 0, 0])):
            xcx = np.where(aps_cube[0].data[i, :, :] == np.nanmax(aps_cube[0].data[i, :, :]))[1]
            ycy = np.where(aps_cube[0].data[i, :, :] == np.nanmax(aps_cube[0].data[i, :, :]))[0]
            if len(xcx) < len(aps_cube[0].data[0, 0, :]):
                for j in np.arange(len(xcx)):
                    xc.append(xcx[j])
                    yc.append(ycy[j])
                    lamc_a.append(lam_a[i])

        xc = np.array(xc)
        yc = np.array(yc)
        lamc_a = np.array(lamc_a)

        ax = plt.subplot(gs[3, 3:5])
        ax.plot(lamc_a, xc, '+', color='blue', ms=2, label='X center')
        ax.plot(lamc_a, yc, '+', color='red', ms=2, label='Y center')
        ax.set_xlabel(r'$\lambda$ [$\AA$]')
        ax.set_ylabel(r'X and Y center')
        ax.set_title('Peak flux spaxel (L2)')
        ax.legend(markerscale=5)

        # ------

        fig_l2 = date + '_' + gal_name + '_' + blue_cube[0].header['MODE'] + '_' + str(blue_cube[0].header['OBID']) + \
                 '_L2.png'

        fig.savefig(fig_l2)

        text = '''
        <html>
            <body style="background-color:white;">
                <div style="text-align: center;">
                    <h1>Night report ''' + date + '''</h1>
                    <h1>CNAME ''' + blue_cube[0].header['CCNAME1'] + '''</h1>
                    <h1>IFUNAME ''' + blue_cube[0].header['IFUNAME'] + '''</h1>
                    <h1>OBID ''' + str(blue_cube[0].header['OBID']) + '''</h1>
                    <h1>LIFU MODE ''' + blue_cube[0].header['MODE'] + '''</h1>
                    <img src="''' + fig_l0 + '''" class="center">
                    <img src="''' + fig_l1 + '''" class="center">
                    <img src="''' + fig_l2 + '''" class="center">
                </div>
            </body>
        </html>
        '''

    else:

        text = '''
        <html>
            <body style="background-color:white;">
                <div style="text-align: center;">
                    <h1>Night report ''' + date + '''</h1>
                    <h1>CNAME ''' + blue_cube[0].header['CCNAME1'] + '''</h1>
                    <h1>IFUNAME ''' + blue_cube[0].header['IFUNAME'] + '''</h1>
                    <h1>OBID ''' + str(blue_cube[0].header['OBID']) + '''</h1>
                    <h1>LIFU MODE ''' + blue_cube[0].header['MODE'] + '''</h1>
                    <img src="''' + fig_l0 + '''" class="center">
                    <img src="''' + fig_l1 + '''" class="center">
                </div>
            </body>
        </html>
        '''

    f = open(date + '_' + gal_name + '_' + blue_cube[0].header['MODE'] + '_' + str(blue_cube[0].header['OBID'])
             + ".html", "w")

    f.write(text)

    f.close()

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(hostname='minos.aip.de', username='gcouto', password=open('minos_pass', 'r').read().splitlines()[0])

    scp = SCPClient(ssh.get_transport())

    if int(config.get('QC_plots', 'aps_flag')) == 1:
        scp.put([date + '_' + gal_name + '_' + blue_cube[0].header['MODE'] + '_' + str(blue_cube[0].header['OBID']) +
                 '.html', fig_l0, fig_l1, fig_l2], '/store/weave/apertif/')
    else:
        scp.put([date + '_' + gal_name + '_' + blue_cube[0].header['MODE'] + '_' + str(blue_cube[0].header['OBID']) +
                 '.html', fig_l0, fig_l1], '/store/weave/apertif/')
