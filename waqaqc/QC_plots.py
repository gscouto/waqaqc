import os
import configparser
import json

import matplotlib.gridspec as gridspec
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
import astropy.units as u
import multiprocessing as mp


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

    blue_cube = fits.open(file_dir + config.get('QC_plots', 'blue_cube'))
    red_cube = fits.open(file_dir + config.get('QC_plots', 'red_cube'))

    blue_cube_data = blue_cube[1].data
    blue_cube_err = blue_cube[2].data

    red_cube_data = red_cube[1].data
    red_cube_err = red_cube[2].data

    gal_name = blue_cube[0].header['CCNAME1']
    date = blue_cube[0].header['DATE-OBS']

    gal_dir = gal_name + '/'
    os.makedirs(gal_dir, exist_ok=True)

    targetSN = np.float(config.get('QC_plots', 'target_SN'))
    levels = np.array(json.loads(config.get('QC_plots', 'levels'))).astype(np.float)  # SNR levels to display

    colap_b_map = np.sum(blue_cube[1].data[:], axis=0)
    colap_r_map = np.sum(red_cube[1].data[:], axis=0)

    lam_r = red_cube[1].header['CRVAL3'] + (np.arange(red_cube[1].header['NAXIS3']) * red_cube[1].header['CD3_3'])
    lam_b = blue_cube[1].header['CRVAL3'] + (np.arange(blue_cube[1].header['NAXIS3']) * blue_cube[1].header['CD3_3'])

    med_b = np.median(blue_cube[1].data[np.where(lam_b == 5100.)[0][0] - 500:np.where(lam_b == 5100.)[0][0] + 500],
                      axis=0)
    sgn_b = np.mean(blue_cube[1].data[np.where(lam_b == 5100.)[0][0] - 500:np.where(lam_b == 5100.)[0][0] + 500],
                    axis=0)
    rms_b = np.std(blue_cube[1].data[np.where(lam_b == 5100.)[0][0] - 500:np.where(lam_b == 5100.)[0][0] + 500], axis=0)
    snr_b = sgn_b / rms_b

    med_r = np.median(red_cube[1].data[np.where(lam_r == 6200.)[0][0] - 500:np.where(lam_r == 6200.)[0][0] + 500],
                      axis=0)
    sgn_r = np.mean(red_cube[1].data[np.where(lam_r == 6200.)[0][0] - 500:np.where(lam_r == 6200.)[0][0] + 500], axis=0)
    rms_r = np.std(red_cube[1].data[np.where(lam_r == 6200.)[0][0] - 500:np.where(lam_r == 6200.)[0][0] + 500], axis=0)
    snr_r = sgn_r / rms_r

    sc = SkyCoord(ra=blue_cube[0].header['RA'], dec=blue_cube[0].header['DEC'], unit=(u.hourangle, u.deg), frame=FK5,
                  equinox='J' + str(blue_cube[0].header['CATEPOCH']))
    nsc = sc.transform_to(FK5(equinox='J2000.0'))

    url = geturl(nsc.ra.value, nsc.dec.value, size=480, filters="grizy", output_size=None, format="jpg", color=True)
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))

    title = blue_cube[0].header['CCNAME1'] + ' - ' + blue_cube[0].header['PLATE'] + ' - ' + blue_cube[0].header['MODE']

    # doing the plots

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

    fig = plt.figure(figsize=(14, 42))

    fig.suptitle(gal_name + ' L1 QC plots', size=22, weight='bold')

    gs = gridspec.GridSpec(9, 5, height_ratios=[1, 1, 0.6, 1, 0.6, 0.6, 1, 0.6, 0.6],
                           width_ratios=[1, 0.06, 0.3, 1, 0.06])
    gs.update(left=0.07, right=0.9, bottom=0.05, top=0.95, wspace=0.0, hspace=0.25)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    cdelt1 = 0.25
    cdelt2 = 0.25
    crpix1 = im.size[1] / 2.
    crpix2 = im.size[1] / 2.

    x_ax = (np.arange(im.size[1]) + 1 - crpix1) * cdelt1
    y_ax = (np.arange(im.size[0]) + 1 - crpix2) * cdelt2

    ax = plt.subplot(gs[0, 0])
    ax.imshow(im, extent=[min(x_ax), max(x_ax), max(y_ax), min(y_ax)])
    ax.plot(0, 0, marker='H', color='red', markerfacecolor='none', markersize=200)
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

    wcs = WCS(axis_header)
    ax = plt.subplot(gs[1, 0], projection=wcs)

    im = ax.imshow(np.arcsinh(colap_b_map), origin='lower')

    ypmax_b = np.where(colap_b_map == np.nanmax(colap_b_map))[0]
    xpmax_b = np.where(colap_b_map == np.nanmax(colap_b_map))[1]

    ax.plot(xpmax_b, ypmax_b, 'x', color='red', markersize=4, label=str(xpmax_b) + ', ' + str(ypmax_b))
    ax.set_title('Collapsed Blue Arm Datacube')
    ax.set_xlabel('Right Ascension (J2000)')
    ax.set_ylabel('Declination (J2000)')
    ax.legend()

    cbax = plt.subplot(gs[1, 1])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('arcsinh scale')

    ax = plt.subplot(gs[1, 3])
    im = ax.imshow(np.arcsinh(colap_r_map), origin='lower')

    ypmax_r = np.where(colap_r_map == np.nanmax(colap_r_map))[0]
    xpmax_r = np.where(colap_r_map == np.nanmax(colap_r_map))[1]

    ax.plot(xpmax_r, ypmax_r, 'x', color='red', markersize=4, label=str(xpmax_r) + ', ' + str(ypmax_r))
    ax.set_title('Collapsed Red Arm Datacube')
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    ax.legend()

    cbax = plt.subplot(gs[1, 4])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('arcsinh scale')

    # ------

    ax = plt.subplot(gs[2, 0])
    ax.plot(lam_b, blue_cube[1].data[:, ypmax_b[0], xpmax_b[0]])
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel('Flux?')
    ax.set_title('Blue spectrum at (' + str(xpmax_b[0]) + ', ' + str(ypmax_b[0]) + ')')

    ax = plt.subplot(gs[2, 3])
    ax.plot(lam_r, red_cube[1].data[:, ypmax_r[0], xpmax_r[0]])
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel('Flux?')
    ax.set_title('Red spectrum at (' + str(xpmax_r[0]) + ', ' + str(ypmax_r[0]) + ')')

    # ------

    ax = plt.subplot(gs[3, 0])
    im = ax.imshow(snr_b, origin='lower')
    cs = ax.contour(snr_b, levels, linestyles=np.array([':', '-']), colors='white')
    cs.collections[0].set_label('SNR = 3')
    cs.collections[1].set_label('SNR = 30')
    leg = ax.legend(framealpha=1, fontsize=8, loc='lower left')
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')

    ax.set_title(r'SNR @5100$\AA$')
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')

    ims_xlims = ax.get_xlim()
    ims_ylims = ax.get_ylim()

    cbax = plt.subplot(gs[3, 1])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('SNR')

    ax = plt.subplot(gs[3, 3])
    im = ax.imshow(snr_r, origin='lower')
    cs = ax.contour(snr_r, levels, linestyles=np.array([':', '-']), colors='white')
    cs.collections[0].set_label('SNR = 3')
    cs.collections[1].set_label('SNR = 30')
    leg = ax.legend(framealpha=1, fontsize=8, loc='lower left')
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')

    ax.set_title(r'SNR @6200$\AA$')
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')

    cbax = plt.subplot(gs[3, 4])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('SNR')

    # ------

    ax = plt.subplot(gs[4, 0])
    ax.plot(med_b, snr_b, 'o', color='blue', alpha=0.3, markeredgecolor='black')
    ax.set_ylabel(r'SNR [@5100$\AA$]')
    ax.set_xlabel(r'Median Flux [@5050-5150$\AA$]')
    ax.grid(True, alpha=0.3, zorder=-1)

    ax = plt.subplot(gs[4, 3])
    ax.plot(med_r, snr_r, 'o', color='red', alpha=0.3, markeredgecolor='black')
    ax.set_ylabel(r'SNR [@6200$\AA$]')
    ax.set_xlabel(r'Median Flux [@6150-6250$\AA$]')
    ax.grid(True, alpha=0.3, zorder=-1)

    # ------

    ax = plt.subplot(gs[5, 0])
    ax.hist(snr_b[snr_b >= 3], 30, histtype='step', lw=2)
    ax.set_yscale('log')
    ax.set_ylabel(r'N pixels [SNR $\geq$ 3]')
    ax.set_xlabel(r'SNR [@5100$\AA$]')

    int_spec_b = np.sum(blue_cube[1].data * ((snr_b >= 3)[np.newaxis, :, :]), axis=(1, 2))
    in_ax = ax.inset_axes([0.55, 0.6, 0.4, 0.3])
    in_ax.set_title(r'          integrated spec [SNR$\geq$3]', fontsize=10)
    in_ax.plot(lam_b, int_spec_b)
    in_ax.axvline(5050, linestyle='--', color='black')
    in_ax.axvline(5150, linestyle='--', color='black')

    ax = plt.subplot(gs[5, 3])
    ax.hist(snr_r[snr_r >= 3], 30, histtype='step', lw=2)
    ax.set_yscale('log')
    ax.set_ylabel(r'N pixels [SNR $\geq$ 3]')
    ax.set_xlabel(r'SNR [@6200$\AA$]')

    int_spec_r = np.sum(red_cube[1].data * ((snr_r >= 3)[np.newaxis, :, :]), axis=(1, 2))
    in_ax = ax.inset_axes([0.55, 0.6, 0.4, 0.3])
    in_ax.set_title(r'          integrated spec [SNR$\geq$3]', fontsize=10)
    in_ax.plot(lam_r, int_spec_r)
    in_ax.axvline(6150, linestyle='--', color='black')
    in_ax.axvline(6250, linestyle='--', color='black')

    # doing voronoi binning

    pixelsize = 1

    yy, xx = np.indices(snr_r.shape)

    x_t = np.ravel(xx)
    y_t = np.ravel(yy)

    sgn_t_r = np.ravel(sgn_r)
    sgn_t_b = np.ravel(sgn_b)
    rms_t_r = np.ravel(rms_r)
    rms_t_b = np.ravel(rms_b)

    x_t_b = x_t[sgn_t_b / rms_t_b > 3]
    y_t_b = y_t[sgn_t_b / rms_t_b > 3]
    sgn_tt_b = sgn_t_b[sgn_t_b / rms_t_b > 3]
    rms_tt_b = rms_t_b[sgn_t_b / rms_t_b > 3]

    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_b, y_t_b, sgn_tt_b, rms_tt_b,
                                                                              targetSN,
                                                                              pixelsize=pixelsize, plot=0, quiet=1)

    ax = plt.subplot(gs[6, 0])

    xmin, xmax = np.min(x_t_b), np.max(x_t_b)
    ymin, ymax = np.min(y_t_b), np.max(y_t_b)
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

    ax = plt.subplot(gs[7, 0])

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
                                                for i in np.arange(np.nanmax(vorbin_map))))

    for i in np.arange(int(np.nanmax(vorbin_map))):
        for j in np.arange(len(nb_cube[i][2][0])):
            nb_cube_data[:, nb_cube[i][2][1][j], nb_cube[i][2][0][j]] = nb_cube[i][0]
            nb_cube_err[:, nb_cube[i][2][1][j], nb_cube[i][2][0][j]] = nb_cube[i][1]

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

    n_cube = fits.HDUList([fits.PrimaryHDU(data=nb_cube_data, header=cube_head),
                           fits.ImageHDU(data=nb_cube_err, header=cube_head, name='ERROR')])

    n_cube.writeto(gal_dir + 'blue_cube_vorbin.fits', overwrite=True)
    #

    x_t_r = x_t[sgn_t_r / rms_t_r > 3]
    y_t_r = y_t[sgn_t_r / rms_t_r > 3]
    sgn_tt_r = sgn_t_r[sgn_t_r / rms_t_r > 3]
    rms_tt_r = rms_t_r[sgn_t_r / rms_t_r > 3]

    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x_t_r, y_t_r, sgn_tt_r, rms_tt_r,
                                                                              targetSN,
                                                                              pixelsize=pixelsize, plot=0, quiet=1)

    ax = plt.subplot(gs[6, 3])

    xmin, xmax = np.min(x_t_r), np.max(x_t_r)
    ymin, ymax = np.min(y_t_r), np.max(y_t_r)
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

    ax = plt.subplot(gs[7, 3])

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
                                                for i in np.arange(np.nanmax(vorbin_map))))

    for i in np.arange(int(np.nanmax(vorbin_map))):
        for j in np.arange(len(nb_cube[i][2][0])):
            nb_cube_data[:, nb_cube[i][2][1][j], nb_cube[i][2][0][j]] = nb_cube[i][0]
            nb_cube_err[:, nb_cube[i][2][1][j], nb_cube[i][2][0][j]] = nb_cube[i][1]

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

    n_cube = fits.HDUList([fits.PrimaryHDU(data=nb_cube_data, header=cube_head),
                           fits.ImageHDU(data=nb_cube_err, header=cube_head, name='ERROR')])

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

    ax = plt.subplot(gs[8, 0])
    ax.plot(lamc_b, xc, '+', color='blue', ms=2, label='X center')
    ax.plot(lamc_b, yc, '+', color='red', ms=2, label='Y center')
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r'X and Y center')
    ax.set_title('Differential Atmosphere Effect (Blue)')
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

    ax = plt.subplot(gs[8, 3])
    ax.plot(lamc_r, xc, '+', color='blue', ms=2)
    ax.plot(lamc_r, yc, '+', color='red', ms=2)
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel(r'X and Y center')
    ax.set_title('Differential Atmosphere Effect (Red)')

    fig_l1 = date + '_' + gal_name + '_L1.png'

    fig.savefig(fig_l1)

    # ==================================================================

    # creating plots for APS

    aps_cube = fits.open(gal_name + '/' + gal_name + '_cube.fits')

    targetSN = np.float(config.get('QC_plots', 'target_SN'))
    levels = np.array(json.loads(config.get('QC_plots', 'levels'))).astype(np.float)  # SNR levels to display

    colap_a_map = np.sum(aps_cube[0].data[:], axis=0)

    lam_a = aps_cube[0].header['CRVAL3'] + (np.arange(aps_cube[0].header['NAXIS3']) * aps_cube[0].header['CDELT3'])

    sgn_a = np.mean(aps_cube[0].data[np.where(lam_a == min(lam_a, key=lambda x: abs(x - 6200)))[0][0] - 50:
                                     np.where(lam_a == min(lam_a, key=lambda x: abs(x - 6200)))[0][0] + 50], axis=0)
    rms_a = np.std(aps_cube[0].data[np.where(lam_a == min(lam_a, key=lambda x: abs(x - 6200)))[0][0] - 50:
                                    np.where(lam_a == min(lam_a, key=lambda x: abs(x - 6200)))[0][0] + 50], axis=0)
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

    fig = plt.figure(figsize=(16, 20))

    fig.suptitle(gal_name + ' L2/APS QC plots', size=22, weight='bold')

    gs = gridspec.GridSpec(4, 5, height_ratios=[1, 1, 1, 1], width_ratios=[1, 0.06, 0.4, 1, 1])
    gs.update(left=0.07, right=0.9, bottom=0.05, top=0.92, wspace=0.0, hspace=0.25)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    wcs = WCS(axis_header)

    ax = plt.subplot(gs[0, 0], projection=wcs)
    im = ax.imshow(np.arcsinh(colap_a_map), origin='lower')

    ypmax_a = np.where(colap_a_map == np.nanmax(colap_a_map))[0]
    xpmax_a = np.where(colap_a_map == np.nanmax(colap_a_map))[1]

    ax.plot(xpmax_a, ypmax_a, 'x', color='red', markersize=4, label=str(xpmax_a) + ', ' + str(ypmax_a))
    ax.set_title('Collapsed APS Datacube')
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    ax.legend()

    cbax = plt.subplot(gs[0, 1])
    cbar = Colorbar(ax=cbax, mappable=im)
    cbar.set_label('arcsinh scale')

    # ------

    ax = plt.subplot(gs[0, 3:5])
    ax.plot(lam_a, aps_cube[0].data[:, ypmax_a[0], xpmax_a[0]])
    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel('Flux?')
    ax.set_title('APS spectrum at (' + str(xpmax_a[0]) + ', ' + str(ypmax_a[0]) + ')')

    # ------

    ax = plt.subplot(gs[1, 0])
    im = ax.imshow(snr_a, origin='lower')
    cs = ax.contour(snr_a, levels, linestyles=np.array([':', '-']), colors='white')
    cs.collections[0].set_label('SNR = 3')
    cs.collections[1].set_label('SNR = 30')
    leg = ax.legend(framealpha=1, fontsize=8, loc='lower left')
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')

    ax.set_title(r'SNR @6200$\AA$')
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
    ax.set_xlabel(r'SNR [@6200$\AA$]')

    int_spec_a = np.sum(aps_cube[0].data * ((snr_a >= 3)[np.newaxis, :, :]), axis=(1, 2))
    in_ax = ax.inset_axes([0.55, 0.6, 0.4, 0.3])
    in_ax.set_title(r'          integrated spec [SNR$\geq$3]', fontsize=10)
    in_ax.plot(lam_a, int_spec_a)
    in_ax.axvline(6150, linestyle='--', color='black')
    in_ax.axvline(6250, linestyle='--', color='black')

    # doing voronoi binning

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

    xmin, xmax = np.min(x_t_a), np.max(x_t_a)
    ymin, ymax = np.min(y_t_a), np.max(y_t_a)
    nx = int(round((xmax - xmin) / pixelsize) + 1)
    ny = int(round((ymax - ymin) / pixelsize) + 1)
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
    
    # ------

    xc = []
    yc = []
    lamc_a = []
    
    for i in np.arange(len(aps_cube[1].data[:, 0, 0])):
        xcx = np.where(aps_cube[1].data[i, :, :] == np.nanmax(aps_cube[1].data[i, :, :]))[1]
        ycy = np.where(aps_cube[1].data[i, :, :] == np.nanmax(aps_cube[1].data[i, :, :]))[0]
        if len(xcx) < len(aps_cube[1].data[0, 0, :]):
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
    ax.set_title('Differential Atmosphere Effect (L2)')
    ax.legend(markerscale=5)
    
    
    # ------


    fig_l2 = date + '_' + gal_name + '_L2.png'

    fig.savefig(fig_l2)

    text = '''
    <html>
        <body style="background-color:white;">
            <div style="text-align: center;">
                <h1>Night report ''' + date + '''</h1>
                <img src="''' + fig_l1 + '''" class="center">
                <img src="''' + fig_l2 + '''" class="center">
            </div>
        </body>
    </html>
    '''

    f = open(date + ".html", "w")

    f.write(text)

    f.close()

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(hostname='minos.aip.de', username='gcouto', password=open('minos_pass', 'r').read().splitlines()[0])

    scp = SCPClient(ssh.get_transport())

    scp.put([date + '.html', fig_l1, fig_l2], '/store/weave/apertif/')
