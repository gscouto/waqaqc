import json
import numpy as np
from astropy.io import fits
import configparser
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm


def cm(self):
    config = configparser.ConfigParser()
    config.read(self)

    file_dir = config.get('APS_cube', 'file_dir')

    blue_cube = fits.open(file_dir + config.get('QC_plots', 'blue_cube'))
    gal_name = blue_cube[0].header['CCNAME1']
    aps_cube = fits.open(gal_name + '/' + gal_name + '_vorbin_cube.fits')

    wave = aps_cube[0].header['CRVAL3'] + (np.arange(aps_cube[0].header['NAXIS3']) * aps_cube[0].header['CDELT3'])

    cen_wav = float(config.get('channel_maps', 'cen_wav'))
    min_vel = float(config.get('channel_maps', 'min_vel'))
    max_vel = float(config.get('channel_maps', 'max_vel'))

    wave_vels = 300000. * (wave - cen_wav) / cen_wav

    grid_side = int(config.get('channel_maps', 'grid_side'))

    step = int(len(np.where((wave_vels > min_vel) & (wave_vels < max_vel))[0]) / (grid_side ** 2.)) + 1

    wave_w_pix = np.arange(np.where((wave_vels - min_vel) == min(wave_vels - min_vel, key=abs))[0][0],
                           np.where((wave_vels - max_vel) == min(wave_vels - max_vel, key=abs))[0][0], step)
    wave_window = wave[wave_w_pix]

    vels = 300000. * (wave_window - cen_wav) / cen_wav

    cont_cen_pix = np.where((wave - int(config.get('channel_maps', 'cont_wave')))
                            == min(wave - int(config.get('channel_maps', 'cont_wave')), key=abs))[0][0]

    cont_map = np.nanmedian(aps_cube[0].data[cont_cen_pix - 50:cont_cen_pix + 50, :, :], axis=0)

    channel_maps = np.zeros((len(wave_window), aps_cube[0].data.shape[1], aps_cube[0].data.shape[2]))

    for i in np.arange(len(wave_window)):
        channel_maps[i] = np.nanmedian(aps_cube[0].data[wave_w_pix[i]:wave_w_pix[i] + step, :, :], axis=0) - cont_map

    f_min = float(config.get('channel_maps', 'flux_min')) * np.nanmax(channel_maps)
    f_max = float(config.get('channel_maps', 'flux_max')) * np.nanmax(channel_maps)

    channel_maps[channel_maps < f_min] = np.nan

    cmap = matplotlib.cm.get_cmap(config.get('channel_maps', 'color_scale'))
    cmap.set_bad(color='black')
    
    if int(config.get('channel_maps', 'contours')):
        levels = np.array(json.loads(config.get('channel_maps', 'c_levels'))).astype(np.float) * np.nanmax(channel_maps)

    fig = plt.figure(figsize=(grid_side * 4, grid_side * 4))

    gs = gridspec.GridSpec(grid_side, grid_side, height_ratios=np.zeros(grid_side) + 1,
                           width_ratios=np.zeros(grid_side) + 1)
    gs.update(left=0.07, right=0.95, bottom=0.05, top=0.95, wspace=0.15, hspace=0.15)

    cont = 0
    for j in np.arange(grid_side):
        for i in np.arange(grid_side):
            if cont + 1 <= len(channel_maps):
                ax = plt.subplot(gs[j, i])
                ax.imshow(channel_maps[cont], cmap=cmap, norm=LogNorm(vmin=f_min, vmax=f_max), origin='lower')
                if int(config.get('channel_maps', 'contours')):
                    ax.contour(channel_maps[cont], levels, linestyles='--', colors='blue')
                ax.annotate(str(int(vels[cont])), (0.85, 0.85), xycoords='axes fraction', color='white')
            cont += 1

    fig.savefig('channel_maps.pdf')
