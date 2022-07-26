import numpy as np
from astropy.io import fits
import os
from astropy.table import Table
import configparser


def tab_cre(self):

    config = configparser.ConfigParser()
    config.read(self)

    gal = config.get('spec_fit', 'gal_id')

    res_dir = config.get('table_creator', 'results_dir') + config.get('spec_fit', 'gal_id') + '/' + \
              np.sort(os.listdir(config.get('table_creator', 'results_dir') +
                                 config.get('spec_fit', 'gal_id') + '/'))[-1] + '/'

    c = fits.open(config.get('APS_cube', 'file_dir') + config.get('APS_cube', 'gal_id') + '.fits')
    rss_file = fits.open(res_dir + gal + '_vorbin_RSS.fits')

    contm_file = fits.open(res_dir + gal + '_vorbin.cont_model.fits')
    contr_file = fits.open(res_dir + gal + '_vorbin.cont_res.fits')
    elinm_file = fits.open(res_dir + gal + '_vorbin.eline_model.fits')
    elinr_file = fits.open(res_dir + gal + '_vorbin.eline_res.fits')

    elint_file = fits.open(res_dir + gal + '_vorbin.eline_table.fits')
    stelt_file = fits.open(res_dir + gal + '_vorbin.stellar_table.fits')

    params_stel = open(res_dir + '/parameters_stellar', 'r')
    lines = params_stel.readlines()

    elint_t = Table(elint_file[1].data)
    stelt_t = Table(stelt_file[1].data)

    xx = c[2].data['X']
    yy = c[2].data['Y']

    bin_id = c[2].data['BIN_ID']
    r_bin_id = c[3].data['BIN_ID']

    stel_template = fits.open(config.get('table_creator', 'template_dir') + lines[1].split()[1])

    contm_data = np.zeros((rss_file[0].shape[1], np.unique(yy).shape[0], np.unique(xx).shape[0]))
    contr_data = np.zeros((rss_file[0].shape[1], np.unique(yy).shape[0], np.unique(xx).shape[0]))
    elinm_data = np.zeros((rss_file[0].shape[1], np.unique(yy).shape[0], np.unique(xx).shape[0]))
    elinr_data = np.zeros((rss_file[0].shape[1], np.unique(yy).shape[0], np.unique(xx).shape[0]))

    xx_map = np.zeros((np.unique(yy).shape[0], np.unique(xx).shape[0]))
    yy_map = np.zeros((np.unique(yy).shape[0], np.unique(xx).shape[0]))
    vorbin_map = np.zeros((np.unique(yy).shape[0], np.unique(xx).shape[0])) * np.nan

    for i in np.arange(np.unique(xx).shape[0]):
        for j in np.arange(np.unique(yy).shape[0]):
            if np.size(np.where((xx == np.unique(xx)[i]) & (yy == np.unique(yy)[j]))) > 0:
                xx_map[j, i] = xx[np.where((xx == np.unique(xx)[i]) & (yy == np.unique(yy)[j]))]
                yy_map[j, i] = yy[np.where((xx == np.unique(xx)[i]) & (yy == np.unique(yy)[j]))]
                vorbin_map[j, i] = bin_id[np.where((xx == np.unique(xx)[i]) & (yy == np.unique(yy)[j]))]
            if vorbin_map[j, i] >= 0:
                contm_data[:, j, i] = contm_file[0].data[r_bin_id == vorbin_map[j, i]][0]
                contr_data[:, j, i] = contr_file[0].data[r_bin_id == vorbin_map[j, i]][0]
                elinm_data[:, j, i] = elinm_file[0].data[r_bin_id == vorbin_map[j, i]][0]
                elinr_data[:, j, i] = elinr_file[0].data[r_bin_id == vorbin_map[j, i]][0]
                print('Rearranging results into datacube formats: ' + str(
                    round(100. * i / np.unique(xx).shape[0], 2)) + '%', end='\r')

    print('')
    nvorbin_map = np.flip(vorbin_map, axis=1)

    tab_el = elint_t.copy()
    tab_st = stelt_t.copy()

    tx = []
    ty = []

    elint_maps = []
    stelt_maps = []

    elint_maps_n = elint_file[1].data.names
    stelt_maps_n = stelt_file[1].data.names
    elint_maps_n.remove('fiber')
    stelt_maps_n.remove('fiber')
    stelt_maps_n.remove('base_coeff')

    base_coeff_t = Table(stel_template[1].data)
    base_coeff_maps = []

    for i in np.arange(len(elint_maps_n)):
        elint_maps.append(nvorbin_map.copy())
    for i in np.arange(len(stelt_maps_n)):
        stelt_maps.append(nvorbin_map.copy())
    for i in np.arange(len(base_coeff_t)):
        base_coeff_maps.append(nvorbin_map.copy())

    elint_maps = np.reshape(elint_maps, (len(elint_maps_n), elint_maps[0].shape[0], elint_maps[0].shape[1]))
    stelt_maps = np.reshape(stelt_maps, (len(stelt_maps_n), stelt_maps[0].shape[0], stelt_maps[0].shape[1]))
    base_coeff_maps = np.reshape(base_coeff_maps,
                                 (len(base_coeff_maps), base_coeff_maps[0].shape[0], base_coeff_maps[0].shape[1]))

    for k in r_bin_id:
        tx.append(np.where(nvorbin_map == k)[1])
        ty.append(np.where(nvorbin_map == k)[0])

        for i in np.arange(len(elint_maps)):
            elint_maps[i][nvorbin_map == k] = elint_file[1].data[elint_maps_n[i]][elint_file[1].data['fiber'] == k]
        for i in np.arange(len(stelt_maps)):
            stelt_maps[i][nvorbin_map == k] = stelt_file[1].data[stelt_maps_n[i]][stelt_file[1].data['fiber'] == k]
        for i in np.arange(len(base_coeff_maps)):
            base_coeff_maps[i][nvorbin_map == k] = \
                stelt_file[1].data['base_coeff'][stelt_file[1].data['fiber'] == k][0][i]

    ttx = np.concatenate(tx)
    tty = np.concatenate(ty)

    for k in r_bin_id:
        for j in np.arange(len(np.where(nvorbin_map == k)[0]) - 1):
            tab_el.add_row(tab_el[tab_el['fiber'] == k][0])
            tab_st.add_row(tab_st[tab_st['fiber'] == k][0])
        print('Organizing tables formats: ' + str(round(100. * k / len(r_bin_id), 2)) + '%', end='\r')

    tab_el = tab_el[tab_el.argsort(['fiber'])]
    tab_st = tab_st[tab_st.argsort(['fiber'])]

    tab_el.add_column(ttx, name='x_cor', index=0)
    tab_el.add_column(tty, name='y_cor', index=1)
    tab_st.add_column(ttx, name='x_cor', index=0)
    tab_st.add_column(tty, name='y_cor', index=1)

    ncontm_data = np.flip(contm_data, axis=2)
    ncontr_data = np.flip(contr_data, axis=2)
    nelinm_data = np.flip(elinm_data, axis=2)
    nelinr_data = np.flip(elinr_data, axis=2)

    # create RSS file

    cube_head = fits.Header()
    cube_head['SIMPLE'] = True
    cube_head['BITPIX'] = -32
    cube_head['NAXIS'] = 3
    cube_head['NAXIS1'] = ncontm_data.shape[2]
    cube_head['NAXIS2'] = ncontm_data.shape[1]
    cube_head['NAXIS3'] = ncontm_data.shape[0]
    cube_head['CTYPE3'] = 'WAVELENGTH'
    cube_head['CUNIT3'] = 'Angstrom'
    cube_head['CDELT3'] = rss_file[0].header['CDELT1']
    cube_head['DISPAXIS'] = rss_file[0].header['DISPAXIS']
    cube_head['CRVAL3'] = rss_file[0].header['CRVAL1']
    cube_head['CRPIX3'] = rss_file[0].header['CRPIX1']
    cube_head['CRPIX1'] = \
        np.where(np.unique(c[2].data['X']) == np.min(np.unique(c[2].data['X'])[np.unique(c[2].data['X']) > 0]))[0][
            0] + 1  # X from APS with lower absolute value
    cube_head['CRPIX2'] = \
        np.where(np.unique(c[2].data['Y']) == np.min(np.unique(c[2].data['Y'])[np.unique(c[2].data['Y']) > 0]))[0][
            0] + 1  # Y from APS with lower absolute value
    cube_head['CRVAL1'] = np.unique(c[2].data['X_0'])[0] + (np.min(
        np.unique(c[2].data['X'])[np.unique(c[2].data['X']) > 0]) / 3600.)  # X from central pixel plus position X_0
    cube_head['CRVAL2'] = np.unique(c[2].data['Y_0'])[0] + (np.min(
        np.unique(c[2].data['Y'])[np.unique(c[2].data['Y']) > 0]) / 3600.)  # Y from central pixel plus position Y_0
    cube_head['CDELT1'] = -(np.unique(c[2].data['X'])[1] - np.unique(c[2].data['X'])[0]) / 3600.
    cube_head['CDELT2'] = (np.unique(c[2].data['Y'])[1] - np.unique(c[2].data['Y'])[0]) / 3600.
    cube_head['CTYPE1'] = 'RA---TAN'
    cube_head['CTYPE2'] = 'DEC--TAN'
    cube_head['CUNIT1'] = 'deg'
    cube_head['CUNIT2'] = 'deg'

    n_elinm = fits.HDUList([fits.PrimaryHDU(data=nelinm_data, header=cube_head)])
    n_contm = fits.HDUList([fits.PrimaryHDU(data=ncontm_data, header=cube_head)])
    n_elinr = fits.HDUList([fits.PrimaryHDU(data=nelinr_data, header=cube_head)])
    n_contr = fits.HDUList([fits.PrimaryHDU(data=ncontr_data, header=cube_head)])

    n_tab_eline = fits.HDUList([elint_file[0].copy(),
                                fits.BinTableHDU(tab_el, header=elint_file[1].header)])
    n_tab_stell = fits.HDUList([stelt_file[0].copy(),
                                fits.BinTableHDU(tab_st, header=stelt_file[1].header)])

    map_head = fits.Header()
    map_head['SIMPLE'] = True
    map_head['BITPIX'] = -32
    map_head['NAXIS'] = 2
    map_head['NAXIS1'] = nvorbin_map.shape[1]
    map_head['NAXIS2'] = nvorbin_map.shape[0]
    map_head['DISPAXIS'] = 1
    map_head['CRPIX1'] = \
        np.where(np.unique(c[2].data['X']) == np.min(np.unique(c[2].data['X'])[np.unique(c[2].data['X']) > 0]))[0][
            0] + 1  # X from APS with lower absolute value
    map_head['CRPIX2'] = \
        np.where(np.unique(c[2].data['Y']) == np.min(np.unique(c[2].data['Y'])[np.unique(c[2].data['Y']) > 0]))[0][
            0] + 1  # Y from APS with lower absolute value
    map_head['CRVAL1'] = np.unique(c[2].data['X_0'])[0] + (np.min(
        np.unique(c[2].data['X'])[np.unique(c[2].data['X']) > 0]) / 3600.)  # X from central pixel plus position X_0
    map_head['CRVAL2'] = np.unique(c[2].data['Y_0'])[0] + (np.min(
        np.unique(c[2].data['Y'])[np.unique(c[2].data['Y']) > 0]) / 3600.)  # Y from central pixel plus position Y_0
    map_head['CDELT1'] = -(np.unique(c[2].data['X'])[1] - np.unique(c[2].data['X'])[0]) / 3600.
    map_head['CDELT2'] = (np.unique(c[2].data['Y'])[1] - np.unique(c[2].data['Y'])[0]) / 3600.
    map_head['CTYPE1'] = 'RA---TAN'
    map_head['CTYPE2'] = 'DEC--TAN'
    map_head['CUNIT1'] = 'deg'
    map_head['CUNIT2'] = 'deg'

    hdu_elint_maps = fits.HDUList([fits.PrimaryHDU()])
    hdu_stelt_maps = fits.HDUList([fits.PrimaryHDU()])
    hdu_base_coeff_maps = fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU(base_coeff_t, name=lines[1].split()[1][:-5])])

    for i in np.arange(len(elint_maps)):
        hdu_elint_maps.append(fits.ImageHDU(data=elint_maps[i], name=elint_maps_n[i], header=map_head))
    for i in np.arange(len(stelt_maps)):
        hdu_stelt_maps.append(fits.ImageHDU(data=stelt_maps[i], name=stelt_maps_n[i], header=map_head))
    for i in np.arange(len(base_coeff_maps)):
        hdu_base_coeff_maps.append(fits.ImageHDU(data=base_coeff_maps[i], name='Template ' + str(i), header=map_head))

    print('Saving data...')

    n_elinm.writeto(res_dir + gal + '_eline_model.fits', overwrite=True)
    n_contm.writeto(res_dir + gal + '_cont_model.fits', overwrite=True)
    n_elinr.writeto(res_dir + gal + '_eline_res.fits', overwrite=True)
    n_contr.writeto(res_dir + gal + '_cont_res.fits', overwrite=True)

    n_tab_eline.writeto(res_dir + gal + '_eline_table.fits', overwrite=True)
    n_tab_stell.writeto(res_dir + gal + '_stellar_table.fits', overwrite=True)

    hdu_elint_maps.writeto(res_dir + gal + '_eline_maps.fits', overwrite=True)
    hdu_stelt_maps.writeto(res_dir + gal + '_stellar_maps.fits', overwrite=True)
    hdu_base_coeff_maps.writeto(res_dir + gal + '_base_coeff_maps.fits', overwrite=True)
