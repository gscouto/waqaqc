import numpy as np
from astropy.io import fits
import os
from astropy.table import Table
import configparser
from astropy.wcs import WCS


def tab_cre(self):
    config = configparser.ConfigParser()
    config.read(self)

    file_dir = config.get('APS_cube', 'file_dir')

    blue_cube = fits.open(file_dir + config.get('QC_plots', 'blue_cube'))

    gal = blue_cube[0].header['CCNAME1']

    res_dir = gal + '/pyp_results/' + np.sort(os.listdir(gal + '/pyp_results/'))[-1] + '/'
    # res_dir = gal + '/pyp_results/' + '2022-12-22_15.08.53/'

    list_file = os.listdir(file_dir)

    if int(config.get('spec_fit', 'aps_fit')) == 1:

        wcs_c = fits.open(gal + '/' + gal + '_cube.fits')
        c = fits.open(gal + '/APS_cube_vorbin.fits')
        rss_file = fits.open(res_dir + gal + '_APS_vorbin_RSS.fits')

        contm_file = fits.open(res_dir + gal + '_APS_vorbin.cont_model.fits')
        contr_file = fits.open(res_dir + gal + '_APS_vorbin.cont_res.fits')
        elinm_file = fits.open(res_dir + gal + '_APS_vorbin.eline_model.fits')
        elinr_file = fits.open(res_dir + gal + '_APS_vorbin.eline_res.fits')

        elint_file = fits.open(res_dir + gal + '_APS_vorbin.eline_table.fits')
        stelt_file = fits.open(res_dir + gal + '_APS_vorbin.stellar_table.fits')

        vorbin_map = fits.getdata(gal + '/vorbin_map_aps.fits')

        params_stel = open(res_dir + '/parameters_stellar', 'r')
        lines = params_stel.readlines()

        elint_t = Table(elint_file[1].data)
        stelt_t = Table(stelt_file[1].data)

        axis_header = fits.Header()
        axis_header['NAXIS1'] = c[1].header['NAXIS1']
        axis_header['NAXIS2'] = c[1].header['NAXIS2']
        axis_header['CDELT1'] = c[1].header['CDELT1']
        axis_header['CDELT2'] = c[1].header['CDELT2']
        axis_header['CRPIX1'] = c[1].header['CRPIX1']
        axis_header['CRPIX2'] = c[1].header['CRPIX2']
        axis_header['CRVAL1'] = c[1].header['CRVAL1']
        axis_header['CRVAL2'] = c[1].header['CRVAL2']
        axis_header['CTYPE1'] = c[1].header['CTYPE1']
        axis_header['CTYPE2'] = c[1].header['CTYPE2']
        axis_header['CUNIT1'] = c[1].header['CUNIT1']
        axis_header['CUNIT2'] = c[1].header['CUNIT2']

        # wcs = WCS(axis_header)

        # aps_ra = c[2].data['X_0'] + (c[2].data['X'] / 3600)
        # aps_dec = c[2].data['Y_0'] + (c[2].data['Y'] / 3600)

        # aps_ra_dec = np.vstack((aps_ra, aps_dec)).T

        # pix_map = np.round(wcs.wcs_world2pix(aps_ra_dec, 0), 0)

        # pix_mapt = pix_map.T.astype(int)
        # pix_mapt[0] = pix_mapt[0] - np.min(pix_mapt[0])
        # pix_mapt[1] = pix_mapt[1] - np.min(pix_mapt[1])

        # pix_mapt = pix_mapt.T

        # x_pix, y_pix = pix_map.T.astype(int)

        # bin_id = c[2].data['BIN_ID']
        # r_bin_id = c[3].data['BIN_ID']

        stel_template = fits.open(config.get('table_creator', 'template_dir') + lines[1].split()[1])

        contm_data = np.zeros((contm_file[0].data.shape[1], c[0].data.shape[1], c[0].data.shape[2]))
        contm_err = np.zeros((contm_file[0].data.shape[1], c[0].data.shape[1], c[0].data.shape[2]))
        contm_badp = np.zeros((contm_file[0].data.shape[1], c[0].data.shape[1], c[0].data.shape[2]))
        contm_norm = np.zeros((contm_file[0].data.shape[1], c[0].data.shape[1], c[0].data.shape[2]))
        contr_data = np.zeros((contm_file[0].data.shape[1], c[0].data.shape[1], c[0].data.shape[2]))
        elinm_data = np.zeros((contm_file[0].data.shape[1], c[0].data.shape[1], c[0].data.shape[2]))
        elinr_data = np.zeros((contm_file[0].data.shape[1], c[0].data.shape[1], c[0].data.shape[2]))

        cnt = 0
        for i in np.unique(vorbin_map[np.isfinite(vorbin_map)]).astype(int):
            contm_data.T[vorbin_map.T == i] = contm_file[0].data[i]
            contm_err.T[vorbin_map.T == i] = contm_file[1].data[i]
            contm_badp.T[vorbin_map.T == i] = contm_file[2].data[i]
            contm_norm.T[vorbin_map.T == i] = contm_file[3].data[i]
            contr_data.T[vorbin_map.T == i] = contr_file[0].data[i]
            elinm_data.T[vorbin_map.T == i] = elinm_file[0].data[i]
            elinr_data.T[vorbin_map.T == i] = elinr_file[0].data[i]
            print('Rearranging into datacube formats: ' + str(
                round(100. * cnt / np.unique(vorbin_map[np.isfinite(vorbin_map)]).shape[0], 2)) + '%', end='\r')
            cnt += 1

        print('')

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
            elint_maps.append(vorbin_map.copy())
        for i in np.arange(len(stelt_maps_n)):
            stelt_maps.append(vorbin_map.copy())
        for i in np.arange(len(base_coeff_t)):
            base_coeff_maps.append(vorbin_map.copy())

        elint_maps = np.reshape(elint_maps, (len(elint_maps_n), elint_maps[0].shape[0], elint_maps[0].shape[1]))
        stelt_maps = np.reshape(stelt_maps, (len(stelt_maps_n), stelt_maps[0].shape[0], stelt_maps[0].shape[1]))
        base_coeff_maps = np.reshape(base_coeff_maps,
                                     (len(base_coeff_maps), base_coeff_maps[0].shape[0], base_coeff_maps[0].shape[1]))

        for k in np.arange(len(stelt_file[1].data['fiber'])):
            tx.append(np.where(vorbin_map == k)[1])
            ty.append(np.where(vorbin_map == k)[0])

            for i in np.arange(len(elint_maps)):
                elint_maps[i][vorbin_map == k] = \
                    elint_file[1].data[elint_maps_n[i]][elint_file[1].data['fiber'] == k][0]
            for i in np.arange(len(stelt_maps)):
                stelt_maps[i][vorbin_map == k] = \
                    stelt_file[1].data[stelt_maps_n[i]][stelt_file[1].data['fiber'] == k][0]
            for i in np.arange(len(base_coeff_maps)):
                base_coeff_maps[i][vorbin_map == k] = \
                    stelt_file[1].data['base_coeff'][stelt_file[1].data['fiber'] == k][0][i]

        ttx = np.concatenate(tx)
        tty = np.concatenate(ty)

        for k in np.arange(len(stelt_file[1].data['fiber'])):
            for j in np.arange(len(np.where(vorbin_map == k)[0]) - 1):
                tab_el.add_row(tab_el[tab_el['fiber'] == k][0])
                tab_st.add_row(tab_st[tab_st['fiber'] == k][0])
            print(
                'Organizing tables formats: ' + str(round(100. * k / np.nanmax(stelt_file[1].data['fiber']), 2)) + '%',
                end='\r')

        print('')

        tab_el = tab_el[tab_el.argsort(['fiber'])]
        tab_st = tab_st[tab_st.argsort(['fiber'])]

        tab_el.add_column(ttx, name='x_cor', index=0)
        tab_el.add_column(tty, name='y_cor', index=1)
        tab_st.add_column(ttx, name='x_cor', index=0)
        tab_st.add_column(tty, name='y_cor', index=1)

        # create RSS file

        cube_head = fits.Header()
        cube_head['SIMPLE'] = True
        cube_head['BITPIX'] = -32
        cube_head['NAXIS'] = 3
        cube_head['NAXIS1'] = contm_data.shape[2]
        cube_head['NAXIS2'] = contm_data.shape[1]
        cube_head['NAXIS3'] = contm_data.shape[0]
        cube_head['CTYPE3'] = 'WAVELENGTH'
        cube_head['CUNIT3'] = 'Angstrom'
        cube_head['CDELT3'] = rss_file[0].header['CDELT1']
        cube_head['DISPAXIS'] = rss_file[0].header['DISPAXIS']
        cube_head['CRVAL3'] = rss_file[0].header['CRVAL1']
        cube_head['CRPIX3'] = rss_file[0].header['CRPIX1']
        cube_head['CRPIX1'] = wcs_c[0].header['CRPIX1']
        cube_head['CRPIX2'] = wcs_c[0].header['CRPIX2']
        cube_head['CRVAL1'] = wcs_c[0].header['CRVAL1']
        cube_head['CRVAL2'] = wcs_c[0].header['CRVAL2']
        cube_head['CDELT1'] = wcs_c[0].header['CDELT1']
        cube_head['CDELT2'] = wcs_c[0].header['CDELT2']
        cube_head['CTYPE1'] = 'RA---TAN'
        cube_head['CTYPE2'] = 'DEC--TAN'
        cube_head['CUNIT1'] = 'deg'
        cube_head['CUNIT2'] = 'deg'

        n_elinm = fits.HDUList([fits.PrimaryHDU(data=elinm_data, header=cube_head)])
        n_contm = fits.HDUList([fits.PrimaryHDU(data=contm_data, header=cube_head),
                                fits.ImageHDU(data=contm_err, header=cube_head, name='ERROR'),
                                fits.ImageHDU(data=contm_badp, header=cube_head, name='BADPIX'),
                                fits.ImageHDU(data=contm_norm, header=cube_head, name='NORMALIZE')])
        n_elinr = fits.HDUList([fits.PrimaryHDU(data=elinr_data, header=cube_head)])
        n_contr = fits.HDUList([fits.PrimaryHDU(data=contr_data, header=cube_head)])

        n_tab_eline = fits.HDUList([elint_file[0].copy(),
                                    fits.BinTableHDU(tab_el, header=elint_file[1].header)])
        n_tab_stell = fits.HDUList([stelt_file[0].copy(),
                                    fits.BinTableHDU(tab_st, header=stelt_file[1].header)])

        map_head = fits.Header()
        map_head['SIMPLE'] = True
        map_head['BITPIX'] = -32
        map_head['NAXIS'] = 2
        map_head['NAXIS1'] = vorbin_map.shape[1]
        map_head['NAXIS2'] = vorbin_map.shape[0]
        map_head['DISPAXIS'] = 1
        map_head['CRPIX1'] = c[0].header['CRPIX1']
        map_head['CRPIX2'] = c[0].header['CRPIX2']
        map_head['CRVAL1'] = c[0].header['CRVAL1']
        map_head['CRVAL2'] = c[0].header['CRVAL2']
        map_head['CDELT1'] = c[0].header['CDELT1']
        map_head['CDELT2'] = c[0].header['CDELT2']
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
            hdu_base_coeff_maps.append(
                fits.ImageHDU(data=base_coeff_maps[i], name='Template ' + str(i), header=map_head))

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

    if int(config.get('spec_fit', 'red_fit')) == 1:

        wcs_c = fits.open(gal + '/red_cube_vorbin.fits')
        # c = fits.open(file_dir + [s for s in list_file if 'APS.fits' in s][0])
        rss_file = fits.open(res_dir + gal + '_red_vorbin_RSS.fits')
        vorbin_map = fits.getdata(gal + '/vorbin_map_red.fits')

        contm_file = fits.open(res_dir + gal + '_red_vorbin.cont_model.fits')
        contr_file = fits.open(res_dir + gal + '_red_vorbin.cont_res.fits')
        elinm_file = fits.open(res_dir + gal + '_red_vorbin.eline_model.fits')
        elinr_file = fits.open(res_dir + gal + '_red_vorbin.eline_res.fits')

        elint_file = fits.open(res_dir + gal + '_red_vorbin.eline_table.fits')
        stelt_file = fits.open(res_dir + gal + '_red_vorbin.stellar_table.fits')

        params_stel = open(res_dir + '/parameters_stellar', 'r')
        lines = params_stel.readlines()

        elint_t = Table(elint_file[1].data)
        stelt_t = Table(stelt_file[1].data)

        axis_header = fits.Header()
        axis_header['NAXIS1'] = wcs_c[1].header['NAXIS1']
        axis_header['NAXIS2'] = wcs_c[1].header['NAXIS2']
        axis_header['CDELT1'] = wcs_c[1].header['CDELT1']
        axis_header['CDELT2'] = wcs_c[1].header['CDELT2']
        axis_header['CRPIX1'] = wcs_c[1].header['CRPIX1']
        axis_header['CRPIX2'] = wcs_c[1].header['CRPIX2']
        axis_header['CRVAL1'] = wcs_c[1].header['CRVAL1']
        axis_header['CRVAL2'] = wcs_c[1].header['CRVAL2']
        axis_header['CTYPE1'] = wcs_c[1].header['CTYPE1']
        axis_header['CTYPE2'] = wcs_c[1].header['CTYPE2']
        axis_header['CUNIT1'] = wcs_c[1].header['CUNIT1']
        axis_header['CUNIT2'] = wcs_c[1].header['CUNIT2']

        # wcs = WCS(axis_header)

        # aps_ra = c[2].data['X_0'] + (c[2].data['X'] / 3600)
        # aps_dec = c[2].data['Y_0'] + (c[2].data['Y'] / 3600)
        #
        # aps_ra_dec = np.vstack((aps_ra, aps_dec)).T
        #
        # pix_map = np.round(wcs.wcs_world2pix(aps_ra_dec, 0), 0)
        #
        # pix_mapt = pix_map.T.astype(int)
        # pix_mapt[0] = pix_mapt[0] - np.min(pix_mapt[0])
        # pix_mapt[1] = pix_mapt[1] - np.min(pix_mapt[1])
        #
        # pix_mapt = pix_mapt.T
        #
        # x_pix, y_pix = pix_map.T.astype(int)

        # bin_id = c[2].data['BIN_ID']
        # r_bin_id = c[3].data['BIN_ID']

        stel_template = fits.open(config.get('table_creator', 'template_dir') + lines[1].split()[1])

        contm_data = np.zeros((contm_file[0].data.shape[1], wcs_c[0].data.shape[1], wcs_c[0].data.shape[2]))
        contr_data = np.zeros((contm_file[0].data.shape[1], wcs_c[0].data.shape[1], wcs_c[0].data.shape[2]))
        elinm_data = np.zeros((contm_file[0].data.shape[1], wcs_c[0].data.shape[1], wcs_c[0].data.shape[2]))
        elinr_data = np.zeros((contm_file[0].data.shape[1], wcs_c[0].data.shape[1], wcs_c[0].data.shape[2]))

        cnt = 0
        for i in np.unique(vorbin_map[np.isfinite(vorbin_map)]).astype(int):
            contm_data.T[vorbin_map.T == i] = contm_file[0].data[i]
            contr_data.T[vorbin_map.T == i] = contr_file[0].data[i]
            elinm_data.T[vorbin_map.T == i] = elinm_file[0].data[i]
            elinr_data.T[vorbin_map.T == i] = elinr_file[0].data[i]
            print('Rearranging into datacube formats: ' + str(
                round(100. * cnt / np.unique(vorbin_map[np.isfinite(vorbin_map)]).shape[0], 2)) + '%', end='\r')
            cnt += 1

        print('')

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
            elint_maps.append(vorbin_map.copy())
        for i in np.arange(len(stelt_maps_n)):
            stelt_maps.append(vorbin_map.copy())
        for i in np.arange(len(base_coeff_t)):
            base_coeff_maps.append(vorbin_map.copy())

        elint_maps = np.reshape(elint_maps, (len(elint_maps_n), elint_maps[0].shape[0], elint_maps[0].shape[1]))
        stelt_maps = np.reshape(stelt_maps, (len(stelt_maps_n), stelt_maps[0].shape[0], stelt_maps[0].shape[1]))
        base_coeff_maps = np.reshape(base_coeff_maps,
                                     (len(base_coeff_maps), base_coeff_maps[0].shape[0], base_coeff_maps[0].shape[1]))

        for k in np.arange(np.nanmax(stelt_file[1].data['fiber'])):
            # for k in np.unique(vorbin_map[np.isfinite(vorbin_map)]).astype(int):
            tx.append(np.where(vorbin_map == k)[1])
            ty.append(np.where(vorbin_map == k)[0])

            # for i in np.arange(len(elint_maps)):
            #     elint_maps[i][vorbin_map == k] = elint_file[1].data[elint_maps_n[i]][elint_file[1].data['fiber'] == k]
            for i in np.arange(len(stelt_maps)):
                stelt_maps[i][vorbin_map == k] = stelt_file[1].data[stelt_maps_n[i]][stelt_file[1].data['fiber'] == k]
            for i in np.arange(len(base_coeff_maps)):
                base_coeff_maps[i][vorbin_map == k] = \
                    stelt_file[1].data['base_coeff'][stelt_file[1].data['fiber'] == k][0][i]

        ttx = np.concatenate(tx)
        tty = np.concatenate(ty)

        for k in np.arange(np.nanmax(stelt_file[1].data['fiber'])):
            # for k in np.unique(vorbin_map[np.isfinite(vorbin_map)]).astype(int):
            for j in np.arange(len(np.where(vorbin_map == k)[0]) - 1):
                # tab_el.add_row(tab_el[tab_el['fiber'] == k][0])
                tab_st.add_row(tab_st[tab_st['fiber'] == k][0])
            print('Organizing tables formats: ' + str(round(100. * k / np.nanmax(stelt_file[1].data['fiber']), 2)) + '%'
                  , end='\r')

        # tab_el = tab_el[tab_el.argsort(['fiber'])]
        tab_st = tab_st[tab_st.argsort(['fiber'])]

        # tab_el.add_column(ttx, name='x_cor', index=0)
        # tab_el.add_column(tty, name='y_cor', index=1)
        tab_st.add_column(ttx, name='x_cor', index=0)
        tab_st.add_column(tty, name='y_cor', index=1)

        # create RSS file

        cube_head = fits.Header()
        cube_head['SIMPLE'] = True
        cube_head['BITPIX'] = -32
        cube_head['NAXIS'] = 3
        cube_head['NAXIS1'] = contm_data.shape[2]
        cube_head['NAXIS2'] = contm_data.shape[1]
        cube_head['NAXIS3'] = contm_data.shape[0]
        cube_head['CTYPE3'] = 'WAVELENGTH'
        cube_head['CUNIT3'] = 'Angstrom'
        cube_head['CDELT3'] = rss_file[0].header['CDELT1']
        cube_head['DISPAXIS'] = rss_file[0].header['DISPAXIS']
        cube_head['CRVAL3'] = rss_file[0].header['CRVAL1']
        cube_head['CRPIX3'] = rss_file[0].header['CRPIX1']
        cube_head['CRPIX1'] = wcs_c[0].header['CRPIX1']
        cube_head['CRPIX2'] = wcs_c[0].header['CRPIX2']
        cube_head['CRVAL1'] = wcs_c[0].header['CRVAL1']
        cube_head['CRVAL2'] = wcs_c[0].header['CRVAL2']
        cube_head['CDELT1'] = wcs_c[0].header['CDELT1']
        cube_head['CDELT2'] = wcs_c[0].header['CDELT2']
        cube_head['CTYPE1'] = 'RA---TAN'
        cube_head['CTYPE2'] = 'DEC--TAN'
        cube_head['CUNIT1'] = 'deg'
        cube_head['CUNIT2'] = 'deg'

        n_elinm = fits.HDUList([fits.PrimaryHDU(data=elinm_data, header=cube_head)])
        n_contm = fits.HDUList([fits.PrimaryHDU(data=contm_data, header=cube_head)])
        n_elinr = fits.HDUList([fits.PrimaryHDU(data=elinr_data, header=cube_head)])
        n_contr = fits.HDUList([fits.PrimaryHDU(data=contr_data, header=cube_head)])

        # n_tab_eline = fits.HDUList([elint_file[0].copy(),
        #                             fits.BinTableHDU(tab_el, header=elint_file[1].header)])
        n_tab_stell = fits.HDUList([stelt_file[0].copy(),
                                    fits.BinTableHDU(tab_st, header=stelt_file[1].header)])

        map_head = fits.Header()
        map_head['SIMPLE'] = True
        map_head['BITPIX'] = -32
        map_head['NAXIS'] = 2
        map_head['NAXIS1'] = vorbin_map.shape[1]
        map_head['NAXIS2'] = vorbin_map.shape[0]
        map_head['DISPAXIS'] = 1
        map_head['CRPIX1'] = wcs_c[0].header['CRPIX1']
        map_head['CRPIX2'] = wcs_c[0].header['CRPIX2']
        map_head['CRVAL1'] = wcs_c[0].header['CRVAL1']
        map_head['CRVAL2'] = wcs_c[0].header['CRVAL2']
        map_head['CDELT1'] = wcs_c[0].header['CDELT1']
        map_head['CDELT2'] = wcs_c[0].header['CDELT2']
        map_head['CTYPE1'] = 'RA---TAN'
        map_head['CTYPE2'] = 'DEC--TAN'
        map_head['CUNIT1'] = 'deg'
        map_head['CUNIT2'] = 'deg'

        # hdu_elint_maps = fits.HDUList([fits.PrimaryHDU()])
        hdu_stelt_maps = fits.HDUList([fits.PrimaryHDU()])
        hdu_base_coeff_maps = fits.HDUList(
            [fits.PrimaryHDU(), fits.BinTableHDU(base_coeff_t, name=lines[1].split()[1][:-5])])

        # for i in np.arange(len(elint_maps)):
        #     hdu_elint_maps.append(fits.ImageHDU(data=elint_maps[i], name=elint_maps_n[i], header=map_head))
        for i in np.arange(len(stelt_maps)):
            hdu_stelt_maps.append(fits.ImageHDU(data=stelt_maps[i], name=stelt_maps_n[i], header=map_head))
        for i in np.arange(len(base_coeff_maps)):
            hdu_base_coeff_maps.append(
                fits.ImageHDU(data=base_coeff_maps[i], name='Template ' + str(i), header=map_head))

        print('Saving data...')

        n_elinm.writeto(res_dir + gal + '_eline_model.fits', overwrite=True)
        n_contm.writeto(res_dir + gal + '_cont_model.fits', overwrite=True)
        n_elinr.writeto(res_dir + gal + '_eline_res.fits', overwrite=True)
        n_contr.writeto(res_dir + gal + '_cont_res.fits', overwrite=True)

        # n_tab_eline.writeto(res_dir + gal + '_eline_table.fits', overwrite=True)
        n_tab_stell.writeto(res_dir + gal + '_stellar_table.fits', overwrite=True)

        # hdu_elint_maps.writeto(res_dir + gal + '_eline_maps.fits', overwrite=True)
        hdu_stelt_maps.writeto(res_dir + gal + '_stellar_maps.fits', overwrite=True)
        hdu_base_coeff_maps.writeto(res_dir + gal + '_base_coeff_maps.fits', overwrite=True)
