import configparser
from astropy import constants as const
from astropy.io import fits


def pp(self):
    config = configparser.ConfigParser()
    config.read(self)

    blue_cube = fits.open(config.get('APS_cube', 'file_dir') + config.get('QC_plots', 'blue_cube'))

    # =======
    # read redshift and input velocity and line flux
    z = float(config.get('pyp_params', 'redshift'))

    vel = float(config.get('pyp_params', 'line_vel'))
    line_flux = float(config.get('pyp_params', 'line_flux'))

    # =================== running for blue cube ===========================

    if int(config.get('pyp_params', 'blue_fit')) == 1:

        # =======
        # create stellar parameters file

        vel = round(vel + (const.c.to('km/s').value * z), 1)

        f = open("parameters_stellar_blue", "w+")

        f.write(
            'tmpldir            ' + config.get('pyp_params', 'temp_dir') + '          '
                                                                           '!Directory with template spec (string)\n')
        f.write(
            'tmplfile           ' + config.get('pyp_params', 'temp_file') + '                         '
                                                                            '!Template library fits file (string)\n')
        f.write(
            'tmplinitspec       10                                          '
            '!Number of the template spectrum as an inital guess (integer)\n')
        f.write('vel_guess          ' + str(vel) +
                '                                  !rough velocity guess for the object in km/s as cz (float)\n')
        f.write('vel_min            ' + str(
            vel - float(config.get('pyp_params', 'vel_range'))) + '                                  '
                                                                  '!minimum velocity in km/s (float)\n')
        f.write('vel_max            ' + str(
            vel + float(config.get('pyp_params', 'vel_range'))) + '                                  '
                                                                  '!maximum velocity in km/s (float)\n')
        f.write(
            'disp_min           10.0                                        '
            '!minimum velocity dispersion in km/s (float)\n')
        f.write('disp_max           ' + config.get('pyp_params', 'disp_max') +
                '                                       !maximum velocity  dispersion in km/s (float)\n')
        f.write(
            'kin_fix            0                                           '
            '!Keep velocity and dispersion fixed at input (0/1)\n')
        f.write(
            'kin_bootstrap      0                                           '
            '!Use kinematic uncertainties when performing bootstrapping (0/1)\n')
        f.write(
            'oversampling       2                                           '
            '!Oversampling in log-wavelength space (int)\n')
        f.write(
            'excl_fit           excl_blue.fit                                    '
            '!Exclude wavelength region during the fitting (string)\n')
        f.write('excl_cont          excl_blue.cont                                   '
                '!Exclude wavelength region during the continuum '
                'normalization (string)\n')
        f.write(
            'nwidth_norm        150                                         '
            '!Width of running mean in pixels for the normalization (int)\n')
        f.write('start_wave         ' + config.get('pyp_params', 'blue_lam_min') +
                '                                   !Lower wavelength limit for the stellar population fitting\n')
        start_wave = float(config.get('pyp_params', 'blue_lam_min'))
        f.write('end_wave           ' + config.get('pyp_params', 'blue_lam_max') +
                '                                   !Upper wavelength limit for the stellar population fitting\n')
        end_wave = float(config.get('pyp_params', 'blue_lam_max'))
        f.write('min_x              1                                           !Minimum x dimension (int)\n')
        f.write('max_x              10000                                       !Maximum x dimension (int)\n')
        f.write('min_y              1                                           !Minimum y dimension (int)\n')
        f.write('max_y              10000                                       !Maximum y dimension (int)\n')
        f.write(
            'mcmc_code          emcee                                       '
            '!The MCMC code to use for determining kinematics (emcee/pymc)\n')
        f.write(
            'iterations         2                                           '
            '!Iterations to establish kinematics and template library coefficients (int)\n')
        f.write(
            'walkers            20                                          '
            '!Number of independent MCMC walkers (int)\n')
        f.write(
            'samples            400                                         '
            '!Number of samples for the MCMC fit (int)\n')
        f.write(
            'burn               150                                         '
            '!Number of excluded samples as for the MCMC fit (int)\n')
        f.write('store_chain        0                                           !???\n')
        f.write(
            'thin               1                                           '
            '!Factor to reduce the total sample number (int)\n')
        f.write(
            'agebinfile         agebins.txt                                 !age bins for SFH output  (string)\n')
        f.write(
            'bootstrap_verb     0                                           !Save the coefficients of every '
            'bootstrap run, will produce large files (0/1)')

        f.close()

        # =======
        # create emission line parameters file

        f = open("parameters_eline_blue", "w+")

        f.write('eCompFile		par_blue.lines				!name of the line parameter file, None '
                'if no emission lines to be fitted\n')
        f.write(
            'vel_guess		' + str(
                vel) + '				!rough velocity guess for the object in km/s as cz (float)\n')
        f.write('line_fit_region	lines_blue.fit				'
                '!Wavelength regions considered during the fitting '
                '(string)\n')
        f.write('efit_method		leastsq					!method for line parameter fitting (leastsq/simplex)\n')
        f.write('efit_ftol		1e-8					!ftol convergence parameter\n')
        f.write('efit_xtol		1e-8					!xtol convergence parameter\n')
        f.write('eguess_window	20	       				!size of the spectral window in pixel to estimate initial '
                'guesses for each line\n')
        f.write('min_x           1                       !Minimum x dimension (int)\n')
        f.write('max_x           10000                   !Maximum x dimension (int)\n')
        f.write('min_y           1                       !Minimum y dimension (int)\n')
        f.write('max_y           10000                   !Maximum y dimension (int)')

        f.close()

        # =======
        # create par.lines and lines.fit
        f = open("lines_blue.fit", "w+")
        g = open("par_blue.lines", "w+")

        f.write('[rest-frame]\n')

        if (start_wave < 4851 * (1 + z)) & (end_wave > 4871 * (1 + z)):
            f.write('4851  4871 !Hb_4861\n')

            g.write('Gauss: Hb_4861\n')
            g.write('restwave 4861.32\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel ' + str(vel) + ' 1\n')
            g.write('disp 100 1\n')
            g.write('\n')

        if (start_wave < 4950 * (1 + z)) & (end_wave > 5017 * (1 + z)):
            f.write('4950  4970 !OIII_4960\n')
            f.write('4997  5017 !OIII_5007\n')

            g.write('Gauss: OIII_5007\n')
            g.write('restwave 5006.77\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel ' + str(vel) + ' 1\n')
            g.write('disp 100 1\n')
            g.write('\n')

            g.write('Gauss: OIII_4960\n')
            g.write('restwave 4958.83\n')
            g.write('flux OIII_5007:0.35\n')
            g.write('vel OIII_5007\n')
            g.write('disp OIII_5007\n')
            g.write('\n')

        if (start_wave < 3717 * (1 + z)) & (end_wave > 3737 * (1 + z)):
            f.write('3716  3736 !OII_3726\n')
            f.write('3718  3738 !OII_3728\n')

            g.write('Gauss: OII_3726\n')
            g.write('restwave 3726.03\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel ' + str(vel) + ' 1\n')
            g.write('disp 100 1\n')
            g.write('\n')

            g.write('Gauss: OII_3728\n')
            g.write('restwave 3728.73\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel OII_3726\n')
            g.write('disp OII_3726\n')
            g.write('\n')

        if (start_wave < 4353 * (1 + z)) & (end_wave > 4373 * (1 + z)):
            f.write('4353  4373 !OIII_4363\n')

            g.write('Gauss: OIII_4363\n')
            g.write('restwave 4363.15\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            if (start_wave < 4950 * (1 + z)) & (end_wave > 5017 * (1 + z)):
                g.write('vel OIII_5007\n')
                g.write('disp OIII_5007\n')
            else:
                g.write('vel ' + str(vel) + ' 1\n')
                g.write('disp 100 1\n')
            g.write('\n')

        f.close()
        g.close()

        # =======
        # create excl.cont
        h = open("excl_blue.cont", "w+")

        h.write('[rest-frame]\n')
        if z < 1.617:
            h.write('3712  3742\n')
        if z < 1.562:
            h.write('3850  3895\n')
        if z < 1.474:
            h.write('4080  4110\n')
        if z < 1.392:
            h.write('4320  4375\n')
        # if z < 1.307:
        # h.write('4600  4750\n')
        if z < 1.242:
            h.write('4840  4880\n')
        if z < 1.217:
            h.write('4940  4980\n')
        if z < 1.205:
            h.write('4990  5022\n')
        if z < 1.026:
            h.write('5856  5926\n')
        h.write('\n')

        h.write('[observed-frame]\n')
        if blue_cube[0].header['MODE'] == 'LOWRES':
            h.write('5470 5590    ! weave (blue gap + red part)\n')
        if blue_cube[0].header['MODE'] == 'HIGHRES':
            h.write('4690 4730    ! weave (blue border)\n')
            h.write('5330 5350    ! weave (blue gap)\n')  # should be 5300-5320
            h.write('5500 5510    ! weave (red border)\n')
        h.close()

        # =======
        # create excl.fit
        f = open("excl_blue.fit", "w+")

        f.write('[rest-frame]\n')
        f.write('3712 3742    !OII, H13\n')
        f.write('3850 3895    !NeIII, HeI, H8\n')
        f.write('4080 4110    !Hd\n')
        f.write('4320 4375    !Hg\n')
        # f.write('4600 4750    !blueWRbump\n')
        f.write('4840 4880    !Hb\n')
        f.write('4940 4980    !OIII\n')
        f.write('4990 5022    !OIII\n')
        f.write('5856 5926    !HeI, ?\n')

        f.write('\n')
        f.write('[observed-frame]\n')
        if blue_cube[0].header['MODE'] == 'LOWRES':
            f.write('5470 5590    ! weave (blue gap + red part)\n')
        if blue_cube[0].header['MODE'] == 'HIGHRES':
            f.write('4690 4730    ! weave (blue border)\n')
            f.write('5330 5350    ! weave (blue gap)\n')  # should be 5300-5320
            f.write('5500 5510    ! weave (red border)\n')
        f.close()

    # =================== running for red cube ===========================

    if int(config.get('pyp_params', 'red_fit')) == 1:

        vel = round(vel + (const.c.to('km/s').value * z), 1)

        # =======
        # create stellar parameters file

        f = open("parameters_stellar_red", "w+")

        f.write(
            'tmpldir            ' + config.get('pyp_params', 'temp_dir') + '          '
                                                                           '!Directory with template spec (string)\n')
        f.write(
            'tmplfile           ' + config.get('pyp_params', 'temp_file') + '                         '
                                                                            '!Template library fits file (string)\n')
        f.write(
            'tmplinitspec       10                                          '
            '!Number of the template spectrum as an inital guess (integer)\n')
        f.write('vel_guess          ' + str(vel) +
                '                                  !rough velocity guess for the object in km/s as cz (float)\n')
        f.write('vel_min            ' + str(vel - 200) + '                                  '
                                                         '!minimum velocity in km/s (float)\n')
        f.write('vel_max            ' + str(vel + 200) + '                                  '
                                                         '!maximum velocity in km/s (float)\n')
        f.write(
            'disp_min           40.0                                        '
            '!minimum velocity dispersion in km/s (float)\n')
        f.write('disp_max           ' + config.get('pyp_params', 'disp_max') +
                '                                       !maximum velocity  dispersion in km/s (float)\n')
        f.write(
            'kin_fix            0                                           '
            '!Keep velocity and dispersion fixed at input (0/1)\n')
        f.write(
            'kin_bootstrap      0                                           '
            '!Use kinematic uncertainties when performing bootstrapping (0/1)\n')
        f.write(
            'oversampling       2                                           '
            '!Oversampling in log-wavelength space (int)\n')
        f.write(
            'excl_fit           excl_red.fit                                    '
            '!Exclude wavelength region during the fitting (string)\n')
        f.write('excl_cont          excl_red.cont                                   '
                '!Exclude wavelength region during the continuum '
                'normalization (string)\n')
        f.write(
            'nwidth_norm        150                                         '
            '!Width of running mean in pixels for the normalization (int)\n')
        f.write('start_wave         ' + config.get('pyp_params', 'red_lam_min') +
                '                                   !Lower wavelength limit for the stellar population fitting\n')
        start_wave = float(config.get('pyp_params', 'red_lam_min'))
        f.write('end_wave           ' + config.get('pyp_params', 'red_lam_max') +
                '                                   !Upper wavelength limit for the stellar population fitting\n')
        end_wave = float(config.get('pyp_params', 'red_lam_max'))
        f.write('min_x              1                                           !Minimum x dimension (int)\n')
        f.write('max_x              10000                                       !Maximum x dimension (int)\n')
        f.write('min_y              1                                           !Minimum y dimension (int)\n')
        f.write('max_y              10000                                       !Maximum y dimension (int)\n')
        f.write(
            'mcmc_code          emcee                                       '
            '!The MCMC code to use for determining kinematics (emcee/pymc)\n')
        f.write(
            'iterations         2                                           '
            '!Iterations to establish kinematics and template library coefficients (int)\n')
        f.write(
            'walkers            20                                          '
            '!Number of independent MCMC walkers (int)\n')
        f.write(
            'samples            400                                         '
            '!Number of samples for the MCMC fit (int)\n')
        f.write(
            'burn               150                                         '
            '!Number of excluded samples as for the MCMC fit (int)\n')
        f.write('store_chain        0                                           !???\n')
        f.write(
            'thin               1                                           '
            '!Factor to reduce the total sample number (int)\n')
        f.write(
            'agebinfile         agebins.txt                                 !age bins for SFH output  (string)\n')
        f.write(
            'bootstrap_verb     0                                           !Save the coefficients of every '
            'bootstrap run, will produce large files (0/1)')

        f.close()

        # =======
        # create emission line parameters file

        f = open("parameters_eline_red", "w+")

        f.write('eCompFile		par_red.lines				!name of the line parameter file, None '
                'if no emission lines to be fitted\n')
        f.write(
            'vel_guess		' + str(
                vel) + '				!rough velocity guess for the object in km/s as cz (float)\n')
        f.write('line_fit_region	lines_red.fit				'
                '!Wavelength regions considered during the fitting '
                '(string)\n')
        f.write('efit_method		leastsq					!method for line parameter fitting (leastsq/simplex)\n')
        f.write('efit_ftol		1e-8					!ftol convergence parameter\n')
        f.write('efit_xtol		1e-8					!xtol convergence parameter\n')
        f.write('eguess_window	20	       				!size of the spectral window in pixel to estimate initial '
                'guesses for each line\n')
        f.write('min_x           1                       !Minimum x dimension (int)\n')
        f.write('max_x           10000                   !Maximum x dimension (int)\n')
        f.write('min_y           1                       !Minimum y dimension (int)\n')
        f.write('max_y           10000                   !Maximum y dimension (int)')

        f.close()

        # =======
        # create par.lines and lines.fit
        f = open("lines_red.fit", "w+")
        g = open("par_red.lines", "w+")

        f.write('[rest-frame]\n')

        if (start_wave < 6538 * (1 + z)) & (end_wave > 6558 * (1 + z)):
            f.write('6573  6593 !NII_6583\n')
            f.write('6553  6573 !Ha_6562\n')
            f.write('6538  6558 !NII_6548\n')

            g.write('Gauss: Ha_6562\n')
            g.write('restwave 6562.80\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel ' + str(vel) + ' 1\n')
            g.write('disp 100 1\n')
            g.write('\n')

            g.write('Gauss: NII_6583\n')
            g.write('restwave 6583.34\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel Ha_6562\n')
            g.write('disp Ha_6562\n')
            g.write('\n')

            g.write('Gauss: NII_6548\n')
            g.write('restwave 6547.96\n')
            g.write('flux NII_6583:0.33\n')
            g.write('vel Ha_6562\n')
            g.write('disp Ha_6562\n')
            g.write('\n')

        if (start_wave < 6707 * (1 + z)) & (end_wave > 6740 * (1 + z)):
            f.write('6707  6727 !SII_6716\n')
            f.write('6720  6740 !SII_6730\n')

            g.write('Gauss: SII_6716\n')
            g.write('restwave 6716.31\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel ' + str(vel) + ' 1\n')
            g.write('disp 100 1\n')
            g.write('\n')

            g.write('Gauss: SII_6730\n')
            g.write('restwave 6730.68\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel SII_6716\n')
            g.write('disp SII_6716\n')
            g.write('\n')

        if (start_wave < 6290 * (1 + z)) & (end_wave > 6374 * (1 + z)):
            f.write('6290  6310 !OI_6300\n')
            f.write('6354  6374 !OI_6364\n')

            g.write('Gauss: OI_6300\n')
            g.write('restwave 6300.20\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel ' + str(vel) + ' 1\n')
            g.write('disp 100 1\n')
            g.write('\n')

            g.write('Gauss: OI_6364\n')
            g.write('restwave 6363.67\n')
            g.write('flux OI_6300:0.33\n')
            g.write('vel OI_6300\n')
            g.write('disp OI_6300\n')
            g.write('\n')

        f.close()
        g.close()

        # =======
        # create excl.cont
        h = open("excl_red.cont", "w+")

        h.write('[rest-frame]\n')

        if (z > 1.559) & (z < 2.562):
            h.write('3712  3742\n')
        if (z > 1.503) & (z < 2.462):
            h.write('3850  3895\n')
        if (z > 1.419) & (z < 2.333):
            h.write('4080  4110\n')
        if (z > 1.340) & (z < 2.192):
            h.write('4320  4375\n')
        if (z > 1.258) & (z < 2.018):
            h.write('4600  4750\n')
        if (z > 1.196) & (z < 1.965):
            h.write('4840  4880\n')
        if (z > 1.172) & (z < 1.925):
            h.write('4940  4980\n')
        if (z > 1.160) & (z < 1.909):
            h.write('4990  5022\n')
        if z < 1.639:
            h.write('5856  5926\n')
        if z < 1.527:
            h.write('6280  6320\n')
        if z < 1.471:
            h.write('6520  6605\n')
        if z < 1.420:
            h.write('6690  6750\n')
        if z < 1.341:
            h.write('7050  7150\n')
        if z < 1.297:
            h.write('7200  7390\n')
        h.write('\n')

        h.write('[observed-frame]\n')
        h.write('6860 6890    ! telluric lines\n')
        if blue_cube[0].header['MODE'] == 'LOWRES':
            h.write('7560 7710    ! weave (red gap)\n')
        if blue_cube[0].header['MODE'] == 'HIGHRES':
            h.write('5930 5975    ! weave (blue border)\n')
            h.write('6410 6440    ! weave (red gap)\n')
            h.write('5930 5975    ! weave (red border)\n')
        h.write('7950 8500    ! sky lines\n')
        h.write('8700 8870    ! sky lines\n')

        h.close()

        # =======
        # create excl.fit
        f = open("excl_red.fit", "w+")

        f.write('[rest-frame]\n')
        f.write('5856 5926    !HeI, ?\n')
        f.write('6280 6320    !OI\n')
        f.write('6520 6605    !NII, Ha\n')
        f.write('6690 6750    !SII\n')
        f.write('7050 7150    !HeI\n')
        f.write('7200 7390    !OII\n')

        f.write('\n')
        f.write('[observed-frame]\n')
        f.write('6860 6890    ! telluric lines\n')
        if blue_cube[0].header['MODE'] == 'LOWRES':
            f.write('7560 7710    ! weave (red gap)\n')
        if blue_cube[0].header['MODE'] == 'HIGHRES':
            f.write('5930 5975    ! weave (blue border)\n')
            f.write('6410 6440    ! weave (red gap)\n')
            f.write('5930 5975    ! weave (red border)\n')
        f.write('7950 8500    ! sky lines\n')
        f.write('8700 8870    ! sky lines\n')

        f.close()

    # =================== running for APS ===========================

    if int(config.get('pyp_params', 'aps_fit')) == 1:

        # =======
        # create stellar parameters file

        f = open("parameters_stellar_aps", "w+")

        f.write(
            'tmpldir            ' + config.get('pyp_params', 'temp_dir') + '          '
                                                                           '!Directory with template spec (string)\n')
        f.write(
            'tmplfile           ' + config.get('pyp_params', 'temp_file') + '                         '
                                                                            '!Template library fits file (string)\n')
        f.write(
            'tmplinitspec       10                                          '
            '!Number of the template spectrum as an inital guess (integer)\n')
        f.write('vel_guess          ' + str(vel) +
                '                                  !rough velocity guess for the object in km/s as cz (float)\n')
        f.write('vel_min            ' + str(vel - 200) + '                                  '
                                                         '!minimum velocity in km/s (float)\n')
        f.write('vel_max            ' + str(vel + 200) + '                                  '
                                                         '!maximum velocity in km/s (float)\n')
        f.write(
            'disp_min           40.0                                        '
            '!minimum velocity dispersion in km/s (float)\n')
        f.write('disp_max           ' + config.get('pyp_params', 'disp_max') +
                '                                       !maximum velocity  dispersion in km/s (float)\n')
        f.write(
            'kin_fix            0                                           '
            '!Keep velocity and dispersion fixed at input (0/1)\n')
        f.write(
            'kin_bootstrap      0                                           '
            '!Use kinematic uncertainties when performing bootstrapping (0/1)\n')
        f.write(
            'oversampling       2                                           '
            '!Oversampling in log-wavelength space (int)\n')
        f.write(
            'excl_fit           excl_aps.fit                                    '
            '!Exclude wavelength region during the fitting (string)\n')
        f.write('excl_cont          excl_aps.cont                                   '
                '!Exclude wavelength region during the continuum '
                'normalization (string)\n')
        f.write(
            'nwidth_norm        150                                         '
            '!Width of running mean in pixels for the normalization (int)\n')
        f.write('start_wave         ' + config.get('pyp_params', 'aps_lam_min') +
                '                                   !Lower wavelength limit for the stellar population fitting\n')
        start_wave = float(config.get('pyp_params', 'aps_lam_min'))
        f.write('end_wave           ' + config.get('pyp_params', 'aps_lam_max') +
                '                                   !Upper wavelength limit for the stellar population fitting\n')
        end_wave = float(config.get('pyp_params', 'aps_lam_max'))
        f.write('min_x              1                                           !Minimum x dimension (int)\n')
        f.write('max_x              10000                                       !Maximum x dimension (int)\n')
        f.write('min_y              1                                           !Minimum y dimension (int)\n')
        f.write('max_y              10000                                       !Maximum y dimension (int)\n')
        f.write(
            'mcmc_code          emcee                                       '
            '!The MCMC code to use for determining kinematics (emcee/pymc)\n')
        f.write(
            'iterations         2                                           '
            '!Iterations to establish kinematics and template library coefficients (int)\n')
        f.write(
            'walkers            20                                          '
            '!Number of independent MCMC walkers (int)\n')
        f.write(
            'samples            400                                         '
            '!Number of samples for the MCMC fit (int)\n')
        f.write(
            'burn               150                                         '
            '!Number of excluded samples as for the MCMC fit (int)\n')
        f.write('store_chain        0                                           !???\n')
        f.write(
            'thin               1                                           '
            '!Factor to reduce the total sample number (int)\n')
        f.write(
            'agebinfile         agebins.txt                                 !age bins for SFH output  (string)\n')
        f.write(
            'bootstrap_verb     0                                           !Save the coefficients of every '
            'bootstrap run, will produce large files (0/1)')

        f.close()

        # =======
        # create emission line parameters file

        f = open("parameters_eline_aps", "w+")

        f.write('eCompFile		par_aps.lines				!name of the line parameter file, None '
                'if no emission lines to be fitted\n')
        f.write(
            'vel_guess		' + str(
                vel) + '				!rough velocity guess for the object in km/s as cz (float)\n')
        f.write('line_fit_region	lines_aps.fit				'
                '!Wavelength regions considered during the fitting '
                '(string)\n')
        f.write('efit_method		leastsq					!method for line parameter fitting (leastsq/simplex)\n')
        f.write('efit_ftol		1e-8					!ftol convergence parameter\n')
        f.write('efit_xtol		1e-8					!xtol convergence parameter\n')
        f.write('eguess_window	20	       				!size of the spectral window in pixel to estimate initial '
                'guesses for each line\n')
        f.write('min_x           1                       !Minimum x dimension (int)\n')
        f.write('max_x           10000                   !Maximum x dimension (int)\n')
        f.write('min_y           1                       !Minimum y dimension (int)\n')
        f.write('max_y           10000                   !Maximum y dimension (int)')

        f.close()

        # =======
        # create par.lines and lines.fit
        f = open("lines_aps.fit", "w+")
        g = open("par_aps.lines", "w+")

        f.write('[rest-frame]\n')

        if (start_wave < 6538) & (end_wave > 6558):
            f.write('6553  6573 !Ha_6562\n')
            f.write('6573  6593 !NII_6583\n')
            f.write('6538  6558 !NII_6548\n')

            g.write('Gauss: Ha_6562\n')
            g.write('restwave 6562.80\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel ' + str(vel) + ' 1\n')
            g.write('disp 100 1\n')
            g.write('\n')

            g.write('Gauss: NII_6583\n')
            g.write('restwave 6583.34\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel Ha_6562\n')
            g.write('disp Ha_6562\n')
            g.write('\n')

            g.write('Gauss: NII_6548\n')
            g.write('restwave 6547.96\n')
            g.write('flux NII_6583:0.33\n')
            g.write('vel Ha_6562\n')
            g.write('disp Ha_6562\n')
            g.write('\n')

        if (start_wave < 4851) & (end_wave > 4871):
            f.write('4851  4871 !Hb_4861\n')

            g.write('Gauss: Hb_4861\n')
            g.write('restwave 4861.32\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            if (start_wave < 6538 * (1 + z)) & (end_wave > 6558 * (1 + z)):
                g.write('vel Ha_6562\n')
                g.write('disp Ha_6562\n')
            else:
                g.write('vel ' + str(vel) + ' 1\n')
                g.write('disp 100 1\n')
            g.write('\n')

        if (start_wave < 6707) & (end_wave > 6740):
            f.write('6707  6727 !SII_6716\n')
            f.write('6720  6740 !SII_6730\n')

            g.write('Gauss: SII_6716\n')
            g.write('restwave 6716.31\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel ' + str(vel) + ' 1\n')
            g.write('disp 100 1\n')
            g.write('\n')

            g.write('Gauss: SII_6730\n')
            g.write('restwave 6730.68\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel SII_6716\n')
            g.write('disp SII_6716\n')
            g.write('\n')

        if (start_wave < 4950) & (end_wave > 5017):
            f.write('4950  4970 !OIII_4960\n')
            f.write('4997  5017 !OIII_5007\n')

            g.write('Gauss: OIII_5007\n')
            g.write('restwave 5006.77\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel ' + str(vel) + ' 1\n')
            g.write('disp 100 1\n')
            g.write('\n')

            g.write('Gauss: OIII_4960\n')
            g.write('restwave 4958.83\n')
            g.write('flux OIII_5007:0.35\n')
            g.write('vel OIII_5007\n')
            g.write('disp OIII_5007\n')
            g.write('\n')

        if (start_wave < 3717) & (end_wave > 3737):
            f.write('3716  3736 !OII_3726\n')
            f.write('3718  3738 !OII_3728\n')

            g.write('Gauss: OII_3726\n')
            g.write('restwave 3726.03\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel ' + str(vel) + ' 1\n')
            g.write('disp 100 1\n')
            g.write('\n')

            g.write('Gauss: OII_3728\n')
            g.write('restwave 3728.73\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel OII_3726\n')
            g.write('disp OII_3726\n')
            g.write('\n')

        if (start_wave < 6290) & (end_wave > 6374):
            f.write('6290  6310 !OI_6300\n')
            f.write('6354  6374 !OI_6364\n')

            g.write('Gauss: OI_6300\n')
            g.write('restwave 6300.20\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            g.write('vel ' + str(vel) + ' 1\n')
            g.write('disp 100 1\n')
            g.write('\n')

            g.write('Gauss: OI_6364\n')
            g.write('restwave 6363.67\n')
            g.write('flux OI_6300:0.33\n')
            g.write('vel OI_6300\n')
            g.write('disp OI_6300\n')
            g.write('\n')

        if (start_wave < 4353) & (end_wave > 4373):
            f.write('4353  4373 !OIII_4363\n')

            g.write('Gauss: OIII_4363\n')
            g.write('restwave 4363.15\n')
            g.write('flux ' + str(line_flux) + ' 1\n')
            if (start_wave < 4950) & (end_wave > 5017):
                g.write('vel OIII_5007\n')
                g.write('disp OIII_5007\n')
            else:
                g.write('vel ' + str(vel) + ' 1\n')
                g.write('disp 100 1\n')
            g.write('\n')

        # =======
        # create excl.cont
        h = open("excl_aps.cont", "w+")

        h.write('[rest-frame]\n')

        if z < 1.517:
            h.write('3712  3742\n')
        if z < 1.413:
            h.write('3850  3895\n')
        if z < 1.287:
            h.write('4080  4110\n')
        if z < 1.148:
            h.write('4320  4375\n')
        if z < 0.979:
            h.write('4600  4750\n')
        if z < 0.926:
            h.write('4840  4880\n')
        if z < 0.887:
            h.write('4940  4980\n')
        if z < 0.872:
            h.write('4990  5022\n')
        if z < 0.586:
            h.write('5856  5926\n')
        if z < 0.487:
            h.write('6280  6320\n')
        if z < 0.423:
            h.write('6520  6605\n')
        if z < 0.392:
            h.write('6690  6750\n')
        if z < 0.315:
            h.write('7050  7150\n')
        if z < 0.272:
            h.write('7200  7390\n')
        h.write('\n')

        h.write('[observed-frame]\n')
        if blue_cube[0].header['MODE'] == 'LOWRES':
            h.write(
                str(round(5470. / (1 + z))) + ' ' + str(round(5590. / (1 + z))) + '    ! weave (blue gap + red part)\n')
        if blue_cube[0].header['MODE'] == 'HIGHRES':
            h.write(str(round(5330. / (1 + z))) + ' ' + str(
                round(5350. / (1 + z))) + '    ! weave (blue gap)\n')  # should be 5300-5320
        h.write(str(round(6860. / (1 + z))) + ' ' + str(round(6890. / (1 + z))) + '    ! telluric lines\n')
        if blue_cube[0].header['MODE'] == 'LOWRES':
            h.write(str(round(7560. / (1 + z))) + ' ' + str(round(7710. / (1 + z))) + '    ! weave (red gap)\n')
        if blue_cube[0].header['MODE'] == 'HIGHRES':
            h.write(str(round(6410. / (1 + z))) + ' ' + str(round(6440. / (1 + z))) + '    ! weave (red gap)\n')
        h.write(str(round(7950. / (1 + z))) + ' ' + str(round(8500. / (1 + z))) + '    ! sky lines\n')
        h.write(str(round(8700. / (1 + z))) + ' ' + str(round(8870. / (1 + z))) + '    ! sky lines\n')

        h.close()

        # =======
        # create excl.fit
        f = open("excl_aps.fit", "w+")

        f.write('[rest-frame]\n')
        f.write('3712 3742    !OII, H13\n')
        f.write('3850 3895    !NeIII, HeI, H8\n')
        f.write('4080 4110    !Hd\n')
        f.write('4320 4375    !Hg\n')
        f.write('4600 4750    !blueWRbump\n')
        f.write('4840 4880    !Hb\n')
        f.write('4940 4980    !OIII\n')
        f.write('4990 5022    !OIII\n')
        f.write('5856 5926    !HeI, ?\n')
        f.write('6280 6320    !OI\n')
        f.write('6520 6605    !NII, Ha\n')
        f.write('6690 6750    !SII\n')
        f.write('7050 7150    !HeI\n')
        f.write('7200 7390    !OII\n')

        f.write('\n')
        f.write('[observed-frame]\n')
        if blue_cube[0].header['MODE'] == 'LOWRES':
            f.write(
                str(round(5470. / (1 + z))) + ' ' + str(round(5590. / (1 + z))) + '    ! weave (blue gap + red part)\n')
        if blue_cube[0].header['MODE'] == 'HIGHRES':
            f.write(str(round(5330. / (1 + z))) + ' ' + str(
                round(5350. / (1 + z))) + '    ! weave (blue gap)\n')  # should be 5300-5320
        f.write(str(round(6860. / (1 + z))) + ' ' + str(round(6890. / (1 + z))) + '    ! telluric lines\n')
        if blue_cube[0].header['MODE'] == 'LOWRES':
            f.write(str(round(7560. / (1 + z))) + ' ' + str(round(7710. / (1 + z))) + '    ! weave (red gap)\n')
        if blue_cube[0].header['MODE'] == 'HIGHRES':
            f.write(str(round(6410. / (1 + z))) + ' ' + str(round(6440. / (1 + z))) + '    ! weave (red gap)\n')
        f.write(str(round(7950. / (1 + z))) + ' ' + str(round(8500. / (1 + z))) + '    ! sky lines\n')
        f.write(str(round(8700. / (1 + z))) + ' ' + str(round(8870. / (1 + z))) + '    ! sky lines\n')

        f.close()
