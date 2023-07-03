from astropy.io import fits
from astropy import constants as const
import configparser
import os


def pp(self):
    config = configparser.ConfigParser()
    config.read(self)

    # estimate source systemic velocity from the redshift

    z = float(config.get('pyp_params', 'redshift'))

    vel = float(config.get('pyp_params', 'line_vel'))
    line_flux = float(config.get('pyp_params', 'line_flux'))
    # vel = const.c.to('km/s').value * z
    # vel_el = const.c.to('km/s').value * ((((1 + z) ** 2) - 1) / (((1 + z) ** 2) + 1))

    # =================== running for APS

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
    f.write('start_wave         ' + config.get('pyp_params', 'lam_min') +
            '                                   !Lower wavelength limit for the stellar population fitting\n')
    f.write('end_wave           ' + config.get('pyp_params', 'lam_max') +
            '                                   !Upper wavelength limit for the stellar population fitting\n')
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

    # create emission line parameters file

    f = open("parameters_eline_aps", "w+")

    f.write('eCompFile		par_aps.lines				!name of the line parameter file, None '
            'if no emission lines to be fitted\n')
    f.write(
        'vel_guess		' + str(vel) + '				!rough velocity guess for the object in km/s as cz (float)\n')
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

    # create par.lines and lines.fit for each object

    f = open("lines_aps.fit", "w+")
    g = open("par_aps.lines", "w+")

    f.write('[rest-frame]\n')

    # if z > 1.522:
    #     f.write('3717  3737 !OII3727\n')
    #
    #     g.write('Gauss: OII3727\n')
    #     g.write('restwave 3727.00\n')
    #     g.write('flux ' + str(line_flux) + ' 1\n')
    #     g.write('vel ' + str(vel) + ' 1\n')
    #     g.write('disp 100 1\n')
    #     g.write('\n')

    # if (z < 1.506):

    # f.write('3740  3760 !H12\n')

    # g.write('Gauss: H12\n')
    # g.write('restwave 3750.15\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.428):
    # g.write('vel Halpha\n')
    # g.write('disp Halpha\n')
    # elif (z < 0.934):
    # g.write('vel Hbeta\n')
    # g.write('disp Hbeta\n')
    # elif (z < 1.165):
    # g.write('vel Hgamma\n')
    # g.write('disp Hgamma\n')
    # elif (z < 1.291):
    # g.write('vel Hdelta\n')
    # g.write('disp Hdelta\n')
    # elif (z < 1.367):
    # g.write('vel H7\n')
    # g.write('disp H7\n')
    # elif (z < 1.417):
    # g.write('vel H8HeI\n')
    # g.write('disp H8HeI\n')
    # elif (z < 1.451):
    # g.write('vel H9\n')
    # g.write('disp H9\n')
    # elif (z < 1.475):
    # g.write('vel H10\n')
    # g.write('disp H10\n')
    # elif (z < 1.492):
    # g.write('vel H11\n')
    # g.write('disp H11\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.492):

    # f.write('3760  3780 !H11\n')

    # g.write('Gauss: H11\n')
    # g.write('restwave 3770.63\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.428):
    # g.write('vel Halpha\n')
    # g.write('disp Halpha\n')
    # elif (z < 0.934):
    # g.write('vel Hbeta\n')
    # g.write('disp Hbeta\n')
    # elif (z < 1.165):
    # g.write('vel Hgamma\n')
    # g.write('disp Hgamma\n')
    # elif (z < 1.291):
    # g.write('vel Hdelta\n')
    # g.write('disp Hdelta\n')
    # elif (z < 1.367):
    # g.write('vel H7\n')
    # g.write('disp H7\n')
    # elif (z < 1.417):
    # g.write('vel H8HeI\n')
    # g.write('disp H8HeI\n')
    # elif (z < 1.451):
    # g.write('vel H9\n')
    # g.write('disp H9\n')
    # elif (z < 1.475):
    # g.write('vel H10\n')
    # g.write('disp H10\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.475):

    # f.write('3788  3808 !H10\n')

    # g.write('Gauss: H10\n')
    # g.write('restwave 3797.92\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.428):
    # g.write('vel Halpha\n')
    # g.write('disp Halpha\n')
    # elif (z < 0.934):
    # g.write('vel Hbeta\n')
    # g.write('disp Hbeta\n')
    # elif (z < 1.165):
    # g.write('vel Hgamma\n')
    # g.write('disp Hgamma\n')
    # elif (z < 1.291):
    # g.write('vel Hdelta\n')
    # g.write('disp Hdelta\n')
    # elif (z < 1.367):
    # g.write('vel H7\n')
    # g.write('disp H7\n')
    # elif (z < 1.417):
    # g.write('vel H8HeI\n')
    # g.write('disp H8HeI\n')
    # elif (z < 1.451):
    # g.write('vel H9\n')
    # g.write('disp H9\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.451):

    # f.write('3825  3845 !H9\n')

    # g.write('Gauss: H9\n')
    # g.write('restwave 3835.39\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.428):
    # g.write('vel Halpha\n')
    # g.write('disp Halpha\n')
    # elif (z < 0.934):
    # g.write('vel Hbeta\n')
    # g.write('disp Hbeta\n')
    # elif (z < 1.165):
    # g.write('vel Hgamma\n')
    # g.write('disp Hgamma\n')
    # elif (z < 1.291):
    # g.write('vel Hdelta\n')
    # g.write('disp Hdelta\n')
    # elif (z < 1.367):
    # g.write('vel H7\n')
    # g.write('disp H7\n')
    # elif (z < 1.417):
    # g.write('vel H8HeI\n')
    # g.write('disp H8HeI\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.429):

    # f.write('3859  3879 !NeIII3869\n')

    # g.write('Gauss: NeIII3869\n')
    # g.write('restwave 3868.75\n')
    # g.write('flux 10.0 1\n')
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.417):

    # f.write('3879  3899 !H8HeI\n')

    # g.write('Gauss: H8HeI\n')
    # g.write('restwave 3888.85\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.428):
    # g.write('vel Halpha\n')
    # g.write('disp Halpha\n')
    # elif (z < 0.934):
    # g.write('vel Hbeta\n')
    # g.write('disp Hbeta\n')
    # elif (z < 1.165):
    # g.write('vel Hgamma\n')
    # g.write('disp Hgamma\n')
    # elif (z < 1.291):
    # g.write('vel Hdelta\n')
    # g.write('disp Hdelta\n')
    # elif (z < 1.367):
    # g.write('vel H7\n')
    # g.write('disp H7\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.367):

    # f.write('3960  3980 !H7\n')

    # g.write('Gauss: H7\n')
    # g.write('restwave 3970.07\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.428):
    # g.write('vel Halpha\n')
    # g.write('disp Halpha\n')
    # elif (z < 0.934):
    # g.write('vel Hbeta\n')
    # g.write('disp Hbeta\n')
    # elif (z < 1.165):
    # g.write('vel Hgamma\n')
    # g.write('disp Hgamma\n')
    # elif (z < 1.291):
    # g.write('vel Hdelta\n')
    # g.write('disp Hdelta\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.334):

    # f.write('4016  4036 !HeI4026\n')

    # g.write('Gauss: HeI4026\n')
    # g.write('restwave 4026.21\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.600):
    # g.write('vel HeI5876\n')
    # g.write('disp HeI5876\n')
    # elif (z < 0.910):
    # g.write('vel HeI4922\n')
    # g.write('disp HeI4922\n')
    # elif (z < 1.102):
    # g.write('vel HeI4472\n')
    # g.write('disp HeI4472\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.310):

    # f.write('4058  4078 !SII4068\n')

    # g.write('Gauss: SII4068\n')
    # g.write('restwave 4068.60\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.396):
    # g.write('vel SII6717\n')
    # g.write('disp SII6717\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.291):

    # f.write('4092  4112 !Hdelta\n')

    # g.write('Gauss: Hdelta\n')
    # g.write('restwave 4101.74\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.428):
    # g.write('vel Halpha\n')
    # g.write('disp Halpha\n')
    # elif (z < 0.934):
    # g.write('vel Hbeta\n')
    # g.write('disp Hbeta\n')
    # elif (z < 1.165):
    # g.write('vel Hgamma\n')
    # g.write('disp Hgamma\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.165):

    # f.write('4330  4350 !Hgamma\n')

    # g.write('Gauss: Hgamma\n')
    # g.write('restwave 4340.47\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.428):
    # g.write('vel Halpha\n')
    # g.write('disp Halpha\n')
    # elif (z < 0.934):
    # g.write('vel Hbeta\n')
    # g.write('disp Hbeta\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.154):

    # f.write('4353  4373 !OIII4363\n')

    # g.write('Gauss: OIII4363\n')
    # g.write('restwave 4363.12\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.877):
    # g.write('vel OIII5007\n')
    # g.write('disp OIII5007\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.102):

    # f.write('4462  4482 !HeI4472\n')

    # g.write('Gauss: HeI4472\n')
    # g.write('restwave 4471.50\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.600):
    # g.write('vel HeI5876\n')
    # g.write('disp HeI5876\n')
    # elif (z < 0.910):
    # g.write('vel HeI4922\n')
    # g.write('disp HeI4922\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.018):

    # f.write('4648  4668 !FeIII4658\n')

    # g.write('Gauss: FeIII4658\n')
    # g.write('restwave 4658.00\n')
    # g.write('flux 10.0 1\n')
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 1.006):

    # f.write('4676  4696 !HeII4686\n')

    # g.write('Gauss: HeII4686\n')
    # g.write('restwave 4686.00\n')
    # g.write('flux 10.0 1\n')
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    f.write('4851  4871 !Hbeta\n')

    g.write('Gauss: Hbeta\n')
    g.write('restwave 4861.33\n')
    g.write('flux ' + str(line_flux) + ' 1\n')
    if z < 0.428:
        g.write('vel Halpha\n')
        g.write('disp Halpha\n')
    else:
        g.write('vel ' + str(vel) + ' 1\n')
        g.write('disp 100 1\n')
    g.write('\n')

    # if (z < 0.910):

    # f.write('4912  4932 !HeI4922\n')

    # g.write('Gauss: HeI4922\n')
    # g.write('restwave 4921.93\n')
    # g.write('flux 10.0 1\n')
    # if (z < 0.600):
    # g.write('vel HeI5876\n')
    # g.write('disp HeI5876\n')
    # else:
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    f.write('4950  4970 !OIII4960\n')
    f.write('4997  5017 !OIII5007\n')

    g.write('Gauss: OIII4960\n')
    g.write('restwave 4958.9\n')
    g.write('flux OIII5007:0.33\n')
    g.write('vel OIII5007\n')
    g.write('disp OIII5007\n')
    g.write('\n')

    g.write('Gauss: OIII5007\n')
    g.write('restwave 5006.84\n')
    g.write('flux ' + str(line_flux) + ' 1\n')
    g.write('vel ' + str(vel) + ' 1\n')
    g.write('disp 100 1\n')
    g.write('\n')

    # if (z < 0.783):

    # f.write('5260  5280 !FeIII5270\n')

    # g.write('Gauss: FeIII5270\n')
    # g.write('restwave 5270.40\n')
    # g.write('flux 10.0 1\n')
    # g.write('vel FeIII4658\n')
    # g.write('disp FeIII4658\n')
    # g.write('\n')

    # if (z < 0.600):

    # f.write('5866  5886 !HeI5876\n')

    # g.write('Gauss: HeI5876\n')
    # g.write('restwave 5875.67\n')
    # g.write('flux 10.0 1\n')
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 0.492):

    # f.write('6290  6310 !OI6300\n')

    # g.write('Gauss: OI6300\n')
    # g.write('restwave 6300.3\n')
    # g.write('flux 10.0 1\n')
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    f.write('6538  6558 !NII6548\n')
    f.write('6553  6573 !Halpha\n')
    f.write('6573  6593 !NII6583\n')

    g.write('Gauss: NII6548\n')
    g.write('restwave 6548.05\n')
    g.write('flux NII6583:0.33\n')
    g.write('vel NII6583\n')
    g.write('disp NII6583\n')
    g.write('\n')

    g.write('Gauss: Halpha\n')
    g.write('restwave 6562.80\n')
    g.write('flux ' + str(line_flux) + ' 1\n')
    g.write('vel ' + str(vel) + ' 1\n')
    g.write('disp 100 1\n')
    g.write('\n')

    g.write('Gauss: NII6583\n')
    g.write('restwave 6583.45\n')
    g.write('flux ' + str(line_flux) + ' 1\n')
    g.write('vel ' + str(vel) + ' 1\n')
    g.write('disp 100 1\n')
    g.write('\n')

    # if (z < 0.407):

    # f.write('6668  6688 !HeI6678\n')

    # g.write('Gauss: HeI6678\n')
    # g.write('restwave 6678.15\n')
    # g.write('flux 10.0 1\n')
    # g.write('vel HeI5876\n')
    # g.write('disp HeI5876\n')
    # g.write('\n')

    f.write('6707  6727 !SII6717\n')
    f.write('6720  6740 !SII6730\n')

    g.write('Gauss: SII6717\n')
    g.write('restwave 6716.44\n')
    g.write('flux ' + str(line_flux) + ' 1\n')
    g.write('vel ' + str(vel) + ' 1\n')
    g.write('disp 100 1\n')
    g.write('\n')

    g.write('Gauss: SII6730\n')
    g.write('restwave 6730.81\n')
    g.write('flux ' + str(line_flux) + ' 1\n')
    g.write('vel ' + str(vel) + ' 1\n')
    g.write('disp 100 1\n')
    g.write('\n')

    # if (z < 0.330):

    # f.write('7055  7075 !HeI7065\n')

    # g.write('Gauss: HeI7065\n')
    # g.write('restwave 7065.25\n')
    # g.write('flux 10.0 1\n')
    # g.write('vel HeI5876\n')
    # g.write('disp HeI5876\n')
    # g.write('\n')

    # if (z < 0.317):

    # f.write('7125  7145 !ArIII7135\n')

    # g.write('Gauss: ArIII7135\n')
    # g.write('restwave 7135.80\n')
    # g.write('flux 10.0 1\n')
    # g.write('vel '+str(vel)+' 1\n')
    # g.write('disp 100 1\n')
    # g.write('\n')

    # if (z < 0.291):

    # f.write('7271  7291 !HeI7281\n')

    # g.write('Gauss: HeI7281\n')
    # g.write('restwave 7281.35\n')
    # g.write('flux 10.0 1\n')
    # g.write('vel HeI5876\n')
    # g.write('disp HeI5876\n')
    # g.write('\n')

    # if (z < 0.282):

    # f.write('7309  7329 !OII7319\n')
    # f.write('7320  7340 !OII7330\n')

    # g.write('Gauss: OII7330\n')
    # g.write('restwave 7330.20\n')
    # g.write('flux 10.0 1\n')
    # g.write('vel OII3727\n')
    # g.write('disp OII3727\n')
    # g.write('\n')

    # g.write('Gauss: OII7319\n')
    # g.write('restwave 7319.46\n')
    # g.write('flux 10.0 1\n')
    # g.write('vel OII3727\n')
    # g.write('disp OII3727\n')
    # g.write('\n')

    f.close()
    g.close()

    h = open("excl_aps.cont", "w+")

    h.write('[rest-frame]\n')

    if z < 1.517:
        h.write('3712  3742\n')

    if z < 1.413:
        h.write('3850  3895\n')

    # if z < 1.383:
    #     h.write('3920  3945\n')
    #
    # if z < 1.362:
    #     h.write('3955  3980\n')

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

    # if z < 0.801:
    #     h.write('5183  5220\n')

    # if z < 0.636:
    #     h.write('5695  5745\n')

    if z < 0.586:
        h.write('5856  5926\n')

    # if z < 0.538:
    #     h.write('6065  6110\n')

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
    h.write(str(round(5470. / (1 + z))) + ' ' + str(round(5590. / (1 + z))) + '    ! weave (blue gap + red part)\n')
    # h.write(str(round(5570. / (1 + z))) + ' ' + str(round(5582. / (1 + z))) + '\n')
    # h.write('5890 5910\n')
    # h.write('6290 6310\n')
    h.write(str(round(6860. / (1 + z))) + ' ' + str(round(6890. / (1 + z))) + '    ! telluric lines\n')
    h.write(str(round(7560. / (1 + z))) + ' ' + str(round(7710. / (1 + z))) + '    ! weave (red gap)\n')
    h.write(str(round(7950. / (1 + z))) + ' ' + str(round(8500. / (1 + z))) + '    ! weave from 4most\n')
    h.write(str(round(8700. / (1 + z))) + ' ' + str(round(8870. / (1 + z))) + '    ! weave from 4most\n')

    h.close()

    f = open("excl_aps.fit", "w+")

    f.write('[rest-frame]\n')
    f.write('3712 3742    !OII, H13\n')
    f.write('3850 3895    !NeIII, HeI, H8\n')
    # f.write('3920 3945    ! ?\n')
    # f.write('3955 3980    !H7\n')
    f.write('4080 4110    !Hd\n')
    f.write('4320 4375    !Hg\n')
    f.write('4600 4750    !blueWRbump\n')
    f.write('4840 4880    !Hb\n')
    f.write('4940 4980    !OIII\n')
    f.write('4990 5022    !OIII\n')
    # f.write('5183 5220    ! ?\n')
    # f.write('5695 5745    ! ?\n')
    f.write('5856 5926    !HeI, ?\n')
    # f.write('6065 6110    ! ?\n')
    f.write('6280 6320    !OI\n')
    f.write('6520 6605    !NII, Ha\n')
    f.write('6690 6750    !SII\n')
    f.write('7050 7150    !HeI\n')
    f.write('7200 7390    !OII\n')

    f.write('\n')

    f.write('[observed-frame]\n')
    f.write(str(round(5495. / (1 + z))) + ' ' + str(round(5572. / (1 + z))) + '    ! weave (blue gap + red part)\n')
    # f.write(str(round(5570. / (1 + z))) + ' ' + str(round(5582. / (1 + z))) + '\n')
    # f.write('5890 5910\n')
    # f.write('6290 6310\n')
    f.write(str(round(6860. / (1 + z))) + ' ' + str(round(6890. / (1 + z))) + '    ! telluric lines\n')
    f.write(str(round(7580. / (1 + z))) + ' ' + str(round(7695. / (1 + z))) + '    ! weave (red gap)\n')
    f.write(str(round(7950. / (1 + z))) + ' ' + str(round(8500. / (1 + z))) + '    ! weave from 4most\n')
    f.write(str(round(8700. / (1 + z))) + ' ' + str(round(8870. / (1 + z))) + '    ! weave from 4most\n')

    f.close()
