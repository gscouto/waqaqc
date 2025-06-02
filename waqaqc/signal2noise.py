# signal2noise.py
#
# Scott C. Trager, based on "signal.f" by Chris Benn
# driving code for "signalWEAVE"
#
# v1.5 15 Feb 2018: added "totalSeeing" to account for blurring by PRI
# v1.4 15 Jan 2018: fixed major bug in Moffat profile normalization
# v1.3 12 Jan 2018: stripped out "blue" and "red" arguements, as no longer needed
# v1.2 19 Jun 2013: now allows for Moffat or Gaussian seeing profile
# v1.1 20 Nov 2011: now calculates dispersion self-consistently based on
#                   resolution and spectrograph parameters
#
# Parameter updates
#
# UPDATE 17.04.2012: central obstruction 1.5m -> 1.7m
# UPDATE 15.10.2012: PFC efficiency dropped to 0.70 in blue arm and 0.75 in red arm
# UPDATE 22.11.2012: added full-well depth (fullwell) of 265000 e- per Iain S's message of 30.10.2012
# UPDATE 02.12.2012: new PFC and spectrograph average efficiencies -
#                    blue  PFC: 0.6944  red  PFC: 0.7310
#                    blue spec: 0.6484  red spec: 0.6698
#                    new CCD efficiencies: blue=0.8612 red=0.9293
#                    new fibre efficiencies: blue=0.7887 red=0.8680
# UPDATE 11.12.2013: central obstruction 1.7m -> 1.83m
# UPDATE 30.01.2014: new efficiencies (blue: 400-600nm)
#                    blue  PFC: 0.7400  red  PFC: 0.7564 low res
#                    blue spec: 0.5394  red spec: 0.6168 low res
#                    blue  fib: 0.7242  red  fib: 0.8014 low res
#                    blue  CCD: 0.9310  red  CCD: 0.9269 low res
#                    blue  PFC: 0.7300  red  PFC: 0.7500 high res
#                    blue spec: 0.4192  red spec: 0.4597 high res
#                    blue  fib: 0.6600  red  fib: 0.7900 high res
#                    blue  CCD: 0.9500  red  CCD: 0.9300 high res
# defaults below are RED SIDE, LOW-RES efficiencies

from sys import argv
from math import atan2, asin, cos
from numpy import *
import scipy
from scipy import special
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq

PRI_FWHM = 0.45


def totalSeeing(seeing):
    return sqrt(seeing ** 2 + PRI_FWHM ** 2)


class FullWellException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class SaturationException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class signal:
    def __init__(self, QE=0.9269, fiberEff=0.8014, specEff=0.6168, pfEff=0.7564,
                 res=5000., offset=0., fiberD=1.3045, fFP=3.2, fcol=3.1, fcam=1.8,
                 R5000fiberD=1.3045, telD=4.2, centralObs=1.83,
                 cwave=4900., gain=1., saturation=6.e4, fullwell=2.65e5,
                 profile='Gaussian', betam=None):
        # WHT parameters
        # telescope area after removing central obstruction
        self.telarea = pi * ((telD / 2) ** 2 - (centralObs / 2) ** 2)
        # M1 reflectivity; average blue (390-600): 0.910; average red (600-980): 0.891
        self.m1Eff = 0.90
        # nominal instrumental parameters
        # detector QE
        self.QE = QE
        # fiber efficiency
        self.fiberEff = fiberEff
        # spectrograph efficiency
        self.specEff = specEff
        # PF corrector efficiency
        self.pfEff = pfEff
        # bands
        self.bands = ['U', 'B', 'V', 'R', 'I', 'z']
        # central wavelengths for UBVRIz in nm
        self.wave = array([360., 430., 550., 650., 820., 950.], 'f')
        # mag -> Jy from Bessell (1979)
        self.fluxJy = array([1810., 4260., 3640., 3080., 2550., 2200.], 'f')
        # photons/m^2/s/Ang for m=0
        self.photons = 10000 * self.fluxJy * 1000. / (6.6 * self.wave)
        # extinction
        self.extinct = array([0.4554, 0.2175, 0.1020, 0.0575, 0.0157, 0.0087], 'f')
        ## self.extinct=array([0.55,0.25,0.15,0.09,0.06,0.05],'f')
        # spectrograph/detector parameters
        self.res = res
        # fiber diameter in arcsec
        self.fiberD = fiberD
        # fiber diameter in arcsec when R=5000
        self.R5000fiberD = R5000fiberD
        # pixel size in microns
        self.pixSize = 15.
        # fiber diameter in microns
        self.fiberDmu = self.fiberD * fFP * 4.2 * 1e6 / 206265
        # fiber diameter on CCD in pixels
        self.fiberCCD = self.fiberDmu * (fcam / fcol) / self.pixSize
        # dispersion at central wavelength, in Angstrom/pixel
        # new 20.11.2011
        self.disp = self.dispersion(cwave, fcam, fcol, beamsize=180.)
        ##         # average dispersion in Angstroms/pixel
        ##         if red:
        ##             self.disp=(5000./self.res)*10.*(983-594)/8132
        ##         else: # blue side
        ##             self.disp=(5000./self.res)*10.*(606-366)/8132
        # offset of fiber from object center
        self.offset = offset
        # CCD parameters
        self.gain = gain
        self.saturation = saturation
        self.fullwell = fullwell
        # seeing profile
        self.profile = profile
        if self.profile == 'Moffat': self.betam = betam

    def dispersion(self, cwave, fcam, fcol, beamsize):
        # compute dispersion in Ang/pixel at central wavelength
        # camera focal length
        camfl = beamsize * fcam
        # grating angle in radians
        gamma = atan2(1.e-3 * self.fiberDmu * self.res, 2 * beamsize * fcol)
        # l/mm of grating
        lpmm = 2 * sin(gamma) / (1.e-7 * cwave)
        lpum = 1.e-3 * lpmm
        ##         # FOV of camera on detector (in radians)
        ##         fovcam=atan2(npix*self.pixSize*1e-3,2*camfl)
        # 1 pixel subtends this number of radians
        pixsubt = self.pixSize * 1.e-3 / camfl
        # diffracted angle of one pixel (approximation good to very high precision!)
        diffangle = asin(sin(gamma) - 1.e-4 * cwave * lpum)
        # dispersion in ang/pixel
        return 1e4 * pixsubt * cos(diffangle) / lpum

    def efficiency(self):
        return self.QE * self.fiberEff * self.specEff * self.pfEff * self.m1Eff

    def effectiveArea(self, eff):
        return eff * self.telarea

    def extinction(self, airmass=1.2):
        return pow(10, -0.4 * self.extinct * airmass)

    def lightfrac_old(self, seeing, fiberD):
        # note that the integral of a circularly-symmetric gaussian over 0 
        # to 2pi and 0 to infinity is just 2 pi sigma^2...
        s = seeing / 2.3548
        rfib = fiberD / 2.
        lf = quad(lambda x: x * exp(-pow(x - self.offset, 2) / (2 * s * s)) / \
                            (s * (s + self.offset * sqrt(pi / 2.))), 0, rfib)[0]
        return lf

    def gaussian_xy(self, y, x, s):
        r = sqrt((x - self.offset) ** 2 + y * y)
        return exp(-pow(r, 2) / (2 * s * s)) / (2 * pi * s * s)

    def moffat_xy(self, y, x, alpha, betam):
        r = sqrt((x - self.offset) ** 2 + y * y)
        # new normalization given by integrating Moffat function in Mathematica (15.01.2018)
        if alpha >= 0. and betam > 1.:
            norm = (2 * pi * alpha ** 2) / (2 * (betam - 1))
        else:
            raise ValueError('alpha and/or beta out of bounds')
        return pow(1. + pow(r / alpha, 2), -betam) / norm

    def lightfrac(self, seeing, fiberD):
        rfib = fiberD / 2.
        if self.profile == 'Gaussian':
            s = seeing / 2.3548
            lf = dblquad(self.gaussian_xy, -rfib, rfib, lambda x: -sqrt(rfib * rfib - x * x), \
                         lambda x: sqrt(rfib * rfib - x * x), args=(s,))[0]
        elif self.profile == 'Moffat':
            self.alpha = seeing / (2. * sqrt(2. ** (1. / self.betam) - 1))
            lf = dblquad(self.moffat_xy, -rfib, rfib, lambda x: -sqrt(rfib * rfib - x * x), \
                         lambda x: sqrt(rfib * rfib - x * x), args=(self.alpha, self.betam))[0]
        return lf

    def objectphotons(self, mag, time, band, airmass, effArea):
        # total number of object photons collected per pixel in wavelength
        # direction, integrated over slit
        return pow(10, -0.4 * mag) * time * \
               self.photons.tolist()[self.bands.index(band)] * \
               effArea * self.disp * \
               self.extinction(airmass).tolist()[self.bands.index(band)]

    def skyphotons(self, skysb, time, band, effArea, fiberD):
        # total number of sky photons collected per pixel in wavelength direction,
        # integrated over slit
        return pow(10, -0.4 * skysb) * time * \
               self.photons.tolist()[self.bands.index(band)] * \
               pi * pow(fiberD / 2., 2) * effArea * self.disp

    def S2N(self, time, mag, band='V', airmass=1.2, fiberD=1.3045,
            seeing=1.0, rn=3., dark=0., eff=None, sb=None, skysb=22.7, skyband="B"):
        # number of pixels along the slit subtended by the fiber
        npix_spatial = self.fiberCCD
        # number of pixels in 1 Angstrom
        npix_spectral = 1. / self.disp
        pixArea = npix_spatial * npix_spectral
        if not eff:
            eff = self.efficiency()
        effArea = self.effectiveArea(eff)
        if sb:
            # total number of photons per spectral pixel in the fiber, uniform SB
            ophot = self.objectphotons(mag, time, band, airmass, effArea) * pi * \
                    pow(fiberD / 2., 2)
        else:
            # total number of photons per spectral pixel in the fiber, circular
            # Gaussian
            ophot = self.objectphotons(mag, time, band, airmass, effArea) * \
                    self.lightfrac(seeing, fiberD)
        # number of photons in the sky in the fiber per pixel
        sphot = self.skyphotons(skysb, time, skyband, effArea, fiberD)
        ## # saturation?
        ## if (sphot+ophot)/self.gain > self.saturation:
        ##    raise SaturationException, 'saturation'
        # full-well depth exceeded?
        if (sphot + ophot) / self.gain > self.fullwell:
            raise FullWellException('exceeded full well depth')
        # S/N in 1 spectral pixel
        SNRpix = ophot / sqrt(ophot + sphot + npix_spatial * (rn * rn + dark * time / 3600.))
        # S/N in 1 Angstrom
        SNR = npix_spectral * ophot / sqrt(
            npix_spectral * ophot + npix_spectral * sphot + pixArea * (rn * rn + dark * time / 3600.))
        # S/N in 1 resolution element
        SNRres = self.fiberCCD * ophot / sqrt(
            self.fiberCCD * ophot + self.fiberCCD * sphot + self.fiberCCD ** 2 * (rn ** 2 + dark * time / 3600.))
        return {'SNR': SNR, 'objectphotons': ophot, 'skyphotons': sphot, \
                'efficiency': eff, 'effectivearea': effArea, \
                'SNRres': SNRres, 'SNRpix': SNRpix}

    def RVaccuracy(self, snr, scale=0.6):
        # formula taken from Munari et al. (2001)
        # scaling taken from Battaglia et al. (2008)
        # note that this S/N per Ang!
        return scale * pow(10, 0.6 * pow(log10(snr), 2) - 2.4 * log10(snr) - 1.75 * log10(self.res) + 9.36)

    def time4S2N(self, S2Ndesired, mag, band, airmass=1.2, fiberD=1.3045, seeing=1.0,
                 rn=3., dark=0., eff=None, sb=None, skysb=22.7, skyband="B",
                 snrtype='SNRres'):
        # determine time required to achieve given S/N ratio S2Ndesired
        # bounds: [0.1,72000] s
        args = (mag, band, airmass, fiberD, seeing, rn, dark, eff, sb, skysb, skyband)

        def bfunc(x, s2n, args):
            return self.S2N(x, *args)[snrtype] - s2n

        return brentq(bfunc, 0.1, 7.2e4, (S2Ndesired, args))

    def eff4timeS2N(self, timeDesired, S2Ndesired, mag, band, airmass=1.2,
                    fiberD=1.3045, seeing=1.0, rn=3., dark=0., sb=None, skysb=22.7,
                    skyband="B", snrtype='SNRres'):
        # determine efficiency required to achieve given S/N ratio S2Ndesired
        # in time timeDesired
        # bounds: [0.01,1.00]
        args = vars()
        del args['self']
        del args['timeDesired']
        del args['S2Ndesired']
        del args['snrtype']
        args['time'] = timeDesired

        def bfunc(x, s2n, args):
            args['eff'] = x
            return self.S2N(**args)[snrtype] - s2n

        return brentq(bfunc, 0.01, 1.0, (S2Ndesired, args))

    def mag4timeS2N(self, S2Ndesired, timeDesired, band, airmass=1.2, eff=None,
                    fiberD=1.3045, seeing=1.0, rn=3., dark=0., sb=None, skysb=22.7,
                    skyband="B", snrtype='SNRres'):
        # determine magnitude reached at given S/N ratio S2Ndesired in time
        # in time timeDesired
        # bounds: [0.01,1.00]
        args = vars()
        del args['self']
        del args['timeDesired']
        del args['S2Ndesired']
        del args['snrtype']
        args['time'] = timeDesired

        def bfunc(x, s2n, args):
            args['mag'] = x
            return self.S2N(**args)[snrtype] - s2n

        return brentq(bfunc, 5, 30, (S2Ndesired, args))
