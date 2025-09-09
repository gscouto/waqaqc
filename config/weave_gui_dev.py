import os
import sys

import numpy as np
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QPushButton, QLineEdit, QLabel, QComboBox
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar

gal = 'WVE_12204755+5805328'
gal_dir = 'WVE_12204755+5805328_LOWRES_11113'
res_dir = 'APS_2024-05-06_22.21.09'
mode = 'aps'   # 'aps', 'blue' or 'red'

# reading RSS data and products from PyParadise fit, for the specific source
datacube = fits.open('/work1/gcouto/weave/WEAVE-Apertif/QC_plots/'+gal_dir+'/'+mode+'_cube_vorbin.fits')
cont_model = fits.open('/work1/gcouto/weave/WEAVE-Apertif/QC_plots/'+gal_dir+'/pyp_results/'+res_dir+'/'+gal+'_'+mode+'_cont_model.fits')
cont_res = fits.open('/work1/gcouto/weave/WEAVE-Apertif/QC_plots/'+gal_dir+'/pyp_results/'+res_dir+'/'+gal+'_'+mode+'_cont_res.fits')
eline_model = fits.open('/work1/gcouto/weave/WEAVE-Apertif/QC_plots/'+gal_dir+'/pyp_results/'+res_dir+'/'+gal+'_'+mode+'_eline_model.fits')
eline_res = fits.open('/work1/gcouto/weave/WEAVE-Apertif/QC_plots/'+gal_dir+'/pyp_results/'+res_dir+'/'+gal+'_'+mode+'_eline_res.fits')

star_tab = fits.open('/work1/gcouto/weave/WEAVE-Apertif/QC_plots/'+gal_dir+'/pyp_results/'+res_dir+'/'+gal+'_'+mode+'_stellar_table.fits')
eline_tab = fits.open('/work1/gcouto/weave/WEAVE-Apertif/QC_plots/'+gal_dir+'/pyp_results/'+res_dir+'/'+gal+'_'+mode+'_eline_table.fits')

# creating wavelength vectors
lam = (np.arange(datacube[1].header['NAXIS3'])*datacube[1].header['CDELT3'])+datacube[1].header['CRVAL3']
lam_cont = (np.arange(cont_model[0].header['NAXIS3'])*cont_model[0].header['CDELT3'])+cont_model[0].header['CRVAL3']

if mode == 'aps':
    redshift = 0
else:
    redshift = float(datacube[1].header['W_Z'])

# class for the main window
class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'WEAVE + PyParadise'
        self.width = 2000
        self.height = 1000
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        m = PlotCanvas(self, width=20, height=10)
        m.move(0,0)

        self.show()
        
# class for plots and labels
class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=7, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        self.x_i = int(datacube[1].data.shape[2]/2.)
        self.y_i = int(datacube[1].data.shape[1]/2.)

        self.w = [np.where((star_tab[1].data['x_cor'] == self.x_i) & (star_tab[1].data['y_cor'] == self.y_i))][0]
        
        self.plot_lims = 0
        self.pix_up_flag = 0
        self.map_up_flag = 0
        self.map_name = star_tab[1].data.names[2:][0]
        self.plot()
        self.configs()


    # doing the first plot when booting the GUI
    def plot(self):

        gs = self.figure.add_gridspec(2, 3)
        gs.update(wspace=0.3)

        ax1 = self.figure.add_subplot(gs[0, 1:])
        
        ax1.plot(lam, datacube[1].data[:,self.y_i,self.x_i], color='black')
        ylim = ax1.get_ylim()
        xlim = ax1.get_xlim()
        ax1.plot(lam_cont, cont_model[0].data[:,self.y_i,self.x_i], color='red')
        if mode == 'aps':
            ax1.set_title('Target '+datacube[1].header['CCNAME']+' / x = '+str(self.x_i)+', y = '+str(self.y_i))
        else:
            ax1.set_title('Target '+datacube[1].header['CNAME']+' / x = '+str(self.x_i)+', y = '+str(self.y_i))
        ax1.fill_between(lam_cont, ylim[0], ylim[1], where=cont_model[2].data[:,self.y_i,self.x_i] > 0.5, facecolor='green', alpha=0.5, zorder=-1)
        self.plot_lines(ax1, ylim, redshift=redshift, name_flag=1)
        ax1.set_ylim(ylim)
        ax1.set_xlim(xlim)
        ax1.set_xlabel(r'$\lambda$ [$\AA$]')

        ax2 = self.figure.add_subplot(gs[1, 1:])
        ax2.plot(lam_cont, cont_res[0].data[:,self.y_i,self.x_i], color='black')
        ax2.plot(lam_cont, eline_model[0].data[:,self.y_i,self.x_i], color='blue')
        ax2.plot(lam_cont, eline_res[0].data[:,self.y_i,self.x_i], linestyle='--', color='magenta', zorder=-1, linewidth=0.5)
        if any('Halpha_b' in s for s in eline_tab[1].data.names):
            flux = eline_tab[1].data['Halpha_b_flux'][self.w][0]
            sigma = 6563.*eline_tab[1].data['Halpha_b_FWHM'][self.w][0]/2.35/300000.
            shift = 6563.*(1+(eline_tab[1].data['Halpha_b_vel'][self.w][0]/300000.))
            gauss = (flux/(sigma*np.sqrt(2*np.pi)))*np.exp(-((lam_cont-shift)**2)/(2*(sigma**2)))
            flux = eline_tab[1].data['NII6583_b_flux'][self.w][0]
            sigma = 6583.*eline_tab[1].data['NII6583_b_FWHM'][self.w][0]/2.35/300000.
            shift = 6583.*(1+(eline_tab[1].data['NII6583_b_vel'][self.w][0]/300000.))
            gauss = gauss + (flux/(sigma*np.sqrt(2*np.pi)))*np.exp(-((lam_cont-shift)**2)/(2*(sigma**2)))
            ax2.plot(lam_cont, gauss, color='brown')
        self.plot_lines(ax2, ylim, redshift=redshift, name_flag=0)
        ax2.set_xlim(xlim)

        if self.plot_lims:
            xlim = [float(self.xlim_min.text()),float(self.xlim_max.text())]
            ylim = [float(self.ylim_min.text()),float(self.ylim_max.text())]

            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)

            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)

        # n_map = np.zeros((np.nanmax(star_tab[1].data['y_cor'])+1, np.nanmax(star_tab[1].data['x_cor'])+1))
        n_map = np.zeros(datacube[1].shape[1:])

        for i in np.arange(np.nanmax(star_tab[1].data['x_cor'])+1):
            for j in np.arange(np.nanmax(star_tab[1].data['y_cor'])+1):
                w = [np.where((star_tab[1].data['x_cor'] == i) & (star_tab[1].data['y_cor'] == j))][0]
                if len(w[0]):
                    if self.map_name in eline_tab[1].data.names:
                        n_map[j,i] = eline_tab[1].data[self.map_name][w][0]
                    if self.map_name in star_tab[1].data.names:
                        n_map[j,i] = star_tab[1].data[self.map_name][w][0]

        ax3 = self.figure.add_subplot(gs[:, 0])
        im = ax3.imshow(n_map, origin='lower', vmin=np.nanmin(n_map), vmax = np.nanmax(n_map))
        if self.map_up_flag:
            im = ax3.imshow(n_map, origin='lower', vmin=float(self.vlim_min.text()), vmax = float(self.vlim_max.text()))
        else:
            im = ax3.imshow(n_map, origin='lower', vmin=np.nanmin(n_map), vmax = np.nanmax(n_map))
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        self.figure.subplots_adjust(left=0.15, bottom=0.2)
        self.draw()


    # creating the buttons, labels and boxes for the left side of the GUI
    def configs(self):

        self.mapbox = QComboBox(self)
        for i in star_tab[1].data.names[2:]:
            if i == 'base_coeff':
                continue
            else:
                self.mapbox.addItem(i)
        for i in eline_tab[1].data.names[2:]:
            self.mapbox.addItem(i)
        self.mapbox.move(400, 85)
        self.maplabel = QLabel('change map:', self)
        self.maplabel.move(300, 90)
        self.mapbox.activated[str].connect(self.map_up)

        self.vlim_min = QLineEdit(self)
        self.vlim_max = QLineEdit(self)
        self.vlim_min_label = QLabel('map min', self)
        self.vlim_max_label = QLabel('map max', self)

        self.vlim_min.move(370, 150)
        self.vlim_min.resize(60, 32)
        self.vlim_min_label.move(300, 155)

        self.vlim_max.move(370, 180)
        self.vlim_max.resize(60, 32)
        self.vlim_max_label.move(300, 185)

        vlim_but = QPushButton('OK', self)
        vlim_but.clicked.connect(self.map_lim_up)
        vlim_but.resize(40, 60)
        vlim_but.move(440, 150)

        self.xlim_min = QLineEdit(self)
        self.xlim_max = QLineEdit(self)
        self.ylim_min = QLineEdit(self)
        self.ylim_max = QLineEdit(self)

        self.xlim_min_label = QLabel('lam min', self)
        self.xlim_max_label = QLabel('lam max', self)
        self.ylim_min_label = QLabel('flux min', self)
        self.ylim_max_label = QLabel('flux max', self)

        self.xlim_min.move(70, 85)
        self.xlim_min.resize(60, 32)
        self.xlim_min_label.move(15, 90)

        self.xlim_max.move(70, 115)
        self.xlim_max.resize(60, 32)
        self.xlim_max_label.move(15, 120)

        self.ylim_min.move(70, 165)
        self.ylim_min.resize(60, 32)
        self.ylim_min_label.move(15, 170)

        self.ylim_max.move(70, 195)
        self.ylim_max.resize(60, 32)
        self.ylim_max_label.move(15, 200)

        lims_but = QPushButton('OK', self)
        lims_but.clicked.connect(self.lims_up)
        lims_but.resize(110, 32)
        lims_but.move(20, 240)

        self.n_xpix = QLineEdit(self)
        self.n_ypix = QLineEdit(self)

        self.n_xpix_label = QLabel('X pixel', self)
        self.n_ypix_label = QLabel('Y pixel', self)

        self.n_xpix.move(70, 300)
        self.n_xpix.resize(60, 32)
        self.n_xpix_label.move(15, 305)

        self.n_ypix.move(70, 330)
        self.n_ypix.resize(60, 32)
        self.n_ypix_label.move(15, 335)

        pix_but = QPushButton('OK', self)
        pix_but.clicked.connect(self.pix_up)
        pix_but.resize(100, 42)
        pix_but.move(20, 375)

        self.b2Label = QLabel('Pyparadise Fit Quality', self)
        myFont = QFont()
        myFont.setBold(True)
        self.b2Label.setFont(myFont)
        self.b2Label.move(15, 750)

        self.b3Label = QLabel('Pyparadise Stellar Parameters', self)
        self.b3Label.setFont(myFont)
        self.b3Label.move(15, 830)

        self.b4Label = QLabel('Pyparadise Emission-Lines Parameters',self)
        myFont = QFont()
        myFont.setBold(True)
        self.b4Label.setFont(myFont)
        self.b4Label.move(300, 750)

        self.lines = np.array([s[:-5] for s in eline_tab[1].data.names if 'flux' in s and 'err' not in s])
        self.el_box1 = QComboBox(self)
        self.el_box2 = QComboBox(self)
        for i in self.lines:
            self.el_box1.addItem(i)
            self.el_box2.addItem(i)
        self.el_box1.move(300, 775)
        self.el_box2.move(300, 875)
        self.el_box1.activated[str].connect(self.el_pyp1)
        self.el_box2.activated[str].connect(self.el_pyp2)

        self.pyp_params()

    # printing pyparadise parameters
    def pyp_params(self):

        self.w = [np.where((star_tab[1].data['x_cor'] == self.x_i) & (star_tab[1].data['y_cor'] == self.y_i))][0]

        if self.pix_up_flag == 0:

            self.chi2Label = QLabel('chi2 = '+str(round(star_tab[1].data['chi2'][self.w][0], 2)), self)
            self.chi2Label.move(15, 770)

            self.rvelLabel = QLabel('Rvel = '+str(round(star_tab[1].data['Rvel'][self.w][0], 5)), self)
            self.rvelLabel.move(15, 790)

            self.rdispLabel = QLabel('Rdisp = '+str(round(star_tab[1].data['Rdisp'][self.w][0], 5)), self)
            self.rdispLabel.move(15, 810)

            self.svelLabel = QLabel('vel_star = '+str(round(star_tab[1].data['vel_fit'][self.w][0], 1))+u' \u00B1 ' +  str(round(star_tab[1].data['vel_fit_err'][self.w][0], 1))+' km/s', self)
            self.svelLabel.move(15, 850)

            self.sdispLabel = QLabel('disp_star = '+str(round(star_tab[1].data['disp_fit'][self.w][0],1))+u' \u00B1 '+str(round(star_tab[1].data['disp_fit_err'][self.w][0],1))+' km/s',self)
            self.sdispLabel.move(15, 870)

            if 'mass_age_total_err' in star_tab[1].data.names:
                self.smassLabel = QLabel('age_massw = '+'{:.1e}'.format(star_tab[1].data['mass_age_total'][self.w][0],1)+u' \u00B1 '+'{:.1e}'.format(star_tab[1].data['mass_age_total_err'][self.w][0])+' yr',self)
                self.smassLabel.move(15, 890)

                self.slumLabel = QLabel('age_lumw = '+'{:.1e}'.format(star_tab[1].data['lum_age_total'][self.w][0],1)+u' \u00B1 '+'{:.1e}'.format(star_tab[1].data['lum_age_total_err'][self.w][0])+' yr',self)
                self.slumLabel.move(15, 910)
            else:
                self.smassLabel = QLabel('age_massw = '+'{:.1e}'.format(star_tab[1].data['mass_age_total'][self.w][0],1)+' yr',self)
                self.smassLabel.move(15, 890)

                self.slumLabel = QLabel('age_lumw = '+'{:.1e}'.format(star_tab[1].data['lum_age_total'][self.w][0],1)+' yr',self)
                self.slumLabel.move(15, 910)


            if self.lines[0]+'_flux_err' in eline_tab[1].data.names:
                self.el_flux1 = QLabel('flux = '+'{:.1e}'.format(eline_tab[1].data[self.lines[0]+'_flux'][self.w][0])+u' \u00B1 '+'{:.1e}'.format(eline_tab[1].data[self.lines[0]+'_flux_err'][self.w][0])+'',self)
                self.el_vel1 = QLabel('vel = '+str(round(eline_tab[1].data[self.lines[0]+'_vel'][self.w][0],1))+u' \u00B1 '+str(round(eline_tab[1].data[self.lines[0]+'_vel_err'][self.w][0],1))+' km/s',self)
                self.el_fwhm1 = QLabel('FWHM = '+str(round(eline_tab[1].data[self.lines[0]+'_fwhm'][self.w][0],1))+u' \u00B1 '+str(round(eline_tab[1].data[self.lines[0]+'_fwhm_err'][self.w][0],1))+' km/s',self)
                self.el_flux2 = QLabel('flux = '+'{:.1e}'.format(eline_tab[1].data[self.lines[0]+'_flux'][self.w][0])+u' \u00B1 '+'{:.1e}'.format(eline_tab[1].data[self.lines[0]+'_flux_err'][self.w][0])+'',self)
                self.el_vel2 = QLabel('vel = '+str(round(eline_tab[1].data[self.lines[0]+'_vel'][self.w][0],1))+u' \u00B1 '+str(round(eline_tab[1].data[self.lines[0]+'_vel_err'][self.w][0],1))+' km/s',self)
                self.el_fwhm2 = QLabel('FWHM = '+str(round(eline_tab[1].data[self.lines[0]+'_fwhm'][self.w][0],1))+u' \u00B1 '+str(round(eline_tab[1].data[self.lines[0]+'_fwhm_err'][self.w][0],1))+' km/s',self)
            else:
                self.el_flux1 = QLabel('flux = '+'{:.1e}'.format(eline_tab[1].data[self.lines[0]+'_flux'][self.w][0])+'',self)
                self.el_vel1 = QLabel('vel = '+str(round(eline_tab[1].data[self.lines[0]+'_vel'][self.w][0],1))+' km/s',self)
                self.el_fwhm1 = QLabel('FWHM = '+str(round(eline_tab[1].data[self.lines[0]+'_fwhm'][self.w][0],1))+' km/s',self)
                self.el_flux2 = QLabel('flux = '+'{:.1e}'.format(eline_tab[1].data[self.lines[0]+'_flux'][self.w][0])+'',self)
                self.el_vel2 = QLabel('vel = '+str(round(eline_tab[1].data[self.lines[0]+'_vel'][self.w][0],1))+' km/s',self)
                self.el_fwhm2 = QLabel('FWHM = '+str(round(eline_tab[1].data[self.lines[0]+'_fwhm'][self.w][0],1))+' km/s',self)
            self.el_flux1.move(300, 810)
            self.el_vel1.move(300, 830)
            self.el_fwhm1.move(300, 850)
            self.el_flux2.move(300, 910)
            self.el_vel2.move(300, 930)
            self.el_fwhm2.move(300, 950)

        if self.pix_up_flag == 1:

            self.chi2Label.setText('chi2 = '+str(round(star_tab[1].data['chi2'][self.w][0],2)))
            self.chi2Label.repaint()
            self.chi2Label.adjustSize()

            self.rvelLabel.setText('Rvel = '+str(round(star_tab[1].data['Rvel'][self.w][0],5)))
            self.rvelLabel.repaint()
            self.rvelLabel.adjustSize()

            self.rdispLabel.setText('Rdisp = '+str(round(star_tab[1].data['Rdisp'][self.w][0],5)))
            self.rdispLabel.repaint()
            self.rdispLabel.adjustSize()

            self.svelLabel.setText('vel_star = '+str(round(star_tab[1].data['vel_fit'][self.w][0],1))+u' \u00B1 '+str(round(star_tab[1].data['vel_fit_err'][self.w][0],1))+' km/s')
            self.svelLabel.repaint()
            self.svelLabel.adjustSize()

            self.sdispLabel.setText('disp_star = '+str(round(star_tab[1].data['disp_fit'][self.w][0],1))+u' \u00B1 '+str(round(star_tab[1].data['disp_fit_err'][self.w][0],1))+' km/s')
            self.sdispLabel.repaint()
            self.sdispLabel.adjustSize()

            if 'mass_age_total_err' in star_tab[1].data.names:
                self.smassLabel.setText('age_massw = '+'{:.1e}'.format(star_tab[1].data['mass_age_total'][self.w][0])+u' \u00B1 '+'{:.1e}'.format(star_tab[1].data['mass_age_total_err'][self.w][0])+u' yr')
                self.smassLabel.repaint()
                self.smassLabel.adjustSize()

                self.slumLabel.setText('age_lumw = '+'{:.1e}'.format(star_tab[1].data['lum_age_total'][self.w][0])+u' \u00B1 '+'{:.1e}'.format(star_tab[1].data['lum_age_total_err'][self.w][0])+u' yr')
                self.slumLabel.repaint()
                self.slumLabel.adjustSize()
            else:
                self.smassLabel.setText('age_massw = '+'{:.1e}'.format(star_tab[1].data['mass_age_total'][self.w][0])+' yr')
                self.smassLabel.repaint()
                self.smassLabel.adjustSize()

                self.slumLabel.setText('age_lumw = '+'{:.1e}'.format(star_tab[1].data['lum_age_total'][self.w][0])+' yr')
                self.slumLabel.repaint()
                self.slumLabel.adjustSize()

            if self.lines[0]+'_flux_err' in eline_tab[1].data.names:
                self.el_flux1.setText('flux = '+'{:.1e}'.format(eline_tab[1].data[self.lines[0]+'_flux'][self.w][0])+u' \u00B1 '+'{:.1e}'.format(eline_tab[1].data[self.lines[0]+'_flux_err'][self.w][0])+'')
                self.el_vel1.setText('vel = '+str(round(eline_tab[1].data[self.lines[0]+'_vel'][self.w][0],1))+u' \u00B1 '+str(round(eline_tab[1].data[self.lines[0]+'_vel_err'][self.w][0],1))+' km/s',)
                self.el_fwhm1.setText('FWHM = '+str(round(eline_tab[1].data[self.lines[0]+'_fwhm'][self.w][0],1))+u' \u00B1 '+str(round(eline_tab[1].data[self.lines[0]+'_fwhm_err'][self.w][0],1))+' km/s')
                self.el_flux2.setText('flux = '+'{:.1e}'.format(eline_tab[1].data[self.lines[0]+'_flux'][self.w][0])+u' \u00B1 '+'{:.1e}'.format(eline_tab[1].data[self.lines[0]+'_flux_err'][self.w][0])+'')
                self.el_vel2.setText('vel = '+str(round(eline_tab[1].data[self.lines[0]+'_vel'][self.w][0],1))+u' \u00B1 '+str(round(eline_tab[1].data[self.lines[0]+'_vel_err'][self.w][0],1))+' km/s')
                self.el_fwhm2.setText('FWHM = '+str(round(eline_tab[1].data[self.lines[0]+'_fwhm'][self.w][0],1))+u' \u00B1 '+str(round(eline_tab[1].data[self.lines[0]+'_fwhm_err'][self.w][0],1))+' km/s')
                self.el_flux1.repaint()
                self.el_vel1.repaint()
                self.el_fwhm1.repaint()
                self.el_flux2.repaint()
                self.el_vel2.repaint()
                self.el_fwhm2.repaint()
                self.el_flux1.adjustSize()
                self.el_vel1.adjustSize()
                self.el_fwhm1.adjustSize()
                self.el_flux2.adjustSize()
                self.el_vel2.adjustSize()
                self.el_fwhm2.adjustSize()
            else:
                self.el_flux1.setText('flux = '+'{:.1e}'.format(eline_tab[1].data[self.lines[0]+'_flux'][self.w][0])+'')
                self.el_vel1.setText('vel = '+str(round(eline_tab[1].data[self.lines[0]+'_vel'][self.w][0],1))+' km/s')
                self.el_fwhm1.setText('FWHM = '+str(round(eline_tab[1].data[self.lines[0]+'_fwhm'][self.w][0],1))+' km/s')
                self.el_flux2.setText('flux = '+'{:.1e}'.format(eline_tab[1].data[self.lines[0]+'_flux'][self.w][0])+'')
                self.el_vel2.setText('vel = '+str(round(eline_tab[1].data[self.lines[0]+'_vel'][self.w][0],1))+' km/s')
                self.el_fwhm2.setText('FWHM = '+str(round(eline_tab[1].data[self.lines[0]+'_fwhm'][self.w][0],1))+' km/s')
                self.el_flux1.repaint()
                self.el_vel1.repaint()
                self.el_fwhm1.repaint()
                self.el_flux2.repaint()
                self.el_vel2.repaint()
                self.el_fwhm2.repaint()
                self.el_flux1.adjustSize()
                self.el_vel1.adjustSize()
                self.el_fwhm1.adjustSize()
                self.el_flux2.adjustSize()
                self.el_vel2.adjustSize()
                self.el_fwhm2.adjustSize()

    # defining the limits for the plot according to input values
    def lims_up(self):
        self.plot_lims = 1
        self.pix_up_flag = 0
        #self.map_up_flag = 0

        self.figure.clf()
        self.plot()

    # updating map displayed
    def map_up(self,text):
        #self.plot_lims = 0
        self.pix_up_flag = 0
        self.map_up_flag = 0

        self.map_name = text

        self.figure.clf()
        self.plot()

    # updating map limits
    def map_lim_up(self):
        #self.plot_lims = 0
        self.pix_up_flag = 0
        self.map_up_flag = 1

        self.figure.clf()
        self.plot()

    # changing line emission parameters shown
    def el_pyp1(self, text):
        if text+'_flux_err' in eline_tab[1].data.names:
            self.el_flux1.setText('flux = '+'{:.1e}'.format(eline_tab[1].data[text+'_flux'][self.w][0])+u' \u00B1 '+'{:.1e}'.format(eline_tab[1].data[text+'_flux_err'][self.w][0])+'')
            self.el_vel1.setText('vel = '+str(round(eline_tab[1].data[text+'_vel'][self.w][0],1))+u' \u00B1 '+str(round(eline_tab[1].data[text+'_vel_err'][self.w][0],1))+' km/s')
            self.el_fwhm1.setText('FWHM = '+str(round(eline_tab[1].data[text+'_fwhm'][self.w][0],1))+u' \u00B1 '+str(round(eline_tab[1].data[text+'_fwhm_err'][self.w][0],1))+' km/s')
        else:
            self.el_flux1.setText('flux = '+'{:.1e}'.format(eline_tab[1].data[text+'_flux'][self.w][0])+'')
            self.el_vel1.setText('vel = '+str(round(eline_tab[1].data[text+'_vel'][self.w][0],1))+' km/s')
            self.el_fwhm1.setText('FWHM = '+str(round(eline_tab[1].data[text+'_fwhm'][self.w][0],1))+' km/s')
        self.el_flux1.repaint()
        self.el_flux1.adjustSize()
        self.el_vel1.repaint()
        self.el_vel1.adjustSize()
        self.el_fwhm1.repaint()
        self.el_fwhm1.adjustSize()

    def el_pyp2(self, text):
        if text+'_flux_err' in eline_tab[1].data.names:
            self.el_flux2.setText('flux = '+'{:.1e}'.format(eline_tab[1].data[text+'_flux'][self.w][0])+u' \u00B1 '+'{:.1e}'.format(eline_tab[1].data[text+'_flux_err'][self.w][0])+'')
            self.el_vel2.setText('vel = '+str(round(eline_tab[1].data[text+'_vel'][self.w][0],1))+u' \u00B1 '+str(round(eline_tab[1].data[text+'_vel_err'][self.w][0],1))+' km/s')
            self.el_fwhm2.setText('FWHM = '+str(round(eline_tab[1].data[text+'_fwhm'][self.w][0],1))+u' \u00B1 '+str(round(eline_tab[1].data[text+'_fwhm_err'][self.w][0],1))+' km/s')
        else:
            self.el_flux2.setText('flux = '+'{:.1e}'.format(eline_tab[1].data[text+'_flux'][self.w][0])+'')
            self.el_vel2.setText('vel = '+str(round(eline_tab[1].data[text+'_vel'][self.w][0],1))+' km/s')
            self.el_fwhm2.setText('FWHM = '+str(round(eline_tab[1].data[text+'_fwhm'][self.w][0],1))+' km/s')
        self.el_flux2.repaint()
        self.el_flux2.adjustSize()
        self.el_vel2.repaint()
        self.el_vel2.adjustSize()
        self.el_fwhm2.repaint()
        self.el_fwhm2.adjustSize()

    # defining the limits for the plot according to input values
    def pix_up(self):

        self.plot_lims = 0
        self.pix_up_flag = 1

        self.x_i = int(self.n_xpix.text())
        self.y_i = int(self.n_ypix.text())

        self.figure.clf()
        self.plot()
        self.pyp_params()

    # plotting emission lines wavelengths
    def plot_lines(self, ax, ylim, redshift, name_flag):
        #lin_wav = np.array([3727., 3750., 3771., 3798., 3835., 3869., 3889., 3970., 4026., 4068., 4102., 4340., 4363., 4472., 4658., 4686., 4861., 4922., 4959., 5007., 5270., 5876., 6300., 6548., 6563., 6583., 6678., 6716., 6731., 7065., 7135., 7281., 7319., 7330.])
        #lines = np.array([r'[OII]3727', r'H12', r'H11', r'H10', r'H9', r'[NeIII]3869', r'H8 + HeI', r'H7', r'HeI 4026', r'[SII]4068', r'H$\delta$', r'H$\gamma$', r'[OIII]4363', r'HeI 4472', r'[FeIII]4658', r'HeII 4686', r'H$\beta$', r'HeI 4922', r'[OIII]4959', r'[OIII]5007', r'[FeIII]5270', r'HeI 5876', r'[OI]6300', r'[NII]6548', r'H$\alpha$', r'[NII]6583', r'HeI 6678', r'[SII]6716', r'[SII]6731', r'HeI 7065', r'[ArIII]7135', r'HeI 7281', r'[OII]7319', r'[OII]7330'])

        lin_wav = np.array([4861., 4959., 5007., 6300., 6548., 6563., 6583., 6716., 6731.])
        lines = np.array([r'H$\beta$', r'[OIII]4959', r'[OIII]5007', r'[OI]6300', r'[NII]6548', r'H$\alpha$', r'[NII]6583', r'[SII]6716', r'[SII]6731'])

        for i in np.arange(len(lines)):
            ax.axvline(lin_wav[i]*(1+redshift), linestyle='--', color='gray')
            if name_flag == 1:
                ax.text(1+(lin_wav[i]*(1+redshift)), 0.1*np.mean(ylim), lines[i], color='gray',
                    rotation=90)

        
# booting GUI interface
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())        
