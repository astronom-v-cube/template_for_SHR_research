#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:36:23 2018

@author: sergey
"""

import sys;
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QTabWidget, QVBoxLayout, QGridLayout, QTableWidget, QTableWidgetItem, QProxyStyle, QStyle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as NP
from srhFitsFile612 import SrhFitsFile
from skimage.transform import warp, AffineTransform
from astropy.io import fits
from astropy.time import Time, TimeDelta
#import casacore.tables as T
from casacore.images import image
#import base2uvw as bl2uvw
#import srhMS2
from mpl_toolkits.mplot3d import Axes3D
import pylab as PL
from scipy.signal import argrelextrema
from sunpy import coordinates
import matplotlib.animation as animation
import matplotlib.colors
import os
import casaDescDicts as desc
#from casatasks import tclean

cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.4, 0.2, 0.2),
                 (0.6, 0.0, 0.0),
                 (0.8, 1.0, 1.0),
                 (0.9, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
        'green':((0.0, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (0.9, 0.0, 0.0),
                 (1.0, 1.0, 1.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (0.2, 1.0, 1.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 0.0, 0.0),
                 (0.8, 0.0, 0.0),
                 (0.9, 0.0, 0.0),
                 (1.0, 1.0, 1.0))}

my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

class CustomStyle(QProxyStyle):
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_SpinBox_KeyPressAutoRepeatRate:
            return 10**7
        elif hint == QStyle.SH_SpinBox_ClickAutoRepeatRate:
            return 10**7
        elif hint == QStyle.SH_SpinBox_ClickAutoRepeatThreshold:
            # You can use only this condition to avoid the auto-repeat,
            # but better safe than sorry ;-)
            return 10**7
        else:
            return super().styleHint(hint, option, widget, returnData)

class ResponseCanvas(FigureCanvas):
    mouseSignal = QtCore.pyqtSignal(float, float, name = 'xyChanged')
    
    def mouse_moved(self, event):
        try:
            self.mouseSignal.emit(event.xdata, event.ydata)
        except:
            pass
        
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.subplot = self.fig.add_subplot(111)
        self.subplot.xaxis.set_visible(True)
        self.subplot.yaxis.set_visible(True)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.mpl_connect('motion_notify_event', self.mouse_moved)
#        self.cmap = 'ocean'
        self.cmap = my_cmap
        
    def setData(self, array):
        self.imageObject.set_data(array)
        self.draw()

    def setCurveXY(self, curveX, curveY):
        self.plotObject[0].setx_data(curveX)
        self.plotObject[0].sety_data(curveY)
        self.draw()

#    def imshow(self, array):
#        self.imageObject = self.subplot.imshow(array, cmap=self.cmap, origin='lower')
#        self.draw()
        
    def imshow(self, array, arrayMin, arrayMax):
        self.imageObject = self.subplot.imshow(array, vmin = arrayMin, vmax = arrayMax, cmap=self.cmap, origin='lower')
        self.draw()

    def contour(self, array, levels):
        self.contourObject = self.subplot.contour(array, levels)
        self.draw()

    def plot(self, data):
        self.plotObject = self.subplot.plot(data)
        self.draw()
        
    def plot_xy(self, x, data):
        self.plotObject = self.subplot.plot(x, data)
        self.draw()
    
    def scatter(self, data):
        self.plotObject = self.subplot.plot(data, '.')
        self.draw()
    
    def clear(self):
        self.subplot.cla()
        self.draw()
    
    def redraw(self):
        self.draw()
    
    def setColormap(self, cmap):
        self.cmap = cmap

class SrhEdik612(QtWidgets.QMainWindow):#MainWindow):
#    def buildEwPhase(self):
#        self.ewLcpPhaseCorrection[:] = 0.
#        self.ewRcpPhaseCorrection[:] = 0.
#        # for j in range(16):
#        #     for i in range(16):
#        #         self.ewLcpPhaseCorrection[16 + i] +=  NP.deg2rad(self.ewPhaseCoefsLcp[self.currentFrequencyChannel, j] * (-1)**(i // (j + 1))) 
#        #         self.ewLcpPhaseCorrection[15 - i]  += -NP.deg2rad(self.ewPhaseCoefsLcp[self.currentFrequencyChannel, j] * (-1)**(i // (j + 1))) 
#        #         self.ewRcpPhaseCorrection[16 + i] +=  NP.deg2rad(self.ewPhaseCoefsRcp[self.currentFrequencyChannel, j] * (-1)**(i // (j + 1))) 
#        #         self.ewRcpPhaseCorrection[15 - i]  += -NP.deg2rad(self.ewPhaseCoefsRcp[self.currentFrequencyChannel, j] * (-1)**(i // (j + 1))) 
#        for j in range(self.ewAntennaNumber):
#                self.ewLcpPhaseCorrection[j] += NP.deg2rad(self.ewLcpPhaseSlope[self.currentFrequencyChannel] * (j - 32)) 
#                self.ewRcpPhaseCorrection[j] += NP.deg2rad(self.ewRcpPhaseSlope[self.currentFrequencyChannel] * (j - 32))
#                # self.ewLcpPhaseCorrection[j] += NP.deg2rad(self.ewLcpPhaseAnt[j])
#                # self.ewRcpPhaseCorrection[j] += NP.deg2rad(self.ewRcpPhaseAnt[j])
#        self.srhFits.changeEastWestPhase(self.ewLcpPhaseCorrection, self.ewRcpPhaseCorrection)
#        
#    def buildNPhase(self):
#        self.nLcpPhaseCorrection[:] = 0.
#        self.nRcpPhaseCorrection[:] = 0.
#        # for j in range(16):
#        #     for i in range(16):
#        #         self.nLcpPhaseCorrection[i] +=  NP.deg2rad(self.nPhaseCoefsLcp[self.currentFrequencyChannel, j] * (-1)**(i // (j + 1))) 
#        #         self.nRcpPhaseCorrection[i] +=  NP.deg2rad(self.nPhaseCoefsRcp[self.currentFrequencyChannel, j] * (-1)**(i // (j + 1))) 
#        for j in range(self.nAntennaNumber):
#                self.nLcpPhaseCorrection[j] += NP.deg2rad(self.nLcpPhaseSlope[self.currentFrequencyChannel] * (j + 1)) 
#                self.nRcpPhaseCorrection[j] += NP.deg2rad(self.nRcpPhaseSlope[self.currentFrequencyChannel] * (j + 1)) 
#                # self.nLcpPhaseCorrection[j] += NP.deg2rad(self.nLcpPhaseAnt[j])
#                # self.nRcpPhaseCorrection[j] += NP.deg2rad(self.nRcpPhaseAnt[j])
#        self.srhFits.changeNorthPhase(self.nLcpPhaseCorrection, self.nRcpPhaseCorrection)
        
    def buildImage(self):

        if self.indexOfImageType == 4:
            self.srhFits.vis2uv(self.currentScan, phaseCorrect=self.phaseCorrect, amplitudeCorrect=self.amplitudeCorrect, PSF=True);
        elif self.indexOfImageType == 5:
            self.srhFits.vis2uv(self.currentScan, phaseCorrect=self.phaseCorrect, amplitudeCorrect=self.amplitudeCorrect, PSF=True, average = self.averScans);
            self.srhFits.uv2lmImage()
            self.psfData = self.srhFits.lcp.real
            self.srhFits.vis2uv(self.currentScan, phaseCorrect=self.phaseCorrect, amplitudeCorrect=self.amplitudeCorrect, PSF=False, average = self.averScans);
        else:
            self.srhFits.vis2uv(self.currentScan, phaseCorrect=self.phaseCorrect, amplitudeCorrect=self.amplitudeCorrect, PSF=False, average = self.averScans);
        self.srhFits.uv2lmImage()

        scaling = self.srhFits.getPQScale(self.uvSize, NP.deg2rad(self.arcsecPerPixel * (self.uvSize - 1)/3600.)*2)
        scale = AffineTransform(scale=(self.uvSize/scaling[0], self.uvSize/scaling[1]))
        shift = AffineTransform(translation=(-self.uvSize/2,-self.uvSize/2))
        rotate = AffineTransform(rotation = self.pAngle)
        matrix = AffineTransform(matrix = self.srhFits.getPQ2HDMatrix())
        back_shift = AffineTransform(translation=(self.uvSize/2,self.uvSize/2))

        if self.indexOfFrameType == 0:
            self.lcpData = self.srhFits.lcp.real.copy()
            self.rcpData = self.srhFits.rcp.real.copy()
            
        else:
            O = self.uvSize//2
            Q = self.uvSize//4
            lcp = self.srhFits.lcp.real.copy()
            rcp = self.srhFits.rcp.real.copy()
            
            dataResult0 = warp(lcp,(shift + (scale + back_shift)).inverse)
            self.lcpData = warp(dataResult0,(shift + (matrix + back_shift)).inverse)
            dataResult0 = warp(rcp,(shift + (scale + back_shift)).inverse)
            self.rcpData = warp(dataResult0,(shift + (matrix + back_shift)).inverse)
            if self.indexOfImageType == 5:
                dataResult0 = warp(self.psfData,(shift + (scale + back_shift)).inverse)
                self.psfData = warp(dataResult0,(shift + (matrix + back_shift)).inverse)
            dataResult0 = 0
            if self.indexOfFrameType == 2:
                self.lcpData = warp(self.lcpData,(shift + (rotate + back_shift)).inverse)[O-Q:O+Q,O-Q:O+Q]
                self.rcpData = warp(self.rcpData,(shift + (rotate + back_shift)).inverse)[O-Q:O+Q,O-Q:O+Q]
            if self.indexOfFrameType == 1:
                self.lcpData = self.lcpData[O-Q:O+Q,O-Q:O+Q]
                self.rcpData = self.rcpData[O-Q:O+Q,O-Q:O+Q]
        self.lcpData = NP.flip(NP.flip(self.lcpData,1), 0)
        self.rcpData = NP.flip(NP.flip(self.rcpData,1), 0)
        
        
        if self.srhFits.hAngle > NP.pi:
            self.lcpData = NP.flip(self.lcpData,1)
            self.rcpData = NP.flip(self.rcpData,1)
        else:
            self.lcpData = NP.flip(self.lcpData,0)
            self.rcpData = NP.flip(self.rcpData,0)
            
        if self.indexOfImageType == 6:
            self.lcpData -= (self.freqDiffLcp_lm * self.diffRelationLcp)
            self.rcpData -= (self.freqDiffRcp_lm * self.diffRelationRcp)
        
        self.iData = self.lcpData + self.lcpRcpRel * self.rcpData
        self.vData = self.lcpData - self.lcpRcpRel * self.rcpData
        
        if self.indexOfImageType == 5:
            self.lcpData += self.psfData  / NP.max(self.psfData) * NP.max(self.lcpData) * .5
            self.rcpData += self.psfData  / NP.max(self.psfData) * NP.max(self.rcpData) * .5

        if self.indexOfPlotType == 1:
            self.hLcp, self.bLcp = NP.histogram(self.lcpData[NP.logical_and(self.lcpData!=0, NP.abs(self.lcpData)<1e5)], bins = self.bins)
            self.hRcp, self.bRcp = NP.histogram(self.rcpData[NP.logical_and(self.rcpData!=0, NP.abs(self.rcpData)<1e5)], bins = self.bins)
            
        elif self.indexOfPlotType == 0:
            lcpCorr = NP.mean(NP.abs(self.srhFits.visLcp[self.currentFrequencyChannel, self.currentScan, :3007]))
            rcpCorr = NP.mean(NP.abs(self.srhFits.visRcp[self.currentFrequencyChannel, self.currentScan, :3007]))
            self.lcpMaxTrace.append(lcpCorr + rcpCorr)
            self.rcpMaxTrace.append(lcpCorr - rcpCorr)
            
        elif self.indexOfPlotType == 3:
            if self.indexOfImageType == 1:
                self.lcpMaxTrace.append(NP.mean(self.iData))
                self.rcpMaxTrace.append(NP.mean(NP.abs(self.vData)))
            else:
                self.lcpMaxTrace.append(NP.mean(self.lcpData))
                self.rcpMaxTrace.append(NP.mean(self.rcpData))
            
    def showImage(self):
        self.lcpCanvas.clear()
        self.rcpCanvas.clear()
        if self.indexOfImageType == 0 or self.indexOfImageType == 4 or self.indexOfImageType == 5 or self.indexOfImageType == 6:
            if self.autoscaleButton.isChecked():
                self.lcpCanvas.imshow(self.lcpData, NP.min(self.lcpData), NP.max(self.lcpData))
                self.rcpCanvas.imshow(self.rcpData, NP.min(self.rcpData), NP.max(self.rcpData))
            else:
                self.lcpCanvas.imshow(self.lcpData*self.imageScale + self.imageOffset, self.vminLcp, self.vmaxLcp)
                self.rcpCanvas.imshow(self.rcpData*self.imageScale + self.imageOffset, self.vminRcp, self.vmaxRcp)
            if self.indexOfImageType == 4:
                self.lcpCanvas.contour(self.psfData*self.imageScale + self.imageOffset, NP.max(self.psfData)*.5*self.imageScale + self.imageOffset)
        elif self.indexOfImageType == 1:
            if self.autoscaleButton.isChecked():
                self.lcpCanvas.imshow(self.iData*.5, NP.min(self.iData), NP.max(self.iData))
                self.rcpCanvas.imshow(self.vData*.5, NP.min(self.vData), NP.max(self.vData))
            else:
                self.lcpCanvas.imshow(self.iData*.5*self.imageScale + self.imageOffset, self.vminI, self.vmaxI)
                self.rcpCanvas.imshow(self.vData*.5*self.imageScale + self.imageOffset, self.vminV, self.vmaxV)
        elif self.indexOfImageType == 2:
            if self.autoscaleButton.isChecked():
                self.lcpCanvas.imshow(NP.abs(self.srhFits.uvLcp[self.uvSize//2-150:self.uvSize//2+150, self.uvSize//2-150:self.uvSize//2+150])**.5, NP.min(NP.abs(self.srhFits.uvLcp)**.5), NP.max(NP.abs(self.srhFits.uvLcp)**.5))
                self.rcpCanvas.imshow(NP.abs(self.srhFits.uvRcp[self.uvSize//2-150:self.uvSize//2+150, self.uvSize//2-150:self.uvSize//2+150])**.5, NP.min(NP.abs(self.srhFits.uvRcp)**.5), NP.max(NP.abs(self.srhFits.uvRcp)**.5))
            else:
                self.lcpCanvas.imshow(NP.abs(self.srhFits.uvLcp[self.uvSize//2-150:self.uvSize//2+150, self.uvSize//2-150:self.uvSize//2+150])**.5 *self.imageScale + self.imageOffset, NP.min(NP.abs(self.srhFits.uvLcp)**.5), NP.max(NP.abs(self.srhFits.uvLcp)**.5))
                self.rcpCanvas.imshow(NP.abs(self.srhFits.uvRcp[self.uvSize//2-150:self.uvSize//2+150, self.uvSize//2-150:self.uvSize//2+150])**.5 *self.imageScale + self.imageOffset, NP.min(NP.abs(self.srhFits.uvRcp)**.5), NP.max(NP.abs(self.srhFits.uvRcp)**.5))
        elif self.indexOfImageType == 3:
            if self.autoscaleButton.isChecked():
                self.lcpCanvas.imshow(NP.angle(self.srhFits.uvLcp[self.uvSize//2-150:self.uvSize//2+150, self.uvSize//2-150:self.uvSize//2+150]), NP.min(NP.angle(self.srhFits.uvLcp)), NP.max(NP.angle(self.srhFits.uvLcp)))
                self.rcpCanvas.imshow(NP.angle(self.srhFits.uvRcp[self.uvSize//2-150:self.uvSize//2+150, self.uvSize//2-150:self.uvSize//2+150]), NP.min(NP.angle(self.srhFits.uvRcp)), NP.max(NP.angle(self.srhFits.uvRcp)))
            else:
                self.lcpCanvas.imshow(NP.angle(self.srhFits.uvLcp[self.uvSize//2-150:self.uvSize//2+150, self.uvSize//2-150:self.uvSize//2+150]) *self.imageScale + self.imageOffset, NP.min(NP.angle(self.srhFits.uvLcp)), NP.max(NP.angle(self.srhFits.uvLcp)))
                self.rcpCanvas.imshow(NP.angle(self.srhFits.uvRcp[self.uvSize//2-150:self.uvSize//2+150, self.uvSize//2-150:self.uvSize//2+150]) *self.imageScale + self.imageOffset, NP.min(NP.angle(self.srhFits.uvRcp)), NP.max(NP.angle(self.srhFits.uvRcp)))
            
        if self.sunContourButton.isChecked():
            self.lcpCanvas.contour(self.qSun, [0.5])
            

        if self.indexOfPlotType == 1:
            self.lcpMaxCanvas.clear()
            self.rcpMaxCanvas.clear()
            self.lcpMaxCanvas.plot_xy((self.bLcp[:-1]+self.bLcp[1:])/2, self.hLcp)
            self.rcpMaxCanvas.plot_xy((self.bRcp[:-1]+self.bRcp[1:])/2, self.hRcp)
            
        elif self.indexOfPlotType == 0:
            self.lcpMaxCanvas.clear()
            self.rcpMaxCanvas.clear()
            self.lcpMaxCanvas.plot(self.lcpMaxTrace)
            self.rcpMaxCanvas.plot(self.rcpMaxTrace)
            
        elif self.indexOfPlotType == 3:
            self.lcpMaxCanvas.clear()
            self.rcpMaxCanvas.clear()
            self.lcpMaxCanvas.plot(self.lcpMaxTrace)
            self.rcpMaxCanvas.plot(self.rcpMaxTrace)
        
    def onEastWestLcpPhaseSlopeChanged(self, value):
        self.srhFits.ewSlopeLcp[self.currentFrequencyChannel] = value
        self.srhFits.buildEwPhase()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()

    def onEastWestRcpPhaseSlopeChanged(self, value):
        self.srhFits.ewSlopeRcp[self.currentFrequencyChannel] = value
        self.srhFits.buildEwPhase()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()

    def onSouthLcpPhaseSlopeChanged(self, value):
        self.srhFits.sSlopeLcp[self.currentFrequencyChannel] = value
        self.srhFits.buildSPhase()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()

    def onSouthRcpPhaseSlopeChanged(self, value):
        self.srhFits.sSlopeRcp[self.currentFrequencyChannel] = value
        self.srhFits.buildSPhase()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
 
    def onSouthPhaseStairLcpChanged(self, value):
        self.srhFits.sLcpStair[self.currentFrequencyChannel] = value
        self.srhFits.buildSPhase()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
        
    def onSouthPhaseStairRcpChanged(self, value):
        self.srhFits.sRcpStair[self.currentFrequencyChannel] = value
        self.srhFits.buildSPhase()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()           
 
    def onCenterDisk(self):
        self.srhFits.centerDisk()
        self.buildImage()
        self.showImage()
        self.ewLcpPhaseSlopeSpin.setValue(self.srhFits.ewSlopeLcp[self.currentFrequencyChannel])
        self.sLcpPhaseSlopeSpin.setValue(self.srhFits.sSlopeLcp[self.currentFrequencyChannel])
        self.ewRcpPhaseSlopeSpin.setValue(self.srhFits.ewSlopeRcp[self.currentFrequencyChannel])
        self.sRcpPhaseSlopeSpin.setValue(self.srhFits.sSlopeRcp[self.currentFrequencyChannel])
        self.sPhaseStairLcp.setValue(self.srhFits.sLcpStair[self.currentFrequencyChannel])
        self.sPhaseStairRcp.setValue(self.srhFits.sRcpStair[self.currentFrequencyChannel])

    def onEastWestPhaseStairLcpChanged(self, value):
        self.ewPhaseCoefsLcp[self.currentFrequencyChannel, self.ewStairLength - 1] = value
        self.srhFits.buildEwPhase()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()

    def onEastWestPhaseStairRcpChanged(self, value):
        self.ewPhaseCoefsRcp[self.currentFrequencyChannel, self.ewStairLength - 1] = value
        self.srhFits.buildEwPhase()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()

    def onEwPhaseStairLengthChanged(self, value):
        self.ewStairLength = value
        self.ewPhaseStairLcp.setValue(self.ewPhaseCoefsLcp[self.currentFrequencyChannel, self.ewStairLength - 1])
        self.ewPhaseStairRcp.setValue(self.ewPhaseCoefsRcp[self.currentFrequencyChannel, self.ewStairLength - 1])

    def onSPhaseStairLengthChanged(self, value):
        self.sStairLength = value
        self.sPhaseStairLcp.setValue(self.sPhaseCoefsLcp[self.currentFrequencyChannel])
        self.sPhaseStairRcp.setValue(self.sPhaseCoefsRcp[self.currentFrequencyChannel])

    def onClear(self):
        self.lcpMaxTrace = []
        self.rcpMaxTrace = []
        self.lcpMaxCanvas.clear()
        self.rcpMaxCanvas.clear()
        
    def onImageUpdate(self, value):
        self.imageUpdate = value
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onCalibrate(self):
        self.srhFits.calibrate()
        self.srhFits.setFrequencyChannel(self.currentFrequencyChannel)
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()

    def onImageAnimate(self, value):
        self.imageAnimate = value
        if (self.imageAnimate):
            self.animTimer.start(100)
        else:
            self.animTimer.stop()

    def onPhaseCorrect(self, value):
        self.phaseCorrect = value
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()

    def onAmplitudeCorrect(self, value):
        self.amplitudeCorrect = value
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onCalculatePhase(self):
        self.srhFits.baselines = 5
        self.srhFits.calculatePhaseCalibration(baselinesNumber = self.srhFits.baselines)
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onCalculatePhaseCurrent(self):
        self.srhFits.updateAntennaPhase(self.srhFits.frequencyChannel, baselinesNumber = self.srhFits.baselines)
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onDoubleBaselinesAmplitude(self, value):
        self.srhFits.useDoubleBaselinesAmp = value
        if not self.srhFits.useNonlinearApproach:
            self.srhFits.updateAntennaAmplitude()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onFrequencyChannelChanged(self, value):
        self.currentFrequencyChannel = value
        self.srhFits.setFrequencyChannel(value)
        self.frequencyList.setCurrentIndex(self.currentFrequencyChannel)
        self.ewLcpPhaseSlopeSpin.setValue(self.srhFits.ewSlopeLcp[self.currentFrequencyChannel])
        self.ewRcpPhaseSlopeSpin.setValue(self.srhFits.ewSlopeRcp[self.currentFrequencyChannel])
        self.sLcpPhaseSlopeSpin.setValue(self.srhFits.sSlopeLcp[self.currentFrequencyChannel])
        self.sRcpPhaseSlopeSpin.setValue(self.srhFits.sSlopeRcp[self.currentFrequencyChannel])
        self.sPhaseStairLcp.setValue(self.srhFits.sLcpStair[self.currentFrequencyChannel])
        self.sPhaseStairRcp.setValue(self.srhFits.sRcpStair[self.currentFrequencyChannel])
        # self.flaggedAntsLabel.setText('Flagged antennas:  LCP ' + str(NP.where(self.srhFits.badAntsLcp[self.currentFrequencyChannel]==1)[0]) + ';  RCP ' + str(NP.where(self.srhFits.badAntsRcp[self.currentFrequencyChannel]==1)[0]))
        # self.flaggedAntsBoxLcp.setText('')
        # self.flaggedAntsBoxRcp.setText('')
        if (self.imageUpdate):
#            self.buildSPhase()
#            self.buildEwPhase()
            self.buildImage()
            self.showImage()

    def onScanChanged(self, value):
        self.currentScan = value
        self.srhFits.getHourAngle(self.currentScan)
        self.timeList.setCurrentIndex(self.currentScan)
        if self.setCalibScanToCurrentScan:
#            self.onCalibScanChanged(value)
            self.calibScan.setValue(value)
        elif (self.imageUpdate):
            self.buildImage()
            self.showImage()

    def onEwAntennaChanged(self, value):
        self.ewAntennaNumber = value
        self.ewLcpAntennaPhase.setValue(self.ewLcpPhaseAnt[self.ewAntennaNumber - 49])
        self.ewRcpAntennaPhase.setValue(self.ewRcpPhaseAnt[self.ewAntennaNumber - 49])
    
    def onSAntennaChanged(self, value):
        self.sAntennaNumber = value
        self.sLcpAntennaPhase.setValue(self.sLcpPhaseAnt[15 - (self.sAntennaNumber - 177)])
        self.sRcpAntennaPhase.setValue(self.sRcpPhaseAnt[15 - (self.sAntennaNumber - 177)])
    
    def onEwLcpAntennaPhaseChanged(self, value):
        self.ewLcpPhaseAnt[self.ewAntennaNumber - 49] = value
        self.buildEwPhase()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
    
    def onSLcpAntennaPhaseChanged(self, value):
        self.sLcpPhaseAnt[15 - (self.sAntennaNumber - 177)] = value
        self.buildSPhase()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
    
    def onEwRcpAntennaPhaseChanged(self, value):
        self.ewRcpPhaseAnt[self.ewAntennaNumber - 49] = value
        self.buildEwPhase()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
    
    def onSRcpAntennaPhaseChanged(self, value):
        self.sRcpPhaseAnt[15 - (self.sAntennaNumber - 177)] = value
        self.buildSPhase()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
    
    def onCalibScanChanged(self, value):
        self.srhFits.setCalibIndex(value)
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()

    def onImageOffsetSlider(self, value):
        if self.indexOfImageType == 2:
            self.imageOffset = value*(NP.max(NP.abs(self.srhFits.uvLcp)**.5) - NP.min(NP.abs(self.srhFits.uvLcp)**.5))*0.01
        elif self.indexOfImageType == 3:
            self.imageOffset = value*(NP.max(NP.angle(self.srhFits.uvLcp)) - NP.min(NP.angle(self.srhFits.uvLcp)))*0.01
        else:
            self.imageOffset = value*(NP.max(self.lcpData) - NP.min(self.lcpData))*0.01
        if (self.imageUpdate):
            self.showImage()

    def onImageScaleSlider(self, value):
        self.imageScale = value*0.1
        if (self.imageUpdate):
            self.showImage()
        
    def onAnimTimer(self):
        if (self.currentScan < self.srhFits.dataLength):
            if self.averScans:
                self.currentScan += self.averScans
            else:
                self.currentScan += 1
            self.scan.setValue(self.currentScan)
        else:
            self.animTimer.stop()
            self.imageAnimateButton.setChecked(False)
            self.scan.setValue(0)
        
    def onFrequencyListSelected(self):
        self.frequencyChannel.setValue(self.frequencyList.currentIndex())
        
    def onTimeListSelected(self):
        self.scan.setValue(self.timeList.currentIndex())
        
    def onCanvasXyChangedLcp(self, x, y):
        self.xInd = int(x)
        self.yInd = int(y)
        if self.fitsIsOpen:
            if self.indexOfImageType == 1:
                self.lcpTextBox.setText('x: ' + str(self.xInd) + ', y: ' + str(self.yInd) + ', z: %.4g' % self.iData[self.yInd,self.xInd])
            else:
                self.lcpTextBox.setText('x: ' + str(self.xInd) + ', y: ' + str(self.yInd) + ', z: %.4g' % self.lcpData[self.yInd,self.xInd])
        
            if self.indexOfPlotType == 2:
                self.lcpMaxCanvas.clear()
                if self.indexOfImageType == 1:
                    self.lcpMaxCanvas.plot(self.iData[:,self.xInd])
                    self.lcpMaxCanvas.plot(self.iData[self.yInd,:])
                else:
                    self.lcpMaxCanvas.plot(self.lcpData[:,self.xInd])
                    self.lcpMaxCanvas.plot(self.lcpData[self.yInd,:])
            
            
#            self.lcpMaxCanvas.clear()
#            if self.indexTypeOfImage == 0:
#                self.lcpMaxCanvas.plot(self.lcpData[self.yInd,:])
#                self.lcpMaxCanvas.plot(self.lcpData[:,self.xInd])
#            else:
#                self.lcpMaxCanvas.plot((self.lcpData[self.yInd,:] + self.rcpData[self.yInd,:]*self.lcpRcpRel)*.5)
#                self.lcpMaxCanvas.plot((self.lcpData[:,self.xInd] + self.rcpData[:,self.xInd]*self.lcpRcpRel)*.5)
#        
#            self.rcpMaxCanvas.clear()
#            if self.indexTypeOfImage == 0:
#                self.rcpMaxCanvas.plot(self.rcpData[self.yInd,:])
#                self.rcpMaxCanvas.plot(self.rcpData[:,self.xInd])
#            else:
#                self.rcpMaxCanvas.plot((self.lcpData[self.yInd,:] - self.rcpData[self.yInd,:]*self.lcpRcpRel)*.5)
#                self.rcpMaxCanvas.plot((self.lcpData[:,self.xInd] - self.rcpData[:,self.xInd]*self.lcpRcpRel)*.5)

    def onCanvasXyChangedRcp(self, x, y):
        self.xInd = int(x)
        self.yInd = int(y)
        if self.fitsIsOpen:
            if self.indexOfImageType == 1:
                self.rcpTextBox.setText('x: ' + str(self.xInd) + ', y: ' + str(self.yInd) + ', z: %.4g' % self.vData[self.yInd,self.xInd])
            else:
                self.rcpTextBox.setText('x: ' + str(self.xInd) + ', y: ' + str(self.yInd) + ', z: %.4g' % self.rcpData[self.yInd,self.xInd])
        
            if self.indexOfPlotType == 2:
                self.rcpMaxCanvas.clear()
                if self.indexOfImageType == 1:
                    self.rcpMaxCanvas.plot(self.vData[:,self.xInd])
                    self.rcpMaxCanvas.plot(self.vData[self.yInd,:])
                else:
                    self.rcpMaxCanvas.plot(self.rcpData[:,self.xInd])
                    self.rcpMaxCanvas.plot(self.rcpData[self.yInd,:])

    def onTypeOfImage(self, index):
        self.indexOfImageType = index
        if index == 6:
            self.onFreqDiff(self.freqDiffSpin.value())
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
        
    def onTypeOfFrame(self, index):
        self.indexOfFrameType = index
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()

    def onLcpRcpRelationChanged(self, value):
        self.lcpRcpRel = value
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onHistogram(self):
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onTypeOfPlot(self, index):
        self.indexOfPlotType = index
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()

    def onBinsChanged(self, value):
        self.bins = value
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onCalibrateBrightness(self):
        h_l, b_l = NP.histogram(self.srhFits.lcp.real, bins = 100)
        h_r, b_r = NP.histogram(self.srhFits.rcp.real, bins = 100)
        sun_l = b_l[NP.argmax(h_l[60:])+60]
        sun_r = b_r[NP.argmax(h_r[60:])+60]
        self.calCoefLcp = 16000./sun_l
        self.calCoefRcp = 16000./sun_r
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
        
    def onSunContour(self, value):
        if (self.imageUpdate):
            self.showImage()
            
    def onAutoscale(self, value):
        if (self.imageUpdate):
            self.buildImage()
            if self.indexOfImageType == 1:
                self.vminI = self.iData.min() * 0.5
                self.vmaxI = self.iData.max() * 0.5
                self.vminV = self.vData.min() * 0.5
                self.vmaxV = self.vData.max() * 0.5
            else:
                self.vminLcp = self.lcpData.min()
                self.vmaxLcp = self.lcpData.max()
                self.vminRcp = self.lcpData.min()
                self.vmaxRcp = self.lcpData.max()
        self.showImage()
            
    def onAutoFlag(self):
        pass
        
    def onObjectChanged(self, obj):
        self.srhFits.changeObject(obj)
        
    def onFlag(self):
        l=self.flaggedAntsBoxLcp.text()
        r=self.flaggedAntsBoxRcp.text()
        try:
            ant_ind = NP.array(l.split(',')).astype(NP.int)
            ew_ind = ant_ind[ant_ind<98] - 1
            self.srhFits.ewAntAmpLcp[self.currentFrequencyChannel, ew_ind] = 1e6
            n_ind = ant_ind[ant_ind>=98] - 98
            self.srhFits.nAntAmpLcp[self.currentFrequencyChannel, n_ind] = 1e6
        except:
            pass
        try:
            ant_ind = NP.array(r.split(',')).astype(NP.int)
            ew_ind = ant_ind[ant_ind<98] - 1
            self.srhFits.ewAntAmpRcp[self.currentFrequencyChannel, ew_ind] = 1e6
            n_ind = ant_ind[ant_ind>=98] - 98
            self.srhFits.nAntAmpRcp[self.currentFrequencyChannel, n_ind] = 1e6
        except:
            pass
        flaggedLcp = str(NP.where(self.srhFits.ewAntAmpLcp[self.currentFrequencyChannel]==1e6)[0] + 1) + ', ' + str(NP.where(self.srhFits.nAntAmpLcp[self.currentFrequencyChannel]==1e6)[0] + 98)
        flaggedRcp = str(NP.where(self.srhFits.ewAntAmpRcp[self.currentFrequencyChannel]==1e6)[0] + 1) + ', ' + str(NP.where(self.srhFits.nAntAmpRcp[self.currentFrequencyChannel]==1e6)[0] + 98)
        self.flaggedAntsLabel.setText('Flagged antennas:  LCP ' + flaggedLcp + ';  RCP ' + flaggedRcp)
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onEwAmpCoefChanged(self, value):
        self.ewAmpCoef = value
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onSAmpCoefChanged(self, value):
        self.sAmpCoef = value
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
        
    def onAverageCalib(self, value):
        self.srhFits.averageCalib = value
        if value:
            self.srhFits.updateAntennaAmplitude(average = self.averCalib)
            self.srhFits.updateAntennaPhaseFull(average = self.averCalib)
        else:
            self.srhFits.updateAntennaAmplitude()
            self.srhFits.updateAntennaPhaseFull()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onAverageScans(self, value):
        self.averScans = value
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onFreqDiff(self, value):
        self.srhFits.setFrequencyChannel(value)
        self.srhFits.vis2uv(self.currentScan, phaseCorrect=self.phaseCorrect, amplitudeCorrect=self.amplitudeCorrect, PSF=False, average = self.averScans);
        self.srhFits.uv2lmImage()
        self.srhFits.lm2Heliocentric()
        self.freqDiffLcp_lm = self.srhFits.lcp.copy()
        self.freqDiffRcp_lm = self.srhFits.rcp.copy()
        self.srhFits.setFrequencyChannel(self.currentFrequencyChannel)
        
    def onfreqDiffRelationLcpChanged(self, value):
        self.diffRelationLcp = value
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
        
    def onfreqDiffRelationRcpChanged(self, value):
        self.diffRelationRcp = value
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
        
    def closeEvent(self, event):
        close = QtWidgets.QMessageBox()
        close.setText("Are you sure to exit?")
        close.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        close = close.exec()

        if close == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()        

#        
    def antennasPhases(self):
        ewPhases = NP.zeros((self.srhFits.dataLength,2, 32))
        sPhases = NP.zeros((self.srhFits.dataLength,2, 16))
        for scan in range(self.srhFits.dataLength):
            self.srhFits.calibIndex = scan
            self.srhFits.updateAntennaPhase()
            ewPhases[scan, 0, :] = self.srhFits.ewAntPhaLcp[1:]
            ewPhases[scan, 1, :] = self.srhFits.ewAntPhaRcp[1:]
            sPhases[scan, 0] = self.srhFits.sAntPhaLcp[1:]
            sPhases[scan, 1] = self.srhFits.sAntPhaRcp[1:]        
        return ewPhases, sPhases
    
    def onNonlinear(self, value):
        self.srhFits.useNonlinearApproach = value
            
    def onWeights(self, value):
        self.useWeights = value
        self.srhFits.updateAntennaAmplitude(average = self.averCalib, useWeights = self.useWeights)
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
            
    def onFringeStopping(self, value):
        self.srhFits.fringeStopping = value
        if(value):
            self.srhFits.calculateDelays()
        else:
            self.srhFits.resetDelays()
        if (self.imageUpdate):
            self.buildImage()
            self.showImage()
    
    def onOpenMS(self):
        self.MSName = QtWidgets.QFileDialog.getExistingDirectory(self)
        self.MSTextbox.setText(self.MSName)
        self.cleanTable.item(0,1).setText(self.MSName)
        self.cleanTable.item(1,1).setText(os.path.dirname(self.MSName)+'/images/imagename')
        self.gaincalTable.item(0,1).setText(self.MSName)
        self.gaincalTable.item(1,1).setText(os.path.dirname(self.MSName)+'/gaintable.gcal')
        self.applycalTable.item(0,1).setText(self.MSName)
        
    def initCleanDict(self):
       self.cleanParams = {'vis = ':'\'\'',
                            'imagename = ':'\'\'',
                            'cell = ':'2.45',
                            'scan = ' : '0~19',
                            'stokes = ' : 'RRLL',
                            'datacolumn = ' : 'corrected',
                            'imsize = ':'1024',
                            'niter = ':'100000',
                            'gain = ' : '0.1',
                            'threshold = ' : '30000Jy',
                            'deconvolver = ' : 'hogbom',
                            'restoringbeam = ' : ''}
#        self.cleanParams = {'vis':self.MSName,
#                            'imagename':os.path.dirname(self.MSName)+'/images/imagename',
#                            'cell':10.0,
#                            'niter':100000,
#                            'gain' : 0.1,
#                            'datacolumn' : 'corrected',
#                            'imsize':[300,300],
#                            'scan' : '12',
#                            'threshold' : '5mJy'}
        
    def createCleanTable(self):
        rowNumber = len(self.cleanParams)
        self.cleanTable = QTableWidget(self)
        self.cleanTable.setRowCount(rowNumber)
        self.cleanTable.setColumnCount(3)
        for i, (k,v) in enumerate(self.cleanParams.items()):
            item = QTableWidgetItem(k)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.cleanTable.setItem(i, 0, item)
            item = QTableWidgetItem(v)
            self.cleanTable.setItem(i, 1, item)
            item = QTableWidgetItem(desc.cleanParamsDesc[k])
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.cleanTable.setItem(i, 2, item)
            
    def onMakeClean(self):
        for i in range(len(self.cleanParams)):
            self.cleanParams[self.cleanTable.item(i,0).text()] = self.cleanTable.item(i,1).text()
        
        tclean(vis = self.cleanParams['vis = '],
               imagename = self.cleanParams['imagename = '],
               cell = float(self.cleanParams['cell = ']),
               scan = self.cleanParams['scan = '],
               stokes = self.cleanParams['stokes = '],
               datacolumn = self.cleanParams['datacolumn = '],
               imsize = int(self.cleanParams['imsize = ']),
               niter = int(self.cleanParams['niter = ']),
               gain = float(self.cleanParams['gain = ']),
               threshold = self.cleanParams['threshold = '],
               deconvolver = self.cleanParams['deconvolver = '],
               restoringbeam = self.cleanParams['restoringbeam = '])

        
        self.casaLcpData = image(self.cleanParams['imagename = ']+'.image').getdata()[0][0]
        self.casaRcpData = image(self.cleanParams['imagename = ']+'.image').getdata()[0][1]
        self.casaImsize = int(self.cleanParams['imsize = '])
        shift = AffineTransform(translation=(-self.casaImsize/2,-self.casaImsize/2))
        rotate = AffineTransform(rotation = -self.pAngle)
        back_shift = AffineTransform(translation=(self.casaImsize/2,self.casaImsize/2))
        self.casaLcpData = warp(self.casaLcpData,(shift + (rotate + back_shift)).inverse)
        self.casaRcpData = warp(self.casaRcpData,(shift + (rotate + back_shift)).inverse)
        self.im_l = self.casaLcpData
        self.im_r = self.casaRcpData
        self.casaLeftCanvas.imshow(self.im_l*self.casaLeftScale + self.casaLeftOffset, NP.min(self.im_l), NP.max(self.im_l))
        self.casaRightCanvas.imshow(self.im_r*self.casaRightScale + self.casaRightOffset, NP.min(self.im_r), NP.max(self.im_r))
        
    def initGaincalDict(self):
       self.gaincalParams = {'vis = ':'\'\'',
                             'caltable = ':'\'\'',
                             'gaintype = ':'\'G\'',
                             'calmode = ':'\'ap\'',
                             'scan = ':'\'0\'',
                             'refant = ':'\'16\'',
                             'minsnr = ':'2.0',
                             'solint = ':'\'int\''}
        
    def createGaincalTable(self):
        rowNumber = len(self.gaincalParams)
        self.gaincalTable = QTableWidget(self)
        self.gaincalTable.setRowCount(rowNumber)
        self.gaincalTable.setColumnCount(3)
        for i, (k,v) in enumerate(self.gaincalParams.items()):
            item = QTableWidgetItem(k)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.gaincalTable.setItem(i, 0, item)
            item = QTableWidgetItem(v)
            self.gaincalTable.setItem(i, 1, item)
            item = QTableWidgetItem(desc.gaincalParamsDesc[k])
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.gaincalTable.setItem(i, 2, item)
            
    def onMakeGaincal(self):
        for i in range(len(self.gaincalParams)):
            self.gaincalParams[self.gaincalTable.item(i,0).text()] = self.gaincalTable.item(i,1).text()
        gaincal_script = os.path.join(os.path.dirname(self.MSName), 'cal.py')
        with open(gaincal_script, 'w') as f:
            for i in range(len(self.gaincalParams)):
                if self.gaincalTable.item(i,1).text() != '' and self.gaincalTable.item(i,1).text() != '\'\'':
                    f.write(self.gaincalTable.item(i,0).text() + self.gaincalTable.item(i,1).text() + '\n')
            f.write('gaincal()')
        os.system('casa -c ' + gaincal_script.replace(' ', '\ '))
        self.applycalTable.item(1,1).setText(self.gaincalParams['vis = '])
        
    def initApplycalDict(self):
       self.applycalParams = {'vis = ':'\'\'',
                             'caltable = ':'\'\'',
                             'calwt = ':'True'}
        
    def createApplycalTable(self):
        rowNumber = len(self.applycalParams)
        self.applycalTable = QTableWidget(self)
        self.applycalTable.setRowCount(rowNumber)
        self.applycalTable.setColumnCount(3)
        for i, (k,v) in enumerate(self.applycalParams.items()):
            item = QTableWidgetItem(k)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.applycalTable.setItem(i, 0, item)
            item = QTableWidgetItem(v)
            self.applycalTable.setItem(i, 1, item)
            item = QTableWidgetItem(desc.applycalParamsDesc[k])
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.applycalTable.setItem(i, 2, item)
            
    def onMakeApplycal(self):
        for i in range(len(self.applycalParams)):
            self.applycalParams[self.applycalTable.item(i,0).text()] = self.applycalTable.item(i,1).text()
        applycal_script = os.path.join(os.path.dirname(self.MSName), 'applycal.py')
        with open(applycal_script, 'w') as f:
            for i in range(len(self.applycalParams)):
                f.write(self.applycalTable.item(i,0).text() + self.applycalTable.item(i,1).text() + '\n')
            f.write('applycal()')
        os.system('casa -c ' + applycal_script.replace(' ', '\ '))
        
    def onPlotms(self):
        os.system('casaplotms')
        
    def onViewer(self):
        os.system('casaviewer')
        
    def onCasaRightOffsetSlider(self, value):
        self.casaRightOffset = value*0.01*NP.max(self.im_r)
        self.buildCasaImage()

    def onCasaRightScaleSlider(self, value):
        self.casaRightScale = value*0.1
        self.buildCasaImage()
        
    def onCasaLeftOffsetSlider(self, value):
        self.casaLeftOffset = value*0.01*NP.max(self.im_l)
        self.buildCasaImage()

    def onCasaLeftScaleSlider(self, value):
        self.casaLeftScale = value*0.1
        self.buildCasaImage()
        
    def onTypeOfLeftCasaImage(self):
        if self.typeOfLeftCasaImage.currentIndex() == 0:
            self.im_l = self.casaLcpData
        if self.typeOfLeftCasaImage.currentIndex() == 1:
            self.im_l = self.casaRcpData
        if self.typeOfLeftCasaImage.currentIndex() == 2:
            self.im_l = self.casaLcpData + self.casaRcpData
        if self.typeOfLeftCasaImage.currentIndex() == 3:
            self.im_l = self.casaLcpData - self.casaRcpData
        if self.typeOfLeftCasaImage.currentIndex() == 4:
            self.im_l = image(self.cleanParams['imagename = ']+'.model').getdata()[0][0]
        if self.typeOfLeftCasaImage.currentIndex() == 5:
            self.im_l = image(self.cleanParams['imagename = ']+'.model').getdata()[0][1]
        if self.typeOfLeftCasaImage.currentIndex() == 6:
            self.im_l = image(self.cleanParams['imagename = ']+'.psf').getdata()[0][0]
        if self.typeOfLeftCasaImage.currentIndex() == 7:
            self.im_l = image(self.cleanParams['imagename = ']+'.pb').getdata()[0][0]
        self.casaLeftCanvas.imshow(self.im_l*self.casaLeftScale + self.casaLeftOffset, NP.min(self.im_l), NP.max(self.im_l))
#        self.buildCasaImage()
        
    def onTypeOfRightCasaImage(self):
        if self.typeOfRightCasaImage.currentIndex() == 0:
            self.im_r = self.casaLcpData
        if self.typeOfRightCasaImage.currentIndex() == 1:
            self.im_r = self.casaRcpData
        if self.typeOfRightCasaImage.currentIndex() == 2:
            self.im_r = self.casaLcpData + self.casaRcpData
        if self.typeOfRightCasaImage.currentIndex() == 3:
            self.im_r = self.casaLcpData - self.casaRcpData
        if self.typeOfRightCasaImage.currentIndex() == 4:
            self.im_r = image(self.cleanParams['imagename = '][1:-1]+'.model').getdata()[0][0]
        if self.typeOfRightCasaImage.currentIndex() == 5:
            self.im_r = image(self.cleanParams['imagename = '][1:-1]+'.model').getdata()[0][1]
        if self.typeOfRightCasaImage.currentIndex() == 6:
            self.im_r = image(self.cleanParams['imagename = '][1:-1]+'.psf').getdata()[0][0]
        if self.typeOfRightCasaImage.currentIndex() == 7:
            self.im_r = image(self.cleanParams['imagename = '][1:-1]+'.pb').getdata()[0][0]
        self.casaRightCanvas.imshow(self.im_r*self.casaRightScale + self.casaRightOffset, NP.min(self.im_r), NP.max(self.im_r))
#        self.buildCasaImage()
    
    def buildCasaImage(self): 
        self.casaLeftCanvas.setData(self.im_l*self.casaLeftScale + self.casaLeftOffset)
        self.casaRightCanvas.setData(self.im_r*self.casaRightScale + self.casaRightOffset)
        
    def onCasaSaveAs(self):
        saveName, _ = QtWidgets.QFileDialog.getSaveFileName(self)        
        if saveName:
            
                    
            scan = self.currentScan
            nameParts = saveName.split('.') 
            fitsName = ('%s_%s.%s' % (nameParts[0], self.timeList.itemText(scan), nameParts[1]))
    
            pHeader = fits.Header();
            pHeader['DATE-OBS']     = self.srhFits.hduList[0].header['DATE-OBS'] + 'T' + self.timeList.itemText(scan)
            pHeader['T-OBS']     = self.srhFits.hduList[0].header['DATE-OBS'] + 'T' + self.timeList.itemText(scan)#self.srhFits.hduList[0].header['TIME-OBS']
            pHeader['INSTRUME']     = self.srhFits.hduList[0].header['INSTRUME']
            pHeader['ORIGIN']       = self.srhFits.hduList[0].header['ORIGIN']
            pHeader['OBS-LAT']      = self.srhFits.hduList[0].header['OBS-LAT']
            pHeader['OBS-LONG']     = self.srhFits.hduList[0].header['OBS-LONG']
            pHeader['OBS-ALT']      = self.srhFits.hduList[0].header['OBS-ALT']
            pHeader['FR_CHAN']      = self.srhFits.hduList[0].header['FR_CHAN']
            pHeader['FREQUENC']     = ('%3.3f') % (self.srhFits.freqList[self.currentFrequencyChannel]*1e3)
            pHeader['CDELT1']       = float(self.cleanParams['cell = '])
            pHeader['CDELT2']       = float(self.cleanParams['cell = '])
            pHeader['CRPIX1']       = float(self.cleanParams['imsize = ']) // 2
            pHeader['CRPIX2']       = float(self.cleanParams['imsize = ']) // 2
            pHeader['CTYPE1']       = 'HPLN-TAN'
            pHeader['CTYPE2']       = 'HPLT-TAN'
            pHeader['CUNIT1']       = 'arcsec'
            pHeader['CUNIT2']       = 'arcsec'
            
    
            savePath, saveExt = fitsName.split('.')
            
            pHdu = fits.PrimaryHDU(self.casaLcpData + self.casaRcpData, header=pHeader);
            hduList = fits.HDUList([pHdu]);
            hduList.writeto(savePath + '_I.' + saveExt, overwrite=True);
            hduList.close();
            
            pHdu = fits.PrimaryHDU(self.casaLcpData - self.casaRcpData, header=pHeader);
            hduList = fits.HDUList([pHdu]);
            hduList.writeto(savePath + '_V.' + saveExt, overwrite=True);
            
            hduList.close();
                
    def onDiskModel(self):
        model_image = image(self.cleanParams['imagename = '][1:-1]+'.model')
        arcsecPerPix = NP.abs(NP.rad2deg(model_image.info()['coordinates']['direction0']['cdelt'][0]))*3600.
        N = model_image.info()['coordinates']['direction0']['_axes_sizes'][0]
        arcsecRadius = 1020
        radius = int(arcsecRadius/arcsecPerPix +0.5)
        disk_level = 1
        self.model = NP.zeros(N)
        for i in range(N):
            for j in range(N):
                x=i - N/2
                y=j - N/2
                if (NP.sqrt(x**2 + y**2) < radius):
                    self.model[i, j] = disk_level
        model_image.putdata(self.model)
        model_image.saveas(self.cleanParams['imagename = '][1:-1]+'_model')
        self.casaModelCanvas.imshow(self.model)
                    
        
    def onPointModel(self):    
        model_image = image(self.cleanParams['imagename = '][1:-1]+'.model')
        N = model_image.info()['coordinates']['direction0']['_axes_sizes'][0]
        point_level = 1
        self.model = NP.zeros(N)
        self.model[N//2, N//2] = point_level
        model_image.putdata(self.model)
        model_image.saveas(self.cleanParams['imagename = '][1:-1]+'_model')
        self.casaModelCanvas.imshow(self.model)
        
    def onSetModel(self):
        setjy_script = os.path.join(os.path.dirname(self.MSName), 'edikSetjy.py')
        with open(setjy_script, 'w') as f:
            f.write('setjy(vis = \''+self.MSName+'\', standard = \'manual\', model = \''+self.cleanParams['imagename = '][1:-1]+'_model\')')
#        os.system('casa -c ' + setjy_script.replace(' ', '\ '))

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self,parent)
        
        self.fitsIsOpen = False

        self.currentFrequencyChannel = 0
        self.currentScan = 0
        self.imageOffset = 0
        self.imageScale = 1
        self.phaseCorrect = True
        self.amplitudeCorrect = True
        self.indexOfFrameType = 0
        self.pAngle = 0.
        self.ewStairLength = 16
        self.sStairLength = 16
        self.lcpRcpRel = 1.
        self.arcsecPerPixel = 4.91104/2
        self.bins = 1000
        self.calCoefLcp = 1.
        self.calCoefRcp = 1.
        self.ewAmpCoef = 1.
        self.sAmpCoef = 1.
        self.averScans = 0
        self.averCalib = 0
        self.indexOfPlotType = 0
        self.indexOfImageType = 0
        self.setCalibScanToCurrentScan = False
        self.useWeights = False
        self.diffRelationLcp = 1.
        self.diffRelationRcp = 1.

        self.lcpMaxTrace = []
        self.rcpMaxTrace = []
        self.ewLcpPhaseAnt = NP.zeros(32, dtype=int)
        self.ewRcpPhaseAnt = NP.zeros(32, dtype=int)
        self.sLcpPhaseAnt = NP.zeros(16, dtype=int)
        self.sRcpPhaseAnt = NP.zeros(16, dtype=int)
        self.ewAntennaNumber = 128
        self.sAntennaNumber = 64
        self.imageUpdate = False
        self.imageAnimate = False
        self.animTimer = QtCore.QTimer(self)
        self.animTimer.timeout.connect(self.onAnimTimer)
        
        self.casaRightOffset = 0
        self.casaRightScale = 1
        self.casaLeftOffset = 0
        self.casaLeftScale = 1
        self.initCleanDict()
        self.initGaincalDict()
        self.initApplycalDict()
        
#        self.setGeometry(100,100,1024,700)
        self.openButton = QtWidgets.QPushButton('Open...', self)
        self.openButton.clicked.connect(self.onOpen)

        self.saveButton = QtWidgets.QPushButton('Save as...', self)
        self.saveButton.clicked.connect(self.onSaveAs)


        self.ewPhaseStairLcp = QtWidgets.QSpinBox(self, prefix='EW L ')
        self.ewPhaseStairLcp.setRange(-180,180)
        self.ewPhaseStairLcp.setStyle(CustomStyle())
        self.ewPhaseStairLcp.valueChanged.connect(self.onEastWestPhaseStairLcpChanged)
        self.ewPhaseStairRcp = QtWidgets.QSpinBox(self, prefix='EW R ')
        self.ewPhaseStairRcp.setRange(-180,180)
        self.ewPhaseStairRcp.setStyle(CustomStyle())
        self.ewPhaseStairRcp.valueChanged.connect(self.onEastWestPhaseStairRcpChanged)

        self.ewPhaseStairLength = QtWidgets.QSpinBox(self, prefix='period ')
        self.ewPhaseStairLength.setRange(1,16)
        self.ewPhaseStairLength.setValue(16)
        self.ewPhaseStairLength.setStyle(CustomStyle())
        self.ewPhaseStairLength.valueChanged.connect(self.onEwPhaseStairLengthChanged)

        self.sPhaseStairLcp = QtWidgets.QDoubleSpinBox(self, prefix='S L ')
        self.sPhaseStairLcp.setRange(-180.,180.)
        self.sPhaseStairLcp.setStyle(CustomStyle())
        self.sPhaseStairLcp.valueChanged.connect(self.onSouthPhaseStairLcpChanged)
        self.sPhaseStairRcp = QtWidgets.QDoubleSpinBox(self, prefix='S R ')
        self.sPhaseStairRcp.setRange(-180.,180.)
        self.sPhaseStairRcp.setStyle(CustomStyle())
        self.sPhaseStairRcp.valueChanged.connect(self.onSouthPhaseStairRcpChanged)

        self.sPhaseStairLength = QtWidgets.QSpinBox(self, prefix = 'period ')
        self.sPhaseStairLength.setRange(1,16)
        self.sPhaseStairLength.setValue(16)
        self.sPhaseStairLength.setStyle(CustomStyle())
        self.sPhaseStairLength.valueChanged.connect(self.onSPhaseStairLengthChanged)

        self.lcpCanvas = ResponseCanvas(self)
        self.lcpCanvas.mouseSignal.connect(self.onCanvasXyChangedLcp)
        self.lcpMaxCanvas = ResponseCanvas(self)
        self.rcpCanvas = ResponseCanvas(self)
        self.rcpCanvas.mouseSignal.connect(self.onCanvasXyChangedRcp)
        self.rcpMaxCanvas = ResponseCanvas(self)

        self.clearButton = QtWidgets.QPushButton('Clear trace', self)
        self.clearButton.clicked.connect(self.onClear)
#        
        self.imageUpdateButton = QtWidgets.QPushButton('Update', self)
        self.imageUpdateButton.setCheckable(True)
        self.imageUpdateButton.setChecked(True)
        self.imageUpdateButton.clicked.connect(self.onImageUpdate)
        
        self.calibrateButton = QtWidgets.QPushButton('Calibrate', self)
        self.calibrateButton.clicked.connect(self.onCalibrate)

        self.frequencyChannel = QtWidgets.QSpinBox(self, prefix='channel ')
        self.frequencyChannel.setRange(0,0)
        self.frequencyChannel.setStyle(CustomStyle())
        self.frequencyChannel.valueChanged.connect(self.onFrequencyChannelChanged)

        self.frequencyList = QtWidgets.QComboBox(self)
        self.frequencyList.currentIndexChanged.connect(self.onFrequencyListSelected)
        self.timeList = QtWidgets.QComboBox(self)
        self.timeList.currentIndexChanged.connect(self.onTimeListSelected)
#        
        self.scan = QtWidgets.QSpinBox(self, prefix='scan ')
        self.scan.setRange(0,0)
        self.scan.setStyle(CustomStyle())
        self.scan.valueChanged.connect(self.onScanChanged)

        self.calibScan = QtWidgets.QSpinBox(self, prefix='calib_scan ')
        self.calibScan.setRange(0,0)
        self.calibScan.setStyle(CustomStyle())
        self.calibScan.valueChanged.connect(self.onCalibScanChanged)

        self.ewLcpPhaseSlopeSpin = QtWidgets.QDoubleSpinBox(self, prefix = 'EW LCP Slope ')
        self.ewLcpPhaseSlopeSpin.setRange(-1800.,1800.)
        self.ewLcpPhaseSlopeSpin.setSingleStep(0.1)
        self.ewLcpPhaseSlopeSpin.setStyle(CustomStyle())
        self.ewLcpPhaseSlopeSpin.valueChanged.connect(self.onEastWestLcpPhaseSlopeChanged)

        self.ewRcpPhaseSlopeSpin = QtWidgets.QDoubleSpinBox(self, prefix='EW RCP Slope ')
        self.ewRcpPhaseSlopeSpin.setRange(-1800.,1800.)
        self.ewRcpPhaseSlopeSpin.setSingleStep(0.1)
        self.ewRcpPhaseSlopeSpin.setStyle(CustomStyle())
        self.ewRcpPhaseSlopeSpin.valueChanged.connect(self.onEastWestRcpPhaseSlopeChanged)

        self.sLcpPhaseSlopeSpin = QtWidgets.QDoubleSpinBox(self, prefix='S LCP Slope ')
        self.sLcpPhaseSlopeSpin.setRange(-1800.,1800.)
        self.sLcpPhaseSlopeSpin.setSingleStep(0.1)
        self.sLcpPhaseSlopeSpin.setStyle(CustomStyle())
        self.sLcpPhaseSlopeSpin.valueChanged.connect(self.onSouthLcpPhaseSlopeChanged)

        self.sRcpPhaseSlopeSpin = QtWidgets.QDoubleSpinBox(self, prefix='S RCP Slope ')
        self.sRcpPhaseSlopeSpin.setRange(-1800.,1800.)
        self.sRcpPhaseSlopeSpin.setSingleStep(0.1)
        self.sRcpPhaseSlopeSpin.setStyle(CustomStyle())
        self.sRcpPhaseSlopeSpin.valueChanged.connect(self.onSouthRcpPhaseSlopeChanged)
        
        self.centerDiskButton = QtWidgets.QPushButton('Center disk', self)
        self.centerDiskButton.clicked.connect(self.onCenterDisk)

        self.sunContourButton = QtWidgets.QPushButton('Sun contour', self)
        self.sunContourButton.setCheckable(True)
        self.sunContourButton.setChecked(False)
        self.sunContourButton.clicked.connect(self.onSunContour)

        self.phaseCorrectButton = QtWidgets.QPushButton('Phase', self)
        self.phaseCorrectButton.setCheckable(True)
        self.phaseCorrectButton.setChecked(True)
        self.phaseCorrectButton.clicked.connect(self.onPhaseCorrect)
        
        self.amplitudeCorrectButton = QtWidgets.QPushButton('Amplitude', self)
        self.amplitudeCorrectButton.setCheckable(True)
        self.amplitudeCorrectButton.setChecked(True)
        self.amplitudeCorrectButton.clicked.connect(self.onAmplitudeCorrect)
        
        self.calculatePhaseButton = QtWidgets.QPushButton('Calculate Phase Cal (all)', self)
        self.calculatePhaseButton.clicked.connect(self.onCalculatePhase)
        
        self.calculatePhaseCurrentButton = QtWidgets.QPushButton('Calculate Phase Cal (current)', self)
        self.calculatePhaseCurrentButton.clicked.connect(self.onCalculatePhaseCurrent)
        
        self.typeOfFrame = QtWidgets.QComboBox(self)
        self.typeOfFrame.currentIndexChanged.connect(self.onTypeOfFrame)
        self.typeOfFrame.addItem('l, m')
        self.typeOfFrame.addItem('h,d')
        self.typeOfFrame.addItem('h-P,d-P')

        self.imageAnimateButton = QtWidgets.QPushButton('Animate', self)
        self.imageAnimateButton.setCheckable(True)
        self.imageAnimateButton.setChecked(self.imageAnimate)
        self.imageAnimateButton.clicked.connect(self.onImageAnimate)
        
        self.imageOffsetSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.imageOffsetSlider.setRange(-100,100)
        self.imageOffsetSlider.setValue(0)
        self.imageOffsetSlider.valueChanged.connect(self.onImageOffsetSlider)
        
        self.imageScaleSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.imageScaleSlider.setRange(1,100)
        self.imageScaleSlider.setValue(10)
        self.imageScaleSlider.valueChanged.connect(self.onImageScaleSlider)
        
        self.typeOfImage = QtWidgets.QComboBox(self)
        self.typeOfImage.currentIndexChanged.connect(self.onTypeOfImage)
        self.typeOfImage.addItem('LCP, RCP')
        self.typeOfImage.addItem('I,V')
        self.typeOfImage.addItem('uvLCP,uvRCP (amp)')
        self.typeOfImage.addItem('uvLCP,uvRCP (phase)')
        self.typeOfImage.addItem('PSF')
        self.typeOfImage.addItem('IV + PSF')
        self.typeOfImage.addItem('freq Diff')
        
        self.lcpRcpRelation = QtWidgets.QDoubleSpinBox (self, prefix = 'LCP/RCP ')
        self.lcpRcpRelation.setSingleStep(0.01)
        self.lcpRcpRelation.setRange(0.,2.)
        self.lcpRcpRelation.valueChanged.connect(self.onLcpRcpRelationChanged)
        self.lcpRcpRelation.setValue(1.)
        self.lcpRcpRelation.setStyle(CustomStyle())
        
        self.typeOfPlot = QtWidgets.QComboBox(self)
        self.typeOfPlot.currentIndexChanged.connect(self.onTypeOfPlot)
        self.typeOfPlot.addItem('Correlation plots')
        self.typeOfPlot.addItem('Brightness histograms')
        self.typeOfPlot.addItem('Image slices')
        self.typeOfPlot.addItem('Image mean')
        
        self.binsSpin = QtWidgets.QSpinBox (self, prefix = 'bins ')
        self.binsSpin.setRange(0,10000)
        self.binsSpin.valueChanged.connect(self.onBinsChanged)
        self.binsSpin.setValue(self.bins)
        self.binsSpin.setStyle(CustomStyle())
        
        self.calibrateBrightnessButton = QtWidgets.QPushButton('Calibrate brightness', self)
        self.calibrateBrightnessButton.clicked.connect(self.onCalibrateBrightness)
        
        self.autoscaleButton = QtWidgets.QPushButton('Autoscale', self)
        self.autoscaleButton.setCheckable(True)
        self.autoscaleButton.setChecked(True)
        self.autoscaleButton.clicked.connect(self.onAutoscale)
        
        self.loadGainsButton = QtWidgets.QPushButton('Load Gains...', self)
        self.loadGainsButton.clicked.connect(self.onLoadGains)
        
        self.saveGainsButton = QtWidgets.QPushButton('Save Gains...', self)
        self.saveGainsButton.clicked.connect(self.onSaveGains)
        
        
        
        # self.nonlinearButton = QtWidgets.QPushButton('Nonlinear', self)
        # self.nonlinearButton.setCheckable(True)
        # self.nonlinearButton.setChecked(False)
        # self.nonlinearButton.clicked.connect(self.onNonlinear)
        
        # self.weightsButton = QtWidgets.QPushButton('Use weights', self)
        # self.weightsButton.setCheckable(True)
        # self.weightsButton.setChecked(False)
        # self.weightsButton.clicked.connect(self.onWeights)
        
        # self.fringeStoppingButton = QtWidgets.QPushButton('Fringe Stopping', self)
        # self.fringeStoppingButton.setCheckable(True)
        # self.fringeStoppingButton.setChecked(False)
        # self.fringeStoppingButton.clicked.connect(self.onFringeStopping)
        
#        self.ewAntenna = QtWidgets.QSpinBox(self, prefix='EW_ant ')
#        self.ewAntenna.setRange(49,80)
#        self.ewAntenna.setStyle(CustomStyle())
#        self.ewAntenna.valueChanged.connect(self.onEwAntennaChanged)
#
#        self.ewLcpAntennaPhase = QtWidgets.QSpinBox(self, prefix = 'LCP phase ')
#        self.ewLcpAntennaPhase.setRange(-180,180)
#        self.ewLcpAntennaPhase.setStyle(CustomStyle())
#        self.ewLcpAntennaPhase.valueChanged.connect(self.onEwLcpAntennaPhaseChanged)
#
#        self.sLcpAntennaPhase = QtWidgets.QSpinBox(self, prefix = 'LCP phase ')
#        self.sLcpAntennaPhase.setRange(-180,180)
#        self.sLcpAntennaPhase.setStyle(CustomStyle())
#        self.sLcpAntennaPhase.valueChanged.connect(self.onSLcpAntennaPhaseChanged)
#
#        self.ewRcpAntennaPhase = QtWidgets.QSpinBox(self, prefix = 'RCP phase ')
#        self.ewRcpAntennaPhase.setRange(-180,180)
#        self.ewRcpAntennaPhase.setStyle(CustomStyle())
#        self.ewRcpAntennaPhase.valueChanged.connect(self.onEwRcpAntennaPhaseChanged)
#
#        self.sRcpAntennaPhase = QtWidgets.QSpinBox(self, prefix = 'RCP phase ')
#        self.sRcpAntennaPhase.setRange(-180,180)
#        self.sRcpAntennaPhase.setStyle(CustomStyle())
#        self.sRcpAntennaPhase.valueChanged.connect(self.onSRcpAntennaPhaseChanged)
#        
#        self.ewAmpCoefSpin = QtWidgets.QDoubleSpinBox (self, prefix = 'EW Amp Coef ')
#        self.ewAmpCoefSpin.setSingleStep(0.01)
#        self.ewAmpCoefSpin.setRange(0.1,5.)
#        self.ewAmpCoefSpin.valueChanged.connect(self.onEwAmpCoefChanged)
#        self.ewAmpCoefSpin.setValue(1.)
#        self.ewAmpCoefSpin.setStyle(CustomStyle())
#        
#        self.sAmpCoefSpin = QtWidgets.QDoubleSpinBox (self, prefix = 'S Amp Coef ')
#        self.sAmpCoefSpin.setSingleStep(0.01)
#        self.sAmpCoefSpin.setRange(0.1,5.)
#        self.sAmpCoefSpin.valueChanged.connect(self.onSAmpCoefChanged)
#        self.sAmpCoefSpin.setValue(1.)
#        self.sAmpCoefSpin.setStyle(CustomStyle())
#        
#        self.sAntenna = QtWidgets.QSpinBox(self, prefix='S_ant ')
#        self.sAntenna.setRange(177,192)
#        self.sAntenna.setStyle(CustomStyle())
#        self.sAntenna.valueChanged.connect(self.onSAntennaChanged)     

        self.autoFlagButton = QtWidgets.QPushButton('Auto', self)
        self.autoFlagButton.clicked.connect(self.onAutoFlag)
        
        self.flaggedAntsLabel = QtWidgets.QLabel(self)
        self.flaggedAntsLabel.setText('Flagged antennas: ')
        
        self.manualLabel = QtWidgets.QLabel(self)
        self.manualLabel.setText('          Manual flagging')
        
        self.manualLcpLabel = QtWidgets.QLabel(self)
        self.manualLcpLabel.setText('LCP')
        self.manualRcpLabel = QtWidgets.QLabel(self)
        self.manualRcpLabel.setText('RCP')
        
        self.flaggedAntsBoxLcp = QtWidgets.QLineEdit(self)
        self.flaggedAntsBoxRcp = QtWidgets.QLineEdit(self)
        
        self.flagButton = QtWidgets.QPushButton('Flag', self)
        self.flagButton.clicked.connect(self.onFlag)

        self.lcpTextBox = QtWidgets.QLineEdit(self)
        self.lcpTextBox.setReadOnly(True)
        
        self.rcpTextBox = QtWidgets.QLineEdit(self)
        self.rcpTextBox.setReadOnly(True)
        
        self.averageCalibButton = QtWidgets.QPushButton('Average calibration', self)
        self.averageCalibButton.setCheckable(True)
        self.averageCalibButton.setChecked(False)
        self.averageCalibButton.clicked.connect(self.onAverageCalib)
        
        self.averageScansSpin = QtWidgets.QSpinBox(self, prefix='Average scans:  ')
        self.averageScansSpin.setStyle(CustomStyle())
        self.averageScansSpin.valueChanged.connect(self.onAverageScans)
        
        self.freqDiffSpin = QtWidgets.QSpinBox(self, prefix='Freq to substract:  ')
        self.freqDiffSpin.setStyle(CustomStyle())
        self.freqDiffSpin.valueChanged.connect(self.onFreqDiff)
        
        self.freqDiffRelationLcp = QtWidgets.QDoubleSpinBox (self, prefix = 'Coef LCP')
        self.freqDiffRelationLcp.setSingleStep(0.01)
        self.freqDiffRelationLcp.setRange(0.,2.)
        self.freqDiffRelationLcp.valueChanged.connect(self.onfreqDiffRelationLcpChanged)
        self.freqDiffRelationLcp.setValue(1.)
        self.freqDiffRelationLcp.setStyle(CustomStyle())
        
        self.freqDiffRelationRcp = QtWidgets.QDoubleSpinBox (self, prefix = 'Coef RCP')
        self.freqDiffRelationRcp.setSingleStep(0.01)
        self.freqDiffRelationRcp.setRange(0.,2.)
        self.freqDiffRelationRcp.valueChanged.connect(self.onfreqDiffRelationRcpChanged)
        self.freqDiffRelationRcp.setValue(1.)
        self.freqDiffRelationRcp.setStyle(CustomStyle())

        self.openMSButton = QtWidgets.QPushButton('Open MS...', self)
        self.openMSButton.clicked.connect(self.onOpenMS)
        
        self.MSTextbox = QtWidgets.QTextEdit(self)
        
        self.createCleanTable()
        self.createGaincalTable()
        self.createApplycalTable()
        
        self.makeCleanButton = QtWidgets.QPushButton('Make CLEAN map', self)
        self.makeCleanButton.clicked.connect(self.onMakeClean)
        
        self.makeGaincalButton = QtWidgets.QPushButton('Make gaincal', self)
        self.makeGaincalButton.clicked.connect(self.onMakeGaincal)
        
        self.makeApplycalButton = QtWidgets.QPushButton('Apply cal', self)
        self.makeApplycalButton.clicked.connect(self.onMakeApplycal)
        
        self.plotmsButton = QtWidgets.QPushButton('CASA plotms', self)
        self.plotmsButton.clicked.connect(self.onPlotms)
        
        self.viewerButton = QtWidgets.QPushButton('CASA viewer', self)
        self.viewerButton.clicked.connect(self.onViewer)
             
        self.casaRightOffsetSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.casaRightOffsetSlider.setRange(-100,100)
        self.casaRightOffsetSlider.setValue(0)
        self.casaRightOffsetSlider.valueChanged.connect(self.onCasaRightOffsetSlider)
        
        self.casaRightScaleSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.casaRightScaleSlider.setRange(1,100)
        self.casaRightScaleSlider.setValue(10)
        self.casaRightScaleSlider.valueChanged.connect(self.onCasaRightScaleSlider)
        
        self.casaLeftOffsetSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.casaLeftOffsetSlider.setRange(-100,100)
        self.casaLeftOffsetSlider.setValue(0)
        self.casaLeftOffsetSlider.valueChanged.connect(self.onCasaLeftOffsetSlider)
        
        self.casaLeftScaleSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.casaLeftScaleSlider.setRange(0,100)
        self.casaLeftScaleSlider.setValue(10)
        self.casaLeftScaleSlider.valueChanged.connect(self.onCasaLeftScaleSlider)
        
        self.typeOfLeftCasaImage = QtWidgets.QComboBox(self)
        self.typeOfLeftCasaImage.addItem('LCP')
        self.typeOfLeftCasaImage.addItem('RCP')
        self.typeOfLeftCasaImage.addItem('I')
        self.typeOfLeftCasaImage.addItem('V')
        self.typeOfLeftCasaImage.addItem('LCP model')
        self.typeOfLeftCasaImage.addItem('RCP model')
        self.typeOfLeftCasaImage.addItem('psf')
        self.typeOfLeftCasaImage.addItem('pb')
        self.typeOfLeftCasaImage.currentIndexChanged.connect(self.onTypeOfLeftCasaImage)
        
        self.typeOfRightCasaImage = QtWidgets.QComboBox(self)
        self.typeOfRightCasaImage.addItem('LCP')
        self.typeOfRightCasaImage.addItem('RCP')
        self.typeOfRightCasaImage.addItem('I')
        self.typeOfRightCasaImage.addItem('V')
        self.typeOfRightCasaImage.addItem('LCP model')
        self.typeOfRightCasaImage.addItem('RCP model')
        self.typeOfRightCasaImage.addItem('psf')
        self.typeOfRightCasaImage.addItem('pb')
        self.typeOfRightCasaImage.setCurrentIndex(1)
        self.typeOfRightCasaImage.currentIndexChanged.connect(self.onTypeOfRightCasaImage)
        
        self.casaSaveAsButton = QtWidgets.QPushButton('Save as...', self)
        self.casaSaveAsButton.clicked.connect(self.onCasaSaveAs)
        
        self.diskModelButton = QtWidgets.QPushButton('Disk Model', self)
        self.diskModelButton.clicked.connect(self.onDiskModel)
        
        self.pointModelButton = QtWidgets.QPushButton('Point Model', self)
        self.pointModelButton.clicked.connect(self.onPointModel)
        
        self.setModelButton = QtWidgets.QPushButton('Set Model', self)
        self.setModelButton.clicked.connect(self.onSetModel)
        
        self.casaLeftCanvas = ResponseCanvas(self)
        self.casaRightCanvas = ResponseCanvas(self)
        
        self.casaModelCanvas = ResponseCanvas(self)


        
        self.cWidget = QtWidgets.QWidget()
        layout = QGridLayout()
        self.cWidget.setLayout(layout)
        
        self.tabs = QTabWidget()
#        self.tabs.setFixedHeight(1500)
        self.srhTabs = QTabWidget()
        self.srhTabs.setFixedHeight(100)
        self.casaTabs = QTabWidget()
#        self.casaTabs.setFixedHeight(600)
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()
        self.tab4 = QtWidgets.QWidget()
        self.tab5 = QtWidgets.QWidget()
        self.tab6 = QtWidgets.QWidget()
        self.tab7 = QtWidgets.QWidget()
        self.tab8 = QtWidgets.QWidget()
        self.tab9 = QtWidgets.QWidget()
        self.tab10 = QtWidgets.QWidget()
        self.tab11 = QtWidgets.QWidget()
        
        self.casaGeneralButtons = QtWidgets.QWidget()
        #self.tabs.resize(1024,50)
        
        
        self.srhTabs.addTab(self.tab1,"General")
        self.srhTabs.addTab(self.tab2,"Phase&Amp Correction")
        self.srhTabs.addTab(self.tab3,"Flagging")
        self.srhTabs.addTab(self.tab4,"Centering")
        self.srhTabs.addTab(self.tab5,"Averaging")
        self.srhTabs.addTab(self.tab6,"Freq Diff")
        self.srhTabs.addTab(self.tab7,"Save...")
        
        self.srhWidget = QtWidgets.QWidget()
        self.srhWidget.layout = QGridLayout()
        self.srhWidget.layout.setSpacing(1)
        self.srhWidget.layout.setVerticalSpacing(1)
        self.srhWidget.layout.addWidget(self.srhTabs,0,0, 1,-1)
        self.srhWidget.layout.addWidget(self.lcpTextBox,1,0)
        self.srhWidget.layout.addWidget(self.rcpTextBox,1,1)
        self.srhWidget.layout.addWidget(self.lcpCanvas,4,0)
        self.srhWidget.layout.addWidget(self.rcpCanvas,4,1)
        self.srhWidget.layout.addWidget(self.lcpMaxCanvas,14,0,5,1)
        self.srhWidget.layout.addWidget(self.rcpMaxCanvas,14,1,5,1)
        self.srhWidget.setLayout(self.srhWidget.layout)
        
        self.casaWidget= QtWidgets.QWidget()
        self.casaWidget.layout = QGridLayout()
        self.casaWidget.layout.setSpacing(1)
        self.casaWidget.layout.setVerticalSpacing(1)
        self.casaWidget.layout.addWidget(self.casaTabs)
#        self.casaWidget.layout.addWidget(self.casaLeftCanvas,5,0)
#        self.casaWidget.layout.addWidget(self.casaRightCanvas,5,1)
        self.casaWidget.setLayout(self.casaWidget.layout)
        self.casaTabs.addTab(self.tab8,"General")
        self.casaTabs.addTab(self.tab9,"CLEAN parameters")
        self.casaTabs.addTab(self.tab10,"Calibration")
        self.casaTabs.addTab(self.tab11,"Model")
        
        self.tabs.addTab(self.srhWidget, 'SRH fits')
        self.tabs.addTab(self.casaWidget, 'CASA')
        
        self.tab1.layout = QGridLayout()
        self.tab1.layout.setSpacing(1)
        self.tab1.layout.setVerticalSpacing(1)
        self.tab1.layout.addWidget(self.openButton, 0, 0)
        self.tab1.layout.addWidget(self.frequencyChannel, 0, 1)
        self.tab1.layout.addWidget(self.scan, 1, 1)
        self.tab1.layout.addWidget(self.frequencyList, 0, 2)
        self.tab1.layout.addWidget(self.timeList, 1, 2)
        self.tab1.layout.addWidget(self.calibScan, 0, 3)
        self.tab1.layout.addWidget(self.clearButton, 1, 3)
        self.tab1.layout.addWidget(self.calibrateButton, 0, 4)
        self.tab1.layout.addWidget(self.imageUpdateButton, 1, 4)
        
        self.tab1.layout.addWidget(self.typeOfFrame, 0, 7)
        self.tab1.layout.addWidget(self.typeOfImage, 1, 7)
        self.tab1.layout.addWidget(self.lcpRcpRelation, 0, 8)
        self.tab1.layout.addWidget(self.imageAnimateButton, 1, 8)
        self.tab1.layout.addWidget(self.imageOffsetSlider, 0, 9)
        self.tab1.layout.addWidget(self.imageScaleSlider, 1, 9)
        self.tab1.layout.addWidget(self.autoscaleButton, 0, 10)
        self.tab1.layout.addWidget(self.typeOfPlot, 0, 11)
        self.tab1.layout.addWidget(self.calibrateBrightnessButton, 1, 11)
        self.tab1.layout.addWidget(self.binsSpin, 0, 12)
        self.tab1.setLayout(self.tab1.layout)
        
        self.tab2.layout = QGridLayout()
        self.tab2.layout.setSpacing(1)
        self.tab2.layout.setVerticalSpacing(1)
        self.tab2.layout.addWidget(self.ewPhaseStairLength, 0, 0)
        self.tab2.layout.addWidget(self.ewPhaseStairLcp, 0, 1)
        self.tab2.layout.addWidget(self.ewPhaseStairRcp, 1, 1)
        self.tab2.layout.addWidget(self.sPhaseStairLength, 0, 2)
        self.tab2.layout.addWidget(self.sPhaseStairLcp, 0, 3)
        self.tab2.layout.addWidget(self.sPhaseStairRcp, 1, 3)
        self.tab2.layout.addWidget(self.phaseCorrectButton, 0, 4)
        self.tab2.layout.addWidget(self.amplitudeCorrectButton, 1, 4)
        self.tab2.layout.addWidget(self.calculatePhaseButton, 0, 5)
        self.tab2.layout.addWidget(self.calculatePhaseCurrentButton, 1, 5)
        self.tab2.layout.addWidget(self.loadGainsButton, 0, 6)
        self.tab2.layout.addWidget(self.saveGainsButton, 1, 6)
        # self.tab2.layout.addWidget(self.nonlinearButton, 0, 6)
        # self.tab2.layout.addWidget(self.weightsButton, 1, 6)
        # self.tab2.layout.addWidget(self.fringeStoppingButton, 0, 7)
#        self.tab2.layout.addWidget(self.ewAntenna, 0, 4)
#        self.tab2.layout.addWidget(self.ewLcpAntennaPhase, 0, 5)
#        self.tab2.layout.addWidget(self.ewRcpAntennaPhase, 0, 6)
#        self.tab2.layout.addWidget(self.sAntenna, 1, 4)
#        self.tab2.layout.addWidget(self.sLcpAntennaPhase, 1, 5)
#        self.tab2.layout.addWidget(self.sRcpAntennaPhase, 1, 6)
#        self.tab2.layout.addWidget(self.ewAmpCoefSpin, 0, 7)
#        self.tab2.layout.addWidget(self.sAmpCoefSpin, 1, 7)
        self.tab2.setLayout(self.tab2.layout)
        
        self.tab3.layout = QGridLayout()
        self.tab3.layout.setSpacing(1)
        self.tab3.layout.setVerticalSpacing(1)
        self.tab3.layout.addWidget(self.autoFlagButton, 0, 0)
        self.tab3.layout.addWidget(self.flaggedAntsLabel, 1, 0)
        self.tab3.layout.addWidget(self.manualLabel, 0, 1)
        self.tab3.layout.addWidget(self.manualLcpLabel, 0, 2)
        self.tab3.layout.addWidget(self.manualRcpLabel, 1, 2)
        self.tab3.layout.addWidget(self.flaggedAntsBoxLcp, 0, 3)
        self.tab3.layout.addWidget(self.flaggedAntsBoxRcp, 1, 3)
        self.tab3.layout.addWidget(self.flagButton, 0, 4)
        self.tab3.setLayout(self.tab3.layout)
        
        self.tab4.layout = QGridLayout()
        self.tab4.layout.setSpacing(1)
        self.tab4.layout.setVerticalSpacing(1)
        self.tab4.layout.addWidget(self.ewLcpPhaseSlopeSpin, 0, 0)
        self.tab4.layout.addWidget(self.ewRcpPhaseSlopeSpin, 0, 1)
        self.tab4.layout.addWidget(self.sLcpPhaseSlopeSpin, 1, 0)
        self.tab4.layout.addWidget(self.sRcpPhaseSlopeSpin, 1, 1)
        self.tab4.layout.addWidget(self.centerDiskButton, 0, 2)
        self.tab4.layout.addWidget(self.sunContourButton, 1, 2)
        self.tab4.setLayout(self.tab4.layout)
        
        self.tab5.layout = QGridLayout()
        self.tab5.layout.setSpacing(1)
        self.tab5.layout.setVerticalSpacing(1)
        self.tab5.layout.addWidget(self.averageCalibButton, 0, 0)
        self.tab5.layout.addWidget(self.averageScansSpin, 1, 0)
        self.tab5.setLayout(self.tab5.layout)
        
        self.tab6.layout = QGridLayout()
        self.tab6.layout.setSpacing(1)
        self.tab6.layout.setVerticalSpacing(1)
        self.tab6.layout.addWidget(self.freqDiffSpin, 0, 0)
        self.tab6.layout.addWidget(self.freqDiffRelationLcp, 1, 0)
        self.tab6.layout.addWidget(self.freqDiffRelationRcp, 1, 1)
        self.tab6.setLayout(self.tab6.layout)
        
        self.tab7.layout = QGridLayout()
        self.tab7.layout.setSpacing(1)
        self.tab7.layout.setVerticalSpacing(1)
        self.tab7.layout.addWidget(self.saveButton, 0, 0)
        self.tab7.setLayout(self.tab7.layout)
        
        self.casaGeneralButtons.layout = QGridLayout()
        self.casaGeneralButtons.layout.setSpacing(1)
        self.casaGeneralButtons.layout.setVerticalSpacing(1)
        self.casaGeneralButtons.layout.addWidget(self.openMSButton, 0, 0)
        self.casaGeneralButtons.layout.addWidget(self.MSTextbox, 1, 0)
        self.casaGeneralButtons.layout.addWidget(self.makeCleanButton, 0, 1)
        self.casaGeneralButtons.layout.addWidget(self.casaSaveAsButton, 1, 1)
        self.casaGeneralButtons.layout.addWidget(self.plotmsButton, 0, 2)
        self.casaGeneralButtons.layout.addWidget(self.viewerButton, 1, 2)
        self.casaGeneralButtons.setLayout(self.casaGeneralButtons.layout)
        
        self.tab8.layout = QGridLayout()
        self.tab8.layout.setSpacing(1)
        self.tab8.layout.setVerticalSpacing(1)
        self.tab8.layout.addWidget(self.casaGeneralButtons,0,0)
        self.tab8.layout.addWidget(self.casaLeftOffsetSlider,1,0)
        self.tab8.layout.addWidget(self.casaLeftScaleSlider,2,0)
        self.tab8.layout.addWidget(self.casaRightOffsetSlider,1,1)
        self.tab8.layout.addWidget(self.casaRightScaleSlider,2,1)
        self.tab8.layout.addWidget(self.typeOfLeftCasaImage,3,0)
        self.tab8.layout.addWidget(self.typeOfRightCasaImage,3,1)
        self.tab8.layout.addWidget(self.casaLeftCanvas,4,0)
        self.tab8.layout.addWidget(self.casaRightCanvas,4,1)
        self.tab8.setLayout(self.tab8.layout)
        
        self.tab9.layout = QGridLayout()
        self.tab9.layout.setSpacing(1)
        self.tab9.layout.setVerticalSpacing(1)
        self.tab9.layout.addWidget(self.cleanTable, 0, 0)
        self.tab9.setLayout(self.tab9.layout)
        
        self.tab10.layout = QGridLayout()
        self.tab10.layout.setSpacing(1)
        self.tab10.layout.setVerticalSpacing(1)
        self.tab10.layout.addWidget(self.makeGaincalButton, 0, 0)
        self.tab10.layout.addWidget(self.gaincalTable, 1, 0)
        self.tab10.layout.addWidget(self.makeApplycalButton, 0, 1)
        self.tab10.layout.addWidget(self.applycalTable, 1, 1)
        self.tab10.setLayout(self.tab10.layout)
        
        self.tab11.layout = QGridLayout()
        self.tab11.layout.setSpacing(1)
        self.tab11.layout.setVerticalSpacing(1)
        self.tab11.layout.addWidget(self.diskModelButton, 0, 0)
        self.tab11.layout.addWidget(self.pointModelButton, 1, 0)
        self.tab11.layout.addWidget(self.setModelButton, 0, 1)
        self.tab11.layout.addWidget(self.casaModelCanvas, 2, 0)
        self.tab11.setLayout(self.tab11.layout)
        
        
        self.lcpCanvas.setMinimumSize(500,500)
        self.rcpCanvas.setMinimumSize(500,500)
        self.casaLeftCanvas.setMinimumSize(500,500)
        self.casaRightCanvas.setMinimumSize(500,500)
        layout.addWidget(self.tabs,0,0,1,2)
        
        self.setCentralWidget(self.cWidget)
        
#        self.setLayout(self.layout)

#        self.openButton.setGeometry(0,0,60,25)
#        self.frequencyChannel.setGeometry(70,0,80,25)
#        self.scan.setGeometry(70,25,80,25)
#        self.calibScan.setGeometry(150,0,80,25)
#        self.clearButton.setGeometry(150,25,80,25)
#        self.ewPhaseStairLcp.setGeometry(235,0,70,25)
#        self.ewPhaseStairRcp.setGeometry(305,0,70,25)
#        self.ewPhaseStairLength.setGeometry(235,25,140,25)
#
#        self.sPhaseStairLcp.setGeometry(375,0,70,25)
#        self.sPhaseStairRcp.setGeometry(445,0,70,25)
#        self.sPhaseStairLength.setGeometry(375,25,140,25)
#
#        self.ewLcpPhaseSlopeSpin.setGeometry(500,0,80,25)
#        self.sLcpPhaseSlopeSpin.setGeometry(500,25,80,25)
#        self.ewRcpPhaseSlopeSpin.setGeometry(580,0,80,25)
#        self.sRcpPhaseSlopeSpin.setGeometry(580,25,80,25)
#
#        self.phaseCorrectButton.setGeometry(660,0,60,25)
#        self.amplitudeCorrectButton.setGeometry(660,25,60,25)
#        
#        self.typeOfFrame.setGeometry(720,0,60,25)
#        
#        self.imageAnimateButton.setGeometry(720,25,60,25)
#
#        self.imageOffsetSlider.setGeometry(780,0,60,25)
#        self.imageScaleSlider.setGeometry(780,25,60,25)
#        self.findPhaseButton.setGeometry(840,0,60,25)
#        self.imageUpdateButton.setGeometry(840,25,60,25)
#        self.frequencyList.setGeometry(900, 0, 100, 25)
#        self.timeList.setGeometry(900, 25, 100, 25)
#        
#        self.typeOfImage.setGeometry(1000, 0, 100, 25)
#        self.lcpRcpRelation.setGeometry(1000, 25, 100, 25)
#
#        self.ewAntenna.setGeometry(1160, 0, 80, 25)
#        self.sAntenna.setGeometry(1160, 25, 80, 25)
#        self.ewLcpAntennaPhase.setGeometry(1240, 0, 50, 25)
#        self.sLcpAntennaPhase.setGeometry(1240, 25, 50, 25)
#        self.ewRcpAntennaPhase.setGeometry(1290, 0, 50, 25)
#        self.sRcpAntennaPhase.setGeometry(1290, 25, 50, 25)
#
#        self.saveButton.setGeometry(0,25,60,25)
#        
#        self.lcpCanvas.setGeometry(0,50,512,512)
#        self.lcpMaxCanvas.setGeometry(0,560,512,100)
#        self.rcpCanvas.setGeometry(512,50,512,512)
#        self.rcpMaxCanvas.setGeometry(512,560,512,100)

    def onOpen(self):
        fitsNames, _ = QtWidgets.QFileDialog.getOpenFileNames(self)    
        fitsNames.sort()
        self.uvSize = 1024
        self.currentScan = 0
        self.scan.setValue(0)
        self.srhFits = SrhFitsFile(fitsNames, self.uvSize)
        self.srhFits.getHourAngle(self.currentScan)
        
        self.pAngle = self.srhFits.pAngle
        # try:
        #     client = Client('http://ephemeris.rao.istp.ac.ru/?wsdl')
        #     result = client.service.Ephemeride('SSRT','sun',self.srhFits.dateObs)
        #     self.pAngle = NP.deg2rad(float(result[0]['PAngle']))
        # except:
        #     self.pAngle = NP.deg2rad(coordinates.sun.P(self.srhFits.dateObs).to_value())
        
        # self.pAngle = NP.deg2rad(coordinates.sun.P(self.srhFits.dateObs).to_value())
        
        
#            self.srhFits.setCalibIndex(0)
        self.srhFits.vis2uv(self.currentScan, phaseCorrect=True, PSF=False);
        self.srhFits.uv2lmImage()

        self.frequencyList.clear()
        for freq in self.srhFits.freqList:
            self.frequencyList.addItem(str(freq))

        self.frequencyChannel.setRange(0, self.srhFits.freqListLength-1)
#            self.frequencyChannel.setValue(self.currentFrequencyChannel)
        self.scan.setRange(0, self.srhFits.dataLength-1)
#            self.scan.setValue(self.currentScan)
        self.calibScan.setRange(0, self.srhFits.dataLength-1)
        
        
        self.ewLcpPhaseCorrection = NP.zeros(self.ewAntennaNumber)
        self.ewRcpPhaseCorrection = NP.zeros(self.ewAntennaNumber)
        self.sLcpPhaseCorrection = NP.zeros(self.sAntennaNumber)
        self.sRcpPhaseCorrection = NP.zeros(self.sAntennaNumber)

        self.ewPhaseCoefsLcp = NP.zeros((self.srhFits.freqListLength))
        self.ewPhaseCoefsRcp = NP.zeros((self.srhFits.freqListLength))
        self.sPhaseCoefsLcp = NP.zeros((self.srhFits.freqListLength))
        self.sPhaseCoefsRcp = NP.zeros((self.srhFits.freqListLength))
        
        self.ewLcpPhaseSlope = NP.zeros((self.srhFits.freqListLength))
        self.ewRcpPhaseSlope = NP.zeros((self.srhFits.freqListLength))
        self.sLcpPhaseSlope = NP.zeros((self.srhFits.freqListLength))
        self.sRcpPhaseSlope = NP.zeros((self.srhFits.freqListLength))
        
        self.imageUpdate = True
        self.qSun = NP.zeros((self.uvSize//2, self.uvSize//2))
        sunRadius = coordinates.sun.angular_radius(self.srhFits.dateObs).to_value() / (self.arcsecPerPixel*2)
        for i in range(self.uvSize//2):
            x = i - self.uvSize/4
            for j in range(self.uvSize//2):
                y = j - self.uvSize/4
                if (NP.sqrt(x*x + y*y) < sunRadius):
                    self.qSun[i , j] = 1.
                    
        data = NP.flip(self.srhFits.lcp.real,0)
        self.lcpCanvas.imshow(data, NP.min(data), NP.max(data))
        self.lcpCanvas.plot([0,1])
        self.lcpMaxTrace.append(NP.max(data))
        self.lcpData = data
#            self.lcpMaxCanvas.plot(self.lcpMaxTrace)

        data = NP.flip(self.srhFits.rcp.real,0)
        self.rcpCanvas.imshow(data, NP.min(data), NP.max(data))
        self.rcpMaxTrace.append(NP.max(data))
        self.rcpData = data
#            self.rcpMaxCanvas.plot(self.rcpMaxTrace)

        self.vminI = (self.lcpData + self.rcpData).min() * 0.5
        self.vmaxI = (self.lcpData + self.rcpData).max() * 0.5
        self.vminV = (self.lcpData - self.rcpData).min() * 0.5
        self.vmaxV = (self.lcpData - self.rcpData).max() * 0.5
        self.vminLcp = self.lcpData.min()
        self.vmaxLcp = self.lcpData.max()
        self.vminRcp = self.lcpData.min()
        self.vmaxRcp = self.lcpData.max()

        self.setWindowTitle('SRH phase edit:' + fitsNames[0])

        # for fitsName in fitsNames[1:]:
        #     self.srhFits.append(fitsName)
        self.scan.setRange(0, self.srhFits.dataLength-1)
        self.calibScan.setRange(0, self.srhFits.dataLength-1)
    
        self.averageScansSpin.setRange(0, self.srhFits.dataLength)
        self.freqDiffSpin.setRange(0, self.srhFits.freqListLength-1)
        
        for tim in self.srhFits.freqTime[0,:]:
            fTime = QtCore.QTime(0,0)
            fTime = fTime.addMSecs(int(tim) * 1000)
            self.timeList.addItem(fTime.toString('hh:mm:ss'))
            
        self.fitsIsOpen = True
                
    def saveAsFits(self, saveName):
        for scan in range(self.srhFits.dataLength):
            self.scan.setValue(scan)
            nameParts = saveName.split('.') 
            fitsName = ('%s_%03d.%s' % (nameParts[0], scan, nameParts[1]))
            print(fitsName)
            if self.indexOfImageType == 3:
                self.lcpData /= NP.max(self.lcpData)
                self.rcpData /= NP.max(self.rcpData)
    
            pHeader = fits.Header();
            t = self.srhFits.hduList[0].header['DATE-OBS']
            pHeader['DATE-OBS']     = self.srhFits.hduList[0].header['DATE-OBS'] + 'T' + self.timeList.itemText(scan)
            pHeader['T-OBS']     = self.srhFits.hduList[0].header['DATE-OBS'] + 'T' + self.timeList.itemText(scan)#self.srhFits.hduList[0].header['TIME-OBS']
            pHeader['INSTRUME']     = self.srhFits.hduList[0].header['INSTRUME']
            pHeader['ORIGIN']       = self.srhFits.hduList[0].header['ORIGIN']
            pHeader['OBS-LAT']      = self.srhFits.hduList[0].header['OBS-LAT']
            pHeader['OBS-LONG']     = self.srhFits.hduList[0].header['OBS-LONG']
            pHeader['OBS-ALT']      = self.srhFits.hduList[0].header['OBS-ALT']
            pHeader['FR_CHAN']      = self.srhFits.hduList[0].header['FR_CHAN']
            pHeader['FREQUENC']     = ('%3.3f') % (self.srhFits.freqList[self.currentFrequencyChannel]*1e3)
            pHeader['CDELT1']       = self.arcsecPerPixel*2
            pHeader['CDELT2']       = self.arcsecPerPixel*2
            pHeader['CRPIX1']       = self.uvSize // 4
            pHeader['CRPIX2']       = self.uvSize // 4
            pHeader['CTYPE1']       = 'HPLN-TAN'
            pHeader['CTYPE2']       = 'HPLT-TAN'
            pHeader['CUNIT1']       = 'arcsec'
            pHeader['CUNIT2']       = 'arcsec'
            

            savePath, saveExt = fitsName.split('.')
            
            pHdu = fits.PrimaryHDU(self.lcpData + self.rcpData, header=pHeader);
            hduList = fits.HDUList([pHdu]);
            hduList.writeto(savePath + '_I.' + saveExt, clobber=True);
            hduList.close();
            
            pHdu = fits.PrimaryHDU(self.lcpData - self.rcpData, header=pHeader);
            hduList = fits.HDUList([pHdu]);
            hduList.writeto(savePath + '_V.' + saveExt, clobber=True);
            
            hduList.close();
        
    def updateCanvas(self, scan):
        self.scan.setValue(scan)
        self.buildImage()
        self.showImage()
        return self.rcpCanvas.imageObject,
        
    def saveAsMp4(self, saveName):
        ani = animation.FuncAnimation(self.rcpCanvas.fig, self.updateCanvas, frames=self.srhFits.dataLength, blit=True, repeat=False)
        ani.save(saveName)
        

    def saveAsMs2(self, saveName):
#        try:
        import srh612MS2
        freqs = [self.currentFrequencyChannel]
        ms2Table = srh612MS2.Srh612Ms2(saveName)
        ms2Table.initDataTable(self.srhFits, freqs, phaseCorrect = self.phaseCorrect, amplitudeCorrect = self.amplitudeCorrect)
        ms2Table.initAntennaTable(self.srhFits)
        ms2Table.initSpectralWindowTable(self.srhFits, freqs)
        ms2Table.initDataDescriptionTable(freqs)
        ms2Table.initPolarizationTable()
        ms2Table.initSourceTable()
        ms2Table.initFieldTable()
        ms2Table.initFeedTable(self.srhFits, freqs)
        ms2Table.initObservationTable(self.srhFits)
#        except:
#            pass

        
    def onSaveAs(self):
        saveName, _ = QtWidgets.QFileDialog.getSaveFileName(self)        
        if saveName:
            saveExt = saveName.split('.')[1]
            if saveExt == 'fit' or saveExt == 'fits':
                self.saveAsFits(saveName)
            elif saveExt == 'ms':
                self.saveAsMs2(saveName)
            elif saveExt == 'mp4':
                self.saveAsMp4(saveName)
                
    def onSaveGains(self):
        saveName, _ = QtWidgets.QFileDialog.getSaveFileName(self)        
        if saveName:
            self.srhFits.saveGains(saveName)
            
    def onLoadGains(self):
        loadName, _ = QtWidgets.QFileDialog.getOpenFileNames(self)
        self.srhFits.loadGains(loadName[0])
        if self.imageUpdate:
            self.buildImage()
            self.showImage()
                               
application = QtWidgets.QApplication.instance();
if not application:
    application = QtWidgets.QApplication(sys.argv);
    
#if sys.platform == 'linux':
#    font = QtGui.QFont()
#    application.setFont(QtGui.QFont(font.defaultFamily(),8));

phaseEdit612 = SrhEdik612();
phaseEdit612.setWindowTitle('SRH editor')
phaseEdit612.show();
sys.exit(application.exec_());