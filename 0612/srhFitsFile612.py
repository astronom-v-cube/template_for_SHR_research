# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 00:18:47 2016

@author: Sergey
"""

from astropy.io import fits
import numpy as NP
from astropy import coordinates
from astropy import constants
from BadaryRAO import BadaryRAO
from scipy.optimize import least_squares
import sunpy.coordinates
import base2uvw_612
from skimage.transform import warp, AffineTransform
import scipy.signal
import time
from ZirinTb import ZirinTb
import json
import scipy.constants
import skimage.measure

class SrhFitsFile():
    def __init__(self, name, sizeOfUv, flux_norm = True):
        self.omegaEarth = coordinates.earth.OMEGA_EARTH.to_value()
        self.isOpen = False;
        self.calibIndex = 0;
        self.frequencyChannel = 0;
        self.centerP = 0.;
        self.deltaP = 0.005;
        self.centerQ = 0.;
        self.deltaQ = 0.005;
        self.centerH = 0.;
        self.deltaH = 4.9/3600.*NP.pi;
        self.centerD = 0.;
        self.deltaD = 4.9/3600.*NP.pi;
        self.antNumberEW = 128
        self.antNumberS = 64
        self.averageCalib = False
        self.useNonlinearApproach = True
        self.obsObject = 'Sun'
        self.fringeStopping = False
        
        
        self.badAntsLcp = NP.zeros(128)
        self.badAntsRcp = NP.zeros(128)
        self.sizeOfUv = sizeOfUv
        self.baselines = 5

        self.flagsIndexes = []
        self.arcsecPerPixel = 4.91104/2

        self.centering_ftol = 1e-14
        self.convolutionNormCoef = 18.7 #16
        self.ZirinQSunTb = ZirinTb()
        self.flux_calibrated = False
        self.corr_amp_exist = False
        
        self.open(name, flux_norm)
        
                                    
    def open(self,fitsNames, flux_norm):
        if type(fitsNames) is not list: fitsNames = [fitsNames]
        if fitsNames[0]:
            try:
                self.hduList = fits.open(fitsNames[0])
                self.isOpen = True
                self.dateObs = self.hduList[0].header['DATE-OBS'] + 'T' + self.hduList[0].header['TIME-OBS']
                self.antennaNumbers = self.hduList[2].data['ant_index']
                self.antennaNumbers = NP.reshape(self.antennaNumbers,self.antennaNumbers.size)
                self.antennaNames = self.hduList[2].data['ant_name']
                self.antennaNames = NP.reshape(self.antennaNames,self.antennaNames.size)
                self.antennaA = self.hduList[4].data['ant_A']
                self.antennaA = NP.reshape(self.antennaA,self.antennaA.size)
                self.antennaB = self.hduList[4].data['ant_B']
                self.antennaB = NP.reshape(self.antennaB,self.antennaB.size)
                self.uvLcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex)
                self.uvRcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex)
                self.freqList = self.hduList[1].data['frequency'];
                self.freqListLength = self.freqList.size;
                self.dataLength = self.hduList[1].data['time'].size // self.freqListLength;
                self.freqTime = self.hduList[1].data['time']
                try:
                    self.freqTimeLcp = self.hduList[1].data['time_lcp']
                    self.freqTimeRcp = self.hduList[1].data['time_rcp']
                except:
                    pass
                self.visListLength = self.hduList[1].data['vis_lcp'].size // self.freqListLength // self.dataLength;
                self.visLcp = NP.reshape(self.hduList[1].data['vis_lcp'],(self.freqListLength,self.dataLength,self.visListLength));
                self.visRcp = NP.reshape(self.hduList[1].data['vis_rcp'],(self.freqListLength,self.dataLength,self.visListLength));
                # self.visLcp /= float(self.hduList[0].header['VIS_MAX'])
                # self.visRcp /= float(self.hduList[0].header['VIS_MAX'])
                self.ampLcp = NP.reshape(self.hduList[1].data['amp_lcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                self.ampRcp = NP.reshape(self.hduList[1].data['amp_rcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                ampScale = float(self.hduList[0].header['VIS_MAX']) / 128.
                self.ampLcp = self.ampLcp.astype(float) / ampScale
                self.ampRcp = self.ampRcp.astype(float) / ampScale
                try:
                    self.ampLcp_c = NP.reshape(self.hduList[1].data['amp_c_lcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                    self.ampRcp_c = NP.reshape(self.hduList[1].data['amp_c_rcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                    # self.ampLcp_c = self.ampLcp_c.astype(float) / ampScale
                    # self.ampRcp_c = self.ampRcp_c.astype(float) / ampScale
                    self.corr_amp_exist = True
                except:
                    pass
                
                self.RAO = BadaryRAO(self.dateObs.split('T')[0], 4.9, observedObject = self.obsObject)
                self.pAngle = NP.deg2rad(sunpy.coordinates.sun.P(self.dateObs).to_value())
                self.getHourAngle(0)
                
                self.ewAntPhaLcp = NP.zeros((self.freqListLength, self.antNumberEW))
                self.sAntPhaLcp = NP.zeros((self.freqListLength, self.antNumberS))
                self.ewAntPhaRcp = NP.zeros((self.freqListLength, self.antNumberEW))
                self.sAntPhaRcp = NP.zeros((self.freqListLength, self.antNumberS))
                self.ewLcpPhaseCorrection = NP.zeros((self.freqListLength, self.antNumberEW))
                self.ewRcpPhaseCorrection = NP.zeros((self.freqListLength, self.antNumberEW))
                self.sLcpPhaseCorrection = NP.zeros((self.freqListLength, self.antNumberS))
                self.sRcpPhaseCorrection = NP.zeros((self.freqListLength, self.antNumberS))
                self.sSolarPhase = NP.zeros(self.freqListLength)
                self.ewSolarPhase = NP.zeros(self.freqListLength)
                
                self.ewAntAmpLcp = NP.ones((self.freqListLength, self.antNumberEW))
                self.sAntAmpLcp = NP.ones((self.freqListLength, self.antNumberS))
                self.ewAntAmpRcp = NP.ones((self.freqListLength, self.antNumberEW))
                self.sAntAmpRcp = NP.ones((self.freqListLength, self.antNumberS))
                
                self.temperatureCoefsLcp = NP.ones(self.freqListLength)
                self.temperatureCoefsRcp = NP.ones(self.freqListLength)
                
                self.sLcpStair = NP.zeros(self.freqListLength)
                self.sRcpStair = NP.zeros(self.freqListLength)
                self.ewSlopeLcp = NP.zeros(self.freqListLength)
                self.sSlopeLcp = NP.zeros(self.freqListLength)
                self.ewSlopeRcp = NP.zeros(self.freqListLength)
                self.sSlopeRcp = NP.zeros(self.freqListLength)
                self.diskLevelLcp = NP.zeros(self.freqListLength)
                self.diskLevelRcp = NP.zeros(self.freqListLength)
                self.lm_hd_relation = NP.ones(self.freqListLength)
                
                self.flags_ew = NP.array((), dtype = int)
                self.flags_s = NP.array((), dtype = int)
                
                self.fluxLcp = NP.zeros(self.freqListLength)
                self.fluxRcp = NP.zeros(self.freqListLength)
                
                x_size = (self.baselines-1)*2 + self.antNumberEW + self.antNumberS
                self.x_ini_lcp = NP.full((self.freqListLength, x_size*2+2), NP.append(NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))),1))
                self.x_ini_rcp = NP.full((self.freqListLength, x_size*2+2), NP.append(NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))),1))
                self.calibrationResultLcp = NP.zeros_like(self.x_ini_lcp)
                self.calibrationResultRcp = NP.zeros_like(self.x_ini_rcp)
                
                self.x_ini_centering_lcp = NP.full((self.freqListLength, 4), NP.array([1,0,0,0]))
                self.x_ini_centering_rcp = NP.full((self.freqListLength, 4), NP.array([1,0,0,0]))
                self.centeringResultLcp = NP.zeros_like(self.x_ini_centering_lcp)
                self.centeringResultRcp = NP.zeros_like(self.x_ini_centering_rcp)
                
                self.beam_sr = NP.ones(self.freqListLength)

            except FileNotFoundError:
                print('File %s  not found'%fitsNames[0]);
                
        if len(fitsNames) > 1:
            for fitsName in fitsNames[1:]:
                self.append(fitsName)
        if flux_norm and self.corr_amp_exist:
            self.normalizeFlux()
        self.beam()
    
    def append(self,name):
        try:
            hduList = fits.open(name);
            freqTime = hduList[1].data['time']
            dataLength = hduList[1].data['time'].size // self.freqListLength;
            visLcp = NP.reshape(hduList[1].data['vis_lcp'],(self.freqListLength,dataLength,self.visListLength));
            visRcp = NP.reshape(hduList[1].data['vis_rcp'],(self.freqListLength,dataLength,self.visListLength));
            # visLcp /= float(self.hduList[0].header['VIS_MAX'])
            # visRcp /= float(self.hduList[0].header['VIS_MAX'])
            ampLcp = NP.reshape(hduList[1].data['amp_lcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
            ampRcp = NP.reshape(hduList[1].data['amp_rcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
            ampScale = float(self.hduList[0].header['VIS_MAX']) / 128.
            ampLcp = ampLcp.astype(float) / ampScale
            ampRcp = ampRcp.astype(float) / ampScale
            try:
                ampLcp_c = NP.reshape(hduList[1].data['amp_c_lcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
                ampRcp_c = NP.reshape(hduList[1].data['amp_c_rcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
                # ampLcp_c = ampLcp_c.astype(float) / ampScale
                # ampRcp_c = ampRcp_c.astype(float) / ampScale
                self.ampLcp_c = NP.concatenate((self.ampLcp_c, ampLcp_c), axis = 1)
                self.ampRcp_c = NP.concatenate((self.ampRcp_c, ampRcp_c), axis = 1)
            except:
                pass

            self.freqTime = NP.concatenate((self.freqTime, freqTime), axis = 1)
            self.visLcp = NP.concatenate((self.visLcp, visLcp), axis = 1)
            self.visRcp = NP.concatenate((self.visRcp, visRcp), axis = 1)
            self.ampLcp = NP.concatenate((self.ampLcp, ampLcp), axis = 1)
            self.ampRcp = NP.concatenate((self.ampRcp, ampRcp), axis = 1)
            self.dataLength += dataLength
            hduList.close()

        except FileNotFoundError:
            print('File %s  not found'%name);
            
    def normalizeFlux(self):
        zerosFits = fits.open('srh_0612_cp_zeros.fits')
        corrZeros = zerosFits[2].data['corrI']
        fluxZeros = zerosFits[2].data['fluxI']

        fluxNormFits = fits.open('srh_0612_cp_fluxNorm.fits')
        fluxNormI = fluxNormFits[2].data['fluxNormI']
        
        # max_amp = float(self.hduList[0].header['VIS_MAX']) / 128.
        
        self.lcpSigmaCSrc = NP.sqrt(self.ampLcp_c)# * max_amp)
        self.rcpSigmaCSrc = NP.sqrt(self.ampRcp_c)# * max_amp)
        
        self.rcpSigmaCSrc[self.rcpSigmaCSrc < 1000] = 1e6
        self.lcpSigmaCSrc[self.lcpSigmaCSrc < 1000] = 1e6
        
        for vis in range(18336):
            AB = self.visIndex2antIndex(vis)
            self.visLcp[:,:,vis] = self.visLcp[:,:,vis] / (self.lcpSigmaCSrc[:,:,AB[0]] * self.lcpSigmaCSrc[:,:,AB[1]])
            self.visRcp[:,:,vis] = self.visRcp[:,:,vis] / (self.rcpSigmaCSrc[:,:,AB[0]] * self.rcpSigmaCSrc[:,:,AB[1]])
    
        ampFluxRcp = NP.mean(self.ampRcp, axis = 2)
        ampFluxLcp = NP.mean(self.ampLcp, axis = 2)
        
        for ff in range(self.freqListLength):
            ampFluxRcp[ff,:] -= fluxZeros[ff]
            ampFluxRcp[ff,:] *= fluxNormI[ff]
            ampFluxLcp[ff,:] -= fluxZeros[ff]
            ampFluxLcp[ff,:] *= fluxNormI[ff]
            
            # lam = scipy.constants.c/(self.freqList[ff]*1e3)
            # self.tempLcp[ff] = NP.mean(ampFluxLcp[ff]) * lam**2 / (2*scipy.constants.k * self.beam_sr[ff])
            # self.tempRcp[ff] = NP.mean(ampFluxRcp[ff]) * lam**2 / (2*scipy.constants.k * self.beam_sr[ff])
            
            self.fluxLcp[ff] = NP.mean(ampFluxLcp[ff])
            self.fluxRcp[ff] = NP.mean(ampFluxRcp[ff])
            
            self.visLcp[ff,:,:] *= NP.mean(self.fluxLcp[ff])
            self.visRcp[ff,:,:] *= NP.mean(self.fluxRcp[ff])
            
            self.visLcp[ff,:,:] *= 2 # flux is divided by 2 for R and L
            self.visRcp[ff,:,:] *= 2
        
        self.flux_calibrated = True
            
    def beam(self):
        self.setFrequencyChannel(0)
        self.vis2uv(0, PSF = True)
        self.uv2lmImage()
        self.lm2Heliocentric(image_scale = 2)
        arcsecPerPix = self.arcsecPerPixel / 2.
        beam = self.lcp
        contours = (skimage.measure.find_contours(beam, 0.5*beam.max()))[0]
        con = NP.zeros_like(contours)
        con[:,1] = contours[:,0]
        con[:,0] = contours[:,1]
        sunEll = skimage.measure.EllipseModel()
        sunEll.estimate(con)
        major = NP.deg2rad(sunEll.params[2] * arcsecPerPix / 3600.)
        minor = NP.deg2rad(sunEll.params[3] * arcsecPerPix / 3600.)
        self.beam_sr[0] = NP.pi * major * minor / NP.log(2)
        for ff in range(1, self.freqListLength):
            self.beam_sr[ff] = self.beam_sr[0] * (self.freqList[0]/self.freqList[ff])**2
        """
        In original formula (https://science.nrao.edu/facilities/vla/proposing/TBconv)
        theta_maj ang theta_min are full widths of an ellipse, that is why there is
        4 in denominator.
        Here major and minor are semi-axes.
        """
            
    def calibrate(self, freq = 'all', phaseCorrect = True, amplitudeCorrect = True, average = 0):
        if freq == 'all':
            self.calculatePhaseCalibration()
            for freq in range(self.freqListLength):
                self.setFrequencyChannel(freq)
                self.vis2uv(scan = self.calibIndex, phaseCorrect=phaseCorrect, amplitudeCorrect=amplitudeCorrect, average=average)
                self.centerDisk()
        else:
            self.setFrequencyChannel(freq)
            self.solarPhase(freq)
            self.updateAntennaPhase(freq)
            self.vis2uv(scan = self.calibIndex, phaseCorrect=phaseCorrect, amplitudeCorrect=amplitudeCorrect, average=average)
            self.centerDisk()
            
    def image(self, freq, scan, average = 0, polarization = 'both', phaseCorrect = True, amplitudeCorrect = True, frame = 'heliocentric'):
        self.setFrequencyChannel(freq)    
        self.vis2uv(scan = scan, phaseCorrect=phaseCorrect, amplitudeCorrect=amplitudeCorrect, average=average)
        self.uv2lmImage()
        if frame == 'heliocentric':
            self.lm2Heliocentric()
        if polarization == 'both':
            return NP.flip(self.lcp, 0), NP.flip(self.rcp, 0)
        elif polarization == 'lcp':
            return NP.flip(self.lcp, 0)
        elif polarization == 'rcp':
            return NP.flip(self.rcp, 0)
        
    def saveCalibrationResult(self):
        currentGainsDict = {}
        currentGainsDict['calibrationResultLcp'] = self.calibrationResultLcp.tolist()
        currentGainsDict['calibrationResultRcp'] = self.calibrationResultRcp.tolist()
        with open('srh612CalibrationResult.json', 'w') as saveGainFile:
            json.dump(currentGainsDict, saveGainFile)
            
    def loadCalibrationResult(self):
        try:
            with open('srh612CalibrationResult.json','r') as readGainFile:
                currentGains = json.load(readGainFile)
            self.x_ini_lcp = NP.array(currentGains['calibrationResultLcp'])
            self.x_ini_rcp = NP.array(currentGains['calibrationResultRcp'])
        except FileNotFoundError:
            print('File \"srh612CalibrationResult.json\" not found')
            
    def saveGains(self, filename):
        currentGainsDict = {}
        currentGainsDict['ewPhaseLcp'] = (self.ewAntPhaLcp + self.ewLcpPhaseCorrection).tolist()
        currentGainsDict['sPhaseLcp'] = (self.sAntPhaLcp + self.sLcpPhaseCorrection).tolist()
        currentGainsDict['ewPhaseRcp'] = (self.ewAntPhaRcp + self.ewRcpPhaseCorrection).tolist()
        currentGainsDict['sPhaseRcp'] = (self.sAntPhaRcp + self.sRcpPhaseCorrection).tolist()
        currentGainsDict['ewAmpLcp'] = self.ewAntAmpLcp.tolist()
        currentGainsDict['sAmpLcp'] = self.sAntAmpLcp.tolist()
        currentGainsDict['ewAmpRcp'] = self.ewAntAmpRcp.tolist()
        currentGainsDict['sAmpRcp'] = self.sAntAmpRcp.tolist()
        currentGainsDict['sAmpRcp'] = self.sAntAmpRcp.tolist()
        currentGainsDict['lm_hd_relation'] = self.lm_hd_relation.tolist()
        with open(filename, 'w') as saveGainFile:
            json.dump(currentGainsDict, saveGainFile)
            
    def loadGains(self, filename):
        with open(filename,'r') as readGainFile:
            currentGains = json.load(readGainFile)
        self.ewAntPhaLcp = NP.array(currentGains['ewPhaseLcp'])
        self.sAntPhaLcp = NP.array(currentGains['sPhaseLcp'])
        self.ewAntPhaRcp = NP.array(currentGains['ewPhaseRcp'])
        self.sAntPhaRcp = NP.array(currentGains['sPhaseRcp'])
        self.ewAntAmpLcp = NP.array(currentGains['ewAmpLcp'])
        self.sAntAmpLcp = NP.array(currentGains['sAmpLcp'])
        self.ewAntAmpRcp = NP.array(currentGains['ewAmpRcp'])
        self.sAntAmpRcp = NP.array(currentGains['sAmpRcp'])
        self.lm_hd_relation = NP.array(currentGains['lm_hd_relation'])

    def changeObject(self, obj):
        self.obsObject = obj
        self.RAO = BadaryRAO(self.dateObs.split('T')[0], observedObject = self.obsObject)
        
    def getHourAngle(self, scan):
        self.hAngle = self.omegaEarth * (self.freqTime[self.frequencyChannel, scan] - self.RAO.culmination)
        if self.hAngle > 1.5*NP.pi:
            self.hAngle -= 2*NP.pi
        return self.hAngle
        
    def setSizeOfUv(self, sizeOfUv):
        self.sizeOfUv = sizeOfUv
        self.uvLcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex);
        self.uvRcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex);
        
    def getDeclination(self):
        return self.RAO.declination

    def getPQScale(self, size, FOV):
        self.cosP = NP.sin(self.hAngle) * NP.cos(self.RAO.declination)
        self.cosQ = NP.cos(self.hAngle) * NP.cos(self.RAO.declination) * NP.sin(self.RAO.observatory.lat) - NP.sin(self.RAO.declination) * NP.cos(self.RAO.observatory.lat)
        FOV_p = 2.*(constants.c / (self.freqList[self.frequencyChannel]*1e3)) / (self.RAO.base*NP.sqrt(1. - self.cosP**2.));
        FOV_q = 2.*(constants.c / (self.freqList[self.frequencyChannel]*1e3)) / (self.RAO.base*NP.sqrt(1. - self.cosQ**2.));
        
        return [int(size*FOV/FOV_p.to_value()), int(size*FOV/FOV_q.to_value())]
        
    def getPQ2HDMatrix(self):
        gP =  NP.arctan(NP.tan(self.hAngle)*NP.sin(self.RAO.declination));
        gQ =  NP.arctan(-(NP.sin(self.RAO.declination) / NP.tan(self.hAngle) + NP.cos(self.RAO.declination) / (NP.sin(self.hAngle)*NP.tan(self.RAO.observatory.lat))));
        
        if self.hAngle > 0:
            gQ = NP.pi + gQ;
        g = gP - gQ;
          
        pqMatrix = NP.zeros((3,3))
        pqMatrix[0, 0] =  NP.cos(gP) - NP.cos(g)*NP.cos(gQ)
        pqMatrix[0, 1] = -NP.cos(g)*NP.cos(gP) + NP.cos(gQ)
        pqMatrix[1, 0] =  NP.sin(gP) - NP.cos(g)*NP.sin(gQ)
        pqMatrix[1, 1] = -NP.cos(g)*NP.sin(gP) + NP.sin(gQ)
        pqMatrix /= NP.sin(g)**2.
        pqMatrix[2, 2] = 1.
        return pqMatrix
        
    def close(self):
        self.hduList.close();
        
    def flag(self, names):
        nameList = names.split(',')
        for i in range(len(nameList)):
            ind = NP.where(self.antennaNames == nameList[i])[0]
            if len(ind):
                self.flagsIndexes.append(int(self.antennaNumbers[ind[0]]))
        self.flagVis = NP.array([], dtype = int)
        for i in range(len(self.flagsIndexes)):
            ind = NP.where(self.antennaA == self.flagsIndexes[i])[0]
            if len(ind):
                self.flagVis = NP.append(self.flagVis, ind)
            ind = NP.where(self.antennaB == self.flagsIndexes[i])[0]
            if len(ind):
                self.flagVis = NP.append(self.flagVis, ind)
        self.visLcp[:,:,self.flagVis] = 0
        self.visRcp[:,:,self.flagVis] = 0
        
    def visIndex2antIndex(self, visIndex):
        if visIndex > self.antennaA.size or visIndex < 0:
            Exception('visIndex is out of range')
        else:
            return self.antennaA[visIndex], self.antennaB[visIndex]
    
    def antIndex2visIndex(self, antA, antB):
        if antA > self.antennaA.size or antA < 0:
            Exception('antA index is out of range')
        elif antB > self.antennaB.size or antB < 0:
            Exception('antB index is out of range')
        else:
            try:
                return NP.where((self.antennaA==antA) & (self.antennaB==antB))[0][0]
            except:
                return NP.where((self.antennaA==antB) & (self.antennaB==antA))[0][0]
        
    def setBaselinesNumber(self, value):
        self.baselines = value
        x_size = (self.baselines-1)*2 + self.antNumberEW + self.antNumberS
        self.x_ini_lcp = NP.full((self.freqListLength, x_size*2+2), NP.append(NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))),1))
        self.x_ini_rcp = NP.full((self.freqListLength, x_size*2+2), NP.append(NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))),1))
        self.calibrationResultLcp = NP.zeros_like(self.x_ini_lcp)
        self.calibrationResultRcp = NP.zeros_like(self.x_ini_rcp)    
    
    def buildEwPhase(self):
        newLcpPhaseCorrection = NP.zeros(self.antNumberEW)
        newRcpPhaseCorrection = NP.zeros(self.antNumberEW)
        for j in range(self.antNumberEW):
                newLcpPhaseCorrection[j] += NP.deg2rad(self.ewSlopeLcp[self.frequencyChannel] * (j - 63.5)) 
                newRcpPhaseCorrection[j] += NP.deg2rad(self.ewSlopeRcp[self.frequencyChannel] * (j - 63.5))
        self.ewLcpPhaseCorrection[self.frequencyChannel, :] = newLcpPhaseCorrection[:]
        self.ewRcpPhaseCorrection[self.frequencyChannel, :] = newRcpPhaseCorrection[:]
        
    def buildSPhase(self):
        newLcpPhaseCorrection = NP.zeros(self.antNumberS)
        newRcpPhaseCorrection = NP.zeros(self.antNumberS)
        for j in range(self.antNumberS):
                newLcpPhaseCorrection[j] += (NP.deg2rad(-self.sSlopeLcp[self.frequencyChannel] * (j + 0.5)) + NP.deg2rad(self.sLcpStair[self.frequencyChannel]))
                newRcpPhaseCorrection[j] += (NP.deg2rad(-self.sSlopeRcp[self.frequencyChannel] * (j + 0.5)) + NP.deg2rad(self.sRcpStair[self.frequencyChannel]))
        self.sLcpPhaseCorrection[self.frequencyChannel, :] = newLcpPhaseCorrection[:]
        self.sRcpPhaseCorrection[self.frequencyChannel, :] = newRcpPhaseCorrection[:]
        
    def phaMatrixGen(self, antNumber):
        phaMatrix = NP.zeros((antNumber, antNumber + 1))
        phaMatrix[:,0] = 1
        for pair in range(antNumber - 1):
            phaMatrix[pair, pair + 1] = 1
            phaMatrix[pair, pair + 2] = -1
        return phaMatrix.copy()
    
    
    def phaMatrixGenPairs(self, pairs, antNumber):
        rows = int(((antNumber - 1) + (antNumber - pairs))/2 * pairs)
        cols = antNumber + pairs
        phaMatrix = NP.zeros((rows, cols))
        for pair in range(pairs):
            row0 = int(((antNumber - 1) + (antNumber - pair))/2 * pair)
            row1 = row0 + (antNumber - pair - 1)
            phaMatrix[row0:row1,pair] = 1
            for phaPair in range(antNumber - pair - 1):
                phaMatrix[phaPair + row0, phaPair + pairs] = 1
                phaMatrix[phaPair + row0, phaPair + pairs + (pair + 1)] = -1
        return phaMatrix.copy()

    def phaMatrixGenPairsEWN(self, pairs, antNumberEW, antNumberN):
        rowsEW = int(((antNumberEW - 1) + (antNumberEW - pairs))/2 * pairs)
        rowsN = int(((antNumberN) + (antNumberN + 1 - pairs))/2 * pairs)
        colsEW = antNumberEW + pairs
        colsN = antNumberN + pairs
        phaMatrix = NP.zeros((rowsEW + rowsN, colsEW + colsN))
        for pair in range(pairs):
            row0 = int(((antNumberEW - 1) + (antNumberEW - pair))/2 * pair)
            row1 = row0 + (antNumberEW - pair - 1)
            phaMatrix[row0:row1,pair] = 1
            for phaPair in range(antNumberEW - pair - 1):
                phaMatrix[phaPair + row0, phaPair + 2*pairs] = 1
                phaMatrix[phaPair + row0, phaPair + 2*pairs + (pair + 1)] = -1
            row0 = int(((antNumberN) + (antNumberN + 1 - pair))/2 * pair)
            row1 = row0 + (antNumberN - pair)
            phaMatrix[row0 + rowsEW:row1 + rowsEW,pairs + pair] = 1
            for phaPair in range(antNumberN - pair):
                if phaPair == 0:
                    phaMatrix[rowsEW + row0, 2*pairs + 32] = 1
                else:
                    phaMatrix[phaPair + rowsEW + row0, 2*pairs + phaPair + antNumberEW - 1] = 1
                phaMatrix[phaPair + rowsEW + row0, 2*pairs + phaPair + antNumberEW - 1 + (pair + 1)] = -1
        return phaMatrix.copy()
    
    def calculatePhaseCalibration(self, baselinesNumber = 5, lcp = True, rcp = True):
       for freq in range(self.freqListLength):
           print('FREQ ' + str(freq))
           # if freq>0:
           #     self.x_ini_rcp[freq] = self.calibrationResultRcp[freq-1]
           #     self.x_ini_lcp[freq] = self.calibrationResultLcp[freq-1]
           self.updateAntennaPhase(freq, baselinesNumber, lcp = lcp, rcp = rcp)
           
    def calculateAmpCalibration(self, baselinesNumber = 5):
       for freq in range(self.freqListLength):
           self.calculateAmplitude_linear(freq, baselinesNumber)

    def updateAntennaPhase(self, freqChannel, baselinesNumber = 5, lcp = True, rcp = True):
        self.solarPhase(freqChannel)
        if self.useNonlinearApproach:
            if lcp:
                self.calculatePhaseLcp_nonlinear_cross_new(freqChannel, baselinesNumber = baselinesNumber)
            if rcp:
                self.calculatePhaseRcp_nonlinear_cross_new(freqChannel, baselinesNumber = baselinesNumber)
            if rcp and lcp:
                flags_ew_lcp = NP.where(self.ewAntAmpLcp[freqChannel] == 1e6)[0]
                flags_ew_rcp = NP.where(self.ewAntAmpRcp[freqChannel] == 1e6)[0]
                self.flags_ew = NP.unique(NP.append(flags_ew_lcp, flags_ew_rcp))
                flags_s_lcp = NP.where(self.sAntAmpLcp[freqChannel] == 1e6)[0]
                flags_s_rcp = NP.where(self.sAntAmpRcp[freqChannel] == 1e6)[0]
                self.flags_s = NP.unique(NP.append(flags_s_lcp, flags_s_rcp))
                self.ewAntAmpLcp[freqChannel][self.flags_ew] = 1e6
                self.sAntAmpLcp[freqChannel][self.flags_s] = 1e6
                self.ewAntAmpRcp[freqChannel][self.flags_ew] = 1e6
                self.sAntAmpRcp[freqChannel][self.flags_s] = 1e6
        else:
            self.calculatePhase_linear(freqChannel)
            self.calculateAmp_linear(freqChannel)
            flags_ew_lcp = NP.where(self.ewAntAmpLcp[freqChannel] == 1e6)[0]
            flags_ew_rcp = NP.where(self.ewAntAmpRcp[freqChannel] == 1e6)[0]
            self.flags_ew = NP.unique(NP.append(flags_ew_lcp, flags_ew_rcp))
            flags_s_lcp = NP.where(self.sAntAmpLcp[freqChannel] == 1e6)[0]
            flags_s_rcp = NP.where(self.sAntAmpRcp[freqChannel] == 1e6)[0]
            self.flags_s = NP.unique(NP.append(flags_s_lcp, flags_s_rcp))
            self.ewAntAmpLcp[freqChannel][self.flags_ew] = 1e6
            self.sAntAmpLcp[freqChannel][self.flags_s] = 1e6
            self.ewAntAmpRcp[freqChannel][self.flags_ew] = 1e6
            self.sAntAmpRcp[freqChannel][self.flags_s] = 1e6
            
    def solarPhase(self, freq):
        u,v,w = base2uvw_612.base2uvw(self.hAngle, self.RAO.declination, 129, 130)
        baseWave = NP.sqrt(u**2+v**2)*self.freqList[freq]*1e3/constants.c.to_value()
        if baseWave > 120:
            self.sSolarPhase[freq] = NP.pi
        else:
            self.sSolarPhase[freq] = 0
        u,v,w = base2uvw_612.base2uvw(self.hAngle, self.RAO.declination, 1, 2)
        baseWave = NP.sqrt(u**2+v**2)*self.freqList[freq]*1e3/constants.c.to_value()
        if baseWave > 120:
            self.ewSolarPhase[freq] = NP.pi
        else:
            self.ewSolarPhase[freq] = 0
            
    def calculatePhase_linear(self, freqChannel, baselinesNumber = 1):
        antNumberS = self.antNumberS
        antNumberEW = self.antNumberEW
        redIndexesS = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberS - baseline):
                redIndexesS.append(NP.where((self.antennaA==128+i) & (self.antennaB==128+i+baseline))[0][0])
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==i) & (self.antennaB==i+baseline))[0][0])
                
        phaMatrixEW = self.phaMatrixGen(antNumberEW)
        phaMatrixS = self.phaMatrixGen(antNumberS)
        
        redundantVisLcpEW = self.visLcp[freqChannel, self.calibIndex, redIndexesEW]
        redundantVisLcpS = self.visLcp[freqChannel, self.calibIndex, redIndexesS]

        phasesLcpEW = NP.concatenate((NP.angle(redundantVisLcpEW), (self.ewSolarPhase[freqChannel],)))
        antPhaLcpEW, c, d, e = NP.linalg.lstsq(phaMatrixEW, phasesLcpEW, rcond=None)
        self.ewAntPhaLcp[freqChannel] = antPhaLcpEW[1:]
        phasesLcpS = NP.concatenate((NP.angle(redundantVisLcpS), (self.sSolarPhase[freqChannel],)))
        antPhaLcpS, c, d, e = NP.linalg.lstsq(phaMatrixS, phasesLcpS, rcond=None)
        self.sAntPhaLcp[freqChannel] = antPhaLcpS[1:]
        
        redundantVisRcpEW = self.visRcp[freqChannel, self.calibIndex, redIndexesEW]
        redundantVisRcpS = self.visRcp[freqChannel, self.calibIndex, redIndexesS]

        phasesRcpEW = NP.concatenate((NP.angle(redundantVisRcpEW), (self.ewSolarPhase[freqChannel],)))
        antPhaRcpEW, c, d, e = NP.linalg.lstsq(phaMatrixEW, phasesRcpEW, rcond=None)
        self.ewAntPhaRcp[freqChannel] = antPhaRcpEW[1:]
        phasesRcpS = NP.concatenate((NP.angle(redundantVisRcpS), (self.sSolarPhase[freqChannel],)))
        antPhaRcpS, c, d, e = NP.linalg.lstsq(phaMatrixS, phasesRcpS, rcond=None)
        self.sAntPhaRcp[freqChannel] = antPhaRcpS[1:]
        
        
    def calculateAmp_linear(self, freqChannel, baselinesNumber = 2):
        antNumberS = self.antNumberS
        antNumberEW = self.antNumberEW
        redIndexesS = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberS - baseline):
                redIndexesS.append(NP.where((self.antennaA==128+i) & (self.antennaB==128+i+baseline))[0][0])
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==i) & (self.antennaB==i+baseline))[0][0])
                
        ampMatrixEW = NP.abs(self.phaMatrixGenPairs(baselinesNumber, antNumberEW))
        ampMatrixS = NP.abs(self.phaMatrixGenPairs(baselinesNumber, antNumberS))
        
        redundantVisLcpEW = self.visLcp[freqChannel, self.calibIndex, redIndexesEW]
        redundantVisLcpS = self.visLcp[freqChannel, self.calibIndex, redIndexesS]
        ampLcpEW = NP.abs(redundantVisLcpEW)
        antAmpLcpEW, c, d, e = NP.linalg.lstsq(ampMatrixEW, NP.log(ampLcpEW), rcond=None)
        ampLcpS = NP.abs(redundantVisLcpS)
        antAmpLcpS, c, d, e = NP.linalg.lstsq(ampMatrixS, NP.log(ampLcpS), rcond=None)

        redundantVisRcpEW = self.visRcp[freqChannel, self.calibIndex, redIndexesEW]
        redundantVisRcpS = self.visRcp[freqChannel, self.calibIndex, redIndexesS]
        ampRcpEW = NP.abs(redundantVisRcpEW)
        antAmpRcpEW, c, d, e = NP.linalg.lstsq(ampMatrixEW, NP.log(ampRcpEW), rcond=None)
        ampRcpS = NP.abs(redundantVisRcpS)
        antAmpRcpS, c, d, e = NP.linalg.lstsq(ampMatrixS, NP.log(ampRcpS), rcond=None)
        
        gains = NP.append(self.ewAntAmpLcp[freqChannel], self.sAntAmpLcp[freqChannel])
        norm = NP.mean(NP.abs(gains[NP.abs(gains)>NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpLcp[freqChannel] = NP.exp(antAmpLcpEW[baselinesNumber:])/norm
        self.sAntAmpLcp[freqChannel] = NP.exp(antAmpLcpS[baselinesNumber:])/norm
        
        lin_gains_lcp = gains * NP.exp(1j * NP.append(self.ewAntPhaLcp[freqChannel], self.sAntPhaLcp[freqChannel]))
        self.x_ini_lcp[freqChannel][(self.baselines-1)*2+1:(self.baselines-1)*2+1+antNumberEW+antNumberS] = NP.real(lin_gains_lcp)
        self.x_ini_lcp[freqChannel][(self.baselines-1)*2+1+antNumberEW+antNumberS+(self.baselines-1)*2 : ] = NP.imag(lin_gains_lcp)
        
        gains = NP.append(self.ewAntAmpRcp[freqChannel], self.sAntAmpRcp[freqChannel])
        norm = NP.mean(NP.abs(gains[NP.abs(gains)>NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpRcp[freqChannel] = NP.exp(antAmpRcpEW[baselinesNumber:])/norm
        self.sAntAmpRcp[freqChannel] = NP.exp(antAmpRcpS[baselinesNumber:])/norm
        
        lin_gains_rcp = gains * NP.exp(1j * NP.append(self.ewAntPhaRcp[freqChannel], self.sAntPhaRcp[freqChannel]))
        self.x_ini_rcp[freqChannel][(self.baselines-1)*2+1:(self.baselines-1)*2+1+antNumberEW+antNumberS] = NP.real(lin_gains_rcp)
        self.x_ini_rcp[freqChannel][(self.baselines-1)*2+1+antNumberEW+antNumberS+(self.baselines-1)*2 : ] = NP.imag(lin_gains_rcp)
        
        self.ewAntAmpLcp[freqChannel][self.ewAntAmpLcp[freqChannel]<NP.median(self.ewAntAmpLcp[freqChannel])*0.5] = 1e6
        self.sAntAmpLcp[freqChannel][self.sAntAmpLcp[freqChannel]<NP.median(self.sAntAmpLcp[freqChannel])*0.5] = 1e6
        self.ewAntAmpRcp[freqChannel][self.ewAntAmpRcp[freqChannel]<NP.median(self.ewAntAmpRcp[freqChannel])*0.5] = 1e6
        self.sAntAmpRcp[freqChannel][self.sAntAmpRcp[freqChannel]<NP.median(self.sAntAmpRcp[freqChannel])*0.5] = 1e6
        
        
    def calculatePhaseLcp_nonlinear(self, freqChannel, baselinesNumber = 2):
        antNumberS = self.antNumberS
        antNumberEW = self.antNumberEW
        redIndexesS = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberS - baseline):
                redIndexesS.append(NP.where((self.antennaA==128+i) & (self.antennaB==128+i+baseline))[0][0])
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==i) & (self.antennaB==i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisS = NP.mean(self.visLcp[freqChannel, :20, redIndexesS], axis = 1)
            redundantVisEW = NP.mean(self.visLcp[freqChannel, :20, redIndexesEW], axis = 1)
            redundantVisAll = NP.append(redundantVisEW, redundantVisS)
        else:
            redundantVisS = self.visLcp[freqChannel, self.calibIndex, redIndexesS]
            redundantVisEW = self.visLcp[freqChannel, self.calibIndex, redIndexesEW]
            redundantVisAll = NP.append(redundantVisEW, redundantVisS)
            
        for i in range(len(redIndexesS)):    
            if NP.any(self.flags_s == self.antennaA[redIndexesS[i]]) or NP.any(self.flags_s == self.antennaB[redIndexesS[i]]):
                redundantVisS[i]=0.
        for i in range(len(redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[redIndexesEW[i]]):
                redundantVisEW[i]=0.

        ls_res = least_squares(self.allGainsFunc_constrained, self.x_ini_lcp[freqChannel], args = (redundantVisAll, antNumberEW, antNumberS, baselinesNumber, freqChannel), max_nfev = 400)
        self.calibrationResultLcp[freqChannel] = ls_res['x']
        self.x_ini_lcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:])[(baselinesNumber-1)*2:]
        self.ew_gains_lcp = gains[:antNumberEW]
        self.ewAntPhaLcp[freqChannel] = NP.angle(self.ew_gains_lcp)
        self.s_gains_lcp = gains[antNumberEW:]
        self.sAntPhaLcp[freqChannel] = NP.angle(self.s_gains_lcp)
        
        norm = NP.mean(NP.abs(gains[NP.abs(gains)>NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpLcp[freqChannel] = NP.abs(self.ew_gains_lcp)/norm
        self.ewAntAmpLcp[freqChannel][self.ewAntAmpLcp[freqChannel]<NP.median(self.ewAntAmpLcp[freqChannel])*0.5] = 1e6
        self.sAntAmpLcp[freqChannel] = NP.abs(self.s_gains_lcp)/norm
        self.sAntAmpLcp[freqChannel][self.sAntAmpLcp[freqChannel]<NP.median(self.sAntAmpLcp[freqChannel])*0.5] = 1e6
    
    def calculatePhaseLcp_nonlinear_cross(self, freqChannel, baselinesNumber = 5):
        antNumberS = self.antNumberS
        antNumberEW = self.antNumberEW
        redIndexesS = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberS - baseline):
                redIndexesS.append(NP.where((self.antennaA==128+i) & (self.antennaB==128+i+baseline))[0][0])
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==i) & (self.antennaB==i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisS = NP.mean(self.visLcp[freqChannel, :20, redIndexesS], axis = 1)
            redundantVisEW = NP.mean(self.visLcp[freqChannel, :20, redIndexesEW], axis = 1)
            crossVis = NP.mean(self.visLcp[freqChannel, :20, 63],)
        else:
            redundantVisS = self.visLcp[freqChannel, self.calibIndex, redIndexesS]
            redundantVisEW = self.visLcp[freqChannel, self.calibIndex, redIndexesEW]
            crossVis = self.visLcp[freqChannel, self.calibIndex, 63]
            
        for i in range(len(redIndexesS)):    
            if NP.any(self.flags_s == self.antennaA[redIndexesS[i]]) or NP.any(self.flags_s == self.antennaB[redIndexesS[i]]):
                redundantVisS[i]=0.
        for i in range(len(redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[redIndexesEW[i]]):
                redundantVisEW[i]=0.
                
        redundantVisAll = NP.append(redundantVisEW, redundantVisS)
        redundantVisAll = NP.append(redundantVisAll, crossVis)

        ls_res = least_squares(self.allGainsFunc_constrained_cross, self.x_ini_lcp[freqChannel], args = (redundantVisAll, antNumberEW, antNumberS, baselinesNumber, freqChannel), max_nfev = 400)
        self.calibrationResultLcp[freqChannel] = ls_res['x']
        self.x_ini_lcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:-1])[(baselinesNumber-1)*2:]
        self.ew_gains_lcp = gains[:antNumberEW]
        self.ewAntPhaLcp[freqChannel] = NP.angle(self.ew_gains_lcp)
        self.s_gains_lcp = gains[antNumberEW:]
        self.sAntPhaLcp[freqChannel] = NP.angle(self.s_gains_lcp)
        
        norm = NP.mean(NP.abs(self.ew_gains_lcp[NP.abs(self.ew_gains_lcp)>NP.median(NP.abs(self.ew_gains_lcp))*0.5]))
        self.ewAntAmpLcp[freqChannel] = NP.abs(self.ew_gains_lcp)/norm
        self.ewAntAmpLcp[freqChannel][self.ewAntAmpLcp[freqChannel]<NP.median(self.ewAntAmpLcp[freqChannel])*0.5] = 1e6
        norm = NP.mean(NP.abs(self.s_gains_lcp[NP.abs(self.s_gains_lcp)>NP.median(NP.abs(self.s_gains_lcp))*0.5]))
        self.sAntAmpLcp[freqChannel] = NP.abs(self.s_gains_lcp)/norm
        self.sAntAmpLcp[freqChannel][self.sAntAmpLcp[freqChannel]<NP.median(self.sAntAmpLcp[freqChannel])*0.5] = 1e6
    
    def calculatePhaseRcp_nonlinear_cross(self, freqChannel, baselinesNumber = 5):
        antNumberS = self.antNumberS
        antNumberEW = self.antNumberEW
        redIndexesS = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberS - baseline):
                redIndexesS.append(NP.where((self.antennaA==128+i) & (self.antennaB==128+i+baseline))[0][0])
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==i) & (self.antennaB==i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisS = NP.mean(self.visRcp[freqChannel, :20, redIndexesS], axis = 1)
            redundantVisEW = NP.mean(self.visRcp[freqChannel, :20, redIndexesEW], axis = 1)
            crossVis = NP.mean(self.visRcp[freqChannel, :20, 63],)
        else:
            redundantVisS = self.visRcp[freqChannel, self.calibIndex, redIndexesS]
            redundantVisEW = self.visRcp[freqChannel, self.calibIndex, redIndexesEW]
            crossVis = self.visRcp[freqChannel, self.calibIndex, 63]
            
        for i in range(len(redIndexesS)):    
            if NP.any(self.flags_s == self.antennaA[redIndexesS[i]]) or NP.any(self.flags_s == self.antennaB[redIndexesS[i]]):
                redundantVisS[i]=0.
        for i in range(len(redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[redIndexesEW[i]]):
                redundantVisEW[i]=0.
                
        redundantVisAll = NP.append(redundantVisEW, redundantVisS)
        redundantVisAll = NP.append(redundantVisAll, crossVis)

        ls_res = least_squares(self.allGainsFunc_constrained_cross, self.x_ini_rcp[freqChannel], args = (redundantVisAll, antNumberEW, antNumberS, baselinesNumber, freqChannel), max_nfev = 400)
        self.calibrationResultRcp[freqChannel] = ls_res['x']
        self.x_ini_rcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:-1])[(baselinesNumber-1)*2:]
        self.ew_gains_rcp = gains[:antNumberEW]
        self.ewAntPhaRcp[freqChannel] = NP.angle(self.ew_gains_rcp)
        self.s_gains_rcp = gains[antNumberEW:]
        self.sAntPhaRcp[freqChannel] = NP.angle(self.s_gains_rcp)
        
        norm = NP.mean(NP.abs(self.ew_gains_rcp[NP.abs(self.ew_gains_rcp)>NP.median(NP.abs(self.ew_gains_rcp))*0.5]))
        self.ewAntAmpRcp[freqChannel] = NP.abs(self.ew_gains_rcp)/norm
        self.ewAntAmpRcp[freqChannel][self.ewAntAmpRcp[freqChannel]<NP.median(self.ewAntAmpRcp[freqChannel])*0.5] = 1e6
        norm = NP.mean(NP.abs(self.s_gains_rcp[NP.abs(self.s_gains_rcp)>NP.median(NP.abs(self.s_gains_rcp))*0.5]))
        self.sAntAmpRcp[freqChannel] = NP.abs(self.s_gains_rcp)/norm
        self.sAntAmpRcp[freqChannel][self.sAntAmpRcp[freqChannel]<NP.median(self.sAntAmpRcp[freqChannel])*0.5] = 1e6
        
    def calculatePhaseLcp_nonlinear_cross_new(self, freqChannel, baselinesNumber = 5):
        antNumberS = self.antNumberS
        antNumberEW = self.antNumberEW
        redIndexesS = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberS - baseline):
                redIndexesS.append(NP.where((self.antennaA==128+i) & (self.antennaB==128+i+baseline))[0][0])
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==i) & (self.antennaB==i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisS = NP.mean(self.visLcp[freqChannel, :20, redIndexesS], axis = 1)
            redundantVisEW = NP.mean(self.visLcp[freqChannel, :20, redIndexesEW], axis = 1)
            crossVis = NP.mean(self.visLcp[freqChannel, :20, 63],)
        else:
            redundantVisS = self.visLcp[freqChannel, self.calibIndex, redIndexesS]
            redundantVisEW = self.visLcp[freqChannel, self.calibIndex, redIndexesEW]
            crossVis = self.visLcp[freqChannel, self.calibIndex, 63]
            
        for i in range(len(redIndexesS)):    
            if NP.any(self.flags_s == self.antennaA[redIndexesS[i]]) or NP.any(self.flags_s == self.antennaB[redIndexesS[i]]):
                redundantVisS[i]=0.
        for i in range(len(redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[redIndexesEW[i]]):
                redundantVisEW[i]=0.
                
        redundantVisAll = NP.append(redundantVisEW, redundantVisS)
        redundantVisAll = NP.append(redundantVisAll, crossVis)
        
        ewAmpSign = 1 if self.ewSolarPhase[freqChannel]==0 else -1
        sAmpSign = 1 if self.sSolarPhase[freqChannel]==0 else -1
        
        res = NP.zeros_like(redundantVisAll, dtype = complex)
        ewSolarAmp = 1 * ewAmpSign
        sAntNumber = antNumberS
        sGainsNumber = antNumberS
        ewGainsNumber = antNumberEW
        sSolVisNumber = baselinesNumber - 1
        ewSolVisNumber = baselinesNumber - 1
        sNum = int((2*(antNumberS-1) - (baselinesNumber-1))/2 * baselinesNumber)
        ewNum = int((2*(antNumberEW-1) - (baselinesNumber-1))/2 * baselinesNumber)
        solVisArrayS = NP.zeros(sNum, dtype = complex)
        antAGainsS = NP.zeros(sNum, dtype = complex)
        antBGainsS = NP.zeros(sNum, dtype = complex)
        solVisArrayEW = NP.zeros(ewNum, dtype = complex)
        antAGainsEW = NP.zeros(ewNum, dtype = complex)
        antBGainsEW = NP.zeros(ewNum, dtype = complex)
        ewSolVis = NP.zeros(baselinesNumber, dtype = complex)
        sSolVis = NP.zeros(baselinesNumber, dtype = complex)
        solVis = NP.zeros_like(redundantVisAll, dtype = complex)
        antAGains = NP.zeros_like(redundantVisAll, dtype = complex)
        antBGains = NP.zeros_like(redundantVisAll, dtype = complex)
        
        args = (redundantVisAll, antNumberEW, antNumberS, baselinesNumber, freqChannel,
                res, ewSolarAmp, sGainsNumber, ewGainsNumber, sSolVisNumber, 
                ewSolVisNumber, sNum, ewNum, solVisArrayS, antAGainsS, antBGainsS, solVisArrayEW, 
                antAGainsEW, antBGainsEW, ewSolVis, sSolVis, solVis, antAGains, antBGains, sAmpSign)

        ls_res = least_squares(self.allGainsFunc_constrained_cross_new, self.x_ini_lcp[freqChannel], args = args, max_nfev = 400)
        self.calibrationResultLcp[freqChannel] = ls_res['x']
        self.x_ini_lcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:-1])[(baselinesNumber-1)*2:]
        self.ew_gains_lcp = gains[:antNumberEW]
        self.ewAntPhaLcp[freqChannel] = NP.angle(self.ew_gains_lcp)
        self.s_gains_lcp = gains[antNumberEW:]
        self.sAntPhaLcp[freqChannel] = NP.angle(self.s_gains_lcp)
        
        norm = NP.mean(NP.abs(self.ew_gains_lcp[NP.abs(self.ew_gains_lcp)>NP.median(NP.abs(self.ew_gains_lcp))*0.5]))
        self.ewAntAmpLcp[freqChannel] = NP.abs(self.ew_gains_lcp)/norm
        self.ewAntAmpLcp[freqChannel][self.ewAntAmpLcp[freqChannel]<NP.median(self.ewAntAmpLcp[freqChannel])*0.5] = 1e6
        norm = NP.mean(NP.abs(self.s_gains_lcp[NP.abs(self.s_gains_lcp)>NP.median(NP.abs(self.s_gains_lcp))*0.5]))
        self.sAntAmpLcp[freqChannel] = NP.abs(self.s_gains_lcp)/norm
        self.sAntAmpLcp[freqChannel][self.sAntAmpLcp[freqChannel]<NP.median(self.sAntAmpLcp[freqChannel])*0.5] = 1e6
    
    def calculatePhaseRcp_nonlinear_cross_new(self, freqChannel, baselinesNumber = 5):
        antNumberS = self.antNumberS
        antNumberEW = self.antNumberEW
        redIndexesS = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberS - baseline):
                redIndexesS.append(NP.where((self.antennaA==128+i) & (self.antennaB==128+i+baseline))[0][0])
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==i) & (self.antennaB==i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisS = NP.mean(self.visRcp[freqChannel, :20, redIndexesS], axis = 1)
            redundantVisEW = NP.mean(self.visRcp[freqChannel, :20, redIndexesEW], axis = 1)
            crossVis = NP.mean(self.visRcp[freqChannel, :20, 63],)
        else:
            redundantVisS = self.visRcp[freqChannel, self.calibIndex, redIndexesS]
            redundantVisEW = self.visRcp[freqChannel, self.calibIndex, redIndexesEW]
            crossVis = self.visRcp[freqChannel, self.calibIndex, 63]
            
        for i in range(len(redIndexesS)):    
            if NP.any(self.flags_s == self.antennaA[redIndexesS[i]]) or NP.any(self.flags_s == self.antennaB[redIndexesS[i]]):
                redundantVisS[i]=0.
        for i in range(len(redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[redIndexesEW[i]]):
                redundantVisEW[i]=0.
                
        redundantVisAll = NP.append(redundantVisEW, redundantVisS)
        redundantVisAll = NP.append(redundantVisAll, crossVis)
        
        ewAmpSign = 1 if self.ewSolarPhase[freqChannel]==0 else -1
        sAmpSign = 1 if self.sSolarPhase[freqChannel]==0 else -1
        
        res = NP.zeros_like(redundantVisAll, dtype = complex)
        ewSolarAmp = 1 * ewAmpSign
        sAntNumber = antNumberS
        sGainsNumber = antNumberS
        ewGainsNumber = antNumberEW
        sSolVisNumber = baselinesNumber - 1
        ewSolVisNumber = baselinesNumber - 1
        sNum = int((2*(antNumberS-1) - (baselinesNumber-1))/2 * baselinesNumber)
        ewNum = int((2*(antNumberEW-1) - (baselinesNumber-1))/2 * baselinesNumber)
        solVisArrayS = NP.zeros(sNum, dtype = complex)
        antAGainsS = NP.zeros(sNum, dtype = complex)
        antBGainsS = NP.zeros(sNum, dtype = complex)
        solVisArrayEW = NP.zeros(ewNum, dtype = complex)
        antAGainsEW = NP.zeros(ewNum, dtype = complex)
        antBGainsEW = NP.zeros(ewNum, dtype = complex)
        ewSolVis = NP.zeros(baselinesNumber, dtype = complex)
        sSolVis = NP.zeros(baselinesNumber, dtype = complex)
        solVis = NP.zeros_like(redundantVisAll, dtype = complex)
        antAGains = NP.zeros_like(redundantVisAll, dtype = complex)
        antBGains = NP.zeros_like(redundantVisAll, dtype = complex)
        
        args = (redundantVisAll, antNumberEW, antNumberS, baselinesNumber, freqChannel,
                res, ewSolarAmp, sGainsNumber, ewGainsNumber, sSolVisNumber, 
                ewSolVisNumber, sNum, ewNum, solVisArrayS, antAGainsS, antBGainsS, solVisArrayEW, 
                antAGainsEW, antBGainsEW, ewSolVis, sSolVis, solVis, antAGains, antBGains, sAmpSign)

        ls_res = least_squares(self.allGainsFunc_constrained_cross_new, self.x_ini_lcp[freqChannel], args = args, max_nfev = 400)
        self.calibrationResultRcp[freqChannel] = ls_res['x']
        self.x_ini_rcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:-1])[(baselinesNumber-1)*2:]
        self.ew_gains_rcp = gains[:antNumberEW]
        self.ewAntPhaRcp[freqChannel] = NP.angle(self.ew_gains_rcp)
        self.s_gains_rcp = gains[antNumberEW:]
        self.sAntPhaRcp[freqChannel] = NP.angle(self.s_gains_rcp)
        
        norm = NP.mean(NP.abs(self.ew_gains_rcp[NP.abs(self.ew_gains_rcp)>NP.median(NP.abs(self.ew_gains_rcp))*0.5]))
        self.ewAntAmpRcp[freqChannel] = NP.abs(self.ew_gains_rcp)/norm
        self.ewAntAmpRcp[freqChannel][self.ewAntAmpRcp[freqChannel]<NP.median(self.ewAntAmpRcp[freqChannel])*0.5] = 1e6
        norm = NP.mean(NP.abs(self.s_gains_rcp[NP.abs(self.s_gains_rcp)>NP.median(NP.abs(self.s_gains_rcp))*0.5]))
        self.sAntAmpRcp[freqChannel] = NP.abs(self.s_gains_rcp)/norm
        self.sAntAmpRcp[freqChannel][self.sAntAmpRcp[freqChannel]<NP.median(self.sAntAmpRcp[freqChannel])*0.5] = 1e6
        
    def allGainsFunc_constrained_cross_new(self, x, obsVis, antNumberEW, antNumberS, baselinesNumber, freqChannel,
                res, ewSolarAmp, sGainsNumber, ewGainsNumber, sSolVisNumber, 
                ewSolVisNumber, sNum, ewNum, solVisArrayS, antAGainsS, antBGainsS, solVisArrayEW, 
                antAGainsEW, antBGainsEW, ewSolVis, sSolVis, solVis, antAGains, antBGains, sAmpSign):

        sSolarAmp = NP.abs(x[0])
        x_complex = self.real_to_complex(x[1:-1])
    
        crossVis = x[-1] * NP.exp(1j * 0)

        ewSolVis[0] = ewSolarAmp
        ewSolVis[1:] = x_complex[: ewSolVisNumber]
        sSolVis[0] = sSolarAmp
        sSolVis[1:] = x_complex[ewSolVisNumber : ewSolVisNumber+sSolVisNumber]
        
        ewGains = x_complex[ewSolVisNumber+sSolVisNumber : ewSolVisNumber+sSolVisNumber+ewGainsNumber]
        sGains = x_complex[ewSolVisNumber+sSolVisNumber+ewGainsNumber :]
  
        # for baseline in range(1, baselinesNumber+1):
        #     solVisArrayS = NP.append(solVisArrayS, NP.full(sAntNumber-baseline, sSolVis[baseline-1]))
        #     antAGainsS = NP.append(antAGainsS, sGains[:sAntNumber-baseline])
        #     antBGainsS = NP.append(antBGainsS, sGains[baseline:])
            
        #     solVisArrayEW = NP.append(solVisArrayEW, NP.full(ewAntNumber-baseline, ewSolVis[baseline-1]))
        #     antAGainsEW = NP.append(antAGainsEW, ewGains[:ewAntNumber-baseline])
        #     antBGainsEW = NP.append(antBGainsEW, ewGains[baseline:])
            
        # solVisArray = NP.append(solVisArrayEW, solVisArrayS)
        # antAGains = NP.append(antAGainsEW, antAGainsS)
        # antBGains = NP.append(antBGainsEW, antBGainsS)
        
        # solVisArray = NP.append(solVisArray, crossVis)
        # antAGains = NP.append(antAGains, ewGains[ewAntNumber//2-1])
        # antBGains = NP.append(antBGains, sGains[0])
            
        # res = solVisArray * antAGains * NP.conj(antBGains) - obsVis
        # return self.complex_to_real(res)  
        
        prev_ind_s = 0
        prev_ind_ew = 0
        for baseline in range(1, baselinesNumber+1):
            solVisArrayS[prev_ind_s:prev_ind_s+antNumberS-baseline] = NP.full(antNumberS-baseline, sSolVis[baseline-1])
            antAGainsS[prev_ind_s:prev_ind_s+antNumberS-baseline] = sGains[:antNumberS-baseline]
            antBGainsS[prev_ind_s:prev_ind_s+antNumberS-baseline] = sGains[baseline:]
            prev_ind_s = prev_ind_s+antNumberS-baseline
            
            solVisArrayEW[prev_ind_ew:prev_ind_ew+antNumberEW-baseline] = NP.full(antNumberEW-baseline, ewSolVis[baseline-1])
            antAGainsEW[prev_ind_ew:prev_ind_ew+antNumberEW-baseline] = ewGains[:antNumberEW-baseline]
            antBGainsEW[prev_ind_ew:prev_ind_ew+antNumberEW-baseline] = ewGains[baseline:]
            prev_ind_ew = prev_ind_ew+antNumberEW-baseline
            
        solVis[:len(solVisArrayEW)] = solVisArrayEW
        solVis[len(solVisArrayEW):-1] = solVisArrayS
        antAGains[:len(antAGainsEW)] = antAGainsEW
        antAGains[len(antAGainsEW):-1] = antAGainsS
        antBGains[:len(antBGainsEW)] = antBGainsEW
        antBGains[len(antBGainsEW):-1] = antBGainsS
        
        solVis[-1] = crossVis
        antAGains[-1] = ewGains[antNumberEW//2-1]
        antBGains[-1] = sGains[0]
        
        res = solVis * antAGains * NP.conj(antBGains) - obsVis
        return self.complex_to_real(res) 
    
    def calculatePhaseRcp_nonlinear(self, freqChannel, baselinesNumber = 2):
        antNumberS = self.antNumberS
        antNumberEW = self.antNumberEW
        redIndexesS = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberS - baseline):
                redIndexesS.append(NP.where((self.antennaA==128+i) & (self.antennaB==128+i+baseline))[0][0])
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==i) & (self.antennaB==i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisS = NP.mean(self.visRcp[freqChannel, :20, redIndexesS], axis = 1)
            redundantVisEW = NP.mean(self.visRcp[freqChannel, :20, redIndexesEW], axis = 1)
            redundantVisAll = NP.append(redundantVisEW, redundantVisS)
        else:
            redundantVisS = self.visRcp[freqChannel, self.calibIndex, redIndexesS]
            redundantVisEW = self.visRcp[freqChannel, self.calibIndex, redIndexesEW]
            redundantVisAll = NP.append(redundantVisEW, redundantVisS)
            
        for i in range(len(redIndexesS)):    
            if NP.any(self.flags_s == self.antennaA[redIndexesS[i]]) or NP.any(self.flags_s == self.antennaB[redIndexesS[i]]):
                redundantVisS[i]=0.
        for i in range(len(redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[redIndexesEW[i]]):
                redundantVisEW[i]=0.

        ls_res = least_squares(self.allGainsFunc_constrained, self.x_ini_rcp[freqChannel], args = (redundantVisAll, antNumberEW, antNumberS, baselinesNumber, freqChannel), max_nfev = 400)
        self.calibrationResultRcp[freqChannel] = ls_res['x']
        self.x_ini_rcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:])[(baselinesNumber-1)*2:]
        self.ew_gains_rcp = gains[:antNumberEW]
        self.ewAntPhaRcp[freqChannel] = NP.angle(self.ew_gains_rcp)
        self.s_gains_rcp = gains[antNumberEW:]
        self.sAntPhaRcp[freqChannel] = NP.angle(self.s_gains_rcp)
        
        norm = NP.mean(NP.abs(gains[NP.abs(gains)>NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpRcp[freqChannel] = NP.abs(self.ew_gains_rcp)/norm
        self.ewAntAmpRcp[freqChannel][self.ewAntAmpRcp[freqChannel]<NP.median(self.ewAntAmpRcp[freqChannel])*0.5] = 1e6
        self.sAntAmpRcp[freqChannel] = NP.abs(self.s_gains_rcp)/norm
        self.sAntAmpRcp[freqChannel][self.sAntAmpRcp[freqChannel]<NP.median(self.sAntAmpRcp[freqChannel])*0.5] = 1e6
        
    def calculatePhaseLcp_nonlinear_flags(self, freqChannel, baselinesNumber = 2):
        
        antNumberS = self.antNumberS
        antNumberEW = self.antNumberEW
        redIndexesS = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberS - baseline):
                # if not (NP.any(self.flags_s == (128+i)) or NP.any(self.flags_s == (128+i+baseline))):
                redIndexesS.append(NP.where((self.antennaA==128+i) & (self.antennaB==128+i+baseline))[0][0])
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                # if not (NP.any(self.flags_ew == i) or NP.any(self.flags_ew == (i+baseline))):
                redIndexesEW.append(NP.where((self.antennaA==i) & (self.antennaB==i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisS = NP.mean(self.visLcp[freqChannel, :20, redIndexesS], axis = 1)
            redundantVisEW = NP.mean(self.visLcp[freqChannel, :20, redIndexesEW], axis = 1) 
        else:
            redundantVisS = self.visLcp[freqChannel, self.calibIndex, redIndexesS]
            redundantVisEW = self.visLcp[freqChannel, self.calibIndex, redIndexesEW]
            
        for i in range(len(redIndexesS)):    
            if NP.any(self.flags_s == self.antennaA[redIndexesS[i]]) or NP.any(self.flags_s == self.antennaB[redIndexesS[i]]):
                redundantVisS[i]=0.
        for i in range(len(redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[redIndexesEW[i]]):
                redundantVisEW[i]=0.
        
        redundantVisAll = NP.append(redundantVisEW, redundantVisS)

        ls_res = least_squares(self.allGainsFunc_constrained, self.x_ini_lcp[freqChannel], args = (redundantVisAll, antNumberEW, antNumberS, baselinesNumber, freqChannel), max_nfev = 400)
        self.calibrationResultLcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:])[(baselinesNumber-1)*2:]
        self.ew_gains_lcp = gains[:antNumberEW]
        self.ewAntPhaLcp[freqChannel] = NP.angle(self.ew_gains_lcp)
        self.s_gains_lcp = gains[antNumberEW:]
        self.sAntPhaLcp[freqChannel] = NP.angle(self.s_gains_lcp)
        
        norm = NP.mean(NP.abs(gains[NP.abs(gains)>NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpLcp[freqChannel] = NP.abs(self.ew_gains_lcp)/norm
        self.ewAntAmpLcp[freqChannel][self.ewAntAmpLcp[freqChannel]<NP.median(self.ewAntAmpLcp[freqChannel])*0.5] = 1e6
        self.sAntAmpLcp[freqChannel] = NP.abs(self.s_gains_lcp)/norm
        self.sAntAmpLcp[freqChannel][self.sAntAmpLcp[freqChannel]<NP.median(self.sAntAmpLcp[freqChannel])*0.5] = 1e6 
    
    def eastWestGainsFunc_constrained(self, x, obsVis, antNumber, baselineNumber, freq):
        res = NP.zeros_like(obsVis, dtype = complex)
        x_complex = self.real_to_complex(x)
        solVis = NP.append((NP.exp(1j*self.ewSolarPhase[freq])), x_complex[:baselineNumber-1])
        gains = NP.insert(x_complex[baselineNumber-1:], 32, (1+0j))
        
        solVisArray = NP.array(())
        antAGains = NP.array(())
        antBGains = NP.array(())
        for baseline in range(1, baselineNumber+1):
            solVisArray = NP.append(solVisArray, NP.full(antNumber-baseline, solVis[baseline-1]))
            antAGains = NP.append(antAGains, gains[:antNumber-baseline])
            antBGains = NP.append(antBGains, gains[baseline:])
        res = solVisArray * antAGains * NP.conj(antBGains) - obsVis
        return self.complex_to_real(res)

    def northGainsFunc_constrained(self, x, obsVis, antNumber, baselineNumber, freq):
        antNumber_c = antNumber + 1
        res = NP.zeros_like(obsVis, dtype = complex)
        x_complex = self.real_to_complex(x)
        solVis = NP.append((NP.exp(1j*self.nSolarPhase[freq])), x_complex[:baselineNumber-1])
        gains = NP.append((1+0j), x_complex[baselineNumber-1:])
        
        solVisArray = NP.array(())
        antAGains = NP.array(())
        antBGains = NP.array(())
        for baseline in range(1, baselineNumber+1):
            solVisArray = NP.append(solVisArray, NP.full(antNumber_c-baseline, solVis[baseline-1]))
            antAGains = NP.append(antAGains, gains[:antNumber_c-baseline])
            antBGains = NP.append(antBGains, gains[baseline:])
        res = solVisArray * antAGains * NP.conj(antBGains) - obsVis
        return self.complex_to_real(res)
    
    def allGainsFunc_constrained(self, x, obsVis, ewAntNumber, sAntNumber, baselineNumber, freq):
        res = NP.zeros_like(obsVis, dtype = complex)
        ewSolarAmp = 1
        sSolarAmp = NP.abs(x[0])
        x_complex = self.real_to_complex(x[1:])
    
        
        sGainsNumber = sAntNumber
        ewGainsNumber = ewAntNumber
        sSolVisNumber = baselineNumber - 1
        ewSolVisNumber = baselineNumber - 1
        ewSolVis = NP.append((ewSolarAmp * NP.exp(1j*self.ewSolarPhase[freq])), x_complex[: ewSolVisNumber])
        sSolVis = NP.append((sSolarAmp * NP.exp(1j*self.sSolarPhase[freq])), x_complex[ewSolVisNumber : ewSolVisNumber+sSolVisNumber])
        ewGains = x_complex[ewSolVisNumber+sSolVisNumber : ewSolVisNumber+sSolVisNumber+ewGainsNumber]
        sGains = x_complex[ewSolVisNumber+sSolVisNumber+ewGainsNumber :]
        
        solVisArrayS = NP.array(())
        antAGainsS = NP.array(())
        antBGainsS = NP.array(())
        solVisArrayEW = NP.array(())
        antAGainsEW = NP.array(())
        antBGainsEW = NP.array(())
        for baseline in range(1, baselineNumber+1):
            solVisArrayS = NP.append(solVisArrayS, NP.full(sAntNumber-baseline, sSolVis[baseline-1]))
            antAGainsS = NP.append(antAGainsS, sGains[:sAntNumber-baseline])
            antBGainsS = NP.append(antBGainsS, sGains[baseline:])
            
            solVisArrayEW = NP.append(solVisArrayEW, NP.full(ewAntNumber-baseline, ewSolVis[baseline-1]))
            antAGainsEW = NP.append(antAGainsEW, ewGains[:ewAntNumber-baseline])
            antBGainsEW = NP.append(antBGainsEW, ewGains[baseline:])
            
        res = NP.append(solVisArrayEW, solVisArrayS) * NP.append(antAGainsEW, antAGainsS) * NP.conj(NP.append(antBGainsEW, antBGainsS)) - obsVis
        return self.complex_to_real(res)  
    
    def allGainsFunc_constrained_cross(self, x, obsVis, ewAntNumber, sAntNumber, baselineNumber, freq):
        res = NP.zeros_like(obsVis, dtype = complex)
        ewSolarAmp = 1
        sSolarAmp = NP.abs(x[0])
        x_complex = self.real_to_complex(x[1:-1])
    
        
        sGainsNumber = sAntNumber
        ewGainsNumber = ewAntNumber
        sSolVisNumber = baselineNumber - 1
        ewSolVisNumber = baselineNumber - 1
        crossVis = x[-1] * NP.exp(1j * 0)
        ewSolVis = NP.append((ewSolarAmp * NP.exp(1j*self.ewSolarPhase[freq])), x_complex[: ewSolVisNumber])
        sSolVis = NP.append((sSolarAmp * NP.exp(1j*self.sSolarPhase[freq])), x_complex[ewSolVisNumber : ewSolVisNumber+sSolVisNumber])
        ewGains = x_complex[ewSolVisNumber+sSolVisNumber : ewSolVisNumber+sSolVisNumber+ewGainsNumber]
        sGains = x_complex[ewSolVisNumber+sSolVisNumber+ewGainsNumber :]
        
        solVisArrayS = NP.array(())
        antAGainsS = NP.array(())
        antBGainsS = NP.array(())
        solVisArrayEW = NP.array(())
        antAGainsEW = NP.array(())
        antBGainsEW = NP.array(())
        for baseline in range(1, baselineNumber+1):
            solVisArrayS = NP.append(solVisArrayS, NP.full(sAntNumber-baseline, sSolVis[baseline-1]))
            antAGainsS = NP.append(antAGainsS, sGains[:sAntNumber-baseline])
            antBGainsS = NP.append(antBGainsS, sGains[baseline:])
            
            solVisArrayEW = NP.append(solVisArrayEW, NP.full(ewAntNumber-baseline, ewSolVis[baseline-1]))
            antAGainsEW = NP.append(antAGainsEW, ewGains[:ewAntNumber-baseline])
            antBGainsEW = NP.append(antBGainsEW, ewGains[baseline:])
            
        solVisArray = NP.append(solVisArrayEW, solVisArrayS)
        antAGains = NP.append(antAGainsEW, antAGainsS)
        antBGains = NP.append(antBGainsEW, antBGainsS)
        
        solVisArray = NP.append(solVisArray, crossVis)
        antAGains = NP.append(antAGains, ewGains[ewAntNumber//2-1])
        antBGains = NP.append(antBGains, sGains[0])
            
        res = solVisArray * antAGains * NP.conj(antBGains) - obsVis
        return self.complex_to_real(res)  
    
    def allGainsFunc_constrained_old(self, x, obsVis, ewAntNumber, nAntNumber, baselineNumber, freq):
        res = NP.zeros_like(obsVis, dtype = complex)
        ewSolarAmp = NP.abs(x[0])
        nSolarAmp = NP.abs(x[1])
        x_complex = self.real_to_complex(x[2:])
        
        nAntNumber_c = nAntNumber + 1
        
        nGainsNumber = nAntNumber
        ewGainsNumber = ewAntNumber
        nSolVisNumber = baselineNumber - 1
        ewSolVisNumber = baselineNumber - 1
        ewSolVis = NP.append((ewSolarAmp * NP.exp(1j*self.ewSolarPhase[freq])), x_complex[: ewSolVisNumber])
        nSolVis = NP.append((nSolarAmp * NP.exp(1j*self.nSolarPhase[freq])), x_complex[ewSolVisNumber : ewSolVisNumber+nSolVisNumber])
        ewGains = x_complex[ewSolVisNumber+nSolVisNumber : ewSolVisNumber+nSolVisNumber+ewGainsNumber]
        nGains = NP.append(ewGains[32], x_complex[ewSolVisNumber+nSolVisNumber+ewGainsNumber :])
        
        solVisArrayN = NP.array(())
        antAGainsN = NP.array(())
        antBGainsN = NP.array(())
        solVisArrayEW = NP.array(())
        antAGainsEW = NP.array(())
        antBGainsEW = NP.array(())
        for baseline in range(1, baselineNumber+1):
            solVisArrayN = NP.append(solVisArrayN, NP.full(nAntNumber_c-baseline, nSolVis[baseline-1]))
            antAGainsN = NP.append(antAGainsN, nGains[:nAntNumber_c-baseline])
            antBGainsN = NP.append(antBGainsN, nGains[baseline:])
            
            solVisArrayEW = NP.append(solVisArrayEW, NP.full(ewAntNumber-baseline, ewSolVis[baseline-1]))
            antAGainsEW = NP.append(antAGainsEW, ewGains[:ewAntNumber-baseline])
            antBGainsEW = NP.append(antBGainsEW, ewGains[baseline:])
            
        res = NP.append(solVisArrayEW, solVisArrayN) * NP.append(antAGainsEW, antAGainsN) * NP.conj(NP.append(antBGainsEW, antBGainsN)) - obsVis
        return self.complex_to_real(res)  
    
    def correctPhaseSlopeRL(self, freq):
        workingAnts_ew = NP.arange(0,128,1)
        workingAnts_ew = NP.delete(workingAnts_ew, NP.append(self.flags_ew, NP.array((64,81,89,0,1,2))))
        phaseDif_ew = NP.unwrap((self.ewAntPhaLcp[freq][workingAnts_ew]+self.ewLcpPhaseCorrection[freq][workingAnts_ew])
                             - (self.ewAntPhaRcp[freq][workingAnts_ew]+self.ewRcpPhaseCorrection[freq][workingAnts_ew]))
        A = NP.vstack([workingAnts_ew, NP.ones(len(workingAnts_ew))]).T
        ew_slope, c = NP.linalg.lstsq(A, phaseDif_ew, rcond=None)[0]
        workingAnts_s = NP.arange(0,64,1)
        workingAnts_s = NP.delete(workingAnts_s, NP.append(self.flags_s, NP.array((35,))))
        phaseDif_s = NP.unwrap((self.sAntPhaLcp[freq][workingAnts_s]+self.sLcpPhaseCorrection[freq][workingAnts_s])
                             - (self.sAntPhaRcp[freq][workingAnts_s]+self.sRcpPhaseCorrection[freq][workingAnts_s]))
        A = NP.vstack([workingAnts_s, NP.ones(len(workingAnts_s))]).T
        s_slope, c = NP.linalg.lstsq(A, phaseDif_s, rcond=None)[0]
        self.ewSlopeRcp[freq] = self.wrap(self.ewSlopeRcp[freq] + NP.rad2deg(ew_slope))
        self.sSlopeRcp[freq] = self.wrap(self.sSlopeRcp[freq] - NP.rad2deg(s_slope))

    
#    def changeEastWestPhase(self, newLcpPhaseCorrection, newRcpPhaseCorrection):
#        self.ewLcpPhaseCorrection[self.frequencyChannel, :] = newLcpPhaseCorrection[:]
#        self.ewRcpPhaseCorrection[self.frequencyChannel, :] = newRcpPhaseCorrection[:]
#        
#    def changeNorthPhase(self, newLcpPhaseCorrection, newRcpPhaseCorrection):
#        self.nLcpPhaseCorrection[self.frequencyChannel, :] = newLcpPhaseCorrection[:]
#        self.nRcpPhaseCorrection[self.frequencyChannel, :] = newRcpPhaseCorrection[:]
    
    def real_to_complex(self, z):
        return z[:len(z)//2] + 1j * z[len(z)//2:]
    
    def complex_to_real(self, z):
        return NP.concatenate((NP.real(z), NP.imag(z)))
    
    def setCalibIndex(self, calibIndex):
        self.calibIndex = calibIndex;

    def setFrequencyChannel(self, channel):
        self.frequencyChannel = channel
        
    def vis2uv(self, scan, phaseCorrect = True, amplitudeCorrect = True, PSF=False, average = 0):
        self.uvLcp[:,:] = complex(0,0)
        self.uvRcp[:,:] = complex(0,0)
        
#        self.uvRcp[O + jj*2, O + (ii - 16)*2] *= 0.01*NP.sqrt(self.ampRcp[self.frequencyChannel,scan,ii + 16] * self.ampRcp[self.frequencyChannel,scan,jj])
#        self.uvLcp[O - 2 - jj*2, O + (15 - ii)*2]
        
        O = self.sizeOfUv//2 + 1
        if average:
            firstScan = scan
            if  self.visLcp.shape[1] < (scan + average):
                lastScan = self.dataLength
            else:
                lastScan = scan + average
            for i in range(self.antNumberS):
                for j in range(self.antNumberEW):
                    if not (NP.any(self.flags_ew == j) or NP.any(self.flags_s == i)):
                        self.uvLcp[O + i*2, O + (j-64)*2] = NP.mean(self.visLcp[self.frequencyChannel, firstScan:lastScan, i*128+j])
                        self.uvRcp[O + i*2, O + (j-64)*2] = NP.mean(self.visRcp[self.frequencyChannel, firstScan:lastScan, i*128+j])
                        if (phaseCorrect):
                            ewPh = self.ewAntPhaLcp[self.frequencyChannel, j]+self.ewLcpPhaseCorrection[self.frequencyChannel, j]
                            sPh = self.sAntPhaLcp[self.frequencyChannel, i]+self.sLcpPhaseCorrection[self.frequencyChannel, i]
                            self.uvLcp[O + i*2, O + (j-64)*2] *= NP.exp(1j * (-ewPh + sPh))
                            ewPh = self.ewAntPhaRcp[self.frequencyChannel, j]+self.ewRcpPhaseCorrection[self.frequencyChannel, j]
                            sPh = self.sAntPhaRcp[self.frequencyChannel, i]+self.sRcpPhaseCorrection[self.frequencyChannel, i]
                            self.uvRcp[O + i*2, O + (j-64)*2] *= NP.exp(1j * (-ewPh + sPh))
                        if (amplitudeCorrect):
                            self.uvLcp[O + i*2, O + (j-64)*2] /= (self.ewAntAmpLcp[self.frequencyChannel, j] * self.sAntAmpLcp[self.frequencyChannel, i])
                            self.uvRcp[O + i*2, O + (j-64)*2] /= (self.ewAntAmpRcp[self.frequencyChannel, j] * self.sAntAmpRcp[self.frequencyChannel, i])
                        self.uvLcp[O - (i+1)*2, O - (j-63)*2] = NP.conj(self.uvLcp[O + i*2, O + (j-64)*2])
                        self.uvRcp[O - (i+1)*2, O - (j-63)*2] = NP.conj(self.uvRcp[O + i*2, O + (j-64)*2])
                        
                    

                
        else:
            for i in range(self.antNumberS):
                for j in range(self.antNumberEW):
                    if not (NP.any(self.flags_ew == j) or NP.any(self.flags_s == i)):
                        self.uvLcp[O + i*2, O + (j-64)*2] = self.visLcp[self.frequencyChannel, scan, i*128+j]
                        self.uvRcp[O + i*2, O + (j-64)*2] = self.visRcp[self.frequencyChannel, scan, i*128+j]
                        if (phaseCorrect):
                            ewPh = self.ewAntPhaLcp[self.frequencyChannel, j]+self.ewLcpPhaseCorrection[self.frequencyChannel, j]
                            sPh = self.sAntPhaLcp[self.frequencyChannel, i]+self.sLcpPhaseCorrection[self.frequencyChannel, i]
                            self.uvLcp[O + i*2, O + (j-64)*2] *= NP.exp(1j * (-ewPh + sPh))
                            ewPh = self.ewAntPhaRcp[self.frequencyChannel, j]+self.ewRcpPhaseCorrection[self.frequencyChannel, j]
                            sPh = self.sAntPhaRcp[self.frequencyChannel, i]+self.sRcpPhaseCorrection[self.frequencyChannel, i]
                            self.uvRcp[O + i*2, O + (j-64)*2] *= NP.exp(1j * (-ewPh + sPh))
                        if (amplitudeCorrect):
                            self.uvLcp[O + i*2, O + (j-64)*2] /= (self.ewAntAmpLcp[self.frequencyChannel, j] * self.sAntAmpLcp[self.frequencyChannel, i])
                            self.uvRcp[O + i*2, O + (j-64)*2] /= (self.ewAntAmpRcp[self.frequencyChannel, j] * self.sAntAmpRcp[self.frequencyChannel, i])
                        if (self.fringeStopping):
                            self.uvLcp[O + i*2, O + (j-64)*2] *= NP.exp(1j * 2*NP.pi*self.freqList[self.frequencyChannel]*1e3 * (-self.nDelays[i, scan] + self.ewDelays[j, scan]))
                            self.uvRcp[O + i*2, O + (j-64)*2] *= NP.exp(1j * 2*NP.pi*self.freqList[self.frequencyChannel]*1e3 * (-self.nDelays[i, scan] + self.ewDelays[j, scan]))
                        self.uvLcp[O - (i+1)*2, O - (j-63)*2] = NP.conj(self.uvLcp[O + i*2, O + (j-64)*2])
                        self.uvRcp[O - (i+1)*2, O - (j-63)*2] = NP.conj(self.uvRcp[O + i*2, O + (j-64)*2])
                    
        if PSF:
            self.uvLcp[NP.abs(self.uvLcp)>1e-8] = 1
            self.uvRcp[NP.abs(self.uvRcp)>1e-8] = 1

        self.uvLcp[NP.abs(self.uvLcp)<1e-6] = 0.
        self.uvRcp[NP.abs(self.uvRcp)<1e-6] = 0.
        self.uvLcp /= NP.count_nonzero(self.uvLcp)
        self.uvRcp /= NP.count_nonzero(self.uvRcp)
        
        
        # self.uvLcp /= self.convolutionNormCoef
        # self.uvRcp /= self.convolutionNormCoef
        # self.uvLcp *= self.temperatureCoefsLcp[self.frequencyChannel]
        # self.uvRcp *= self.temperatureCoefsRcp[self.frequencyChannel]
                    
 
    def uv2lmImage(self):
        self.lcp = NP.fft.fft2(NP.roll(NP.roll(self.uvLcp,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        self.lcp = NP.roll(NP.roll(self.lcp,self.sizeOfUv//2,0),self.sizeOfUv//2,1);
        self.rcp = NP.fft.fft2(NP.roll(NP.roll(self.uvRcp,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        self.rcp = NP.roll(NP.roll(self.rcp,self.sizeOfUv//2,0),self.sizeOfUv//2,1);
        if self.flux_calibrated:
            lam = scipy.constants.c/(self.freqList[self.frequencyChannel]*1e3)
            self.lcp = self.lcp * lam**2 * 1e-22 / (2*scipy.constants.k * self.beam_sr[self.frequencyChannel])
            self.rcp = self.rcp * lam**2 * 1e-22 / (2*scipy.constants.k * self.beam_sr[self.frequencyChannel])
        
    def lm2Heliocentric(self, image_scale = 0.5):
        scaling = self.getPQScale(self.sizeOfUv, NP.deg2rad(self.arcsecPerPixel * (self.sizeOfUv - 1)/3600.)/image_scale)
        scale = AffineTransform(scale=(self.sizeOfUv/scaling[0], self.sizeOfUv/scaling[1]))
        shift = AffineTransform(translation=(-self.sizeOfUv/2,-self.sizeOfUv/2))
        rotate = AffineTransform(rotation = self.pAngle)
        matrix = AffineTransform(matrix = self.getPQ2HDMatrix())
        back_shift = AffineTransform(translation=(self.sizeOfUv/2,self.sizeOfUv/2))

        O = self.sizeOfUv//2
        Q = self.sizeOfUv//4
        dataResult0 = warp(self.lcp.real,(shift + (scale + back_shift)).inverse)
        self.lcp = warp(dataResult0,(shift + (matrix + back_shift)).inverse)
        dataResult0 = warp(self.rcp.real,(shift + (scale + back_shift)).inverse)
        self.rcp = warp(dataResult0,(shift + (matrix + back_shift)).inverse)
        dataResult0 = 0
        self.lcp = warp(self.lcp,(shift + (rotate + back_shift)).inverse)[O-Q:O+Q,O-Q:O+Q]
        self.rcp = warp(self.rcp,(shift + (rotate + back_shift)).inverse)[O-Q:O+Q,O-Q:O+Q]
        self.lcp = NP.flip(self.lcp, 1)
        self.rcp = NP.flip(self.rcp, 1)
        
        
    def calculateDelays(self):
        self.nDelays = NP.zeros((self.antNumberN, self.dataLength))
        self.ewDelays = NP.zeros((self.antNumberEW, self.dataLength))
        base = 9.8
        ew_center = 32
        ns_center = 0
        _PHI = 0.903338787600965
        for scan in range(self.dataLength):
            hourAngle = self.omegaEarth * (self.freqTime[self.frequencyChannel, scan] - self.RAO.culmination)
            dec = self.getDeclination()
            for ant in range(self.antNumberN):
                cosQ = NP.cos(hourAngle) * NP.cos(dec)*NP.sin(_PHI) - NP.sin(dec)*NP.cos(_PHI);
                M = ns_center - ant - 1;
                self.nDelays[ant, scan] = base * M * cosQ / constants.c.to_value();
            for ant in range(self.antNumberEW):
                cosP = NP.sin(hourAngle) * NP.cos(dec);
                N = ew_center - ant;
                self.ewDelays[ant, scan] = base * N * cosP / constants.c.to_value();
                
    def resetDelays(self):
        self.nDelays = NP.zeros((self.antNumberN, self.dataLength))
        self.ewDelays = NP.zeros((self.antNumberEW, self.dataLength))
        
    def createDisk(self, radius, arcsecPerPixel = 2.45552):
        qSun = NP.zeros((self.sizeOfUv, self.sizeOfUv))
        sunRadius = radius / (arcsecPerPixel)
        for i in range(self.sizeOfUv):
            x = i - self.sizeOfUv//2 - 1
            for j in range(self.sizeOfUv):
                y = j - self.sizeOfUv//2 - 1
                if (NP.sqrt(x*x + y*y) < sunRadius):
                    qSun[i , j] = 1
                    
        dL = 2*( 6//2) + 1
        arg_x = NP.linspace(-1.,1,dL)
        arg_y = NP.linspace(-1.,1,dL)
        xx, yy = NP.meshgrid(arg_x, arg_y)
        
        scaling = self.getPQScale(self.sizeOfUv, NP.deg2rad(arcsecPerPixel*(self.sizeOfUv-1)/3600.))
        scale = AffineTransform(scale=(scaling[0]/self.sizeOfUv, scaling[1]/self.sizeOfUv))
        back_shift = AffineTransform(translation=(self.sizeOfUv/2, self.sizeOfUv/2))
        shift = AffineTransform(translation=(-self.sizeOfUv/2, -self.sizeOfUv/2))
        matrix = AffineTransform(matrix = NP.linalg.inv(self.getPQ2HDMatrix()))
        rotate = AffineTransform(rotation = -self.pAngle)
        
        gKern =   NP.exp(-0.5*(xx**2 + yy**2))
        qSmoothSun = scipy.signal.fftconvolve(qSun,gKern) / dL**2
        qSmoothSun = qSmoothSun[dL//2:dL//2+self.sizeOfUv,dL//2:dL//2+self.sizeOfUv]
        smoothCoef = qSmoothSun[512, 512]
        qSmoothSun /= smoothCoef
        self.qSun_el_hd = warp(qSmoothSun,(shift + (rotate + back_shift)).inverse)
        
        res = warp(self.qSun_el_hd, (shift + (matrix + back_shift)).inverse)
        self.qSun_lm = warp(res,(shift + (scale + back_shift)).inverse)
        qSun_lm_fft = NP.fft.fft2(NP.roll(NP.roll(self.qSun_lm,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        qSun_lm_fft = NP.roll(NP.roll(qSun_lm_fft,self.sizeOfUv//2,0),self.sizeOfUv//2,1)# / self.sizeOfUv;
        # qSun_lm_fft = NP.flip(qSun_lm_fft, 0)
#        qSun_lm_uv = qSun_lm_fft * uvPsf
#        qSun_lm_conv = NP.fft.fft2(NP.roll(NP.roll(qSun_lm_uv,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
#        qSun_lm_conv = NP.roll(NP.roll(qSun_lm_conv,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1);
#        qSun_lm_conv = NP.flip(NP.flip(qSun_lm_conv, 1), 0)
        self.lm_hd_relation[self.frequencyChannel] = NP.sum(self.qSun_lm)/NP.sum(self.qSun_el_hd)
        self.fftDisk = qSun_lm_fft #qSun_lm_conv, 
    
    def createUvUniform(self):
        self.uvUniform = NP.zeros((self.sizeOfUv, self.sizeOfUv), dtype = complex)
        flags_ew = NP.where(self.ewAntAmpLcp[self.frequencyChannel]==1e6)[0]
        flags_s = NP.where(self.sAntAmpLcp[self.frequencyChannel]==1e6)[0]
        O = self.sizeOfUv//2 + 1
        for i in range(self.antNumberS):
            for j in range(self.antNumberEW):
                if not (NP.any(flags_ew == j) or NP.any(flags_s == i)):
                    self.uvUniform[O + i*2, O + (j-64)*2] = 1
                    self.uvUniform[O - (i+1)*2, O - (j-63)*2] = 1
        self.uvUniform /= NP.count_nonzero(self.uvUniform)

                    
    def createUvPsf(self, T, ewSlope, sSlope, stair = 0):
        self.uvPsf = self.uvUniform.copy()
        O = self.sizeOfUv//2 + 1
        ewSlope = NP.deg2rad(ewSlope)
        sSlope = NP.deg2rad(sSlope)
        ewSlopeUv = NP.linspace(-O * ewSlope/2., O * ewSlope/2., self.sizeOfUv)
        sSlopeUv = NP.linspace(-O * sSlope/2., O * sSlope/2., self.sizeOfUv)
        ewGrid,sGrid = NP.meshgrid(ewSlopeUv, sSlopeUv)
        slopeGrid = ewGrid + sGrid
        slopeGrid[self.uvUniform == 0] = 0
        self.uvPsf *= T * NP.exp(1j * slopeGrid)
        self.uvPsf[O:,:] *= NP.exp(-1j * NP.deg2rad(stair))
        self.uvPsf[:O,:] *= NP.exp(1j * NP.deg2rad(stair))
    
    def diskDiff(self, x, pol):
        self.createUvPsf(x[0], x[1], x[2])
        uvDisk = self.fftDisk * self.uvPsf
        if pol == 0:
            diff = self.uvLcp - uvDisk
        if pol == 1:
            diff = self.uvRcp - uvDisk
        return self.complex_to_real(diff[self.uvUniform!=0])
#        qSun_lm_conv = NP.fft.fft2(NP.roll(NP.roll(diff,uvSize//2+1,0),uvSize//2+1,1));
#        return NP.abs(NP.reshape(qSun_lm_conv, uvSize**2))

    def diskDiff_stair(self, x, pol):
        self.createUvPsf(x[0], x[1], x[2], x[3])
        uvDisk = self.fftDisk * self.uvPsf
        if pol == 0:
            diff = self.uvLcp - uvDisk
        if pol == 1:
            diff = self.uvRcp - uvDisk
        return self.complex_to_real(diff[self.uvUniform!=0])
    
    def diskDiff_stair_fluxNorm(self, x, pol, T):
        self.createUvPsf(T, x[0], x[1], x[2])
        uvDisk = self.fftDisk * self.uvPsf
        if pol == 0:
            diff = self.uvLcp - uvDisk
        if pol == 1:
            diff = self.uvRcp - uvDisk
        return self.complex_to_real(diff[self.uvUniform!=0])
    
    def diskDiff_stair_radius(self, x, pol):
        print(x)
        self.createDisk(x[4]*1e3)
        self.createUvPsf(x[0], x[1], x[2], x[3])
        uvDisk = self.fftDisk * self.uvPsf
        if pol == 0:
            diff = self.uvLcp - uvDisk
        if pol == 1:
            diff = self.uvRcp - uvDisk
        return self.complex_to_real(diff[self.uvUniform!=0])
    
    def findDisk(self):
        self.createDisk(980)
        self.createUvUniform()
        x_ini = [1,0,0]
        ls_res = least_squares(self.diskDiff, x_ini, args = (0,))
        self.diskLevelLcp, self.ewSlopeLcp, self.sSlopeLcp = ls_res['x']
        ls_res = least_squares(self.diskDiff, x_ini, args = (1,))
        self.diskLevelRcp, self.ewSlopeRcp, self.sSlopeRcp = ls_res['x']
        
    def wrap(self, value):
        while value<-180:
            value+=360
        while value>180:
            value-=360
        return value
    
    def findDisk_stair(self):
        self.createDisk(980)
        self.createUvUniform()
        
        # if self.flux_calibrated:
        #     Tb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6) * 1e3
        #     x_ini = [0,0,0]
        #     ls_res = least_squares(self.diskDiff_stair_fluxNorm, x_ini, args = (0,Tb/self.convolutionNormCoef), ftol=self.centering_ftol)
        #     # self.centeringResultLcp[self.frequencyChannel] = ls_res['x']
        #     _ewSlopeLcp, _sSlopeLcp, _sLcpStair = ls_res['x']
        #     ls_res = least_squares(self.diskDiff_stair_fluxNorm, x_ini, args = (1,Tb/self.convolutionNormCoef), ftol=self.centering_ftol)
        #     _ewSlopeRcp, _sSlopeRcp, _sRcpStair = ls_res['x']
        #     # self.centeringResultRcp[self.frequencyChannel] = ls_res['x']
            
        #     self.diskLevelLcp[self.frequencyChannel] = Tb/self.convolutionNormCoef
        #     self.diskLevelRcp[self.frequencyChannel] = Tb/self.convolutionNormCoef
        
        # else:
        Tb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6) * 1e3
        self.x_ini_centering_lcp[self.frequencyChannel][0] = Tb/self.convolutionNormCoef
        self.x_ini_centering_rcp[self.frequencyChannel][0] = Tb/self.convolutionNormCoef
        
        ls_res = least_squares(self.diskDiff_stair, self.x_ini_centering_lcp[self.frequencyChannel], args = (0,), ftol=self.centering_ftol)
        self.centeringResultLcp[self.frequencyChannel] = ls_res['x']
        _diskLevelLcp, _ewSlopeLcp, _sSlopeLcp, _sLcpStair = ls_res['x']
        ls_res = least_squares(self.diskDiff_stair, self.x_ini_centering_rcp[self.frequencyChannel], args = (1,), ftol=self.centering_ftol)
        _diskLevelRcp, _ewSlopeRcp, _sSlopeRcp, _sRcpStair = ls_res['x']
        self.centeringResultRcp[self.frequencyChannel] = ls_res['x']
        
        if _diskLevelLcp<0:
            _sLcpStair += 180
            _diskLevelLcp *= -1
        if _diskLevelRcp<0:
            _sRcpStair += 180
            _diskLevelRcp *= -1
        
        self.diskLevelLcp[self.frequencyChannel] = _diskLevelLcp
        self.diskLevelRcp[self.frequencyChannel] = _diskLevelRcp
        
        if not self.corr_amp_exist:
            self.ewAntAmpLcp[self.frequencyChannel][self.ewAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
            self.sAntAmpLcp[self.frequencyChannel][self.sAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
            self.ewAntAmpRcp[self.frequencyChannel][self.ewAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
            self.sAntAmpRcp[self.frequencyChannel][self.sAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
        
        self.ewSlopeLcp[self.frequencyChannel] = self.wrap(self.ewSlopeLcp[self.frequencyChannel] + _ewSlopeLcp)
        self.sSlopeLcp[self.frequencyChannel] = self.wrap(self.sSlopeLcp[self.frequencyChannel] + _sSlopeLcp)
        self.ewSlopeRcp[self.frequencyChannel] = self.wrap(self.ewSlopeRcp[self.frequencyChannel] + _ewSlopeRcp)
        self.sSlopeRcp[self.frequencyChannel] = self.wrap(self.sSlopeRcp[self.frequencyChannel] + _sSlopeRcp)
        self.sLcpStair[self.frequencyChannel] = self.wrap(self.sLcpStair[self.frequencyChannel] + _sLcpStair)
        self.sRcpStair[self.frequencyChannel] = self.wrap(self.sRcpStair[self.frequencyChannel] + _sRcpStair)

            
    def findDisk_stair_radius(self):
        self.createUvUniform()
        x_ini = [1,0,0,0,0.68]
        ls_res = least_squares(self.diskDiff_stair_radius, x_ini, args = (0,), ftol=self.centering_ftol)
        print(NP.sum(ls_res['fun']**2))
        self.diskLevelLcp, self.ewSlopeLcp, self.sSlopeLcp, self.sLcpStair, self.radius_lcp = ls_res['x']
        ls_res = least_squares(self.diskDiff_stair_radius, x_ini, args = (1,), ftol=self.centering_ftol)
        print(NP.sum(ls_res['fun']**2))
        self.diskLevelRcp, self.ewSlopeRcp, self.sSlopeRcp, self.sRcpStair, self.radius_rcp = ls_res['x']
        
        if self.diskLevelLcp<0:
            self.sLcpStair += 180
            self.diskLevelLcp *= -1
        if self.diskLevelRcp<0:
            self.sRcpStair += 180
            self.diskLevelRcp *= -1
        
        while self.ewSlopeLcp<-180:
            self.ewSlopeLcp+=360
        while self.ewSlopeLcp>180:
            self.ewSlopeLcp-=360
        while self.sSlopeLcp<-180:
            self.sSlopeLcp+=360
        while self.sSlopeLcp>180:
            self.sSlopeLcp-=360
        while self.ewSlopeRcp<-180:
            self.ewSlopeRcp+=360
        while self.ewSlopeRcp>180:
            self.ewSlopeRcp-=360
        while self.sSlopeRcp<-180:
            self.sSlopeRcp+=360
        while self.sSlopeRcp>180:
            self.sSlopeRcp-=360
        while self.sLcpStair<-180:
            self.sLcpStair+=360
        while self.sLcpStair>180:
            self.sLcpStair-=360
        while self.sRcpStair<-180:
            self.sRcpStair+=360
        while self.sRcpStair>180:
            self.sRcpStair-=360
        
    def centerDisk(self):
        self.findDisk_stair()
        self.buildEwPhase()
        self.buildSPhase()
        self.correctPhaseSlopeRL(self.frequencyChannel)
        self.buildEwPhase()
        self.buildSPhase()
        self.ewAntPhaLcp[self.frequencyChannel] += self.ewLcpPhaseCorrection[self.frequencyChannel]
        self.ewAntPhaRcp[self.frequencyChannel] += self.ewRcpPhaseCorrection[self.frequencyChannel]
        self.sAntPhaLcp[self.frequencyChannel] += self.sLcpPhaseCorrection[self.frequencyChannel]
        self.sAntPhaRcp[self.frequencyChannel] += self.sRcpPhaseCorrection[self.frequencyChannel]
        self.ewLcpPhaseCorrection[self.frequencyChannel] = NP.zeros(self.antNumberEW)
        self.ewRcpPhaseCorrection[self.frequencyChannel] = NP.zeros(self.antNumberEW)
        self.sLcpPhaseCorrection[self.frequencyChannel] = NP.zeros(self.antNumberS)
        self.sRcpPhaseCorrection[self.frequencyChannel] = NP.zeros(self.antNumberS)
        self.ewSlopeLcp[self.frequencyChannel] = 0
        self.ewSlopeRcp[self.frequencyChannel] = 0
        self.sSlopeLcp[self.frequencyChannel] = 0
        self.sSlopeRcp[self.frequencyChannel] = 0
        self.sLcpStair[self.frequencyChannel] = 0
        self.sRcpStair[self.frequencyChannel] = 0
        
    def correctRL(self):
        self.correctPhaseSlopeRL(self.frequencyChannel)
        self.buildEwPhase()
        self.buildSPhase()
        self.ewAntPhaLcp[self.frequencyChannel] += self.ewLcpPhaseCorrection[self.frequencyChannel]
        self.ewAntPhaRcp[self.frequencyChannel] += self.ewRcpPhaseCorrection[self.frequencyChannel]
        self.sAntPhaLcp[self.frequencyChannel] += self.sLcpPhaseCorrection[self.frequencyChannel]
        self.sAntPhaRcp[self.frequencyChannel] += self.sRcpPhaseCorrection[self.frequencyChannel]
        self.ewLcpPhaseCorrection[self.frequencyChannel] = NP.zeros(self.antNumberEW)
        self.ewRcpPhaseCorrection[self.frequencyChannel] = NP.zeros(self.antNumberEW)
        self.sLcpPhaseCorrection[self.frequencyChannel] = NP.zeros(self.antNumberS)
        self.sRcpPhaseCorrection[self.frequencyChannel] = NP.zeros(self.antNumberS)
        self.ewSlopeLcp[self.frequencyChannel] = 0
        self.ewSlopeRcp[self.frequencyChannel] = 0
        self.sSlopeLcp[self.frequencyChannel] = 0
        self.sSlopeRcp[self.frequencyChannel] = 0
        self.sLcpStair[self.frequencyChannel] = 0
        self.sRcpStair[self.frequencyChannel] = 0
      
    def modelDiskConv(self):
        # self.createUvPsf(self.diskLevelLcp,0,0,0)
        currentDiskTb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6)*1e3
        self.createUvPsf(currentDiskTb/self.convolutionNormCoef,0,0,0)
        self.uvDiskConv = self.fftDisk * self.uvPsf# - self.uvLcp
        qSun_lm = NP.fft.fft2(NP.roll(NP.roll(self.uvDiskConv,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        qSun_lm = NP.roll(NP.roll(qSun_lm,self.sizeOfUv//2,0),self.sizeOfUv//2,1)# / self.sizeOfUv;
        qSun_lm = NP.flip(qSun_lm, 0)
        self.modelDisk = qSun_lm