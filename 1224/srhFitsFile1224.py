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
from scipy.optimize import least_squares, basinhopping
import sunpy.coordinates
import base2uvw_1224
from skimage.transform import warp, AffineTransform
import scipy.signal
import time
from ZirinTb import ZirinTb
import json
import skimage.measure
from pathlib import Path
from threadpoolctl import threadpool_limits


class SrhFitsFile():
    def __init__(self, name, sizeOfUv, flux_norm = True):
        self.base = 2450
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
        self.averageCalib = False
        self.useNonlinearApproach = True
        self.obsObject = 'Sun'
        self.fringeStopping = False
        self.centering_ftol = 1e-3
        self.sizeOfUv = sizeOfUv
        self.baselines = 8
        self.flagsIndexes = []
        self.arcsecPerPixel = 4.91104/2
        self.ZirinQSunTb = ZirinTb()
        self.convolutionNormCoef = 14.
        self.flags_ew = NP.array(())
        self.flags_s = NP.array(())
        self.flux_calibrated = False
        self.corr_amp_exist = False
        
        self.open(name, flux_norm = flux_norm)
                         
    def open(self,fitsNames, flux_norm):
        if type(fitsNames) is not list: fitsNames = [fitsNames]
        if fitsNames[0]:
            try:
                self.hduList = fits.open(fitsNames[0])
                self.isOpen = True
                self.dateObs = self.hduList[0].header['DATE-OBS'] + 'T' + self.hduList[0].header['TIME-OBS']
                self.antennaNumbers = self.hduList[3].data['ant_index']
                self.antennaNumbers = NP.reshape(self.antennaNumbers,self.antennaNumbers.size)
                self.antennaNames = self.hduList[2].data['ant_name']
                self.antennaNames = NP.reshape(self.antennaNames,self.antennaNames.size)
                self.antennaA = self.hduList[4].data['ant_A']
                self.antennaA = NP.reshape(self.antennaA,self.antennaA.size)
                self.antennaB = self.hduList[4].data['ant_B']
                self.antennaB = NP.reshape(self.antennaB,self.antennaB.size)
                self.antX = self.hduList[3].data['ant_X']
                self.antY = self.hduList[3].data['ant_Y']
                self.uvLcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex)
                self.uvRcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex)
                self.freqList = self.hduList[1].data['frequency'];
                self.freqListLength = self.freqList.size;
                self.dataLength = self.hduList[1].data['time'].size // self.freqListLength;
                self.freqTime = self.hduList[1].data['time']
                self.validScansLcp = NP.ones((self.freqListLength,self.dataLength), dtype = bool)
                self.validScansRcp = NP.ones((self.freqListLength,self.dataLength), dtype = bool)
                self.visListLength = self.hduList[1].data['vis_lcp'].size // self.freqListLength // self.dataLength;
                self.visLcp = NP.reshape(self.hduList[1].data['vis_lcp'],(self.freqListLength,self.dataLength,self.visListLength));
                self.visRcp = NP.reshape(self.hduList[1].data['vis_rcp'],(self.freqListLength,self.dataLength,self.visListLength));
                try:
                    self.freqTimeLcp = self.hduList[1].data['time_lcp']
                    self.freqTimeRcp = self.hduList[1].data['time_rcp']
                except:
                    pass
                try:
                    self.correctSubpacketsNumber = int(self.hduList[0].header['SUBPACKS'])
                    self.subpacketLcp = self.hduList[1].data['spacket_lcp']
                    self.subpacketRcp = self.hduList[1].data['spacket_rcp']
                    self.validScansLcp = self.subpacketLcp==self.correctSubpacketsNumber
                    self.validScansRcp = self.subpacketRcp==self.correctSubpacketsNumber
                    self.visLcp[~self.validScansLcp] = 0
                    self.visRcp[~self.validScansRcp] = 0
                    self.ampLcp[~self.validScansLcp] = 1
                    self.ampRcp[~self.validScansRcp] = 1
                except:
                    pass
                # self.visLcp /= float(self.hduList[0].header['VIS_MAX'])
                # self.visRcp /= float(self.hduList[0].header['VIS_MAX'])
                self.ampLcp = NP.reshape(self.hduList[1].data['amp_lcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                self.ampRcp = NP.reshape(self.hduList[1].data['amp_rcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                ampScale = float(self.hduList[0].header['VIS_MAX']) / 128.
                self.ampLcp = self.ampLcp.astype(float) / ampScale
                self.ampRcp = self.ampRcp.astype(float) / ampScale
                
                try:
                    file = Path(__file__).resolve()
                    parent = str(file.parent)
                    zerosFits = fits.open(parent + '/srh_1224_cp_zeros.fits')
                    skyLcp = zerosFits[2].data['skyLcp_c']
                    skyRcp = zerosFits[2].data['skyRcp_c']
                    self.ampLcp_c = NP.reshape(self.hduList[1].data['amp_c_lcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                    self.ampRcp_c = NP.reshape(self.hduList[1].data['amp_c_rcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                    self.corr_amp_exist = True
                    self.ampLcp_c[self.ampLcp_c <= 0.01] = 1e6
                    self.ampRcp_c[self.ampRcp_c <= 0.01] = 1e6
 
                    # for tt in range(self.dataLength):
                    #     self.ampLcp_c[:,tt,:] = self.ampLcp_c[:,tt,:] - skyLcp
                    #     self.ampRcp_c[:,tt,:] = self.ampRcp_c[:,tt,:] - skyRcp
                    # self.ampLcp_c[self.ampLcp_c <= 1e5] = 1e8
                    # self.ampRcp_c[self.ampRcp_c <= 1e5] = 1e8
                    self.visLcp = self.visLcp / ((NP.sqrt(self.ampLcp_c[:,:,self.antennaA] * self.ampLcp_c[:,:,self.antennaB])))
                    self.visRcp = self.visRcp / ((NP.sqrt(self.ampRcp_c[:,:,self.antennaA] * self.ampRcp_c[:,:,self.antennaB])))
                except Exception as error:
                    print('Visibilities are not corrected:   ', error)
                
                
                # try:
                #     self.ampLcp_c = NP.reshape(self.hduList[1].data['amp_c_lcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                #     self.ampRcp_c = NP.reshape(self.hduList[1].data['amp_c_rcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                #     # self.ampLcp_c = self.ampLcp_c.astype(float) / ampScale
                #     # self.ampRcp_c = self.ampRcp_c.astype(float) / ampScale
                #     self.ampLcp_c[self.ampLcp_c <= 0.01] = 1e6
                #     self.ampRcp_c[self.ampRcp_c <= 0.01] = 1e6
                #     self.corr_amp_exist = True
                #     self.visLcp = self.visLcp/NP.sqrt(NP.abs(self.ampLcp_c[:,:,self.antennaA]) * NP.abs(self.ampLcp_c[:,:,self.antennaB]))
                #     self.visRcp = self.visRcp/NP.sqrt(NP.abs(self.ampRcp_c[:,:,self.antennaA]) * NP.abs(self.ampRcp_c[:,:,self.antennaB]))
                # except:
                #     pass
                
                self.antNumberEW = NP.count_nonzero(self.antY) + 1
                self.antNumberS = NP.count_nonzero(self.antX)
                self.center_ant = NP.where(self.antennaNames=='C1001')[0][0]
                
                self.antZeroRow = []
                for ant in range(self.antNumberEW):
                    if ant<self.center_ant:
                        self.antZeroRow.append(NP.where((self.antennaA==ant) & (self.antennaB==self.center_ant))[0][0])
                    if ant>self.center_ant:
                        self.antZeroRow.append(NP.where((self.antennaA==self.center_ant) & (self.antennaB==ant))[0][0])
                # self.antZeroRow = self.hduList[3].data['ant_zero_row'][:97]
                self.RAO = BadaryRAO(self.dateObs.split('T')[0], self.base*1e-3, observedObject = self.obsObject)
                # try:
                #     client = Client('http://ephemeris.rao.istp.ac.ru/?wsdl')
                #     result = client.service.Ephemeride('SSRT','sun',self.dateObs)
                #     self.pAngle = NP.deg2rad(float(result[0]['PAngle']))
                # except:
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
                
                self.sLcpStair = NP.zeros(self.freqListLength)
                self.sRcpStair = NP.zeros(self.freqListLength)
                self.ewSlopeLcp = NP.zeros(self.freqListLength)
                self.sSlopeLcp = NP.zeros(self.freqListLength)
                self.ewSlopeRcp = NP.zeros(self.freqListLength)
                self.sSlopeRcp = NP.zeros(self.freqListLength)
                self.diskLevelLcp = NP.ones(self.freqListLength)
                self.diskLevelRcp = NP.ones(self.freqListLength)
                self.lm_hd_relation = NP.ones(self.freqListLength)
                
                self.beam_sr = NP.ones(self.freqListLength)
                
                self.fluxLcp = NP.zeros(self.freqListLength)
                self.fluxRcp = NP.zeros(self.freqListLength)
                
                x_size = (self.baselines-1)*2 + self.antNumberEW + self.antNumberS
                self.x_ini_lcp = NP.full((self.freqListLength, x_size*2+1), NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))))
                self.x_ini_rcp = NP.full((self.freqListLength, x_size*2+1), NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))))
                self.calibrationResultLcp = NP.zeros_like(self.x_ini_lcp)
                self.calibrationResultRcp = NP.zeros_like(self.x_ini_rcp)
                
                self.lcpShift = NP.ones(self.freqListLength) # 0-frequency component in the spectrum
                self.rcpShift = NP.ones(self.freqListLength)
                
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
            validScansLcp = NP.ones((self.freqListLength, dataLength), dtype = bool)
            validScansRcp = NP.ones((self.freqListLength, dataLength), dtype = bool)
            try:
                subpacketLcp = self.hduList[1].data['spacket_lcp']
                subpacketRcp = self.hduList[1].data['spacket_rcp']
                self.subpacketLcp = NP.concatenate((self.subpacketLcp, subpacketLcp), axis = 1)
                self.subpacketRcp = NP.concatenate((self.subpacketRcp, subpacketRcp), axis = 1)
                validScansLcp = subpacketLcp==self.correctSubpacketsNumber
                validScansRcp = subpacketRcp==self.correctSubpacketsNumber
                visLcp[~validScansLcp] = 0
                visRcp[~validScansRcp] = 0
                ampLcp[~validScansLcp] = 1
                ampRcp[~validScansRcp] = 1
            except:
                pass
            
            try:
                file = Path(__file__).resolve()
                parent = str(file.parent)
                zerosFits = fits.open(parent + '/srh_1224_cp_zeros.fits')
                skyLcp = zerosFits[2].data['skyLcp_c']
                skyRcp = zerosFits[2].data['skyRcp_c']
                ampLcp_c = NP.reshape(hduList[1].data['amp_c_lcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
                ampRcp_c = NP.reshape(hduList[1].data['amp_c_rcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
                ampLcp_c[ampLcp_c <= 0.01] = 1e6
                ampRcp_c[ampRcp_c <= 0.01] = 1e6
                # for tt in range(dataLength):
                #     ampLcp_c[:,tt,:] = ampLcp_c[:,tt,:] - skyLcp
                #     ampRcp_c[:,tt,:] = ampRcp_c[:,tt,:] - skyRcp
                # ampLcp_c[ampLcp_c <= 1e5] = 1e8
                # ampRcp_c[ampRcp_c <= 1e5] = 1e8
                visLcp = visLcp / ((NP.sqrt(ampLcp_c[:,:,self.antennaA] * ampLcp_c[:,:,self.antennaB])))
                visRcp = visRcp / ((NP.sqrt(ampRcp_c[:,:,self.antennaA] * ampRcp_c[:,:,self.antennaB])))
                self.ampLcp_c = NP.concatenate((self.ampLcp_c, ampLcp_c), axis = 1)
                self.ampRcp_c = NP.concatenate((self.ampRcp_c, ampRcp_c), axis = 1)
            except Exception as error:
                print('Visibilities are not corrected:   ', error)
            
            # try:
            #     ampLcp_c = NP.reshape(hduList[1].data['amp_c_lcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
            #     ampRcp_c = NP.reshape(hduList[1].data['amp_c_rcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
            #     # ampLcp_c = ampLcp_c.astype(float) / ampScale
            #     # ampRcp_c = ampRcp_c.astype(float) / ampScale
            #     ampLcp_c[ampLcp_c <= 0.01] = 1e6
            #     ampRcp_c[ampRcp_c <= 0.01] = 1e6
            #     self.ampLcp_c = NP.concatenate((self.ampLcp_c, ampLcp_c), axis = 1)
            #     self.ampRcp_c = NP.concatenate((self.ampRcp_c, ampRcp_c), axis = 1)
            #     visLcp = visLcp/NP.sqrt(NP.abs(ampLcp_c[:,:,self.antennaA]) * NP.abs(ampLcp_c[:,:,self.antennaB]))
            #     visRcp = visRcp/NP.sqrt(NP.abs(ampRcp_c[:,:,self.antennaA]) * NP.abs(ampRcp_c[:,:,self.antennaB]))
            # except:
            #     pass
            try:
                freqTimeLcp = hduList[1].data['time_lcp']
                freqTimeRcp = hduList[1].data['time_rcp']
                self.freqTimeLcp = NP.concatenate((self.freqTimeLcp, freqTimeLcp), axis = 1)
                self.freqTimeRcp = NP.concatenate((self.freqTimeRcp, freqTimeRcp), axis = 1)
            except:
                pass
            

            self.freqTime = NP.concatenate((self.freqTime, freqTime), axis = 1)
            self.visLcp = NP.concatenate((self.visLcp, visLcp), axis = 1)
            self.visRcp = NP.concatenate((self.visRcp, visRcp), axis = 1)
            self.ampLcp = NP.concatenate((self.ampLcp, ampLcp), axis = 1)
            self.ampRcp = NP.concatenate((self.ampRcp, ampRcp), axis = 1)
            self.validScansLcp = NP.concatenate((self.validScansLcp, validScansLcp), axis = 1)
            self.validScansRcp = NP.concatenate((self.validScansRcp, validScansRcp), axis = 1)
            self.dataLength += dataLength
            hduList.close()

        except FileNotFoundError:
            print('File %s  not found'%name);
            
    def normalizeFlux(self):
        file = Path(__file__).resolve()
        parent = str(file.parent)
        zerosFits = fits.open(parent + '/srh_1224_cp_zeros.fits')
        fluxZerosLcp = zerosFits[2].data['skyLcp']
        fluxZerosRcp = zerosFits[2].data['skyRcp']

        fluxNormFits = fits.open(parent + '/srh_1224_cp_fluxNorm.fits')
        fluxNormLcp = fluxNormFits[2].data['fluxNormLcp']
        fluxNormRcp = fluxNormFits[2].data['fluxNormRcp']
        
        ampFluxRcp = NP.mean(self.ampRcp, axis = 2)
        ampFluxLcp = NP.mean(self.ampLcp, axis = 2)
        
        for ff in range(self.freqListLength):
            ampFluxRcp[ff,:] -= fluxZerosRcp[ff]
            ampFluxRcp[ff,:] *= fluxNormRcp[ff] 
            ampFluxLcp[ff,:] -= fluxZerosLcp[ff]
            ampFluxLcp[ff,:] *= fluxNormLcp[ff] 
            
            self.fluxLcp[ff] = NP.mean(ampFluxLcp[ff])
            self.fluxRcp[ff] = NP.mean(ampFluxRcp[ff])
            
            self.visLcp[ff,:,:] *= NP.mean(self.fluxLcp[ff])
            self.visRcp[ff,:,:] *= NP.mean(self.fluxRcp[ff])
            
            self.visLcp[ff,:,:] *= 2 # flux is divided by 2 for R and L
            self.visRcp[ff,:,:] *= 2
            
        self.flux_calibrated = True
            
    def beam(self):
        self.setFrequencyChannel(0)
        self.vis2uv(0, average= 20, PSF = True)
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
        currentGainsDict['rcpShift'] = self.rcpShift.tolist()
        currentGainsDict['lcpShift'] = self.lcpShift.tolist()
        currentGainsDict['lm_hd_relation'] = self.lm_hd_relation.tolist()
        with open(filename, 'w') as saveGainFile:
            json.dump(currentGainsDict, saveGainFile)
            
    def loadGains(self, filename):
        with open(filename,'r') as readGainFile:
            currentGains = json.load(readGainFile)
        self.ewAntPhaLcp = NP.array(currentGains['ewPhaseLcp'])
        self.ewAntPhaRcp = NP.array(currentGains['ewPhaseRcp'])
        self.ewAntAmpLcp = NP.array(currentGains['ewAmpLcp'])
        self.ewAntAmpRcp = NP.array(currentGains['ewAmpRcp'])
        
        try:
            self.sAntPhaLcp = NP.array(currentGains['sPhaseLcp'])
            self.sAntPhaRcp = NP.array(currentGains['sPhaseRcp'])
            self.sAntAmpLcp = NP.array(currentGains['sAmpLcp'])
            self.sAntAmpRcp = NP.array(currentGains['sAmpRcp'])
        except:
            pass
        try:
            self.sAntPhaLcp = NP.array(currentGains['nsPhaseLcp'])
            self.sAntPhaRcp = NP.array(currentGains['nsPhaseRcp'])
            self.sAntAmpLcp = NP.array(currentGains['nsAmpLcp'])
            self.sAntAmpRcp = NP.array(currentGains['nsAmpRcp'])
        except:
            pass
        
        self.rcpShift = NP.array(currentGains['rcpShift'])
        self.lcpShift = NP.array(currentGains['lcpShift'])
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
    
    def base2uvw(self, ant0, ant1):
        phi = 0.903338787600965
        base = NP.array([self.antY[ant1]-self.antY[ant0], self.antX[ant1]-self.antX[ant0], 0.])
        base *= 1e-3;
        
        phi_operator = NP.array([
            [-NP.sin(phi), 0., NP.cos(phi)],
            [0., 1., 0.],
            [NP.cos(phi), 0., NP.sin(phi)]
            ])
    
        uvw_operator = NP.array([
            [ NP.sin(self.hAngle),		 NP.cos(self.hAngle),		0.	  ],
            [-NP.sin(self.RAO.declination)*NP.cos(self.hAngle),  NP.sin(self.RAO.declination)*NP.sin(self.hAngle), NP.cos(self.RAO.declination)], 
            [ NP.cos(self.RAO.declination)*NP.cos(self.hAngle), -NP.cos(self.RAO.declination)*NP.sin(self.hAngle), NP.sin(self.RAO.declination)]  
            ])
    
        return NP.dot(uvw_operator, NP.dot(phi_operator, base))
        
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
        self.x_ini_lcp = NP.full((self.freqListLength, x_size*2+1), NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))))
        self.x_ini_rcp = NP.full((self.freqListLength, x_size*2+1), NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))))
        self.calibrationResultLcp = NP.zeros_like(self.x_ini_lcp)
        self.calibrationResultRcp = NP.zeros_like(self.x_ini_rcp)

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
    
    def calculatePhaseCalibration(self, lcp = True, rcp = True):
       for freq in range(self.freqListLength):
           self.solarPhase(freq)
           self.updateAntennaPhase(freq, lcp = lcp, rcp = rcp)
           
    def calculateAmpCalibration(self, baselinesNumber = 5):
       for freq in range(self.freqListLength):
           self.calculateAmplitude_linear(freq, baselinesNumber)

    def updateAntennaPhase(self, freqChannel, lcp = True, rcp = True):
        if self.useNonlinearApproach:
            if lcp:
                self.calculatePhaseLcp_nonlinear_new(freqChannel)
            if rcp:
                self.calculatePhaseRcp_nonlinear_new(freqChannel)
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
            
    def solarPhase(self, freq):
        # antAInd = NP.where(self.antennaNames=='S1001')[0][0]
        # antBInd = NP.where(self.antennaNames=='S1002')[0][0]
        u,v,w = base2uvw_1224.base2uvw(self.hAngle, self.RAO.declination, 150, 151)
        baseWave = NP.sqrt(u**2+v**2)*self.freqList[freq]*1e3/constants.c.to_value()
        if baseWave > 120:
            self.sSolarPhase[freq] = NP.pi
        else:
            self.sSolarPhase[freq] = 0
        # antAInd = NP.where(self.antennaNames=='E1001')[0][0]
        # antBInd = NP.where(self.antennaNames=='E1002')[0][0]
        u,v,w = base2uvw_1224.base2uvw(self.hAngle, self.RAO.declination, 1, 2)
        baseWave = NP.sqrt(u**2+v**2)*self.freqList[freq]*1e3/constants.c.to_value()
        if baseWave > 120:
            self.ewSolarPhase[freq] = NP.pi
        else:
            self.ewSolarPhase[freq] = 0
            
    def calculatePhase_linear(self, freqChannel, baselinesNumber = 1):
        antNumberN = 31
        antNumberEW = 97
        redIndexesN = []
        for baseline in range(1, baselinesNumber+1):
            redIndexesN.append(NP.where((self.antennaA==98-1+baseline) & (self.antennaB==33))[0][0])
            for i in range(antNumberN - baseline):
                redIndexesN.append(NP.where((self.antennaB==98+i) & (self.antennaA==98+i+baseline))[0][0])
    
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==1+i) & (self.antennaB==1+i+baseline))[0][0])
                
        phaMatrix = self.phaMatrixGenPairsEWN(baselinesNumber, antNumberEW, antNumberN)
        redundantVisLcp = self.visLcp[freqChannel, self.calibIndex, NP.append(redIndexesEW, redIndexesN)]
        sunVisPhases = NP.zeros(2)
#        if self.freqList[freqChannel] > 4e6:
#            sunVisPhases = NP.array((NP.pi, NP.pi))
        phasesLcp = NP.concatenate((NP.angle(redundantVisLcp), sunVisPhases))
        antPhaLcp, c, d, e = NP.linalg.lstsq(phaMatrix, phasesLcp, rcond=None)
        self.ewAntPhaLcp[freqChannel] = antPhaLcp[baselinesNumber*2:baselinesNumber*2+antNumberEW]
        self.nAntPhaLcp[freqChannel] = antPhaLcp[baselinesNumber*2+antNumberEW:]
        
        redundantVisRcp = self.visRcp[freqChannel, self.calibIndex, NP.append(redIndexesEW, redIndexesN)]
        phasesRcp = NP.concatenate((NP.angle(redundantVisRcp), NP.array((0,0))))
        antPhaRcp, c, d, e = NP.linalg.lstsq(phaMatrix, phasesRcp, rcond=None)
        self.ewAntPhaRcp[freqChannel] = antPhaRcp[baselinesNumber*2:baselinesNumber*2+antNumberEW]
        self.nAntPhaRcp[freqChannel] = antPhaRcp[baselinesNumber*2+antNumberEW:]
        
    def calculatePhaseLcp_nonlinear(self, freqChannel):
        antY_diff = (self.antY[self.antennaB] - self.antY[self.antennaA])/self.base
        antX_diff = (self.antX[self.antennaB] - self.antX[self.antennaA])/self.base
        self.redIndexesS = NP.array((), dtype=int)
        self.redIndexesEW = NP.array((), dtype=int)
        self.redIndexesS_len = []
        self.redIndexesEW_len = []
        for baseline in range(1, self.baselines+1):
            ind = NP.intersect1d(NP.where(NP.abs(antX_diff)==baseline)[0], NP.where(antY_diff == 0)[0])
            self.redIndexesS = NP.append(self.redIndexesS, ind)
            self.redIndexesS_len.append(len(ind))
            ind = NP.intersect1d(NP.where(NP.abs(antY_diff)==baseline)[0], NP.where(antX_diff == 0)[0])
            self.redIndexesEW = NP.append(self.redIndexesEW, ind)
            self.redIndexesEW_len.append(len(ind))

        if self.averageCalib:
            redundantVisS = NP.mean(self.visLcp[freqChannel, :20, self.redIndexesS.astype(int)], axis = 1)
            redundantVisEW = NP.mean(self.visLcp[freqChannel, :20, self.redIndexesEW.astype(int)], axis = 1)
            redundantVisAll = NP.append(redundantVisEW, redundantVisS)
        else:
            redundantVisS = self.visLcp[freqChannel, self.calibIndex, self.redIndexesS.astype(int)]
            redundantVisEW = self.visLcp[freqChannel, self.calibIndex, self.redIndexesEW.astype(int)]
            redundantVisAll = NP.append(redundantVisEW, redundantVisS)
            
        for i in range(len(self.redIndexesS)):    
            if NP.any((self.flags_s+self.antNumberEW) == self.antennaA[self.redIndexesS[i]]) or NP.any((self.flags_s+self.antNumberEW) == self.antennaB[self.redIndexesS[i]]):
                redundantVisS[i]=0.
        for i in range(len(self.redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[self.redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[self.redIndexesEW[i]]):
                redundantVisEW[i]=0.

        ls_res = least_squares(self.allGainsFunc_constrained, self.x_ini_lcp[freqChannel], args = (redundantVisAll, self.antNumberEW, self.antNumberS, self.baselines, freqChannel), max_nfev = 400)
        self.calibrationResultLcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:])[(self.baselines-1)*2:]
        self.ew_gains_lcp = gains[:self.antNumberEW]
        self.ewAntPhaLcp[freqChannel] = NP.angle(self.ew_gains_lcp)
        self.s_gains_lcp = gains[self.antNumberEW:]
        self.sAntPhaLcp[freqChannel] = NP.angle(self.s_gains_lcp)
        
        norm = NP.mean(NP.abs(gains[NP.abs(gains)>NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpLcp[freqChannel] = NP.abs(self.ew_gains_lcp)/norm
        self.ewAntAmpLcp[freqChannel][self.ewAntAmpLcp[freqChannel]<NP.median(self.ewAntAmpLcp[freqChannel])*0.6] = 1e6
        self.sAntAmpLcp[freqChannel] = NP.abs(self.s_gains_lcp)/norm
        self.sAntAmpLcp[freqChannel][self.sAntAmpLcp[freqChannel]<NP.median(self.sAntAmpLcp[freqChannel])*0.6] = 1e6
        
    def calculatePhaseRcp_nonlinear(self, freqChannel):
        antY_diff = (self.antY[self.antennaB] - self.antY[self.antennaA])/self.base
        antX_diff = (self.antX[self.antennaB] - self.antX[self.antennaA])/self.base
        self.redIndexesS = NP.array((), dtype=int)
        self.redIndexesEW = NP.array((), dtype=int)
        self.redIndexesS_len = []
        self.redIndexesEW_len = []
        for baseline in range(1, self.baselines+1):
            ind = NP.intersect1d(NP.where(NP.abs(antX_diff)==baseline)[0], NP.where(antY_diff == 0)[0])
            self.redIndexesS = NP.append(self.redIndexesS, ind)
            self.redIndexesS_len.append(len(ind))
            ind = NP.intersect1d(NP.where(NP.abs(antY_diff)==baseline)[0], NP.where(antX_diff == 0)[0])
            self.redIndexesEW = NP.append(self.redIndexesEW, ind)
            self.redIndexesEW_len.append(len(ind))

        if self.averageCalib:
            redundantVisS = NP.mean(self.visRcp[freqChannel, :20, self.redIndexesS.astype(int)], axis = 1)
            redundantVisEW = NP.mean(self.visRcp[freqChannel, :20, self.redIndexesEW.astype(int)], axis = 1)
            redundantVisAll = NP.append(redundantVisEW, redundantVisS)
        else:
            redundantVisS = self.visRcp[freqChannel, self.calibIndex, self.redIndexesS.astype(int)]
            redundantVisEW = self.visRcp[freqChannel, self.calibIndex, self.redIndexesEW.astype(int)]
            redundantVisAll = NP.append(redundantVisEW, redundantVisS)
            
        for i in range(len(self.redIndexesS)):    
            if NP.any((self.flags_s+self.antNumberEW) == self.antennaA[self.redIndexesS[i]]) or NP.any((self.flags_s+self.antNumberEW) == self.antennaB[self.redIndexesS[i]]):
                redundantVisS[i]=0.
        for i in range(len(self.redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[self.redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[self.redIndexesEW[i]]):
                redundantVisEW[i]=0.

        ls_res = least_squares(self.allGainsFunc_constrained, self.x_ini_rcp[freqChannel], args = (redundantVisAll, self.antNumberEW, self.antNumberS, self.baselines, freqChannel), max_nfev = 400)
        self.calibrationResultRcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:])[(self.baselines-1)*2:]
        self.ew_gains_rcp = gains[:self.antNumberEW]
        self.ewAntPhaRcp[freqChannel] = NP.angle(self.ew_gains_rcp)
        self.s_gains_rcp = gains[self.antNumberEW:]
        self.sAntPhaRcp[freqChannel] = NP.angle(self.s_gains_rcp)
        
        norm = NP.mean(NP.abs(gains[NP.abs(gains)>NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpRcp[freqChannel] = NP.abs(self.ew_gains_rcp)/norm
        self.ewAntAmpRcp[freqChannel][self.ewAntAmpRcp[freqChannel]<NP.median(self.ewAntAmpRcp[freqChannel])*0.6] = 1e6
        self.sAntAmpRcp[freqChannel] = NP.abs(self.s_gains_rcp)/norm
        self.sAntAmpRcp[freqChannel][self.sAntAmpRcp[freqChannel]<NP.median(self.sAntAmpRcp[freqChannel])*0.6] = 1e6
        
    def calculatePhaseLcp_nonlinear_new(self, freqChannel):
        antY_diff = (self.antY[self.antennaB] - self.antY[self.antennaA])/self.base
        antX_diff = (self.antX[self.antennaB] - self.antX[self.antennaA])/self.base
        self.redIndexesS = NP.array((), dtype=int)
        self.redIndexesEW = NP.array((), dtype=int)
        self.redIndexesS_len = []
        self.redIndexesEW_len = []
        for baseline in range(1, self.baselines+1):
            ind = NP.intersect1d(NP.where(NP.abs(antX_diff)==baseline)[0], NP.where(antY_diff == 0)[0])
            self.redIndexesS = NP.append(self.redIndexesS, ind)
            self.redIndexesS_len.append(len(ind))
            ind = NP.intersect1d(NP.where(NP.abs(antY_diff)==baseline)[0], NP.where(antX_diff == 0)[0])
            self.redIndexesEW = NP.append(self.redIndexesEW, ind)
            self.redIndexesEW_len.append(len(ind))
            
        validScansBoth = NP.intersect1d(NP.where(self.validScansLcp[freqChannel]), NP.where(self.validScansRcp[freqChannel]))
        ind = NP.argmin(NP.abs(validScansBoth - self.calibIndex))
        calibIndex = validScansBoth[ind]

        if self.averageCalib:
            redundantVisS = NP.sum(self.visLcp[freqChannel, :20, self.redIndexesS.astype(int)], axis = 1)/NP.sum(self.validScansLcp[freqChannel])
            redundantVisEW = NP.sum(self.visLcp[freqChannel, :20, self.redIndexesEW.astype(int)], axis = 1)/NP.sum(self.validScansLcp[freqChannel])
            redundantVisAll = NP.append(redundantVisEW, redundantVisS)
        else:
            redundantVisS = self.visLcp[freqChannel, calibIndex, self.redIndexesS.astype(int)]
            redundantVisEW = self.visLcp[freqChannel, calibIndex, self.redIndexesEW.astype(int)]
            redundantVisAll = NP.append(redundantVisEW, redundantVisS)
            
        for i in range(len(self.redIndexesS)):    
            if NP.any((self.flags_s+self.antNumberEW) == self.antennaA[self.redIndexesS[i]]) or NP.any((self.flags_s+self.antNumberEW) == self.antennaB[self.redIndexesS[i]]):
                redundantVisS[i]=0.
        for i in range(len(self.redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[self.redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[self.redIndexesEW[i]]):
                redundantVisEW[i]=0.

        baselinesNumber = self.baselines
        antNumberS = self.antNumberS
        antNumberEW = self.antNumberEW

        ewAmpSign = 1 if self.ewSolarPhase[freqChannel]==0 else -1
        sAmpSign = 1 if self.sSolarPhase[freqChannel]==0 else -1
        
        res = NP.zeros_like(redundantVisAll, dtype = complex)
        ewSolarAmp = 1 * ewAmpSign
        sAntNumber_c = antNumberS + 1
        sGainsNumber = antNumberS
        ewGainsNumber = antNumberEW
        sSolVisNumber = baselinesNumber - 1
        ewSolVisNumber = baselinesNumber - 1
        sNum = len(self.redIndexesS)
        ewNum = len(self.redIndexesEW)
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
        
        args = (redundantVisAll, freqChannel,
                res, ewSolarAmp, sAntNumber_c, sGainsNumber, ewGainsNumber, sSolVisNumber, 
                ewSolVisNumber, solVisArrayS, antAGainsS, antBGainsS, solVisArrayEW, 
                antAGainsEW, antBGainsEW, ewSolVis, sSolVis, solVis, antAGains, antBGains, sAmpSign)
        
        with threadpool_limits(limits=1, user_api='blas'):
            ls_res = least_squares(self.allGainsFunc_constrained_new, self.x_ini_lcp[freqChannel], args = args, max_nfev = 400)
        self.calibrationResultLcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:])[(self.baselines-1)*2:]
        self.ew_gains_lcp = gains[:self.antNumberEW]
        self.ewAntPhaLcp[freqChannel] = NP.angle(self.ew_gains_lcp)
        self.s_gains_lcp = gains[self.antNumberEW:]
        self.sAntPhaLcp[freqChannel] = NP.angle(self.s_gains_lcp)
        
        norm = NP.mean(NP.abs(gains[NP.abs(gains)>NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpLcp[freqChannel] = NP.abs(self.ew_gains_lcp)/norm
        self.ewAntAmpLcp[freqChannel][self.ewAntAmpLcp[freqChannel]<NP.median(self.ewAntAmpLcp[freqChannel])*0.6] = 1e6
        self.sAntAmpLcp[freqChannel] = NP.abs(self.s_gains_lcp)/norm
        self.sAntAmpLcp[freqChannel][self.sAntAmpLcp[freqChannel]<NP.median(self.sAntAmpLcp[freqChannel])*0.6] = 1e6
        
    def calculatePhaseRcp_nonlinear_new(self, freqChannel):
        antY_diff = (self.antY[self.antennaB] - self.antY[self.antennaA])/self.base
        antX_diff = (self.antX[self.antennaB] - self.antX[self.antennaA])/self.base
        self.redIndexesS = NP.array((), dtype=int)
        self.redIndexesEW = NP.array((), dtype=int)
        self.redIndexesS_len = []
        self.redIndexesEW_len = []
        for baseline in range(1, self.baselines+1):
            ind = NP.intersect1d(NP.where(NP.abs(antX_diff)==baseline)[0], NP.where(antY_diff == 0)[0])
            self.redIndexesS = NP.append(self.redIndexesS, ind)
            self.redIndexesS_len.append(len(ind))
            ind = NP.intersect1d(NP.where(NP.abs(antY_diff)==baseline)[0], NP.where(antX_diff == 0)[0])
            self.redIndexesEW = NP.append(self.redIndexesEW, ind)
            self.redIndexesEW_len.append(len(ind))

        validScansBoth = NP.intersect1d(NP.where(self.validScansLcp[freqChannel]), NP.where(self.validScansRcp[freqChannel]))
        ind = NP.argmin(NP.abs(validScansBoth - self.calibIndex))
        calibIndex = validScansBoth[ind]

        if self.averageCalib:
            redundantVisS = NP.sum(self.visRcp[freqChannel, :20, self.redIndexesS.astype(int)], axis = 1)/NP.sum(self.validScansRcp[freqChannel])
            redundantVisEW = NP.sum(self.visRcp[freqChannel, :20, self.redIndexesEW.astype(int)], axis = 1)/NP.sum(self.validScansRcp[freqChannel])
            redundantVisAll = NP.append(redundantVisEW, redundantVisS)
        else:
            redundantVisS = self.visRcp[freqChannel, calibIndex, self.redIndexesS.astype(int)]
            redundantVisEW = self.visRcp[freqChannel, calibIndex, self.redIndexesEW.astype(int)]
            redundantVisAll = NP.append(redundantVisEW, redundantVisS)
            
        for i in range(len(self.redIndexesS)):    
            if NP.any((self.flags_s+self.antNumberEW) == self.antennaA[self.redIndexesS[i]]) or NP.any((self.flags_s+self.antNumberEW) == self.antennaB[self.redIndexesS[i]]):
                redundantVisS[i]=0.
        for i in range(len(self.redIndexesEW)):    
            if NP.any(self.flags_ew == self.antennaA[self.redIndexesEW[i]]) or NP.any(self.flags_ew == self.antennaB[self.redIndexesEW[i]]):
                redundantVisEW[i]=0.

        baselinesNumber = self.baselines
        antNumberS = self.antNumberS
        antNumberEW = self.antNumberEW

        ewAmpSign = 1 if self.ewSolarPhase[freqChannel]==0 else -1
        sAmpSign = 1 if self.sSolarPhase[freqChannel]==0 else -1
        
        res = NP.zeros_like(redundantVisAll, dtype = complex)
        ewSolarAmp = 1 * ewAmpSign
        sAntNumber_c = antNumberS + 1
        sGainsNumber = antNumberS
        ewGainsNumber = antNumberEW
        sSolVisNumber = baselinesNumber - 1
        ewSolVisNumber = baselinesNumber - 1
        sNum = len(self.redIndexesS)
        ewNum = len(self.redIndexesEW)
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
        
        args = (redundantVisAll, freqChannel,
                res, ewSolarAmp, sAntNumber_c, sGainsNumber, ewGainsNumber, sSolVisNumber, 
                ewSolVisNumber, solVisArrayS, antAGainsS, antBGainsS, solVisArrayEW, 
                antAGainsEW, antBGainsEW, ewSolVis, sSolVis, solVis, antAGains, antBGains, sAmpSign)
        
        with threadpool_limits(limits=1, user_api='blas'):
            ls_res = least_squares(self.allGainsFunc_constrained_new, self.x_ini_rcp[freqChannel], args = args, max_nfev = 400)
        self.calibrationResultRcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:])[(self.baselines-1)*2:]
        self.ew_gains_rcp = gains[:self.antNumberEW]
        self.ewAntPhaRcp[freqChannel] = NP.angle(self.ew_gains_rcp)
        self.s_gains_rcp = gains[self.antNumberEW:]
        self.sAntPhaRcp[freqChannel] = NP.angle(self.s_gains_rcp)
        
        norm = NP.mean(NP.abs(gains[NP.abs(gains)>NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpRcp[freqChannel] = NP.abs(self.ew_gains_rcp)/norm
        self.ewAntAmpRcp[freqChannel][self.ewAntAmpRcp[freqChannel]<NP.median(self.ewAntAmpRcp[freqChannel])*0.6] = 1e6
        self.sAntAmpRcp[freqChannel] = NP.abs(self.s_gains_rcp)/norm
        self.sAntAmpRcp[freqChannel][self.sAntAmpRcp[freqChannel]<NP.median(self.sAntAmpRcp[freqChannel])*0.6] = 1e6
        
    def allGainsFunc_constrained_new(self, x, obsVis, freq,
                    res, ewSolarAmp, sAntNumber_c, sGainsNumber, ewGainsNumber, sSolVisNumber, 
                    ewSolVisNumber, solVisArrayS, antAGainsS, antBGainsS, solVisArrayEW, 
                    antAGainsEW, antBGainsEW, ewSolVis, sSolVis, solVis, antAGains, antBGains, sAmpSign):


        sSolarAmp = NP.abs(x[0]) * sAmpSign
        x_complex = self.real_to_complex(x[1:])
        
        ewSolVis[0] = ewSolarAmp
        ewSolVis[1:] = x_complex[: ewSolVisNumber]
        sSolVis[0] = sSolarAmp
        sSolVis[1:] = x_complex[ewSolVisNumber : ewSolVisNumber+sSolVisNumber]
        
        ewGains = x_complex[ewSolVisNumber+sSolVisNumber : ewSolVisNumber+sSolVisNumber+ewGainsNumber]
        sGains = NP.append(ewGains[self.center_ant], x_complex[ewSolVisNumber+sSolVisNumber+ewGainsNumber :])

        prev_ind_s = 0
        prev_ind_ew = 0
        for baseline in range(1, self.baselines+1):
            solVisArrayS[prev_ind_s:prev_ind_s+self.redIndexesS_len[baseline-1]] = NP.full(self.redIndexesS_len[baseline-1], sSolVis[baseline-1])
            prev_ind_s = prev_ind_s+self.redIndexesS_len[baseline-1]
            
            solVisArrayEW[prev_ind_ew:prev_ind_ew+self.redIndexesEW_len[baseline-1]] = NP.full(self.redIndexesEW_len[baseline-1], ewSolVis[baseline-1])
            prev_ind_ew = prev_ind_ew+self.redIndexesEW_len[baseline-1]
            
        solVis[:len(solVisArrayEW)] = solVisArrayEW
        solVis[len(solVisArrayEW):] = solVisArrayS

        antA = self.antennaA[self.redIndexesS.astype(int)] - self.antNumberEW + 1
        antB = self.antennaB[self.redIndexesS.astype(int)] - self.antNumberEW + 1
        antA[antA<0] = 0
        antAGainsS = sGains[antA]
        antBGainsS = sGains[antB]
        
        
        antA = self.antennaA[self.redIndexesEW.astype(int)]
        antB = self.antennaB[self.redIndexesEW.astype(int)]
        
        antAGainsEW = ewGains[antA]
        antBGainsEW = ewGains[antB]
        
        antAGains[:len(antAGainsEW)] = antAGainsEW
        antAGains[len(antAGainsEW):] = antAGainsS
        antBGains[:len(antBGainsEW)] = antBGainsEW
        antBGains[len(antBGainsEW):] = antBGainsS
        
        res = solVis * antAGains * NP.conj(antBGains) - obsVis
        return self.complex_to_real(res)
    
    def allGainsFunc_constrained(self, x, obsVis, ewAntNumber, sAntNumber, baselineNumber, freq):
        res = NP.zeros_like(obsVis, dtype = complex)
        ewSolarAmp = 1
        sSolarAmp = NP.abs(x[0])
        x_complex = self.real_to_complex(x[1:])
        
        sAntNumber_c = sAntNumber + 1
        
        sGainsNumber = sAntNumber
        ewGainsNumber = ewAntNumber
        sSolVisNumber = baselineNumber - 1
        ewSolVisNumber = baselineNumber - 1
        ewSolVis = NP.append((ewSolarAmp * NP.exp(1j*self.ewSolarPhase[freq])), x_complex[: ewSolVisNumber])
        sSolVis = NP.append((sSolarAmp * NP.exp(1j*self.sSolarPhase[freq])), x_complex[ewSolVisNumber : ewSolVisNumber+sSolVisNumber])
        ewGains = x_complex[ewSolVisNumber+sSolVisNumber : ewSolVisNumber+sSolVisNumber+ewGainsNumber]
        sGains = NP.append(ewGains[self.center_ant], x_complex[ewSolVisNumber+sSolVisNumber+ewGainsNumber :])
        
        solVisArrayS = NP.array(())
        antAGainsS = NP.array(())
        antBGainsS = NP.array(())
        solVisArrayEW = NP.array(())
        antAGainsEW = NP.array(())
        antBGainsEW = NP.array(())
        for baseline in range(1, baselineNumber+1):
            solVisArrayS = NP.append(solVisArrayS, NP.full(self.redIndexesS_len[baseline-1], sSolVis[baseline-1]))
            solVisArrayEW = NP.append(solVisArrayEW, NP.full(self.redIndexesEW_len[baseline-1], ewSolVis[baseline-1]))
        
        antA = self.antennaA[self.redIndexesS.astype(int)] - self.antNumberEW + 1
        antB = self.antennaB[self.redIndexesS.astype(int)] - self.antNumberEW + 1
        antA[antA<0] = 0
        antAGainsS = NP.append(antAGainsS, sGains[antA])
        antBGainsS = NP.append(antBGainsS, sGains[antB])
        
        antAGainsEW = NP.append(antAGainsEW, ewGains[self.antennaA[self.redIndexesEW.astype(int)]])
        antBGainsEW = NP.append(antBGainsEW, ewGains[self.antennaB[self.redIndexesEW.astype(int)]])
            
        res = NP.append(solVisArrayEW, solVisArrayS) * NP.append(antAGainsEW, antAGainsS) * NP.conj(NP.append(antBGainsEW, antBGainsS)) - obsVis
        return self.complex_to_real(res)  
    
    def buildEwPhase(self):
        newLcpPhaseCorrection = NP.zeros(self.antNumberEW)
        newRcpPhaseCorrection = NP.zeros(self.antNumberEW)
        antY = self.antY[:self.antNumberEW]/self.base
        newLcpPhaseCorrection = NP.deg2rad(self.ewSlopeLcp[self.frequencyChannel] * antY) 
        newRcpPhaseCorrection = NP.deg2rad(self.ewSlopeRcp[self.frequencyChannel] * antY)
        self.ewLcpPhaseCorrection[self.frequencyChannel, :] = newLcpPhaseCorrection[:]
        self.ewRcpPhaseCorrection[self.frequencyChannel, :] = newRcpPhaseCorrection[:]
        
    def buildSPhase(self):
        newLcpPhaseCorrection = NP.zeros(self.antNumberS)
        newRcpPhaseCorrection = NP.zeros(self.antNumberS)
        antX = self.antX[self.antNumberEW:]/self.base
        newLcpPhaseCorrection = NP.deg2rad(-self.sSlopeLcp[self.frequencyChannel] * antX) 
        newRcpPhaseCorrection = NP.deg2rad(-self.sSlopeRcp[self.frequencyChannel] * antX)
        self.sLcpPhaseCorrection[self.frequencyChannel, :] = newLcpPhaseCorrection[:]
        self.sRcpPhaseCorrection[self.frequencyChannel, :] = newRcpPhaseCorrection[:]
        
    def correctPhaseSlopeRL(self, freq):
        workingAnts_ew = NP.arange(0,self.antNumberEW,1)
        y_diff = NP.append(0, self.antY[1:self.antNumberEW]/self.base - self.antY[:self.antNumberEW-1]/self.base)
        workingAnts_ew = NP.delete(workingAnts_ew, NP.append(self.flags_ew, NP.where((y_diff)!=1)))
        # check antennas to delete
        self.phaseDif_ew = NP.unwrap((self.ewAntPhaLcp[freq][workingAnts_ew]+self.ewLcpPhaseCorrection[freq][workingAnts_ew])
                             - (self.ewAntPhaRcp[freq][workingAnts_ew]+self.ewRcpPhaseCorrection[freq][workingAnts_ew]))
        A = NP.vstack([workingAnts_ew, NP.ones(len(workingAnts_ew))]).T
        ew_slope, c = NP.linalg.lstsq(A, self.phaseDif_ew, rcond=None)[0]
        
        workingAnts_s = NP.arange(0,self.antNumberS,1)
        workingAnts_s = NP.delete(workingAnts_s, NP.append(self.flags_s, NP.array((25,64))))
        s_ants = NP.abs(self.antX[self.antNumberEW:]/self.base)
        s_ants = NP.delete(s_ants, NP.append(self.flags_s, NP.array((25,64))))
        self.phaseDif_s = NP.unwrap((self.sAntPhaLcp[freq][workingAnts_s]+self.sLcpPhaseCorrection[freq][workingAnts_s])
                             - (self.sAntPhaRcp[freq][workingAnts_s]+self.sRcpPhaseCorrection[freq][workingAnts_s]))
        A = NP.vstack([s_ants, NP.ones(len(s_ants))]).T
        s_slope, c = NP.linalg.lstsq(A, self.phaseDif_s, rcond=None)[0]
        
        # workingAnts_s = NP.arange(0,self.antNumberS,1)
        # workingAnts_s = NP.delete(workingAnts_s, NP.append(self.flags_s, NP.arange(23,67)))
        # self.phaseDif_s = NP.unwrap((self.sAntPhaLcp[freq][workingAnts_s]+self.sLcpPhaseCorrection[freq][workingAnts_s])
        #                      - (self.sAntPhaRcp[freq][workingAnts_s]+self.sRcpPhaseCorrection[freq][workingAnts_s]))
        # A = NP.vstack([workingAnts_s, NP.ones(len(workingAnts_s))]).T
        # s_slope, c = NP.linalg.lstsq(A, self.phaseDif_s, rcond=None)[0]
        print(ew_slope, s_slope)
        self.ewSlopeRcp[freq] = self.wrap(self.ewSlopeRcp[freq] + NP.rad2deg(ew_slope))
        self.sSlopeRcp[freq] = self.wrap(self.sSlopeRcp[freq] + NP.rad2deg(s_slope))
    
    def real_to_complex(self, z):
        return z[:len(z)//2] + 1j * z[len(z)//2:]
    
    def complex_to_real(self, z):
        return NP.concatenate((NP.real(z), NP.imag(z)))
    
    def setCalibIndex(self, calibIndex):
        self.calibIndex = calibIndex;

    def setFrequencyChannel(self, channel):
        self.frequencyChannel = channel
        
    def vis2uv(self, scan, phaseCorrect = True, amplitudeCorrect = False, PSF=False, average = 0):

        antX = (self.antX/self.base).astype(int)
        antY = (self.antY/self.base).astype(int)
        
        self.uvLcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex)
        self.uvRcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex)
        
        flags_ew = NP.where(self.ewAntAmpLcp[self.frequencyChannel]==1e6)[0]
        flags_s = NP.where(self.sAntAmpLcp[self.frequencyChannel]==1e6)[0]
        
        if average:
            firstScan = scan
            if  self.visLcp.shape[1] < (scan + average):
                lastScan = self.dataLength
            else:
                lastScan = scan + average
        
        O = self.sizeOfUv//2
        for i in range(self.antNumberS):
            for j in range(self.antNumberEW):
                if not (NP.any(flags_ew == j) or NP.any(flags_s == i)):
                    vis = i*self.antNumberEW + j
                    antA = min(self.antennaA[vis], self.antennaB[vis])
                    antB = max(self.antennaA[vis], self.antennaB[vis])
                    antAInd = NP.where(self.antennaNumbers==antA)[0][0]
                    antBInd = NP.where(self.antennaNumbers==antB)[0][0]
                    antAX, antAY = antX[antAInd], antY[antAInd]
                    antBX, antBY = antX[antBInd], antY[antBInd]
                    u = antAY - antBY
                    v = antBX - antAX
                                    
                    if average:
                        self.uvLcp[O + v*2, O + u*2] = NP.sum(self.visLcp[self.frequencyChannel, firstScan:lastScan, vis])/NP.sum(self.validScansLcp[self.frequencyChannel][firstScan:lastScan])
                        self.uvRcp[O + v*2, O + u*2] = NP.sum(self.visRcp[self.frequencyChannel, firstScan:lastScan, vis])/NP.sum(self.validScansRcp[self.frequencyChannel][firstScan:lastScan])
                    else:
                        self.uvLcp[O + v*2, O + u*2] = self.visLcp[self.frequencyChannel, scan, vis]
                        self.uvRcp[O + v*2, O + u*2] = self.visRcp[self.frequencyChannel, scan, vis]
                    
                    if (phaseCorrect):
                        ewPh = self.ewAntPhaLcp[self.frequencyChannel, j]+self.ewLcpPhaseCorrection[self.frequencyChannel, j]
                        sPh = self.sAntPhaLcp[self.frequencyChannel, i]+self.sLcpPhaseCorrection[self.frequencyChannel, i]
                        self.uvLcp[O + v*2, O + u*2] *= NP.exp(1j * (-ewPh + sPh))
                        ewPh = self.ewAntPhaRcp[self.frequencyChannel, j]+self.ewRcpPhaseCorrection[self.frequencyChannel, j]
                        sPh = self.sAntPhaRcp[self.frequencyChannel, i]+self.sRcpPhaseCorrection[self.frequencyChannel, i]
                        self.uvRcp[O + v*2, O + u*2] *= NP.exp(1j * (-ewPh + sPh))
                    if (amplitudeCorrect):
                        self.uvLcp[O + v*2, O + u*2] /= (self.ewAntAmpLcp[self.frequencyChannel, j] * self.sAntAmpLcp[self.frequencyChannel, i])
                        self.uvRcp[O + v*2, O + u*2] /= (self.ewAntAmpRcp[self.frequencyChannel, j] * self.sAntAmpRcp[self.frequencyChannel, i])
                    
                    self.uvLcp[O - v*2, O - u*2] = NP.conj(self.uvLcp[O + v*2, O + u*2])
                    self.uvRcp[O - v*2, O - u*2] = NP.conj(self.uvRcp[O + v*2, O + u*2])
                    
                    
        for i in range(len(self.antZeroRow)):
            vis = self.antZeroRow[i]
            if not (NP.any(flags_ew == i) or NP.any(flags_ew == self.center_ant)):
                if i<self.center_ant:
                    antA = self.antennaA[vis]
                    antB = self.antennaB[vis]
                    antAInd = NP.where(self.antennaNumbers==antA)[0][0]
                    antBInd = NP.where(self.antennaNumbers==antB)[0][0]
                    antAX, antAY = antX[antAInd], antY[antAInd]
                    antBX, antBY = antX[antBInd], antY[antBInd]
                    u = antAY - antBY
                    v = antBX - antAX
                    
                    if average:
                        self.uvLcp[O + v*2, O + u*2] = NP.sum(self.visLcp[self.frequencyChannel, firstScan:lastScan, vis])/NP.sum(self.validScansLcp[self.frequencyChannel][firstScan:lastScan])
                        self.uvRcp[O + v*2, O + u*2] = NP.sum(self.visRcp[self.frequencyChannel, firstScan:lastScan, vis])/NP.sum(self.validScansRcp[self.frequencyChannel][firstScan:lastScan])
                    else:
                        self.uvLcp[O + v*2, O + u*2] = self.visLcp[self.frequencyChannel, scan, vis]
                        self.uvRcp[O + v*2, O + u*2] = self.visRcp[self.frequencyChannel, scan, vis]
                    
                    if (phaseCorrect):
                        ewPh1 = self.ewAntPhaLcp[self.frequencyChannel, i]+self.ewLcpPhaseCorrection[self.frequencyChannel, i]
                        ewPh2 = self.ewAntPhaLcp[self.frequencyChannel, self.center_ant]+self.ewLcpPhaseCorrection[self.frequencyChannel, self.center_ant]
                        self.uvLcp[O + v*2, O + u*2] *= NP.exp(1j * (-ewPh1 + ewPh2))
                        ewPh1 = self.ewAntPhaRcp[self.frequencyChannel, i]+self.ewRcpPhaseCorrection[self.frequencyChannel, i]
                        ewPh2 = self.ewAntPhaRcp[self.frequencyChannel, self.center_ant]+self.ewRcpPhaseCorrection[self.frequencyChannel, self.center_ant]
                        self.uvRcp[O + v*2, O + u*2] *= NP.exp(1j * (-ewPh1 + ewPh2))
                    if (amplitudeCorrect):
                        self.uvLcp[O + v*2, O + u*2] /= (self.ewAntAmpLcp[self.frequencyChannel, i] * self.ewAntAmpLcp[self.frequencyChannel, self.center_ant])
                        self.uvRcp[O + v*2, O + u*2] /= (self.ewAntAmpRcp[self.frequencyChannel, i] * self.ewAntAmpRcp[self.frequencyChannel, self.center_ant])
                    
                    self.uvLcp[O - v*2, O - u*2] = NP.conj(self.uvLcp[O + v*2, O + u*2])
                    self.uvRcp[O - v*2, O - u*2] = NP.conj(self.uvRcp[O + v*2, O + u*2])
                
            # else:
            #     antA = self.antennaB[vis]
            #     antB = self.antennaA[vis]
            #     antAInd = NP.where(self.antennaNumbers==str(antA))[0][0]
            #     antBInd = NP.where(self.antennaNumbers==str(antB))[0][0]
            #     antAX, antAY = antX[antAInd], antY[antAInd]
            #     antBX, antBY = antX[antBInd], antY[antBInd]
            #     u = antAY - antBY
            #     v = antBX - antAX
                
            #     if average:
            #         self.uvLcp[O + v*2, O + u*2] =  NP.conj(NP.mean(self.visLcp[self.frequencyChannel, firstScan:lastScan, vis]))
            #         self.uvRcp[O + v*2, O + u*2] =  NP.conj(NP.mean(self.visRcp[self.frequencyChannel, firstScan:lastScan, vis]))
            #     else:
            #         self.uvLcp[O + v*2, O + u*2] = NP.conj(self.visLcp[self.frequencyChannel, scan, vis])
            #         self.uvRcp[O + v*2, O + u*2] = NP.conj(self.visRcp[self.frequencyChannel, scan, vis])
                
            #     if (phaseCorrect):
            #         ewPh1 = self.ewAntPhaLcp[self.frequencyChannel, i]+self.ewLcpPhaseCorrection[self.frequencyChannel, i]
            #         ewPh2 = self.ewAntPhaLcp[self.frequencyChannel, 69]+self.ewLcpPhaseCorrection[self.frequencyChannel, 69]
            #         self.uvLcp[O + v*2, O + u*2] *= NP.exp(1j * (ewPh1 - ewPh2))
            #         ewPh1 = self.ewAntPhaRcp[self.frequencyChannel, i]+self.ewRcpPhaseCorrection[self.frequencyChannel, i]
            #         ewPh2 = self.ewAntPhaRcp[self.frequencyChannel, 69]+self.ewRcpPhaseCorrection[self.frequencyChannel, 69]
            #         self.uvRcp[O + v*2, O + u*2] *= NP.exp(1j * (ewPh1 - ewPh2))
            #     if (amplitudeCorrect):
            #         self.uvLcp[O + v*2, O + u*2] /= (self.ewAntAmpLcp[self.frequencyChannel, i] * self.ewAntAmpLcp[self.frequencyChannel, 69])
            #         self.uvRcp[O + v*2, O + u*2] /= (self.ewAntAmpRcp[self.frequencyChannel, i] * self.ewAntAmpRcp[self.frequencyChannel, 69])
                
        if (amplitudeCorrect):
            self.uvLcp[O,O] = self.fluxLcp[self.frequencyChannel]
            self.uvRcp[O,O] = self.fluxRcp[self.frequencyChannel]
        
        if PSF:
            self.uvLcp[NP.abs(self.uvLcp)>1e-8] = 1
            self.uvRcp[NP.abs(self.uvRcp)>1e-8] = 1
            
        self.uvLcp[NP.abs(self.uvLcp)<1e-6] = 0.
        self.uvRcp[NP.abs(self.uvRcp)<1e-6] = 0.
        self.uvLcp /= NP.count_nonzero(self.uvLcp)
        self.uvRcp /= NP.count_nonzero(self.uvRcp)

    def uv2lmImage(self):
        self.lcp = NP.fft.fft2(NP.roll(NP.roll(self.uvLcp,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
        self.lcp = NP.roll(NP.roll(self.lcp,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1);
        self.lcp = NP.flip(self.lcp, 1)
        self.rcp = NP.fft.fft2(NP.roll(NP.roll(self.uvRcp,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
        self.rcp = NP.roll(NP.roll(self.rcp,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1);
        self.rcp = NP.flip(self.rcp, 1)
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
        
    def wrap(self, value):
        while value<-180:
            value+=360
        while value>180:
            value-=360
        return value
    
    def createDisk(self, radius, arcsecPerPixel = 2.45552):
        qSun = NP.zeros((self.sizeOfUv, self.sizeOfUv))
        sunRadius = radius / (arcsecPerPixel*2)
        for i in range(self.sizeOfUv):
            x = i - self.sizeOfUv//2 - 1
            for j in range(self.sizeOfUv):
                y = j - self.sizeOfUv//2 - 1
                if (NP.sqrt(x*x + y*y) < sunRadius):
                    qSun[i , j] = 1
                    
        dL = 2*( 12//2) + 1
        arg_x = NP.linspace(-1.,1,dL)
        arg_y = NP.linspace(-1.,1,dL)
        xx, yy = NP.meshgrid(arg_x, arg_y)
        
        scaling = self.getPQScale(self.sizeOfUv, NP.deg2rad(arcsecPerPixel*(self.sizeOfUv-1)/3600.)*2)
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
        qSun_el_hd = warp(qSmoothSun,(shift + (rotate + back_shift)).inverse)
        
        res = warp(qSun_el_hd, (shift + (matrix + back_shift)).inverse)
        qSun_lm = warp(res,(shift + (scale + back_shift)).inverse)
        qSun_lm_fft = NP.fft.fft2(NP.roll(NP.roll(qSun_lm,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        qSun_lm_fft = NP.roll(NP.roll(qSun_lm_fft,self.sizeOfUv//2,0),self.sizeOfUv//2,1) / self.sizeOfUv;
        qSun_lm_fft = NP.flip(qSun_lm_fft, 0)
#        qSun_lm_uv = qSun_lm_fft * uvPsf
#        qSun_lm_conv = NP.fft.fft2(NP.roll(NP.roll(qSun_lm_uv,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
#        qSun_lm_conv = NP.roll(NP.roll(qSun_lm_conv,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1);
#        qSun_lm_conv = NP.flip(NP.flip(qSun_lm_conv, 1), 0)
        self.fftDisk = qSun_lm_fft #qSun_lm_conv, 
        self.lm_hd_relation[self.frequencyChannel] = NP.sum(qSun_lm)/NP.sum(qSun_el_hd)
    
    def createUvUniform(self):
        self.uvUniform = NP.zeros((self.sizeOfUv, self.sizeOfUv), dtype = complex)
        flags_ew = NP.where(self.ewAntAmpLcp[self.frequencyChannel]==1e6)[0]
        flags_s = NP.where(self.sAntAmpLcp[self.frequencyChannel]==1e6)[0]
        O = self.sizeOfUv//2
        antX = (self.antX/self.base).astype(int)
        antY = (self.antY/self.base).astype(int)
        for i in range(self.antNumberS):
            for j in range(self.antNumberEW):
                 if not (NP.any(flags_ew == j) or NP.any(flags_s == i)):
                    vis = i*self.antNumberEW + j
                    antA = min(self.antennaA[vis], self.antennaB[vis])
                    antB = max(self.antennaA[vis], self.antennaB[vis])
                    antAInd = NP.where(self.antennaNumbers==antA)[0][0]
                    antBInd = NP.where(self.antennaNumbers==antB)[0][0]
                    antAX, antAY = antX[antAInd], antY[antAInd]
                    antBX, antBY = antX[antBInd], antY[antBInd]
                    u = antAY - antBY
                    v = antBX - antAX
                    self.uvUniform[O + v*2, O + u*2] = 1
                    self.uvUniform[O - v*2, O - u*2] = 1
        for i in range(self.center_ant):
            if not (NP.any(flags_ew == i) or NP.any(flags_ew == self.center_ant)):
                vis = self.antZeroRow[i]
                antA = min(self.antennaA[vis], self.antennaB[vis])
                antB = max(self.antennaA[vis], self.antennaB[vis])
                antAInd = NP.where(self.antennaNumbers==antA)[0][0]
                antBInd = NP.where(self.antennaNumbers==antB)[0][0]
                antAX, antAY = antX[antAInd], antY[antAInd]
                antBX, antBY = antX[antBInd], antY[antBInd]
                u = antAY - antBY
                v = antBX - antAX
                self.uvUniform[O + v*2, O + u*2] = 1
                self.uvUniform[O - v*2, O - u*2] = 1
        self.uvUniform[O, O] = 1
                    
    def createUvPsf(self, T, ewSlope, sSlope, shift):
        self.uvPsf = self.uvUniform.copy()
        O = self.sizeOfUv//2
        ewSlope = NP.deg2rad(ewSlope)
        sSlope = NP.deg2rad(sSlope)
        ewSlopeUv = NP.linspace(-O * ewSlope/2., O * ewSlope/2., self.sizeOfUv)
        sSlopeUv = NP.linspace(-O * sSlope/2., O * sSlope/2., self.sizeOfUv)
        ewGrid,sGrid = NP.meshgrid(ewSlopeUv, sSlopeUv)
        slopeGrid = ewGrid + sGrid
        slopeGrid[self.uvUniform == 0] = 0
        self.uvPsf *= T * NP.exp(1j * slopeGrid)
        self.uvPsf[O,O] = shift
    
    def diskDiff(self, x, pol):
        self.createUvPsf(x[0], x[1], x[2], x[3])
        uvDisk = self.fftDisk * self.uvPsf
        if pol == 0:
            diff = self.uvLcp - uvDisk
        if pol == 1:
            diff = self.uvRcp - uvDisk
        return self.complex_to_real(diff[self.uvUniform!=0])
#        qSun_lm_conv = NP.fft.fft2(NP.roll(NP.roll(diff,uvSize//2+1,0),uvSize//2+1,1));
#        return NP.abs(NP.reshape(qSun_lm_conv, uvSize**2))
    
    def findDisk(self):
        with threadpool_limits(limits=1, user_api='blas'):
            Tb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6) * 1e3
            self.createDisk(sunpy.coordinates.sun.angular_radius(self.dateObs).to_value())
            self.createUvUniform()
            self.x_ini = [Tb/self.convolutionNormCoef,0,0,1]
            # x_ini = [1,0,0]
            self.center_ls_res_lcp = least_squares(self.diskDiff, self.x_ini, args = (0,))
            _diskLevelLcp, _ewSlopeLcp, _sSlopeLcp, _shiftLcp = self.center_ls_res_lcp['x']
            self.center_ls_res_rcp = least_squares(self.diskDiff, self.x_ini, args = (1,))
            _diskLevelRcp, _ewSlopeRcp, _sSlopeRcp, _shiftRcp = self.center_ls_res_rcp['x']
            
            self.diskLevelLcp[self.frequencyChannel] = _diskLevelLcp
            self.diskLevelRcp[self.frequencyChannel] = _diskLevelRcp
            
            
            
            self.lcpShift[self.frequencyChannel] = self.lcpShift[self.frequencyChannel]/(_shiftLcp * self.convolutionNormCoef / Tb)
            self.rcpShift[self.frequencyChannel] = self.rcpShift[self.frequencyChannel]/(_shiftRcp * self.convolutionNormCoef / Tb)
            
            if not self.corr_amp_exist:
                self.ewAntAmpLcp[self.frequencyChannel][self.ewAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
                self.nAntAmpLcp[self.frequencyChannel][self.nAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
                self.ewAntAmpRcp[self.frequencyChannel][self.ewAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
                self.nAntAmpRcp[self.frequencyChannel][self.nAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
            
            self.ewSlopeLcp[self.frequencyChannel] = self.wrap(self.ewSlopeLcp[self.frequencyChannel] + _ewSlopeLcp)
            self.sSlopeLcp[self.frequencyChannel] = self.wrap(self.sSlopeLcp[self.frequencyChannel] + _sSlopeLcp)
            self.ewSlopeRcp[self.frequencyChannel] = self.wrap(self.ewSlopeRcp[self.frequencyChannel] + _ewSlopeRcp)
            self.sSlopeRcp[self.frequencyChannel] = self.wrap(self.sSlopeRcp[self.frequencyChannel] + _sSlopeRcp)
              
    def findDisk_long(self):
        with threadpool_limits(limits=1, user_api='blas'):
            Tb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6) * 1e3
            self.createDisk(sunpy.coordinates.sun.angular_radius(self.dateObs).to_value())
            self.createUvUniform()
            fun_lcp = 10
            fun_rcp = 10
      
            for i in range(3):
                for j in range(3):
                    start_time = time.time()
                    
                    self.x_ini = [Tb/self.convolutionNormCoef, -90+i*90, -90+j*90, 1]
                    
                    ls_res = least_squares(self.diskDiff, self.x_ini, args = (0,), ftol=1e-3)
                    print(NP.sum(ls_res['fun']**2))
                    if i==0 and j==0:
                        _diskLevelLcp, _ewSlopeLcp, _sSlopeLcp, _shiftLcp = ls_res['x']
                        fun_lcp = NP.sum(ls_res['fun']**2)
                        _diskLevelRcp, _ewSlopeRcp, _sSlopeRcp, _shiftRcp = ls_res['x']
                        fun_rcp = NP.sum(ls_res['fun']**2)
                        
                    else:
                        if NP.sum(ls_res['fun']**2)<fun_lcp and ls_res['x'][0]>0:
                            print('min updated')
                            _diskLevelLcp, _ewSlopeLcp, _sSlopeLcp, _shiftLcp = ls_res['x']
                            fun_lcp = NP.sum(ls_res['fun']**2)
                            print((_diskLevelLcp, _ewSlopeLcp, _sSlopeLcp, _shiftLcp))
     
                        self.x_ini = [Tb/self.convolutionNormCoef, -90+i*90, -90+j*90, 1]
                        ls_res = least_squares(self.diskDiff, self.x_ini, args = (1,), ftol=1e-3)
                        if NP.sum(ls_res['fun']**2)<fun_rcp and ls_res['x'][0]>0:
                            _diskLevelRcp, _ewSlopeRcp, _sSlopeRcp, _shiftRcp = ls_res['x']
                            fun_rcp = NP.sum(ls_res['fun']**2)
                        
                    print("ITER " + str(i*3+j) + " --- %s seconds ---" % (time.time() - start_time))
                    
            self.x_ini = [_diskLevelLcp, _ewSlopeLcp, _sSlopeLcp, _shiftLcp]               
            ls_res = least_squares(self.diskDiff, self.x_ini, args = (0,), ftol=1e-10)
            _diskLevelLcp, _ewSlopeLcp, _sSlopeLcp, _shiftLcp = ls_res['x']
            
            self.x_ini = [_diskLevelRcp, _ewSlopeRcp, _sSlopeRcp, _shiftRcp]               
            ls_res = least_squares(self.diskDiff, self.x_ini, args = (1,), ftol=1e-10)
            _diskLevelRcp, _ewSlopeRcp, _sSlopeRcp, _shiftRcp = ls_res['x']
            
            self.diskLevelLcp[self.frequencyChannel] = _diskLevelLcp
            self.diskLevelRcp[self.frequencyChannel] = _diskLevelRcp
     
            self.lcpShift[self.frequencyChannel] = self.lcpShift[self.frequencyChannel]/(_shiftLcp * self.convolutionNormCoef / Tb)
            self.rcpShift[self.frequencyChannel] = self.rcpShift[self.frequencyChannel]/(_shiftRcp * self.convolutionNormCoef / Tb)
                
            if not self.corr_amp_exist:
                self.ewAntAmpLcp[self.frequencyChannel][self.ewAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
                self.sAntAmpLcp[self.frequencyChannel][self.sAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
                self.ewAntAmpRcp[self.frequencyChannel][self.ewAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
                self.sAntAmpRcp[self.frequencyChannel][self.sAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
            
            self.ewSlopeLcp[self.frequencyChannel] = self.wrap(self.ewSlopeLcp[self.frequencyChannel] + _ewSlopeLcp)
            self.sSlopeLcp[self.frequencyChannel] = self.wrap(self.sSlopeLcp[self.frequencyChannel] + _sSlopeLcp)
            self.ewSlopeRcp[self.frequencyChannel] = self.wrap(self.ewSlopeRcp[self.frequencyChannel] + _ewSlopeRcp)
            self.sSlopeRcp[self.frequencyChannel] = self.wrap(self.sSlopeRcp[self.frequencyChannel] + _sSlopeRcp)
              
        
    def centerDisk(self, long = False):
        if long:
            self.findDisk_long()
        else:
            self.findDisk()
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
        
    def modelDiskConv(self):
        # self.createUvPsf(self.diskLevelLcp,0,0,0)
        currentDiskTb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6)*1e3
        self.createUvPsf(currentDiskTb/self.convolutionNormCoef,0,0,currentDiskTb/self.convolutionNormCoef)
        self.uvDiskConv = self.fftDisk * self.uvPsf# - self.uvLcp
        qSun_lm = NP.fft.fft2(NP.roll(NP.roll(self.uvDiskConv,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
        qSun_lm = NP.roll(NP.roll(qSun_lm,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1)# / self.sizeOfUv;
        qSun_lm = NP.flip(qSun_lm, 0)
        self.modelDisk = qSun_lm
        
    def modelDiskConv_unity(self):
        self.createDisk(980)
        self.createUvUniform()
        self.createUvPsf(1,0,0,1)
        self.uvDiskConv = self.fftDisk * self.uvPsf# - self.uvLcp
        qSun_lm = NP.fft.fft2(NP.roll(NP.roll(self.uvDiskConv,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
        qSun_lm = NP.roll(NP.roll(qSun_lm,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1)# / self.sizeOfUv;
        qSun_lm = NP.flip(qSun_lm, 0)
        self.modelDisk = qSun_lm