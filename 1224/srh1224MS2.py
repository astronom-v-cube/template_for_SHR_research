#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 04:27:13 2018

@author: sergey
"""

import casacore.tables as T
import numpy as NP
from astropy.time import Time, TimeDelta
import base2uvw_1224 as UVW

class Srh1224Ms2():
    def __init__(self, tableName):
        a = 6378137.0000;
        b = 6356752.3141;
        e2 = (a*a - b*b) / (a*a);
        ssrtLat = NP.deg2rad(51.  + 45.562/60.);
        ssrtLon = NP.deg2rad(102. + 13.160/60.);
        ssrtHeight = 832.;
        v = a / (NP.sqrt(1. - e2*(NP.sin(ssrtLat) * NP.sin(ssrtLat))));
        self.x = (v + ssrtHeight)*NP.cos(ssrtLat)*NP.cos(ssrtLon);
        self.y = (v + ssrtHeight)*NP.cos(ssrtLat)*NP.sin(ssrtLon);
        self.z = (((1. - e2)*v) + ssrtHeight)*NP.sin(ssrtLat);
        
        self.dataTable = T.default_ms(name = tableName)
        self.antennaTable = T.default_ms_subtable('ANTENNA',name = tableName + '/ANTENNA')
        self.spectralWindowTable = T.default_ms_subtable('SPECTRAL_WINDOW',name = tableName + '/SPECTRAL_WINDOW')
        self.dataDescriptionTable = T.default_ms_subtable('DATA_DESCRIPTION',name = tableName + '/DATA_DESCRIPTION')
        self.polarizationTable = T.default_ms_subtable('POLARIZATION',name = tableName + '/POLARIZATION')
        self.sourceTable = T.default_ms_subtable('SOURCE',name = tableName + '/SOURCE')
        self.fieldTable = T.default_ms_subtable('FIELD',name = tableName + '/FIELD')
        self.feedTable = T.default_ms_subtable('FEED',name = tableName + '/FEED')
        self.observationTable = T.default_ms_subtable('OBSERVATION',name = tableName + '/OBSERVATION')

    def createMS(self, srhFits, frequencyChannel = [], phaseCorrect = True, amplitudeCorrect = False):
        self.initDataTable(srhFits, frequencyChannel = frequencyChannel, phaseCorrect = phaseCorrect, amplitudeCorrect = amplitudeCorrect)
        self.initAntennaTable(srhFits)
        self.initSpectralWindowTable(srhFits, frequencyChannel = frequencyChannel)
        self.initDataDescriptionTable(frequencyChannel = frequencyChannel)
        self.initPolarizationTable()
        self.initSourceTable()
        self.initFieldTable()
        self.initFeedTable(srhFits, frequencyChannel = frequencyChannel)
        self.initObservationTable(srhFits)

    def initDataTable(self, srhFits, frequencyChannel = [], phaseCorrect = True, amplitudeCorrect = False):
        declination = srhFits.getDeclination()
        noon = srhFits.RAO.culmination
        
        antEW = 139
        antNS = 68
        zero_row = 69
        
        visibilityNumber = antEW * antNS + zero_row + 1
        freqNumber = len(frequencyChannel)
        
        dataDesc = T.makearrcoldesc('DATA',0.+0j, shape=[1,2],valuetype='complex')
        correctedDataDesc = T.makearrcoldesc('CORRECTED_DATA',0.+0j, shape=[1,2],valuetype='complex')
        self.dataTable.addcols(dataDesc)
        self.dataTable.addcols(correctedDataDesc)
        uv = NP.zeros((visibilityNumber,1,2),dtype='complex64')
        uv_corr = NP.zeros((visibilityNumber,2),dtype='complex64')
        flag = NP.zeros((visibilityNumber,1,2),dtype='bool')
        weight = NP.array([100.,100.],dtype='float32')
        sigma = NP.array([.01,.01],dtype='float32')
       
        fitsDate =srhFits.hduList[0].header['DATE-OBS'] + 'T00:00:00';
        
        firstRow = 0
        for freqInd in range(freqNumber):
            freq = frequencyChannel[freqInd]
            
            # flags_ew_lcp = NP.where(srhFits.ewAntAmpLcp[freq] == 1e6)[0]
            # flags_ew_rcp = NP.where(srhFits.ewAntAmpRcp[freq] == 1e6)[0]
            # flags_ew = NP.unique(NP.append(flags_ew_lcp, flags_ew_rcp))
            # flags_s_lcp = NP.where(srhFits.sAntAmpLcp[freq] == 1e6)[0]
            # flags_s_rcp = NP.where(srhFits.sAntAmpRcp[freq] == 1e6)[0]
            # flags_s = NP.unique(NP.append(flags_s_lcp, flags_s_rcp))
            
            flags_arr = NP.zeros((antNS, antEW), dtype = 'bool')
            # flags_arr[flags_s,:] = True
            # flags_arr[:,flags_ew] = True
            flags_arr = NP.reshape(flags_arr, (antEW * antNS))
            flags_zero_row = NP.zeros(zero_row+1, dtype = 'bool')
            # flags_zero_row[flags_ew[flags_ew<zero_row]] = True
            flags_arr = NP.append(flags_arr, flags_zero_row)
            flag[:,0,0] = flags_arr
            flag[:,0,1] = flags_arr
            
            for scan in range(srhFits.dataLength):
                scanDate = Time(fitsDate, format='isot',scale='utc');
                scanTime = srhFits.freqTime[freq, scan]
                scanDate += TimeDelta(scanTime,format='sec')
                hourAngle = NP.deg2rad((scanTime - noon)*15./3600.)
                if hourAngle > NP.pi:
                    hourAngle -= 2*NP.pi
                    
                self.dataTable.addrows(nrows = visibilityNumber)
                uv[:visibilityNumber-zero_row-1,0,0] = srhFits.visLcp[freq,scan,:visibilityNumber-zero_row-1].copy()
                uv[:visibilityNumber-zero_row-1,0,1] = srhFits.visRcp[freq,scan,:visibilityNumber-zero_row-1].copy()
                uv[visibilityNumber-zero_row-1:-1,0,0] = srhFits.visLcp[freq,scan,srhFits.antZeroRow[:zero_row]].copy()
                uv[visibilityNumber-zero_row-1:-1,0,1] = srhFits.visRcp[freq,scan,srhFits.antZeroRow[:zero_row]].copy()
                uv_corr = uv.copy()
                if phaseCorrect:
                    antA = NP.append(NP.tile(srhFits.ewAntPhaLcp[freq]+srhFits.ewLcpPhaseCorrection[freq], antNS), srhFits.ewAntPhaLcp[freq, :zero_row]+srhFits.ewLcpPhaseCorrection[freq, :zero_row])
                    antB = NP.append(NP.repeat(srhFits.sAntPhaLcp[freq]+srhFits.sLcpPhaseCorrection[freq], antEW), NP.full(zero_row, srhFits.ewAntPhaLcp[freq, zero_row]+srhFits.ewLcpPhaseCorrection[freq, zero_row]))
                    uv_corr[:-1,0,0] *= NP.exp(1j* (-antA + antB))
                    antA = NP.append(NP.tile(srhFits.ewAntPhaRcp[freq]+srhFits.ewRcpPhaseCorrection[freq], antNS), srhFits.ewAntPhaRcp[freq, :zero_row]+srhFits.ewRcpPhaseCorrection[freq, :zero_row])
                    antB = NP.append(NP.repeat(srhFits.sAntPhaRcp[freq]+srhFits.sRcpPhaseCorrection[freq], antEW), NP.full(zero_row, srhFits.ewAntPhaRcp[freq, zero_row]+srhFits.ewRcpPhaseCorrection[freq, zero_row]))
                    uv_corr[:-1,0,1] *= NP.exp(1j* (-antA + antB))
                if amplitudeCorrect:
                    antA = NP.append(NP.tile(srhFits.ewAntAmpLcp[freq], antNS), srhFits.ewAntAmpLcp[freq, :zero_row])
                    antB = NP.append(NP.repeat(srhFits.sAntAmpLcp[freq], antEW), NP.full(zero_row, srhFits.ewAntAmpLcp[freq, zero_row]))
                    uv_corr[:-1,0,0] /= (antA * antB)
                    antA = NP.append(NP.tile(srhFits.ewAntAmpRcp[freq], antNS), srhFits.ewAntAmpRcp[freq, :zero_row])
                    antB = NP.append(NP.repeat(srhFits.sAntAmpRcp[freq], antEW), NP.full(zero_row, srhFits.ewAntAmpRcp[freq, zero_row]))
                    uv_corr[:-1,0,1] /= (antA * antB)
                    
                uv_corr[-1,0,0] = srhFits.lcpShift[freq]*2
                uv_corr[-1,0,1] = srhFits.rcpShift[freq]*2
                
                ant1 = NP.append(srhFits.antennaA[:visibilityNumber-zero_row-1], NP.arange(1,zero_row+1, dtype='int32'))
                ant2 = NP.append(srhFits.antennaB[:visibilityNumber-zero_row-1], NP.full(zero_row,zero_row+1))
                ant1 = NP.append(ant1, 1)
                ant2 = NP.append(ant2, 1)

                self.dataTable.putcol("DATA", uv, startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("CORRECTED_DATA", uv_corr, startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("ANTENNA1", ant1, startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("ANTENNA2", ant2, startrow=firstRow, nrow=visibilityNumber)
                
                # uvw = NP.zeros((visibilityNumber,3))
                # for i in range(visibilityNumber):
                #     uvw[i] = UVW.base2uvw(hourAngle,declination, ant1[i], ant2[i])
                
                uvw = [UVW.base2uvw(hourAngle,declination, ant1[i]+1, ant2[i]+1) for i in range(visibilityNumber)]
                self.dataTable.putcol("UVW", NP.array(uvw), startrow=firstRow, nrow=visibilityNumber)

                # self.dataTable.putcol("UVW", NP.full((visibilityNumber,3),UVW.base2uvw(hourAngle,declination,srhFits.antennaA[0], srhFits.antennaB[0])), startrow=firstRow, nrow=visibilityNumber)

                self.dataTable.putcol("SCAN_NUMBER", NP.full(visibilityNumber, scan), startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("ARRAY_ID", NP.full(visibilityNumber, 0), startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("STATE_ID", NP.full(visibilityNumber, -1), startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("PROCESSOR_ID", NP.full(visibilityNumber, -1), startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("FEED1", NP.full(visibilityNumber, 0), startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("FEED2", NP.full(visibilityNumber, 0), startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("WEIGHT", NP.full((visibilityNumber,2), weight), startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("SIGMA", NP.full((visibilityNumber,2), sigma), startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("EXPOSURE", NP.full(visibilityNumber, 0.28), startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("INTERVAL", NP.full(visibilityNumber, 0.28), startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("FLAG", flag, startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("FLAG_ROW", flags_arr, startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("DATA_DESC_ID", NP.full(visibilityNumber, freqInd), startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("TIME", NP.full(visibilityNumber, scanDate.mjd*(24.*3600)), startrow=firstRow, nrow=visibilityNumber)
                self.dataTable.putcol("TIME_CENTROID", NP.full(visibilityNumber, scanDate.mjd*(24.*3600)), startrow=firstRow, nrow=visibilityNumber)
                
                firstRow += visibilityNumber    

        self.dataTable.close()

    def initDataTable_old(self, srhFits, frequencyChannel = [], phaseCorrect = True, amplitudeCorrect = False):
        declination = srhFits.getDeclination()
        noon = srhFits.RAO.culmination
        
        visibilityNumber = 69 * 139 + 69
        freqNumber = len(frequencyChannel)
        
        dataDesc = T.makearrcoldesc('DATA',0.+0j, shape=[1,2],valuetype='complex')
        correctedDataDesc = T.makearrcoldesc('CORRECTED_DATA',0.+0j, shape=[1,2],valuetype='complex')
        self.dataTable.addcols(dataDesc)
        self.dataTable.addcols(correctedDataDesc)
        uv = NP.zeros((1,2),dtype='complex64')
        uv_corr = NP.zeros((1,2),dtype='complex64')
        flag = NP.zeros((1,2),dtype='bool')
        weight = NP.array([100.,100.],dtype='float32')
        sigma = NP.array([.01,.01],dtype='float32')
       
        fitsDate =srhFits.hduList[0].header['DATE-OBS'] + 'T00:00:00';

        for freqInd in range(freqNumber):
            freq = frequencyChannel[freqInd]
            for scan in range(srhFits.dataLength):
                scanDate = Time(fitsDate, format='isot',scale='utc');
                scanTime = srhFits.freqTime[freq, scan]
                scanDate += TimeDelta(scanTime,format='sec')
                hourAngle = NP.deg2rad((scanTime - noon)*15./3600.)
                if hourAngle > NP.pi:
                    hourAngle -= 2*NP.pi
                for vis in range(visibilityNumber):
                    self.dataTable.addrows()
                    row = freqInd*srhFits.dataLength*visibilityNumber + scan*visibilityNumber + vis;
                    if vis < 69 * 139:
                        i = vis // 139
                        j = vis % 139 
                        # uv[0,0] = NP.conj(srhFits.visLcp[freq,scan,vis])
                        # uv[0,1] = NP.conj(srhFits.visRcp[freq,scan,vis])
                        uv[0,0] = 1 + 0j
                        uv[0,1] = 1 + 0j
                        uv_corr[0,0] = 1 + 0j
                        uv_corr[0,1] = 1 + 0j
                        # if phaseCorrect:
                        #     uv_corr[0,0] *= NP.exp(1j*(-(srhFits.ewAntPhaLcp[freq, j]+srhFits.ewLcpPhaseCorrection[freq, j]) + (srhFits.nAntPhaLcp[freq, i] + srhFits.nLcpPhaseCorrection[freq, i])))
                        #     uv_corr[0,1] *= NP.exp(1j*(-(srhFits.ewAntPhaRcp[freq, j]+srhFits.ewRcpPhaseCorrection[freq, j]) + (srhFits.nAntPhaRcp[freq, i] + srhFits.nRcpPhaseCorrection[freq, i])))
                        # if amplitudeCorrect:
                        #     uv_corr[0,0] /= (srhFits.ewAntAmpLcp[freq, j] * srhFits.nAntAmpLcp[freq, i])
                        #     uv_corr[0,1] /= (srhFits.ewAntAmpRcp[freq, j] * srhFits.nAntAmpRcp[freq, i])
                        self.dataTable.col('ANTENNA1')[row] = j+1
                        self.dataTable.col('ANTENNA2')[row] = i+140
                        self.dataTable.col('DATA')[row] = uv
                        self.dataTable.col('CORRECTED_DATA')[row] = uv_corr
                        self.dataTable.col('UVW')[row] = UVW.base2uvw(hourAngle,declination,j+1, i+140)#, correct_positions = False)
                    else:
                        # uv[0,0] = NP.conj(srhFits.visLcp[freq,scan,srhFits.antZeroRow[vis - 31*97]])
                        # uv[0,1] = NP.conj(srhFits.visRcp[freq,scan,srhFits.antZeroRow[vis - 31*97]])
                        uv[0,0] = 1 + 0j
                        uv[0,1] = 1 + 0j
                        uv_corr[0,0] = 1 + 0j
                        uv_corr[0,1] = 1 + 0j
                        # if vis - 31*97 >= 32:
                        #     if phaseCorrect:
                        #         uv_corr[0,0] *= NP.exp(1j*(-(srhFits.ewAntPhaLcp[freq, vis - 31*97 + 1] + srhFits.ewLcpPhaseCorrection[freq, vis - 31*97 + 1]) + (srhFits.ewAntPhaLcp[freq, 32] + srhFits.ewLcpPhaseCorrection[freq, 32])))
                        #         uv_corr[0,1] *= NP.exp(1j*(-(srhFits.ewAntPhaRcp[freq, vis - 31*97 + 1] + srhFits.ewRcpPhaseCorrection[freq, vis - 31*97 + 1]) + (srhFits.ewAntPhaRcp[freq, 32] + srhFits.ewRcpPhaseCorrection[freq, 32])))
                        #     if amplitudeCorrect:
                        #         uv_corr[0,0] /= (srhFits.ewAntAmpLcp[freq, vis - 31*97 + 1] * srhFits.ewAntAmpLcp[freq, 32])
                        #         uv_corr[0,1] /= (srhFits.ewAntAmpRcp[freq, vis - 31*97 + 1] * srhFits.ewAntAmpRcp[freq, 32])
                        self.dataTable.col('ANTENNA1')[row] = 70
                        self.dataTable.col('ANTENNA2')[row] = vis - 69 * 139 + 1
                        self.dataTable.col('DATA')[row] = uv
                        self.dataTable.col('CORRECTED_DATA')[row] = uv_corr
                        self.dataTable.col('UVW')[row] = UVW.base2uvw(hourAngle,declination,vis - 69 * 139 + 1, 70)
                        # else:
                        #     if phaseCorrect:
                        #         uv_corr[0,0] *= NP.exp(1j*(-(srhFits.ewAntPhaLcp[freq, vis - 31*97] + srhFits.ewLcpPhaseCorrection[freq, vis - 31*97]) + (srhFits.ewAntPhaLcp[freq, 32] + srhFits.ewLcpPhaseCorrection[freq, 32])))
                        #         uv_corr[0,1] *= NP.exp(1j*(-(srhFits.ewAntPhaRcp[freq, vis - 31*97] + srhFits.ewRcpPhaseCorrection[freq, vis - 31*97]) + (srhFits.ewAntPhaRcp[freq, 32] + srhFits.ewRcpPhaseCorrection[freq, 32])))
                        #     if amplitudeCorrect:
                        #         uv_corr[0,0] /= (srhFits.ewAntAmpLcp[freq, vis - 31*97] * srhFits.ewAntAmpLcp[freq, 32])
                        #         uv_corr[0,1] /= (srhFits.ewAntAmpRcp[freq, vis - 31*97] * srhFits.ewAntAmpRcp[freq, 32])
                        #     self.dataTable.col('ANTENNA1')[row] = vis - 31*97 + 1
                        #     self.dataTable.col('ANTENNA2')[row] = 33
                        #     self.dataTable.col('DATA')[row] = uv
                        #     self.dataTable.col('CORRECTED_DATA')[row] = uv_corr
                        #     self.dataTable.col('UVW')[row] = UVW.base2uvw(hourAngle,declination,vis - 31*97 + 1, 33)
                            
                    self.dataTable.col('SCAN_NUMBER')[row] = scan
                    self.dataTable.col('ARRAY_ID')[row] = 0
                    self.dataTable.col('STATE_ID')[row] = -1
                    self.dataTable.col('PROCESSOR_ID')[row] = -1
                    self.dataTable.col('FEED1')[row] = 0
                    self.dataTable.col('FEED2')[row] = 0
                    self.dataTable.col('WEIGHT')[row] = weight
                    self.dataTable.col('SIGMA')[row] = sigma
                    self.dataTable.col('EXPOSURE')[row] = 0.28
                    self.dataTable.col('INTERVAL')[row] = 0.28
                    self.dataTable.col('FLAG')[row] = flag
                    self.dataTable.col('FLAG_ROW')[row] = False
                    self.dataTable.col('DATA_DESC_ID')[row] = freqInd
                    self.dataTable.col('TIME')[row] = scanDate.mjd*(24.*3600)
                    self.dataTable.col('TIME_CENTROID')[row] = scanDate.mjd*(24.*3600)
        
        self.dataTable.close()           
        
    def initDataTableRedundant(self, srhFits, frequencyChannel = [], phaseCorrect = True, amplitudeCorrect = False):
        declination = srhFits.getDeclination()
        noon = srhFits.RAO.culmination
        
        visibilityNumber = srhFits.visLcp.shape[2]
        freqNumber = len(frequencyChannel)
        
        dataDesc = T.makearrcoldesc('DATA',0.+0j, shape=[1,2],valuetype='complex')
        correctedDataDesc = T.makearrcoldesc('CORRECTED_DATA',0.+0j, shape=[1,2],valuetype='complex')
        self.dataTable.addcols(dataDesc)
        self.dataTable.addcols(correctedDataDesc)
        uv = NP.zeros((1,2),dtype='complex64')
        flag = NP.zeros((1,2),dtype='bool')
        weight = NP.array([100.,100.],dtype='float32')
        sigma = NP.array([.01,.01],dtype='float32')
       
        fitsDate =srhFits.hduList[0].header['DATE-OBS'] + 'T00:00:00';

        for freqInd in range(freqNumber):
            freq = frequencyChannel[freqInd]
            for scan in range(srhFits.dataLength):
                scanDate = Time(fitsDate, format='isot',scale='utc');
                scanTime = srhFits.freqTime[freq, scan]
                scanDate += TimeDelta(scanTime,format='sec')
                hourAngle = NP.deg2rad((scanTime - noon)*15./3600.)
                for vis in range(visibilityNumber):
                    self.dataTable.addrows()
                    row = freqInd*srhFits.dataLength*visibilityNumber + scan*visibilityNumber + vis;
                    uv[0,0] = srhFits.visLcp[freq,scan,vis]
                    uv[0,1] = srhFits.visRcp[freq,scan,vis]
                    antA_ind = srhFits.antennaA[vis]
                    if phaseCorrect:
                        uv[0,0] *= NP.exp(1j*(-(srhFits.ewAntPhaLcp[freq, j]+srhFits.ewLcpPhaseCorrection[freq, j]) + (srhFits.nAntPhaLcp[freq, i] + srhFits.nLcpPhaseCorrection[freq, i])))
                        uv[0,1] *= NP.exp(1j*(-(srhFits.ewAntPhaRcp[freq, j]+srhFits.ewRcpPhaseCorrection[freq, j]) + (srhFits.nAntPhaRcp[freq, i] + srhFits.nRcpPhaseCorrection[freq, i])))
                    if amplitudeCorrect:
                        uv[0,0] /= (srhFits.ewAntAmpLcp[freq, j] * srhFits.sAntAmpLcp[freq, i])
                        uv[0,1] /= (srhFits.ewAntAmpRcp[freq, j] * srhFits.sAntAmpRcp[freq, i])
                    self.dataTable.col('ANTENNA1')[row] = srhFits.antennaA[vis]
                    self.dataTable.col('ANTENNA2')[row] = srhFits.antennaB[vis]
                    self.dataTable.col('DATA')[row] = uv
                    self.dataTable.col('CORRECTED_DATA')[row] = uv
                    self.dataTable.col('UVW')[row] = UVW.base2uvw(hourAngle,declination,srhFits.antennaA[vis], srhFits.antennaB[vis])#, correct_positions = False)

                        
                    self.dataTable.col('SCAN_NUMBER')[row] = scan
                    self.dataTable.col('ARRAY_ID')[row] = 0
                    self.dataTable.col('STATE_ID')[row] = -1
                    self.dataTable.col('PROCESSOR_ID')[row] = -1
                    self.dataTable.col('FEED1')[row] = 0
                    self.dataTable.col('FEED2')[row] = 0
                    self.dataTable.col('WEIGHT')[row] = weight
                    self.dataTable.col('SIGMA')[row] = sigma
                    self.dataTable.col('EXPOSURE')[row] = 0.28
                    self.dataTable.col('INTERVAL')[row] = 0.28
                    self.dataTable.col('FLAG')[row] = flag
                    self.dataTable.col('FLAG_ROW')[row] = False
                    self.dataTable.col('DATA_DESC_ID')[row] = freqInd
                    self.dataTable.col('TIME')[row] = scanDate.mjd*(24.*3600)
                    self.dataTable.col('TIME_CENTROID')[row] = scanDate.mjd*(24.*3600)
                    
                
        self.dataTable.close()
                        
    
    def distFromCenter(self, ant):
        if ant < 140:
            n = NP.abs(ant - 70)
            if n<25:
                baseDist = n
            else:
                baseDist = 26
                baseDist += (n-24)*2
                if n>48:
                    baseDist += (n-48)*2
            return baseDist * NP.sign(ant-70)
        else:
            n = ant - 139
            if n<25:
                baseDist = n
            else:
                baseDist = 24
                baseDist += (n-24)*2
                if n>48:
                    baseDist += (n-48)*2
            return baseDist
        
    def initAntennaTable(self, srhFits):
        self.antennaTable.addrows(208)
        
        for ant in range(139):
            self.antennaTable.col('POSITION')[ant] = (self.x, self.y + self.distFromCenter(ant) * 2.45, self.z)
            self.antennaTable.col('DISH_DIAMETER')[ant] = 2.0
            self.antennaTable.col('TYPE')[ant] = 'GROUND-BASED'
            self.antennaTable.col('MOUNT')[ant] = 'ALTAZ'
            self.antennaTable.col('STATION')[ant] = 'WE%03d' % (ant+1)
            self.antennaTable.col('NAME')[ant] = 'WE%03d' % (ant+1)
            
        for ant in range(69):
            self.antennaTable.col('POSITION')[139 + ant] = (self.x + self.distFromCenter(ant) * 4.9, self.y, self.z)
            self.antennaTable.col('DISH_DIAMETER')[139 + ant] = 2.0
            self.antennaTable.col('TYPE')[139 + ant] = 'GROUND-BASED'
            self.antennaTable.col('MOUNT')[139 + ant] = 'ALTAZ'
            self.antennaTable.col('STATION')[139 + ant] = 'S%03d' % (140 + ant)
            self.antennaTable.col('NAME')[139 + ant] = 'S%03d' % (140 + ant)
        self.antennaTable.close()
    
    def initSpectralWindowTable(self,  srhFits, frequencyChannel = []):
        frequenciesAmount = len(frequencyChannel)
        for i in range(frequenciesAmount):
            self.spectralWindowTable.addrows()
            freqChan = 1e7
            
            frequencies = srhFits.freqList[frequencyChannel[i]]*1e3
            
            self.spectralWindowTable.col('CHAN_FREQ')[i] = frequencies
            self.spectralWindowTable.col('CHAN_WIDTH')[i] = freqChan
            self.spectralWindowTable.col('MEAS_FREQ_REF')[i] = 4
            self.spectralWindowTable.col('EFFECTIVE_BW')[i] = freqChan
            self.spectralWindowTable.col('RESOLUTION')[i] = freqChan
            self.spectralWindowTable.col('TOTAL_BANDWIDTH')[i] = 2.5e7
            self.spectralWindowTable.col('NUM_CHAN')[i] = 1
            self.spectralWindowTable.col('NET_SIDEBAND')[i] = 1
            self.spectralWindowTable.col('REF_FREQUENCY')[i] = frequencies
        self.spectralWindowTable.close()

    def initDataDescriptionTable(self, frequencyChannel = []):
        frequenciesAmount = len(frequencyChannel)
        for i in range(frequenciesAmount):
            self.dataDescriptionTable.addrows()
            self.dataDescriptionTable.col('SPECTRAL_WINDOW_ID')[i] = i
            self.dataDescriptionTable.col('POLARIZATION_ID')[i] = 0
            self.dataDescriptionTable.col('FLAG_ROW')[i] = 0
        self.dataDescriptionTable.close()
        
    def initPolarizationTable(self):
        self.polarizationTable.addrows()
        self.polarizationTable.col('NUM_CORR')[0] = 2
        self.polarizationTable.col('CORR_TYPE')[0] = NP.array([5, 8])
        self.polarizationTable.col('CORR_PRODUCT')[0] = NP.array([[0, 1], [0, 1]])
        self.polarizationTable.close()

    def initSourceTable(self):
        self.sourceTable.addrows()
        self.sourceTable.col('NAME')[0] = 'Sun'
        self.sourceTable.close()

    def initFieldTable(self):
        self.fieldTable.addrows()
        self.fieldTable.col('PHASE_DIR')[0] = NP.array([[1.0], [0.1]])
        self.fieldTable.col('DELAY_DIR')[0] = NP.array([[1.0], [0.1]])
        self.fieldTable.col('REFERENCE_DIR')[0] = NP.array([[1.0], [0.1]])
        self.fieldTable.col('CODE')[0] = 'S'
        self.fieldTable.col('NAME')[0] = 'Sun'
        self.fieldTable.close()

    def initFeedTable(self, srhFits, frequencyChannel = []):
        self.feedTable.addrows(128)
        for ant in range(128):
            self.feedTable.col('POSITION')[ant] = NP.array([0.,0.,0.])
            self.feedTable.col('BEAM_OFFSET')[ant] = NP.array([[0.],[0.]])
            self.feedTable.col('POLARIZATION_TYPE')[ant] = ['R','L']
            self.feedTable.col('POL_RESPONSE')[ant] = NP.array([[1,0],[0,1]],dtype='complex')
            self.feedTable.col('RECEPTOR_ANGLE')[ant] = NP.array([0.,0.])
            self.feedTable.col('ANTENNA_ID')[ant] = ant
            self.feedTable.col('TIME')[ant] = srhFits.freqTime[frequencyChannel[0], 0]
            self.feedTable.col('NUM_RECEPTORS')[ant] = 2
            self.feedTable.col('SPECTRAL_WINDOW_ID')[ant] = 0
        self.feedTable.close()

    def initObservationTable(self, srhFits, observer = 'Olga V. Melnikova'):
        self.observationTable.addrows()
        dateStart = srhFits.hduList[0].header['DATE-OBS']
        dateFinish = srhFits.hduList[0].header['DATE-OBS']
        timeStart = srhFits.hduList[0].header['TIME-OBS'].split('.')[0]
        timeFinish = srhFits.hduList[0].header['TIME-OBS'].split('.')[0]
        
        t_start = Time(dateStart + ' ' + timeStart, scale='utc')
        t_finish = Time(dateFinish + ' ' + timeFinish, scale='utc')
        self.observationTable.col('TELESCOPE_NAME')[0] = 'SRH'
        self.observationTable.col('OBSERVER')[0] = observer
        self.observationTable.col('RELEASE_DATE')[0] = t_start.jd1
        self.observationTable.col('TIME_RANGE')[0] = [t_start.jd1,t_finish.jd1]
        self.observationTable.close()
        
    def antennaName2Index(self, ant):
        ind = -1
        if ant >=49 and ant <=80:
            ind = ant - 49
        if ant >= 176 and ant <=192:
            ind = 192 - ant + 32
        return ind
            