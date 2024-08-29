#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:08:30 2021

@author: maria
"""

import numpy as NP
import pylab as PL
import random
from BadaryRAO import BadaryRAO
from scipy.optimize import least_squares
import base2uvw_1224
from astropy.time import Time, TimeDelta
import time
from srhFitsFile1224 import SrhFitsFile

def real_to_complex(z):
    return z[:len(z)//2] + 1j * z[len(z)//2:]
    
def complex_to_real(z):
    return NP.concatenate((NP.real(z), NP.imag(z)))
    
def allGainsFunc_constrained(x, obsVis, ewAntNumber, sAntNumber, baselineNumber):
    res = NP.zeros_like(obsVis, dtype = complex)
    ewSolarAmp = 1
    sSolarAmp = NP.abs(x[0])
    x_complex = real_to_complex(x[1:])
    
    sAntNumber_c = sAntNumber + 1
    
    ewSolarPhase = 0
    sSolarPhase = 0
    
    sGainsNumber = sAntNumber
    ewGainsNumber = ewAntNumber
    sSolVisNumber = baselineNumber - 1
    ewSolVisNumber = baselineNumber - 1
    ewSolVis = NP.append((ewSolarAmp * NP.exp(1j*ewSolarPhase)), x_complex[: ewSolVisNumber])
    sSolVis = NP.append((sSolarAmp * NP.exp(1j*sSolarPhase)), x_complex[ewSolVisNumber : ewSolVisNumber+sSolVisNumber])
    ewGains = x_complex[ewSolVisNumber+sSolVisNumber : ewSolVisNumber+sSolVisNumber+ewGainsNumber]
    sGains = NP.append(ewGains[69], x_complex[ewSolVisNumber+sSolVisNumber+ewGainsNumber :])
    
    solVisArrayS = NP.array(())
    antAGainsS = NP.array(())
    antBGainsS = NP.array(())
    solVisArrayEW = NP.array(())
    antAGainsEW = NP.array(())
    antBGainsEW = NP.array(())
    for baseline in range(1, baselineNumber+1):
        solVisArrayS = NP.append(solVisArrayS, NP.full(redIndexesS_len[baseline-1], sSolVis[baseline-1]))
        solVisArrayEW = NP.append(solVisArrayEW, NP.full(redIndexesEW_len[baseline-1], ewSolVis[baseline-1]))
        
    antA = antennaA[redIndexesS.astype(int)] - 138
    antB = antennaB[redIndexesS.astype(int)] - 138
    # ind2swap = NP.where(antB < 0)[0]
    # antB[ind2swap] = 0
    # antA[ind2swap], antB[ind2swap] = antB[ind2swap], antA[ind2swap]
    antAGainsS = sGains[antA]
    antBGainsS = sGains[antB]
    
    antAGainsEW = ewGains[antennaA[redIndexesEW.astype(int)]]
    antBGainsEW = ewGains[antennaB[redIndexesEW.astype(int)]]
        
    res = NP.append(solVisArrayEW, solVisArrayS) * NP.append(antAGainsEW, antAGainsS) * NP.conj(NP.append(antBGainsEW, antBGainsS)) - obsVis
    return complex_to_real(res)  

srhFits = SrhFitsFile('/home/maria/Work/SRH imaging/12-24/fits/20230711/srh_1224_20230711T050206.fit', 1025)
baselinesNumber = 8
antNumberEW = 139
antNumberS = 68
antNumber = antNumberEW + antNumberS

N = 2048
arcsecPerPix = 2
radPerPix = NP.deg2rad(arcsecPerPix/3600.)
arcsecRadius = 1020
frequency = 12e9
degRadius = NP.deg2rad(arcsecRadius/3600)
radius = int(arcsecRadius/arcsecPerPix +0.5)
model = NP.zeros((N, N))
for i in range(N):
    for j in range(N):
        x=i - N/2
        y=j - N/2
        if (NP.sqrt(x**2 + y**2) < radius):
            model[i, j] = 1.
#model[1024,1024]= 10000
RAO = BadaryRAO('2021-03-15', 2.45)
gains = NP.ones(antNumber, dtype = 'complex')
for i in range(antNumber):
    gains[i] = random.uniform(1., 2) * NP.exp(1j * random.uniform(-NP.pi/20., NP.pi/20.))
            
ewGains = gains[:antNumberEW]
sGains = gains[antNumberEW:]

# ewGains[16] = 0.001
# ewGains[17] = 0.001

declination = RAO.declination
noon = RAO.culmination

#uvPlane = NP.zeros((128,128),dtype=complex);
uvPlane2 = NP.zeros((1024,1024),dtype=complex);
#uvPlaneCorrected = NP.zeros((1024,1024),dtype=complex);
uvPlanePSF = NP.zeros((1024,1024),dtype=complex);

antennaA = srhFits.antennaA
antennaB = srhFits.antennaB
antY_diff = (srhFits.antY[antennaB] - srhFits.antY[antennaA])/srhFits.base
antX_diff = (srhFits.antX[antennaB] - srhFits.antX[antennaA])/srhFits.base
redIndexesS = NP.array(())
redIndexesEW = NP.array(())
redIndexesS_len = []
redIndexesEW_len = []
for baseline in range(1, srhFits.baselines+1):
    ind = NP.intersect1d(NP.where(NP.abs(antX_diff)==baseline)[0], NP.where(antY_diff == 0)[0])
    redIndexesS = NP.append(redIndexesS, ind)
    redIndexesS_len.append(len(ind))
    ind = NP.intersect1d(NP.where(NP.abs(antY_diff)==baseline)[0], NP.where(antX_diff == 0)[0])
    redIndexesEW = NP.append(redIndexesEW, ind)
    redIndexesEW_len.append(len(ind))



   
fitsDate ='2021-05-14T00:00:00';
scan = 0
baselinesNumber
scanDate = Time(fitsDate, format='isot',scale='utc');
#scanTime = noon
scanTime = 25200.
#scanTime = 28800.
scanDate += TimeDelta(scanTime,format='sec')
hourAngle = NP.deg2rad((scanTime - noon)*15./3600.)
O = 1024//2
FOV = N * radPerPix
x,y = NP.meshgrid(NP.linspace(-.5,.5,N), NP.linspace(-.5,.5,N))
ewSolVis = NP.zeros(baselinesNumber, dtype = 'complex')
sSolVis = NP.zeros(baselinesNumber, dtype = 'complex')
for i in range(baselinesNumber):
    baseline = i+1
    uvw = base2uvw_1224.base2uvw(hourAngle,declination, 69, 69+baseline)
    cos_uv = NP.cos(2. * NP.pi * ((uvw[0]*frequency/3e8)*x + (uvw[1]*frequency/3e8)*y) * FOV)
    sin_uv = NP.sin(2. * NP.pi * ((uvw[0]*frequency/3e8)*x + (uvw[1]*frequency/3e8)*y) * FOV)
    real = NP.sum(cos_uv * model)
    imag = NP.sum(sin_uv * model)
    ewSolVis[i] = (real + imag * 1j)/1e6
    
    uvw = base2uvw_1224.base2uvw(hourAngle,declination, 140, 140+baseline)
    cos_uv = NP.cos(2. * NP.pi * ((uvw[0]*frequency/3e8)*x + (uvw[1]*frequency/3e8)*y) * FOV)
    sin_uv = NP.sin(2. * NP.pi * ((uvw[0]*frequency/3e8)*x + (uvw[1]*frequency/3e8)*y) * FOV)
    real = NP.sum(cos_uv * model)
    imag = NP.sum(sin_uv * model)
    sSolVis[i] = (real + imag * 1j)/1e6

sGains_C0 = NP.append(ewGains[69], sGains)

ewRedundantVis = NP.array(())
sRedundantVis = NP.array(())
noiseLevel = 0.005
phaseNoiseLevel = 0.005

for i in range(baselinesNumber):
    baseline = i+1
    sRedundantVis = NP.append(sRedundantVis, NP.full(redIndexesS_len[i], sSolVis[i]))
    ewRedundantVis = NP.append(ewRedundantVis, NP.full(redIndexesEW_len[i], ewSolVis[i]))
    
ewRedundantVis_gains = ewRedundantVis * gains[antennaA[redIndexesEW.astype(int)]] * NP.conj(gains[antennaB[redIndexesEW.astype(int)]])
sRedundantVis_gains = sRedundantVis * gains[antennaA[redIndexesS.astype(int)]] * NP.conj(gains[antennaB[redIndexesS.astype(int)]])

ew_noise = NP.abs(NP.random.normal(0, noiseLevel, len(ewRedundantVis))) * NP.exp(1j * NP.random.normal(0, phaseNoiseLevel, len(ewRedundantVis)))
s_noise = NP.abs(NP.random.normal(0, noiseLevel, len(sRedundantVis))) * NP.exp(1j * NP.random.normal(0, phaseNoiseLevel, len(sRedundantVis)))
ew_amp_noise = NP.abs(NP.random.normal(0, noiseLevel, len(ewRedundantVis)))
ew_phase_noise = NP.random.normal(0, phaseNoiseLevel, len(ewRedundantVis))
s_amp_noise = NP.abs(NP.random.normal(0, noiseLevel, len(sRedundantVis)))
s_phase_noise = NP.random.normal(0, phaseNoiseLevel, len(sRedundantVis))

ewRedundantVis_gains_noise = (NP.abs(ewRedundantVis_gains) + ew_amp_noise) * NP.exp(1j * (NP.angle(ewRedundantVis_gains) + ew_phase_noise))
sRedundantVis_gains_noise = (NP.abs(sRedundantVis_gains) + s_amp_noise) * NP.exp(1j * (NP.angle(sRedundantVis_gains) + s_phase_noise))

# ewRedundantVis_gains_noise = ewRedundantVis_gains + ew_noise
# sRedundantVis_gains_noise = sRedundantVis_gains + s_noise

redundantVisAll = NP.append(ewRedundantVis_gains_noise, sRedundantVis_gains_noise)

# NONLINEAR

start_time = time.time()

x_size = (baselinesNumber-1)*2 + antNumberEW + antNumberS
x_ini = NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size)))
ls_res = least_squares(allGainsFunc_constrained, x_ini, args = (redundantVisAll, antNumberEW, antNumberS, baselinesNumber), max_nfev = 1500)

print("--- %s seconds ---" % (time.time() - start_time))

gains_calc = real_to_complex(ls_res['x'][1:])[(baselinesNumber-1)*2:]
ew_gains_lcp = gains_calc[:antNumberEW]
ewAntPhaLcp= NP.angle(ew_gains_lcp)
s_gains_lcp = gains_calc[antNumberEW:]
sAntPhaLcp = NP.angle(s_gains_lcp)

ewAntAmpLcp = NP.abs(ew_gains_lcp)#/NP.min(NP.abs(ew_gains_lcp))
# ewAntAmpLcp[freqChannel][ewAntAmpLcp[freqChannel]<NP.median(ewAntAmpLcp[freqChannel])*0.6] = 1e6
sAntAmpLcp = NP.abs(s_gains_lcp)#/NP.min(NP.abs(n_gains_lcp))
# nAntAmpLcp[freqChannel][snAntAmpLcp[freqChannel]<NP.median(nAntAmpLcp[freqChannel])*0.6] = 1e6


# LINEAR

# allAmp = NP.abs(redundantVisAll)
# ampMatrix = NP.abs(phaMatrixGenPairsEWN(baselinesNumber, antNumberEW, antNumberN))

# antAmp, c, d, e = NP.linalg.lstsq(ampMatrix,NP.log(allAmp), rcond=None)
# antAmp= NP.exp(antAmp[baselinesNumber*2:])
# ewAmp = antAmp[:antNumberEW]
# nAmp = antAmp[antNumberEW:]
