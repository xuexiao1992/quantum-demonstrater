# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 14:22:26 2018

@author: jmboter
"""

#%% importing packages
#
##more general packages:
#import numpy as np
#import os
#import time
#from datetime import datetime, timedelta
#import matplotlib.pyplot as plt
#from qcodes.instrument.parameter import Parameter
#from pycqed.measurement.waveform_control.pulsar import Pulsar #version which is on measurement computer in F006!
#from qcodes.plots.qcmatplotlib import MatPlot#, QtPlot
#
##from projects.calibration_2qubit.functions.twoqubitcalibrationfunctions import fit_resonance_freq
#
#from qtt.measurements.scans import scanjob_t, scan1D, scan2D
#from qtt.tools import addPPTslide
#
#from experiment_version2 import Experiment
#from manipulation_library import Rabi,Ramsey,Rabi_all,CRot
from manipulation_library_Jelmer import ChargeNoiseBob_Jelmer2
#from plot_functions import plot1D

from shutil import copyfile

#%% Define measurements types

# This is done in the regular script before starting the scripted overnight measurements.

#%% Define experiment parameters
ExpParams = []
for DFS in [False, True]:
    ExpParam = [DFS]
    ExpParams.append(ExpParam)

#for correlated in [True, False]:
#    for DFS in [False, True]:
#        for sigma in np.arange(0.2E6, 0.41E6, 0.2E6):
#            ExpParam = [DFS, 'None', True, correlated, sigma]
#            ExpParams.append(ExpParam)

#ExpParamsTest = [[False, 'None', False, True, 0E6]]

#%% Define the experiments
def setExperimentSettings(experiment):
    experiment.qubit_number = 2
    experiment.readnames = ['Qubit2', 'Qubit1']
    
    experiment.calibration_qubit = 'all'
    experiment.saveraw = True
    
    experiment.readout_time = 0.0018
    experiment.threshold = 0.040
    
    off_resonance_amplitude = 1.1;
    
    print("Experiment settings are set.")
    
    return off_resonance_amplitude
    
def newRabiExperiment():
    experiment = Experiment(name = 'NoiseCorrelations', label = 'Rabi', qubits = [qubit_1, qubit_2], #name + label will determine folder on k-drive
                            awg = awg, awg2 = awg2, pulsar = pulsar, vsg = vsg, vsg2 = vsg2, digitizer = digitizer)
    
    off_resonance_amplitude = setExperimentSettings(experiment)
    experiment.seq_repetition = 100
    
    experiment.add_measurement('Rabi_Scan', ['Rabi12','CRot'], [rabi12, crot], sequence1_cfg, sequence1_cfg_type)
    experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 31), element = 'Rabi12')
    
    experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 3))
    
    print('sweep parameter set')
    experiment.set_sweep(repetition = False, plot_average = False, count = 1)
    print('loading sequence')
    experiment.generate_1D_sequence()
    experiment.load_sequence()
    print('sequence loaded')
    
    return experiment

def newNoiseExperiment(DFS = False, add_dephase = False, sigma = 0.0E6, correlated = True, DD = 'None', project = False):
#    label = 'DFS_{0}'.format(DFS)
    experiment = Experiment(name = 'NoiseCorrelations', label = '', qubits = [qubit_1, qubit_2], #name + label will determine folder on k-drive
                            awg = awg, awg2 = awg2, pulsar = pulsar, vsg = vsg, vsg2 = vsg2, digitizer = digitizer)
    
    off_resonance_amplitude = setExperimentSettings(experiment)
        
    # Modify experiment with/without DD
    if DD == 'None':
        det_freq = 6E6
        evolve_time = 0.8E-6
    else:
        det_freq = 1E6
        evolve_time = 5E-6
    
    # Modify expiment with/without added noise
    if add_dephase:
        experiment.seq_repetition = 15
        sweeps = 100
        det_freq = 12E6
        evolve_time = 0.4E-6
    else:
        experiment.seq_repetition = 60
        sweeps = 50
    
    # CPhase settings
    AMP_T = 30*0.5*-0.02615
    AMP_LP = 30*0.5*-0.006
    detuning_time = 90E-9;
    phase_1 = 167 #for target = 'qubit_1'
    phase_2 = 280 #for target = 'qubit_2'
    
    # Define noise experiment
    CN_manip = ChargeNoiseBob_Jelmer2(name = 'Charge_Noise', pulsar = pulsar, phase_1 = phase_1, phase_2 = phase_2, amplitude_control = AMP_T,
               amplitude_target = AMP_LP, detuning_time = detuning_time, off_resonance_amplitude = off_resonance_amplitude, det_freq = det_freq,
               DFS = DFS, add_dephase = add_dephase, sigma = sigma, correlated = correlated, DD = DD, project = project)
    
    # Ramsey frequency update
    experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
    experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')
    
    # Charge noise experiment
    experiment.add_measurement('CN', ['Charge_Noise', 'CRot'], [CN_manip, crot], sequence1_cfg, sequence1_cfg_type)
    experiment.add_X_parameter('CN', parameter = 'waiting_time', sweep_array = sweep_array(1e-9, evolve_time, 81), element = 'Charge_Noise')
    #experiment.add_X_parameter('CN', parameter = 'frequency_shift', sweep_array = sweep_array(-1e6, 1e6, 31), element = 'Charge_Noise')
    
    # Rabi normalisation measurement
    experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
    experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')
    
    
    experiment.add_Y_parameter('CN', parameter = Count, sweep_array = sweep_array(1, 10, sweeps), with_calibration = True)
    #experiment.add_Y_parameter('CN', parameter = 'DFS', sweep_array = sweep_array(0, 39, 40), element = 'Charge_Noise', with_calibration = True)
    
    experiment.set_sweep(repetition = False, plot_average = False, count = 1)
    
    print('loading sequence')
    experiment.generate_1D_sequence()
    experiment.load_sequence()
    print('sequence loaded')
    
    return experiment

#%% Run experiments

repetitions = 10
i = 0

while i < repetitions:
    for ExpParam in ExpParams:
        experiment = newNoiseExperiment(DFS = ExpParam[0]) #initialize noise experiment
        dataset = experiment.run_experiment() #run the noise experiment
        copyfile('C:\\Github\\quantum-demonstrater\\qcodes_xiao\\NoiseCorrelations_overnight.py' , experiment.full_path+'\\NoiseCorrelations_overnight.py')
        plot1D(dataset)
        
        time.sleep(0.5)
        
        experiment = newRabiExperiment() #initialize frequency scan to check calibration
        dataset = experiment.run_experiment() #run the frequency measurement
        freqs_rabi, result_rabi = fit_resonance_freq(dataset=dataset, tag='Rabi', plot=1, ppt=True)
        
        if result_rabi['Fit param qubit 1'] == 'Fitting failed' or result_rabi['Fit param qubit 2'] == 'Fitting failed':
            print('Fitting failed. Measurement aborted. Last measurement: {0} in repetition {1}.'.format(ExpParam,i))
            i = repetitions;
            break
        else:
            vsg.frequency.increment(freqs_rabi[1])
            vsg2.frequency.increment(freqs_rabi[0])
        
        i += 1
        
        time.sleep(0.5)
        
        plt.close('all')
        
        time.sleep(0.5)