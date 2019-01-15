# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:43:14 2018

@author: TUD278306
"""

#%% Experiment reset
def reset_experiment():
    experiment.reset()
    
    experiment.qubit_number = 1
    experiment.threshold = 0.020
    experiment.seq_repetition = 100
    
    print('experiment reset')

    return True

#%% Create experiment

experiment = Experiment(name = 'RB_experiment', label = 'AllXY_sequence', qubits = [qubit_1, qubit_2], 
                        awg = awg, awg2 = awg2, pulsar = pulsar, 
                        vsg = vsg, vsg2 = vsg2, digitizer = digitizer)

calibration = Calibration(name = 'ramsey_calibration', label = 'Ramsey_scan', qubits = [qubit_1, qubit_2], 
                         awg = awg, awg2 = awg2, pulsar = pulsar,
                         vsg = vsg, vsg2 = vsg2, digitizer = digitizer)

print('experiment initialized')

#%% Apply configuration settings
runfile('C:/Github/quantum-demonstrater/qcodes_xiao/experimentConfiguration.py', wdir='C:/Github/quantum-demonstrater/qcodes_xiao')

#%% calibrate readout Q2
'''
experiment.qubit_number = 1
experiment.calibration_qubit = 'qubit_2'

#experiment.add_measurement('Rabi_Scan', ['Finding_resonance'], [finding_resonance,], sequence3_cfg, sequence3_cfg_type)
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.2e9, 20.0e9, 81))

experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi,], sequence3_cfg, sequence3_cfg_type)
experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 51), element = 'Rabi')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(1e-9, 3e-6, 101), element = 'Rabi')

experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 5), with_calibration = False)
#experiment.add_Y_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.475e9, 19.490e9, 31))
#experiment.add_Y_parameter(measurement = 'Rabi_Scan', parameter = vsg2.power, sweep_array = sweep_array(5, 15, 41))

experiment.set_sweep(repetition = False, plot_average = False, count = 1)

'''
'''
% For regular measurement
ds = experiment.data_set
pt = MatPlot()
pt.add(x = ds.index3_set[0,0,0,0,:],y=ds.raw_data[0,0,0,0,:])
for i in range(1,10):
    for j in range(10,20):
        pt.add(x = ds.index3_set[0,0,0,0,:],y=ds.raw_data[0,0,i,j,:])

% For finding resonance scan
ds = experiment.data_set
pt = MatPlot()
pt.add(x = ds.index3_set[0,0,0,0,0,:],y=ds.raw_data[0,0,0,0,0,:])
for i in range(20,30):
    for j in range(10,30):
        pt.add(x = ds.index3_set[0,0,0,0,0,:],y=ds.raw_data[0,i,0,0,j,:])

plot1D(experiment.data_set, measurements = ['Rabi_Scan'], fitfunction = fitcos)

import matplotlib.pyplot as plt
plt.hist(yy.reshape(3100*48,))
'''
#%%     T1 Q2
'''
experiment.qubit_number = 1

experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi,], sequenceT1_cfg2, sequenceT1_cfg2_type)
experiment.add_X_parameter('Rabi_Scan', parameter = 'time', sweep_array = sweep_array(1e-6, 20e-3, 40), element = 'read0_step1')
experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 10, 20),)

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
'''
#%% Measure Q2 as a function of detuing
'''
experiment.add_measurement('Rabi_Scan', ['Rabi2','CRot'], [rabi2_det, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Rabi_Scan', parameter = 'amplitude', sweep_array = sweep_array(30*0.5*-0.023, 30*0.5*-0.033, 21), element = 'Rabi2')

experiment.add_Y_parameter('Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.640e9, 19.655e9, 16))

experiment.set_sweep(repetition = False, plot_average = False, count = 1)
'''
#%% Simultaneous pulse measure Q1 and Q2 frequency Rabi
'''
#experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

experiment.add_measurement('Rabi_Scan', ['Rabi12','CRot'], [rabi12, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 31), element = 'Rabi12')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 1.0e-6, 31), element = 'Rabi12')\
#experiment.add_X_parameter('Rabi_Scan', parameter = 'voltage_1', sweep_array = sweep_array(30*0.5*-0.008, 30*0.5*-0.014, 61), element = 'init_step3')

#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.045e9, 0.075e9, 31), element = 'CRot')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 31), element = 'CRot')

#experiment.add_measurement('Ramsey_Scan2', ['Ramsey12_2','CRot'], [ramsey12, crot], sequence11_cfg, sequence11_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan2', parameter = 'waiting_time', sweep_array = sweep_array(50e-9, 2e-6, 100), element = 'Ramsey12_2')

#experiment.add_measurement('Rabi_Scan2', ['Rabi12','CRot'], [rabi12, crot], sequence11_cfg, sequence11_cfg_type)
#experiment.add_X_parameter('Rabi_Scan2', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi12')

#experiment.add_measurement('Rabi_Scan2', ['Rabi12_2','CRot'], [rabi12, crot], sequence23_cfg, sequence23_cfg_type)
#experiment.add_X_parameter('Rabi_Scan2', parameter = 'duration_time', sweep_array = sweep_array(0, 3.0e-6, 151), element = 'Rabi12_2')

experiment.add_Y_parameter('Rabi_Scan', parameter = Count, sweep_array = sweep_array(1, 5, 5), with_calibration = False)
'''







#%%
print('sweep parameter set')
experiment.set_sweep(repetition = False, plot_average = False, count = 1)
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')