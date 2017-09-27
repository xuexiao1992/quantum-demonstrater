# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:02:29 2017

@author: X.X
"""
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
#from experiment import Experiment
from experiment_version2 import Experiment
from calibration_version2 import Calibration
from data_set_plot import convert_to_ordered_data, convert_to_01_state, convert_to_probability

import stationF006
from manipulation import Manipulation
from manipulation_library import Ramsey, Finding_Resonance, Rabi, CRot, AllXY, Ramsey_all, AllXY_all, CPhase_Calibrate, Charge_Noise, DCZ
#from digitizer_setting import digitizer_param

import qcodes.instrument_drivers.Spectrum.M4i as M4i
from qcodes.instrument_drivers.Spectrum import pyspcm
from qcodes.instrument.parameter import ArrayParameter, StandardParameter
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.loops import Loop, ActiveLoop
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.data.data_set import new_data, DataSet,load_data
from qcodes.data.data_array import DataArray
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from qcodes.plots.qcmatplotlib import MatPlot
from qcodes.plots.pyqtgraph import QtPlot

from mpldatacursor import datacursor
import sys
sys.path.append('C:\\Users\\LocalAdmin\\Documents\\GitHub\\PycQED_py3\\pycqed\\measurement\\waveform_control')
import pulsar as ps
import element as ele
import time

#%% make experiment

def set_step(time = 0, qubits = [], voltages = [], **kw):

    if len(qubits)!=len(voltages):
        raise ValueError('qubits should be same length with voltages')

    step = {'time' : time}

    for i in range(len(qubits)):
        qubit = qubits[i]
        step['voltage_%d'%(i+1)] = voltages[i]

    return step


def set_manip(time = 0, qubits = [], voltages = [], **kw):

    if len(qubits)!=len(voltages):
        raise ValueError('qubits should be same length with voltages')

    parameter1 = kw.pop('parameter1', 0)
    parameter2 = kw.pop('parameter2', 0)
    manip = kw.pop('manip_elem', 0)

    step = {'time' : time}
    step.update(kw)

    step['manip_elem'] = manip
    for i in range(len(qubits)):
        qubit = qubits[i]
        step['voltage_%d'%(i+1)] = voltages[i]

        step['parameter1'] = parameter1
        step['parameter2'] = parameter2

    return step

#%%  Sweep

def sweep_array(start, stop, points):

    sweep_array = np.linspace(start, stop, points)

    return list(sweep_array)

#%%
def make_manipulation_cfg():

    manipulation_cfg = {
            'gate1': ['X','Y'],
            'gate2': ['Y','X'],
            'gate3': ['CPhase'],
            'gate4': ['Z', 'X']
            }

    return manipulation_cfg

#%% make pulsar

def set_5014pulsar(awg, awg2):
    
    awg = awg.name
    awg2 = awg2.name
    pulsar = Pulsar(name = 'PuLsAr', default_AWG = awg, master_AWG = awg)

    marker1highs = [2, 2, 2, 2, 2, 2, 2, 2]
    for i in range(8):
        pulsar.define_channel(id='ch{}'.format(i%4 + 1),
                              name='ch{}'.format(i + 1), type='analog',
                              high=2, low=-2,
                              offset=0.0, delay=0, active=True, AWG = awg if i<4 else awg2)
        pulsar.define_channel(id='ch{}_marker1'.format(i%4 + 1),
                              name='ch{}_marker1'.format(i + 1),
                              type='marker',
                              high=marker1highs[i], low=0, offset=0.,
                              delay=0, active=True, AWG = awg if i<4 else awg2)
        pulsar.define_channel(id='ch{}_marker2'.format(i%4 + 1),
                              name='ch{}_marker2'.format(i + 1),
                              type='marker',
                              high=2, low=0, offset=0.,
                              delay=0, active=True, AWG = awg if i<4 else awg2)
    return pulsar

#%%

def Func_Sin(x,amp,omega,phase,offset):
    return amp*np.sin(omega*x+phase)+offset


def Func_Gaussian(x, a, x0, ):
    sigma = 1e6
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

#pars, pcov = curve_fit(Func_Sin, x, y,)

#%%

#station = stationF006.initialize()

#pulsar = set_5014pulsar(awg = station.awg, awg2 = station.awg2)
vsg = station.vsg
vsg2 = station.vsg2

#vsg.frequency(18.390587130383e9)
#vsg2.frequency(19.6754e9)
#vsg2.frequency(19.6754e9-1.95871e6)
#vsg.power(17.31)
#vsg2.power(1.4)

time.sleep(1)
awg = station.awg
awg2 = station.awg2
#    awg.clock_freq(1e9)
#    awg2.clock_freq(1e9)
    
digitizer = station.digitizer
#    awg.ch3_amp

qubit_1 = station.qubit_1
qubit_2 = station.qubit_2

qubits = [qubit_1, qubit_2]

    
init_cfg = {
        'step1' : set_step(time = 1.5e-3, qubits = qubits, voltages = [30*0.5*0.004, 30*0.5*-0.001]),
        'step2' : set_step(time = 1.5e-3, qubits = qubits, voltages = [30*0.5*-0.004, 30*0.5*0]),
        'step3' : set_step(time = 0.1e-3, qubits = qubits, voltages = [30*0.5*-0.008, 30*0.5*0]),
        'step4' : set_step(time = 4e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        }

manip_cfg = {
        'step1' : set_manip(time = 1.5e-6, qubits = qubits, voltages = [30*0.5*-0.004,30*0.5*0.016],)
        }

read_cfg = {
        'step1' : set_step(time = 0.05e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step2' : set_step(time = 0.95e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        }
init2_cfg = {
        'step1' : set_step(time = 1e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        }
manip2_cfg = {
        'step1' : set_manip(time = 1e-6, qubits = qubits, voltages = [30*0.5*-0.004,30*0.5*0.016],)
        }
read2_cfg = {
        'step1' : set_step(time = 0.05e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step2' : set_step(time = 0.95e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        }

read3_cfg = {
        'step1' : set_step(time = 0.05e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step2' : set_step(time = 0.95e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step3' : set_step(time = 0.5e-3, qubits = qubits, voltages = [30*0.5*0.004, 30*0.5*0.00]),
        }


manip3_cfg = deepcopy(manip2_cfg)
manip4_cfg = deepcopy(manip2_cfg)

sequence_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence_cfg_type = ['init', 'manip','read', 'init2', 'manip2', 'read2']

sequence1_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence1_cfg_type = ['init', 'manip21','read', 'init2', 'manip22', 'read2']

sequence2_cfg = [init_cfg, manip2_cfg, read_cfg,]
sequence2_cfg_type = ['init', 'manip','read',]

sequence3_cfg = [init_cfg, manip2_cfg, read_cfg,]
sequence3_cfg_type = ['init', 'manip2','read',]

sequenceBill_cfg = [init_cfg, manip_cfg, manip2_cfg, read3_cfg, manip3_cfg, read3_cfg, manip4_cfg, read3_cfg]
sequenceBill_cfg_type = ['init', 'manip', 'manip2', 'read', 'manip3', 'read2', 'manip4', 'read3']

sequenceBill1_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg, read2_cfg]    
sequenceBill1_cfg_type = ['init', 'manip21','read', 'init2', 'manip22', 'read2', 'read3']

#%%

#digitizer, dig = set_digitizer(experiment.digitizer)


qubit_1.Pi_pulse_length = 250e-9
qubit_2.Pi_pulse_length = 250e-9
qubit_1.halfPi_pulse_length = 125e-9
qubit_2.halfPi_pulse_length = 125e-9
qubit_2.CRot_pulse_length = 286e-9
def reset_experiment():
    experiment.reset()
    
    experiment.qubit_number = 1
    experiment.threshold = 0.017
    experiment.seq_repetition = 100
    
    print('experiment reset')

    return True
#%%        Set Experiment

experiment = Experiment(name = 'allxy_experiment', label = 'AllXY_sequence', qubits = [qubit_1, qubit_2], 
                        awg = awg, awg2 = awg2, pulsar = pulsar, 
                        vsg = vsg, vsg2 = vsg2, digitizer = digitizer)

calibration = Calibration(name = 'ramsey_calibration', label = 'Ramsey_scan', qubits = [qubit_1, qubit_2], 
                         awg = awg, awg2 = awg2, pulsar = pulsar,
                         vsg = vsg, vsg2 = vsg2, digitizer = digitizer)


print('experiment initialized')

#self.digitizer_trigger_channel = 'ch5_marker1'
#self.digitier_readout_marker = 'ch6_marker1'

experiment.qubit_number = 2
experiment.threshold = 0.021
experiment.seq_repetition = 100

calibration.qubit_number = 1
calibration.threshold = 0.013
calibration.seq_repetition = 100

ramsey = Ramsey(name = 'Ramsey', pulsar = pulsar, duration_time = 125e-9, waiting_time = 300e-9)

ramsey2 = Ramsey(name = 'Ramsey2', pulsar = pulsar, qubit = 'qubit_1', duration_time = 125e-9, waiting_time = 300e-9)

allxy = AllXY(name = 'AllXY', pulsar = pulsar,)

allxy2 = AllXY(name = 'AllXY2', pulsar = pulsar, qubit = 'qubit_1')


ramsey12 = Ramsey_all(name = 'Ramsey2', pulsar = pulsar, qubit = 'qubit_1', off_resonance_amplitude = 1.2, 
                      duration_time = 125e-9, waiting_time = 300e-9,)
allxy12 = AllXY_all(name = 'AllXY2', pulsar = pulsar, qubit = 'qubit_1',)

finding_resonance = Finding_Resonance(name = 'Finding_resonance', pulsar = pulsar)

rabi = Rabi(name = 'Rabi', pulsar = pulsar)
crot = CRot(name = 'CRot', pulsar = pulsar, amplitude = 30*0.5*-0.0267, frequency_shift = 0.0580536e9, duration_time = 261.677e-9)
rabi2 = Rabi(name = 'Rabi2', pulsar = pulsar, amplitude = 1, qubit = 'qubit_1',)
crot_freq_bare = 0.0450677e9
crot_freq = 0.0566355e9
q2_shift = 1.77419e6
#qubit_1.frequency = vsg.frequency()
#qubit_2.frequency = vsg2.frequency()
#q1f = vsg.frequency()
#q2f = vsg2.frequency()

#%%     Bill Coish
"""
experiment.qubit_number = 3

experiment.add_measurement('Ramsey_Scan', ['Ramsey2','CRot',], [ramsey2, crot,], sequenceBill1_cfg, sequenceBill1_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey2')


experiment.add_measurement('Bill_Coish', ['Rabi2', 'CRot', 'CRot', 'CRot'], [rabi2, crot, crot, crot], sequenceBill_cfg, sequenceBill_cfg_type)
experiment.add_X_parameter('Bill_Coish', parameter = 'amplitude', sweep_array = sweep_array(0, 1, 2), element = 'Rabi2')

print('sweep parameter set')

experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 10)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)
"""

#%%     CPhase
"""
cphase = CPhase_Calibrate(name = 'CPhase', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude = 30*0.5*-0.023, 
                          detuning_time = 71.5e-9, phase = 0, off_resonance_amplitude = 1.2)

experiment.calibration_qubit = 'all'

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('CPhase_Calibration', ['CPhase','CRot'], [cphase, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('CPhase_Calibration', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 100e-9, 50), element = 'CPhase')
#experiment.add_X_parameter('CPhase_Calibration', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'CPhase')
print('sweep parameter set')
#experiment.set_sweep(repetition = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)
"""
#%%        DCZ
"""
dcz = DCZ(name = 'DCZ', pulsar = pulsar, Pi_amplitude = 0, detuning_amplitude = 30*0.5*-0.026, detuning_time = 72e-9, phase = 90, off_resonance_amplitude = 1.2)
experiment.calibration_qubit = 'all'

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('DCZ', ['DCZ','CRot'], [dcz, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('DCZ', parameter = 'detuning_time', sweep_array = sweep_array(2e-9, 100e-9, 50), element = 'DCZ')
#experiment.add_X_parameter('DCZ', parameter = 'phase', sweep_array = sweep_array(0, 360, 21), element = 'DCZ')


print('sweep parameter set')
#experiment.set_sweep(repetition = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)
"""
#%%     charge noise
"""
charge_noise = Charge_Noise(name = 'Charge_Noise', pulsar = pulsar, Pi_amplitude = 1, detuning_amplitude = 30*0.5*-0.024, detuning_time = 50e-9, phase = 0)
experiment.calibration_qubit = 'all'

experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')


experiment.add_measurement('Charge_Noise', ['Charge_Noise','CRot'], [charge_noise, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('Charge_Noise', parameter = 'detuning_time', sweep_array = sweep_array(0, 60e-9, 31), element = 'Charge_Noise')
#experiment.add_Y_parameter('Charge_Noise', parameter = 'voltage_1', sweep_array = sweep_array(30*0.5*-0.010, 30*0.5*-0.004, 7), element = 'Charge_Noise')

print('sweep parameter set')
#experiment.set_sweep(repetition = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
"""

#%%        Two Qubit Gate

#experiment.add_measurement('Ramsey_Scan', ['Ramsey2','CRot'], [ramsey2, crot], sequence_cfg, sequence_cfg_type)
##
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1e-6, 31), element = 'Ramsey2')
#experiment.add_Y_parameter('Ramsey_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.386e9, 18.396e9, 51))
"""
experiment.calibration_qubit = 'all'
experiment.add_measurement('Ramsey_Scan', ['Ramsey12','CRot'], [ramsey12, crot], sequence_cfg, sequence_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.001e9, 0.001e9, 11), element = 'Ramsey12')

#experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1e-6, 31), element = 'Ramsey2')
#experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.386e9, 18.396e9, 21))
#

experiment.add_measurement('AllXY_calibration', ['AllXY12', 'CRot'], [allxy12, crot], sequence1_cfg, sequence1_cfg_type)
experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21.5, 42), element = 'AllXY12')
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.frequency, sweep_array = sweep_array(19.667e9, 19.687e9, 11))
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.power, sweep_array = sweep_array(0.9, 1.9, 21))


print('sweep parameter set')
#experiment.set_sweep(repetition = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
time.sleep(0.5)

"""
#%%     CRot
"""
experiment.qubit_number = 1

experiment.add_measurement('2D_Rabi_Scan', ['CRot'], [crot], sequence2_cfg, sequence2_cfg_type)

experiment.add_X_parameter('2D_Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.04e9, 0.06e9, 31), element = 'CRot')
#experiment.add_X_parameter('2D_Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 31), element = 'CRot')

#experiment.add_X_parameter(measurement = '2D_Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.385e9, 18.40e9, 16))
#experiment.add_X_parameter(measurement = '2D_Rabi_Scan', parameter = vsg.power, sweep_array = sweep_array(17.25, 17.35, 51))
print('sweep parameter set')
experiment.set_sweep(repetition = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')
"""
#%%     Ramsey and AllXY

experiment.add_measurement('Rabi_Scan', ['Rabi2','CRot'], [rabi2, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 31), element = 'Rabi2')
#experiment.add_X_parameter('Rabi_Scan', parameter = vsg.frequency, sweep_array = sweep_array(18.388e9, 18.393e9, 11))
experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.04e9, 0.06e9, 31), element = 'CRot')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.03e9, 0.03e9, 31), element = 'Rabi2')
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 31), element = 'CRot')

#experiment.calibration_qubit = 'qubit_1'
#
#experiment.add_measurement('Ramsey_Scan', ['Ramsey2','CRot'], [ramsey2, crot], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.005e9, 0.005e9, 51), element = 'Ramsey2')

#experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1e-6, 31), element = 'Ramsey2')
#

#experiment.add_measurement('AllXY_calibration', ['AllXY2', 'CRot'], [allxy2, crot], sequence1_cfg, sequence1_cfg_type)
#experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21.5, 42), element = 'AllXY2')
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.frequency, sweep_array = sweep_array(19.667e9, 19.687e9, 11))
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.power, sweep_array = sweep_array(0.9, 1.9, 21))


print('sweep parameter set')
experiment.set_sweep(repetition = True, plot_average = False, count = 5)
##experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 5)
#
print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()

print('sequence loaded')

#%%        Rabi frequency spectroscopy

experiment.qubit_number = 1
experiment.threshold = 0.021
experiment.seq_repetition = 100

experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi,], sequence2_cfg, sequence2_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 21), element = 'Rabi')
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.678e9, 19.683e9, 11))
experiment.add_X_parameter('Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.03e9, 0.03e9, 41), element = 'Rabi')

experiment.set_sweep(repetition = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
time.sleep(0.5)

#%%        AllXY calibration   qubit_2
"""
experiment.qubit_number = 1
experiment.threshold = 0.021
experiment.seq_repetition = 100

experiment.calibration_qubit = 'qubit_2'

experiment.add_measurement('Ramsey_Scan', ['Ramsey'], [ramsey,], sequence3_cfg, sequence3_cfg_type)
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.003e9, 0.003e9, 21), element = 'Ramsey')


#experiment.add_measurement('AllXY_calibration', ['AllXY'], [allxy,], sequence2_cfg, sequence2_cfg_type)
#experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21.5, 42), element = 'AllXY')
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.frequency, sweep_array = sweep_array(19.667e9, 19.687e9, 11))
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.power, sweep_array = sweep_array(0.9, 1.9, 21))

print('sweep parameter set')

#experiment.set_sweep(repetition = True, with_calibration = True, plot_average = False, count = 5)
experiment.set_sweep(repetition = True, plot_average = False, count = 3)

print('loading sequence')
experiment.generate_1D_sequence()
experiment.load_sequence()
print('sequence loaded')
time.sleep(0.5)
"""
#%%
"""
calibration.run_experiment()

calibration.plot_average_probability()

DS = calibration.averaged_data

x = DS.frequency_shift_set.ndarray
y = DS.digitizerqubit_1.ndarray

pars, pcov = curve_fit(Func_Gaussian, x, y,)
frequency_shift = pars[1]

frequency = vsg2.frequency()+frequency_shift
vsg2.frequency(frequency)
#
#experiment.plot_average_probability()
#experiment.plot_save()

calibration.close()
reset_experiment()
"""
#%%        Active calibration

#experiment.add_measurement('Active_Calibration', ['Ramsey'], [ramsey], sequence2_cfg, sequence2_cfg_type)
#experiment.add_X_parameter('Active_Calibration',)

#%%
"""
print('sweep parameter set')

experiment.set_sweep(repetition = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
print('sequence loaded')

#experiment.load_sequence()
time.sleep(1)
#data_set = experiment.run_experiment()

experiment.convert_to_probability()
experiment.calculate_average_data()
experiment.plot_average_probability()

"""