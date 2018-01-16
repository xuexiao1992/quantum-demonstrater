# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 18:24:26 2017

@author: think
"""

import math
import numpy as np
from scipy import constants as C

from pycqed.measurement.waveform_control.element import Element
from pycqed.measurement.waveform_control.pulse import CosPulse, SquarePulse, LinearPulse
from pycqed.measurement.waveform_control.sequence import Sequence

from qubit import Qubit
from manipulation import Manipulation
from gate import Single_Qubit_Gate
from experiment_version2 import Experiment

from manipulation_library import Ramsey, Finding_Resonance, Rabi, CRot, AllXY

from scipy.optimize import curve_fit

from pycqed.measurement.waveform_control.pulsar import Pulsar
from experiment_version2 import Experiment
#from calibration import Calibration
from data_set_plot import convert_to_ordered_data, convert_to_01_state, convert_to_probability

import dummy_M4i as M4i
#import qcodes.instrument_drivers.Spectrum.M4i as M4i
#from qcodes.instrument_drivers.Spectrum import pyspcm
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

#from mpldatacursor import datacursor
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


#%% make pulsar

def set_5014pulsar():
    
    pulsar = Pulsar(name = 'PuLsAr',)

    marker1highs = [2, 2, 2, 2, 2, 2, 2, 2]
    for i in range(8):
        pulsar.define_channel(id='ch{}'.format(i + 1),
                              name='ch{}'.format(i + 1), type='analog',
                              high=2, low=-2,
                              offset=0.0, delay=0, active=True,)
        pulsar.define_channel(id='ch{}_marker1'.format(i + 1),
                              name='ch{}_marker1'.format(i + 1),
                              type='marker',
                              high=marker1highs[i], low=0, offset=0.,
                              delay=0, active=True, )
        pulsar.define_channel(id='ch{}_marker2'.format(i + 1),
                              name='ch{}_marker2'.format(i + 1),
                              type='marker',
                              high=2, low=0, offset=0.,
                              delay=0, active=True,)
    return pulsar

#%%
Qubit_1 = Qubit(name = 'qubit_1')

Qubit_2 = Qubit(name = 'qubit_2')

Qubit_1.define_gate(gate_name = 'Microwave1', gate_number = 1, microwave = 1, channel_I = 'ch1', channel_Q = 'ch2', channel_PM = 'ch1_marker1')

Qubit_1.define_gate(gate_name = 'T', gate_number = 3, gate_function = 'plunger', channel_VP = 'ch7')

Qubit_2.define_gate(gate_name = 'Microwave2', gate_number = 4, microwave = 1, channel_I = 'ch3', channel_Q = 'ch4', channel_PM = 'ch1_marker2')

Qubit_2.define_gate(gate_name = 'LP', gate_number = 5, gate_function = 'plunger', channel_VP = 'ch6')
    
Qubit_1.define_neighbor(neighbor_qubit = 'qubit_2', pulse_delay = 10e-9)

Qubit_2.define_neighbor(neighbor_qubit = 'qubit_2', pulse_delay = 0)

qubits = [Qubit_1, Qubit_2]

digitizer = M4i.dummy_M4i(name='digitizer')

pulsar = set_5014pulsar()
#%%


init_cfg = {
        'step1' : set_step(time = 1.5e-3, qubits = qubits, voltages = [30*0.5*0.004, 30*0.5*-0.001]),
        'step2' : set_step(time = 1.5e-3, qubits = qubits, voltages = [30*0.5*-0.004, 30*0.5*0]),
        'step3' : set_step(time = 0.1e-3, qubits = qubits, voltages = [30*0.5*-0.008, 30*0.5*0]),
        'step4' : set_step(time = 4e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        }

manip_cfg = {
        'step1' : set_manip(time = 1e-6, qubits = qubits, voltages = [30*0.5*-0.004,30*0.5*0.016],)
        }

read_cfg = {
        'step1' : set_step(time = 0.1e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step2' : set_step(time = 0.9e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        }
init2_cfg = {
        'step1' : set_step(time = 1e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        }
manip2_cfg = {
        'step1' : set_manip(time = 1.6e-6, qubits = qubits, voltages = [30*0.5*-0.004,30*0.5*0.016],)
        }
read2_cfg = {
        'step1' : set_step(time = 0.1e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        'step2' : set_step(time = 0.9e-3, qubits = qubits, voltages = [30*0.5*0, 30*0.5*0]),
        }

sequence_cfg = [init_cfg, manip_cfg, read_cfg, init2_cfg, manip2_cfg, read2_cfg]         ## the NAME here in this list is not important , only the order matters
sequence_cfg_type = ['init', 'manip','read', 'init2', 'manip2', 'read2']

sequence2_cfg = [init_cfg, manip2_cfg, read_cfg,]
sequence2_cfg_type = ['init', 'manip','read',]

sequence3_cfg = [init_cfg, manip2_cfg, read_cfg,]
sequence3_cfg_type = ['init', 'manip2','read',]

#%%

experiment = Experiment(name = 'tast', label = 'test', qubits = [Qubit_1,Qubit_2], 
                        awg = None, awg2 = None, pulsar=pulsar, digitizer = digitizer)


ramsey = Ramsey(name = 'Ramsey', pulsar = pulsar, duration_time = 125e-9, waiting_time = 10e-9)

#ramsey2 = Ramsey(name = 'Ramsey2', pulsar = pulsar, duration_time = 125e-9, qubit = 'qubit_1')

experiment.add_measurement('Ramsey_Scan', ['Ramsey'], [ramsey,], sequence2_cfg, sequence2_cfg_type)


allxy = AllXY(name = 'AllXY', pulsar = pulsar,)

experiment.add_measurement('AllXY_calibration', ['AllXY'], [allxy,], sequence2_cfg, sequence3_cfg_type)

#experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1e-6, 31), element = 'Ramsey')
#experiment.add_X_parameter(measurement = 'Ramsey_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.673e9, 19.676e9, 21))
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 11), element = 'Ramsey')

print('sweep parameter set')

#experiment.set_sweep(repetition = False, plot_average = False, count = 5)

print('loading sequence')
#experiment.generate_1D_sequence()
#calibration.load_sequence()

print('sequence loaded')
time.sleep(0.5)




experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21.5, 42), element = 'AllXY')
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.frequency, sweep_array = sweep_array(19.667e9, 19.687e9, 11))
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.power, sweep_array = sweep_array(0.9, 1.9, 21))

print('sweep parameter set')

#experiment.set_sweep(repetition = False, plot_average = False, count = 5)

print('loading sequence')
#experiment.generate_1D_sequence()
#experiment.load_sequence()
print('sequence loaded')
time.sleep(0.5)


#%%



"""
x = DS.frequency_shift_set.ndarray
y = DS.digitizerqubit_1.ndarray

pars, pcov = curve_fit(Func_Gaussian, x, y,)
frequency_shift = pars[1]

frequency = vsg2.frequency()+frequency_shift
#vsg2.frequency(frequency)
#
#experiment.plot_average_probability()
#experiment.plot_save()
"""










#%%
"""
Manip_1 = Manipulation(name = 'Manip_1', qubits_name = ['Qubit_1', 'Qubit_2'])

Manip_1.add_single_qubit_gate(name = 'X1_Q1', qubit = Qubit_1)

Manip_1.add_single_qubit_gate(name = 'X1_Q2', refgate = 'X1_Q1', qubit = Qubit_2)

Manip_1.add_X(name = 'X2_Q1', refgate = 'X1_Q2', qubit = Qubit_1)

Manip_1.add_Z(name = 'Z1_Q1', refgate = 'X2_Q1', qubit = Qubit_1, degree = np.pi/4)

Manip_1.add_X(name = 'X3_Q1', refgate = 'X2_Q1', qubit = Qubit_1)

"""