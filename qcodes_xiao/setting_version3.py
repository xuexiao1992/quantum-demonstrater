# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:02:29 2017

@author: X.X
"""
import numpy as np
from scipy.optimize import curve_fit

from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
#from experiment import Experiment
from experiment_version2 import Experiment
from calibration import Calibration
from data_set_plot import convert_to_ordered_data, convert_to_01_state, convert_to_probability

import stationF006
from manipulation import Manipulation
from manipulation_library import Ramsey, Finding_Resonance, Rabi, CRot, AllXY
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

#%% digitizer parameter

class digitizer_param(ArrayParameter):
    
    def __init__(self, name, mV_range, memsize, seg_size, 
                posttrigger_size, label=None, unit=None, instrument=None,
                **kwargs):
        
        global digitizer
        channel_amount = bin(digitizer.enable_channels()).count('1')
       
        super().__init__(name=name, shape=(channel_amount*memsize,), instrument=instrument, **kwargs)
        
        self.mV_range = mV_range
        self.memsize = memsize
        self.seg_size =seg_size
        self.posttrigger_size = posttrigger_size
        
    def get(self):
#        res = digitizer.single_trigger_acquisition(self.mV_range,self.memsize,self.posttrigger_size)
        time.sleep(0.2)
        res = digitizer.multiple_trigger_acquisition(self.mV_range,self.memsize,self.seg_size,self.posttrigger_size)
        
#        res = multiple_trigger_acquisition(digitizer, self.mV_range,self.memsize,self.seg_size,self.posttrigger_size)
#        res = digitizer.single_software_trigger_acquisition(self.mV_range,self.memsize,self.posttrigger_size)
        print(res.shape)
        return res
        
    def __getitem__(self, keys):
        """
        Slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        """
        return SweepFixedValues(self, keys)

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
formatter = HDF5FormatMetadata()
data_IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
#data_location = '2017-08-18/20-40-19_T1_Vread_sweep'
experiment_name = 'Finding_Resonance'
sweep_type = 'Rabi_Sweep'
data_location = time.strftime("%Y-%m-%d/%H-%M-%S") + experiment_name + sweep_type

def data_set_plot(data_set, data_location, sweep_type):
    
    Plot = MatPlot()
    
    threshold = 0.025
    name = 'Rabi'
    
    raw_data_set = load_data(location = data_location, io = data_IO,)
    """
    for 1D sweep outside a sequence, sequence is only one unit sequence
    """
    if sweep_type == 1:
        
#        raw_data_set = load_data(location = data_location, io = data_IO,)
        
        data_set_P = convert_to_probability(raw_data_set, threshold = threshold, name = name)
        
        x_data = data_set_P.arrays['vsg2_frequency_set'].ndarray
        P_data = data_set_P.arrays['digitizer'].ndarray.T[0]
        
        x = x_data
        y = P_data
        
        plt.plot(x, y)
        datacursor()
    """
    for 2D sweep both inside sequence and outside a sequence
    """
    if sweep_type == 2:
#        raw_data_set = load_data(location = data_location, io = data_IO,)
        
        data_set_P = convert_to_probability(raw_data_set, threshold = threshold, name = name)
        
        x_data = data_set_P.arrays['vsg2_frequency_set'].ndarray
        y_data = data_set_P.arrays[name+'_set'].ndarray[0]
        P_data = data_set_P.arrays['digitizer'].ndarray
        
        X, Y = np.meshgrid(x_data, y_data)
        
        plt.pcolor(X,Y,P_data.T)
        datacursor()
    """
    for 1D sweep inside a sequence, no qcodes-loop function
    """
    if sweep_type == 0:
        data_set_P = convert_to_probability(raw_data_set, threshold = threshold, name = name)
        x_data = data_set_P.arrays[name+'_set'].ndarray[0]

        P_data = data_set_P.arrays['digitizer'].ndarray[0]
        
        plt.plot(x_data, P_data)
        datacursor()

        return 0
#%% plot with cursor

#import matplotlib.pyplot as plt
#import numpy as np
#from mpldatacursor import datacursor
'''
data = np.outer(range(10), range(1, 5))

fig, ax = plt.subplots()
lines = ax.plot(data)
ax.set_title('Click somewhere on a line')

datacursor(lines)

plt.show()
'''
#%%

def Func_Sin(x,amp,omega,phase,offset):
    return amp*np.sin(omega*x+phase)+offset


def Func_Gaussian(x, a, x0, ):
#    x_new = x/1e6
    sigma = 1e6
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

#pars, pcov = curve_fit(Func_Sin, x, y,)


#%%

#station = stationF006.initialize()

#pulsar = set_5014pulsar(awg = station.awg, awg2 = station.awg2)

time.sleep(1)
awg = station.awg
awg2 = station.awg2
#    awg.clock_freq(1e9)
#    awg2.clock_freq(1e9)
    
vsg = station.vsg
vsg2 = station.vsg2
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

#%%

#digitizer, dig = set_digitizer(experiment.digitizer)

vsg.frequency(18.392e9)
#vsg.frequency(18.372e9)
vsg2.frequency(19.6521e9)
vsg.power(17.4)
vsg2.power(1.4)

qubit_1.Pi_pulse_length = 250e-9
qubit_2.Pi_pulse_length = 250e-9
qubit_1.halfPi_pulse_length = 125e-9
qubit_2.halfPi_pulse_length = 125e-9

finding_resonance = Finding_Resonance(name = 'Finding_resonance', pulsar = pulsar)

rabi = Rabi(name = 'Rabi', pulsar = pulsar)
crot = CRot(name = 'CRot', pulsar = pulsar, amplitude = 30*0.5*-0.026, frequency_shift = 0.054e9, duration_time = 275e-9)
rabi2 = Rabi(name = 'Rabi2', pulsar = pulsar, qubit = 'qubit_1',)


def reset_experiment():
    experiment.reset()
    
    experiment.qubit_number = 1
    experiment.threshold = 0.025
    experiment.seq_repetition = 100
    
    print('experiment reset')

    return True
#%%        Set Experiment

experiment = Experiment(name = 'ramsey_allxy', label = 'AllXY_sequence', qubits = [qubit_1, qubit_2], 
                        awg = awg, awg2 = awg2, pulsar = pulsar, 
                        vsg = vsg, vsg2 = vsg2, digitizer = digitizer)

calibration = Experiment(name = 'ramsey_calibration', label = 'Ramsey_scan', qubits = [qubit_1, qubit_2], 
                         awg = awg, awg2 = awg2, pulsar = pulsar,
                         vsg = vsg, vsg2 = vsg2, digitizer = digitizer)

exp = Experiment(name = 'tast', label = 'test', qubits = [None, None], None, None)

print('experiment initialized')

#self.digitizer_trigger_channel = 'ch5_marker1'
#self.digitier_readout_marker = 'ch6_marker1'

experiment.qubit_number = 1
experiment.threshold = 0.017
experiment.seq_repetition = 100

calibration.qubit_number = 1
calibration.threshold = 0.017
calibration.seq_repetition = 100

#%%

#experiment.add_measurement('2D_Rabi_Scan', ['Rabi2','CRot'], [rabi2, crot], sequence_cfg, sequence_cfg_type)

#experiment.add_measurement('2D_Rabi_Scan', ['Rabi'], [rabi], sequence_cfg, sequence_cfg_type)
#experiment.add_X_parameter('2D_Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 21), element = 'Rabi')
#experiment.add_X_parameter('2D_Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.015e9, 0.005e9, 21), element = 'Rabi2')
#experiment.add_X_parameter('2D_Rabi_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(0.04e9, 0.06e9, 31), element = 'CRot')

#experiment.add_measurement('2D_Rabi_Scan', ['CRot'], [crot], sequence_cfg, sequence_cfg_type)

#experiment.add_X_parameter('2D_Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.5e-6, 26), element = 'CRot')

#experiment.add_X_parameter(measurement = '2D_Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(18.385e9, 18.40e9, 16))
#experiment.add_X_parameter(measurement = '2D_Rabi_Scan', parameter = vsg2.power, sweep_array = sweep_array(1, 1.8, 17))
#%%        Rabi frequency spectroscopy

#experiment.add_measurement('Rabi_Scan', ['Rabi'], [rabi,], sequence2_cfg, sequence2_cfg_type)
#experiment.add_X_parameter('Rabi_Scan', parameter = 'duration_time', sweep_array = sweep_array(0, 0.6e-6, 31), element = 'Rabi')
#experiment.add_X_parameter(measurement = 'Rabi_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.64e9, 19.660e9, 21))

#%%        Ramsey

ramsey = Ramsey(name = 'Ramsey', pulsar = pulsar, duration_time = 125e-9, waiting_time = 10e-9)

#ramsey2 = Ramsey(name = 'Ramsey2', pulsar = pulsar, duration_time = 125e-9, qubit = 'qubit_1')

experiment.add_measurement('Ramsey_Scan', ['Ramsey'], [ramsey,], sequence2_cfg, sequence2_cfg_type)
#experiment.add_X_parameter('Ramsey_Scan', parameter = 'waiting_time', sweep_array = sweep_array(0, 1e-6, 31), element = 'Ramsey')
#experiment.add_X_parameter(measurement = 'Ramsey_Scan', parameter = vsg2.frequency, sweep_array = sweep_array(19.673e9, 19.676e9, 21))
experiment.add_X_parameter('Ramsey_Scan', parameter = 'frequency_shift', sweep_array = sweep_array(-0.01e9, 0.01e9, 41), element = 'Ramsey')

print('sweep parameter set')

experiment.set_sweep(repetition = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
#calibration.load_sequence()

print('sequence loaded')
time.sleep(0.5)

#%%        AllXY calibration

allxy = AllXY(name = 'AllXY', pulsar = pulsar,)

experiment.add_measurement('AllXY_calibration', ['AllXY'], [allxy,], sequence2_cfg, sequence2_cfg_type)
experiment.add_X_parameter('AllXY_calibration', parameter = 'gate', sweep_array = sweep_array(1, 21.5, 42), element = 'AllXY')
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.frequency, sweep_array = sweep_array(19.667e9, 19.687e9, 11))
#experiment.add_Y_parameter('AllXY_calibration', parameter = vsg2.power, sweep_array = sweep_array(0.9, 1.9, 21))

print('sweep parameter set')

experiment.set_sweep(repetition = True, plot_average = False, count = 5)

print('loading sequence')
experiment.generate_1D_sequence()
#experiment.load_sequence()
print('sequence loaded')
time.sleep(0.5)


#%%

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

"""