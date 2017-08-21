# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:02:29 2017

@author: X.X
"""
import numpy as np
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
#from experiment import Experiment
from experiment_version3 import Experiment

import stationF006
from manipulation import Manipulation
from manipulation_library import Ramsey
#from digitizer_setting import digitizer_param

import qcodes.instrument_drivers.Spectrum.M4i as M4i
from qcodes.instrument_drivers.Spectrum import pyspcm
from qcodes.instrument.parameter import ArrayParameter, StandardParameter
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.loops import Loop, ActiveLoop
from qcodes.data.hdf5_format import HDF5Format, HDF5FormatMetadata
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.data.data_set import new_data, DataSet
from qcodes.data.data_array import DataArray

import sys
sys.path.append('C:\\Users\\LocalAdmin\\Documents\\GitHub\\PycQED_py3\\pycqed\\measurement\\waveform_control')
import pulsar as ps
import element as ele


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

    microwave_frequencies = kw.pop('microwave_ferquencies', [qubit.frequency for qubit in qubits])
    microwave_powers = kw.pop('amplitudes', [qubit.microwave_power for qubit in qubits])
    Pi_pulse_lengths = kw.pop('Pi_pulse_lengths', [qubit.Pi_pulse_length for qubit in qubits])
    IQ_amplitudes = kw.pop('IQ_amplitudes', [qubit.IQ_amplitude for qubit in qubits])
    IQ_frequencies = kw.pop('IQ_frequencies', [qubit.IQ_frequency for qubit in qubits])
    waiting_time = kw.pop('waiting_time', 0)
    duration_time = kw.pop('duration_time', 0)
    parameter1 = kw.pop('parameter1', 0)
    parameter2 = kw.pop('parameter2', 0)

    step = {'time' : time}

    for i in range(len(qubits)):
        qubit = qubits[i]
        step['voltage_%d'%(i+1)] = voltages[i]
        step['microwave_frequency_%d'%(i+1)] = microwave_frequencies[i]
        step['microwave_power_%d'%(i+1)] = microwave_powers[i]
        step['Pi_pulse_length_%d'%(i+1)] = Pi_pulse_lengths[i]
        step['IQ_amplitude_%d'%(i+1)] = IQ_amplitudes[i]
        step['IQ_frequency_%d'%(i+1)] = IQ_frequencies[i]
        step['waiting_time'] = waiting_time
        step['duration_time'] = duration_time
        step['parameter1'] = parameter1
        step['parameter2'] = parameter2

    return step



#%%  Sweep

def sweep_array(start, stop, points):

    sweep_array = np.linspace(start, stop, points)

    return list(sweep_array)


#%% make manipulation
#def make_manipulation(manipulation = Manipulation(name = 'Manip'), qubits = [], **kw):
#
#    waiting_time = kw.pop('waiting_time', None)
#    amplitude = kw.pop('amplitude', None)
#
#    manip = make_Ramsey(manipulation = manipulation, qubits = qubits, waiting_time = waiting_time)
#
#    return manip
#


#%%
def make_manipulation_cfg():

    manipulation_cfg = {
            'gate1': ['X','Y'],
            'gate2': ['Y','X'],
            'gate3': ['CPhase'],
            'gate4': ['Z','X']
            }

    return manipulation_cfg

#%%

def make_experiment_cfg():

    station = stationF006.initialize()
    awg = station.awg
    awg2 = station.awg2
    awg.clock_freq(1e9)
    awg2.clock_freq(1e9)

    digitizer = station.digitizer
#    awg.ch3_amp
    pulsar = set_5014pulsar(awg = awg, awg2= awg2)

    qubit_1 = station.qubit_1
    qubit_2 = station.qubit_2

    qubits = [qubit_1, qubit_2]

    experiment = Experiment(name = 'experiment_test', qubits = [qubit_1, qubit_2], awg = awg, awg2 = awg2, pulsar = pulsar)

    experiment.sweep_loop1 = {
#            'para1': [0.8,0.2,0.53,0.14,0.3],
            'para1': sweep_array(start = 50e-9, stop = 250e-9, points = 5),
            'para2': sweep_array(start = 0.1, stop = 0.5, points = 5),
            }

    experiment.sweep_loop2 = {
            'para1': [0.4,0.5,0.3,0.8,0.5],
            }

#    loop1_para1 = [1,2,344,553,3]
#    loop1_para2 = [33,2,11,22,3]
#    loop2_para1 = [1,2,3,4]

    loop1_para1 = 'loop1_para1'
    loop1_para2 = 'loop1_para2'
    loop2_para1 = 'loop2_para1'
    
    init_cfg = {
            'step1' : set_step(time = 2e-6, qubits = qubits, voltages = [0.2, 0.8]),
            'step2' : set_step(time = 5e-6, qubits = qubits, voltages = [loop2_para1, 0.3]),
            'step3' : set_step(time = 1e-6, qubits = qubits, voltages = [0.4, 0.5]),
#            'step4' : set_step(time = 1000e-6, qubits = qubits, voltages = [0.4, 0.5]),
#            'step5' : set_step(time = 500e-6, qubits = qubits, voltages = [0.4, 0.5]),
#            'step6' : set_step(time = 2000e-6, qubits = qubits, voltages = [0.4, 0.5]),
#            'step7' : set_step(time = 1e-6, qubits = qubits, voltages = [0.4, 0.5]),
#            'step8' : set_step(time = 1000e-6, qubits = qubits, voltages = [0.4, 0.5]),
#            'step9' : set_step(time = 500e-6, qubits = qubits, voltages = [0.4, 0.5]),
#            'step10' : set_step(time = 2000e-6, qubits = qubits, voltages = [0.4, 0.5]),
            }

    manip_cfg = {
            'step1' : set_manip(time = 10e-6, qubits = qubits, voltages = [loop1_para2,0.6], waiting_time = loop1_para1, parameter1 = 0)
            }

    read_cfg = {
            'step1' : set_step(time = 1e-6, qubits = qubits, voltages = [0.3, 0.2]),
            'step2' : set_step(time = 1e-6, qubits = qubits, voltages = [0.4, 0.2]),
            'step3' : set_step(time = 1e-6, qubits = qubits, voltages = [0.5, 0.2]),
            }

#    experiment.sequence_cfg = [init_cfg, manip_cfg, read_cfg]
#    experiment.sequence_cfg_type = ['init', 'manip','read',]

    experiment.sequence_cfg = [init_cfg, manip_cfg,]
    experiment.sequence_cfg_type = ['init','manip',]

    experiment.manip_elem = Ramsey(name = 'Ramsey', pulsar = pulsar)
    
    experiment.manip_elem.pulsar = None

    experiment.set_sweep()
    
#    experiment.manipulation_element(name = 'manip', manip_element = Ramsey(name = 'Ramsey', qubits = qubits, waiting_time = 1e-6))

    return experiment


#%% sweep  outside a  sequence

def function(x):
    return True

Count = StandardParameter(name = 'Count', set_cmd = function)
Sweep_Count = Count[1:5:1]
#Sweep_VSGFreq = vsg.frequency[5e9:15e9:3e9]
#LP = Loop(sweep_values = Sweep_Count).each(dig)
#LP2 = Loop(sweep_values = Sweep_VSGFreq).loop(sweep_values = Sweep_Count).each(dig)

formatter = HDF5FormatMetadata()
data_IO = DiskIO(base_location = 'C:\\Users\\LocalAdmin\\Documents')
data_location = '2017-08-18/20-40-19_T1_Vread_sweep'

#data_set = LP.get_data_set(location=None, loc_record = {'name':'T1', 'label':'Vread_sweep'}, io = data_IO,)

#print('loop.data_set: %s' % LP.data_set)

#data_set = LP.run()

#dig = digitizer_param(name='digitizer', mV_range = mV_range, memsize=memsize, seg_size=seg_size, posttrigger_size=posttrigger_size)
def scan_outside_awg(name, set_parameter, measured_parameter, start, end, step):
    
    Sweep_Value = parameter[start:end:step]
    
    LOOP = Loop(sweep_values = Sweep_Value).each(measured_parameter)
    
    data_set = LOOP.get_data_set(location = None, loc_record = {'name': name, 'label': 'V_sweep'}, io = data_IO,)
    
    data_set = Loop.run()
    
    return data_set


#%% set VSG

def set_vector_signal_generator(VSG):
    
    VSG.frequency.set(0)
    VSG.phase.set(0)
    VSG.power.set(0)
    VSG.frequency.set(0)
    
    return VSG
#%% set digitizer

def set_digitizer(digitizer):
    
    pretrigger=16
    mV_range=1000
    rate = int(np.floor(250000000/1))
    #seg_size = int(np.floor((rate * (10e-6))/16)*16 + pretrigger )
    seg_size = 1040
    memsize = 5*seg_size
    posttrigger_size = 1024
    
    #digitizer.enable_channels(pyspcm.CHANNEL0 | pyspcm.CHANNEL3)
    digitizer.clock_mode(pyspcm.SPC_CM_INTPLL)
    #digitizer.clock_mode(pyspcm.SPC_CM_EXTREFCLOCK)
    
    digitizer.enable_channels(pyspcm.CHANNEL1 | pyspcm.CHANNEL2)
    
    digitizer.data_memory_size(memsize)
    
    digitizer.segment_size(seg_size)
    
    digitizer.posttrigger_memory_size(posttrigger_size)
    
    digitizer.timeout(60000)
    
    digitizer.sample_rate(250000000)
    
    digitizer.set_channel_settings(1,1000, input_path = 0, termination = 0, coupling = 0, compensation = None)
    
    #trig_mode = pyspcm.SPC_TMASK_SOFTWARE
    #trig_mode = pyspcm.SPC_TM_POS
    trig_mode = pyspcm.SPC_TM_POS | pyspcm.SPC_TM_REARM
    
    digitizer.set_ext0_OR_trigger_settings(trig_mode = trig_mode, termination = 0, coupling = 0, level0 = 800, level1 = 900)
    
    dig = digitizer_param(name='digitizer', mV_range = mV_range, memsize=memsize, seg_size=seg_size, posttrigger_size=posttrigger_size )

    return digitizer, dig

#%% make pulsar


def set_5014pulsar(awg, awg2):
    
    awg = awg.name
    awg2 = awg2.name
    pulsar = Pulsar(name = 'PuLsAr', default_AWG = awg, master_AWG = awg)

    marker1highs = [2, 2, 2.7, 2, 2, 2, 2.7, 2]
    for i in range(8):
        pulsar.define_channel(id='ch{}'.format(i%4 + 1),
                              name='ch{}'.format(i + 1), type='analog',
                              high=1, low=-1,
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


#%% test
experiment = make_experiment_cfg()

experiment.generate_1D_sequence()

#experiment.load_sequence()

experiment.run_experiment()
