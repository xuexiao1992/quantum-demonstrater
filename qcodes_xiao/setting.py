# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:02:29 2017

@author: X.X
"""
import numpy as np
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
from experiment import Experiment
from manipulation import Manipulation
import stationF006
from manipulation_library import Ramsey
#from stationF006 import station

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

    return step



#%%  Sweep

def sweep_array(start, stop, points):

    sweep_array = np.linspace(start, stop, points)

    return list(sweep_array)


#%% make manipulation
def make_manipulation(manipulation = Manipulation(name = 'Manip'), qubits = [], **kw):

    waiting_time = kw.pop('waiting_time', None)
    amplitude = kw.pop('amplitude', None)

    manip = make_Ramsey(manipulation = manipulation, qubits = qubits, waiting_time = waiting_time)

    return manip



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
    awg.clock_freq(1e9)

    digitizer = station.digitizer
#    awg.ch3_amp
    pulsar = set_5014pulsar(awg = awg)

    qubit_1 = station.qubit_1
    qubit_2 = station.qubit_2

    qubits = [qubit_1, qubit_2]


    experiment = Experiment(name = 'experiment_test', qubits = [qubit_1, qubit_2], awg = awg, pulsar = pulsar)
    

    experiment.sweep_loop1 = {
#            'para1': [0.8,0.2,0.53,0.14,0.3],
            'para1': sweep_array(start = 50e-9, stop = 250e-9, points = 5),
            'para2': sweep_array(start = 0.1, stop = 0.5, points = 5),
            }

    experiment.sweep_loop2 = {
#            'para1': [0,0.5,0.5,0,0.5],
            }

#    loop1_para1 = [1,2,344,553,3]
#    loop1_para2 = [33,2,11,22,3]
#    loop2_para1 = [1,2,3,4]

    loop1_para1 = 'loop1_para1'
    loop1_para2 = 'loop1_para2'
    loop2_para1 = 'loop2_para1'
    
    init_cfg = {
            'step1' : set_step(time = 2e-6, qubits = qubits, voltages = [0.2, 0.3]),
            'step2' : set_step(time = 5e-6, qubits = qubits, voltages = [0.1, 0.3]),
            'step3' : set_step(time = 1e-6, qubits = qubits, voltages = [0.4, 0.5]),
            }

    manip_cfg = {
            'step1' : set_manip(time = 2e-6, qubits = qubits, voltages = [loop1_para2,0.6], waiting_time = loop1_para1,)
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

    experiment.set_sweep()
    
    experiment.manip_elem = Ramsey(name = 'Ramsey',)
    
#    experiment.manipulation_element(name = 'manip', manip_element = Ramsey(name = 'Ramsey', qubits = qubits, waiting_time = 1e-6))


    return experiment



#%% make pulsar


def set_5014pulsar(awg):
    pulsar = Pulsar()
    pulsar.AWG = awg

    marker1highs = [2, 2, 2.7, 2]
#    marker2highs = [2, 2, 2.7, 2]
    for i in range(4):
        # Note that these are default parameters and should be kept so.
        # the channel offset is set in the AWG itself. For now the amplitude is
        # hardcoded. You can set it by hand but this will make the value in the
        # sequencer different.
        pulsar.define_channel(id='ch{}'.format(i + 1),
                              name='ch{}'.format(i + 1), type='analog',
                              # max safe IQ voltage
                              high=1, low=-1,
                              offset=0.0, delay=0, active=True)
        pulsar.define_channel(id='ch{}_marker1'.format(i + 1),
                              name='ch{}_marker1'.format(i + 1),
                              type='marker',
                              high=marker1highs[i], low=0, offset=0.,
                              delay=0, active=True)
        pulsar.define_channel(id='ch{}_marker2'.format(i + 1),
                              name='ch{}_marker2'.format(i + 1),
                              type='marker',
                              high=2, low=0, offset=0.,
                              delay=0, active=True)

        pulsar.AWG_sequence_cfg = {
            'SAMPLING_RATE': 1e9,
            'CLOCK_SOURCE': 1,  # Internal | External
            'REFERENCE_SOURCE': 1,  # Internal | External
            'EXTERNAL_REFERENCE_TYPE': 1,  # Fixed | Variable
            'REFERENCE_CLOCK_FREQUENCY_SELECTION': 1,  # 10 MHz | 20 MHz | 100 MHz
            'TRIGGER_SOURCE': 1,  # External | Internal
            'TRIGGER_INPUT_IMPEDANCE': 1,  # 50 ohm | 1 kohm
            'TRIGGER_INPUT_SLOPE': 1,  # Positive | Negative
            'TRIGGER_INPUT_POLARITY': 1,  # Positive | Negative
            'TRIGGER_INPUT_THRESHOLD': 0.6,  # V
            'EVENT_INPUT_IMPEDANCE': 2,  # 50 ohm | 1 kohm
            'EVENT_INPUT_POLARITY': 1,  # Positive | Negative
            'EVENT_INPUT_THRESHOLD': 1.4,  # V
            'JUMP_TIMING': 1,  # Sync | Async
            'RUN_MODE': 4,  # Continuous | Triggered | Gated | Sequence
            'RUN_STATE': 0,  # On | Off
        }

    return pulsar



#%% test

experiment = make_experiment_cfg()

experiment.generate_sequence()

experiment.load_sequence()

experiment.run_experiment()
