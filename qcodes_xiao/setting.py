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

#%% make experiment

def set_step(time = 0, qubits = [], voltages = [], **kw):

    if len(qubits)!=len(voltages):
        raise ValueError('qubits should be same length with voltages')

    microwave_frequencies = kw.pop('microwave_ferquencies', [qubit.frequency for qubit in qubits])
    microwave_powers = kw.pop('amplitudes', [qubit.microwave_power for qubit in qubits])
    Pi_pulse_lengths = kw.pop('Pi_pulse_lengths', [qubit.Pi_pulse_lenght for qubit in qubits])
    IQ_amplitudes = kw.pop('IQ_amplitudes', [qubit.IQ_amplitude for qubit in qubits])
    IQ_frequencies = kw.pop('IQ_frequencies', [qubit.IQ_frequency for qubit in qubits])


    step = {'time' : time}

    for i in range(len(qubits)):
        qubit = qubits[i]
        step['voltage_%d'%(i+1)] = voltages[i]
        step['microwave_frequency_%d'%(i+1)] = microwave_frequencies[i]
        step['microwave_power_%d'%(i+1)] = microwave_powers[i]
        step['Pi_pulse_length_%d'%(i+1)] = Pi_pulse_lengths[i]
        step['IQ_amplitude_%d'%(i+1)] = IQ_amplitudes[i]
        step['IQ_frequency_%d'%(i+1)] = IQ_frequencies[i]

    return step
#%%  Sweep

def sweep_array(start, stop, points):

    sweep_array = np.linspace(start, stop, points)

    return list(sweep_array)


#%% make manipulation

def make_manipulation(manipulation = Manipulation(name = 'Manip'),):

    manipulation.add_X(name='X1_Q1',)
    manipulation.add_Y(name='Y1_Q1')
    manipulation.add_CPhase(name='CP1')
    manipulation.addX(name='X2_Q1')

    return manipulation


#%%

def make_experiment_cfg(station):

    experiment = Experiment(name = 'experiment_test', qubits = [Qubit_1, Qubit_2],)

    qubits = experiment.qubits_name

#    experiment.Sweep_2D(parameter1 = 'time_step2', start1 = 0, stop1 = 10, points1 = 11,
#                        parameter2 = 'voltage_q1_step1', start2 = 10, stop2=30, points2 =21)

    experiment.sweep_loop1 = {
            'para1': [1,2,3,4,5],
            'para2': sweep_array(start = 1, stop = 5, points = 5),
            }

    experiment.sweep_loop2 = {
            'para1': [2,45,55],
            }

#    loop1_para1 = [1,2,344,553,3]
#    loop1_para2 = [33,2,11,22,3]
#    loop2_para1 = [1,2,3,4]

#    experiment.construct_sweep_matrix(loop1,loop2)

    init_cfg = {
            'step1' : set_step(time = 10e-6, qubits = qubits, voltages = [0.3, 'loop1_para1']),
            'step2' : set_step(time = experiment.sweep_point1, qubits = qubits, voltages = [0.3, 'loop1_para2']),
            'setp3' : set_step(time = 10e-6, qubits = qubits, voltages = [0.3, experiment.sweep_point2]),
            }

    manip_cfg = {
            'step1' : set_step(time = 10e-6, qubits = qubits, microwave_frequencies = [ ], IQ_amplitudes = [],
                               microwave_powers = [], IQ_frequencies = [], Pi_pulse_lengths = []),
#            'step2' : set_step(time = 10e-6, qubits = qubits, operations = ['X', 'Y']),
#            'step3' : [],
            }

    read_cfg = {
            'step1' : set_step(time = 10e-6, qubits = qubits, voltages = [0.3, 0.2]),
            'step2' : set_step(time = 10e-6, qubits = qubits, voltages = [0.3, 0.2]),
            'step3' : set_step(time = 10e-6, qubits = qubits, voltages = [0.3, 0.2]),
            }

    experiment.sequence_cfg = [init_cfg, manip_cfg, read_cfg]
    experiment.sequence_cfg_type = ['init', 'manip','read',]


#    experiment.Sweep_2D(parameter1 = T, name1 = 'time_step2', )
#    experiment.Sweep_2D(parameter1 = 'time_step2', start1 = 0, stop1 = 10, points1 = 11,
#                        parameter2 = 'voltage_q1_step1', start2 = 10, stop2=30, points2 =21)
#
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
                              high=.7, low=-.7,
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



#%%
