# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:02:29 2017

@author: X.X
"""
import numpy as np
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.element import Element
from experiment import Experiment


#%% make experiment

def set_step(time = 0, qubits = [], voltages = []):

    if len(qubits)!=len(voltages):
        raise ValueError('qubits should be same length with voltages')

    step = {'time' : time}

    for i in range(len(qubits)):
        qubit = qubits[i]
        step[qubit] = voltages[i]

    return step

#%%

def make_experiment_cfg(station):

    experiment = Experiment()

    qubits = experiment.qubits_name

#    experiment.Sweep_2D(parameter1 = 'time_step2', start1 = 0, stop1 = 10, points1 = 11,
#                        parameter2 = 'voltage_q1_step1', start2 = 10, stop2=30, points2 =21)

    experiment.sweep_set1 = np.array([1,2,3])
    experiment.sweep_set2 = np.array([2,45,55])

    init_cfg = {
            'step1' : set_step(time = 10e-6, qubits = qubits, voltages = [0.3, 0.2]),
            'step2' : set_step(time = experiment.sweep_point1, qubits = qubits, voltages = [0.3, 0.2]),
            'setp3' : set_step(time = 10e-6, qubits = qubits, voltages = [0.3, experiment.sweep_point2]),
            }

    manip_cfg = {
            'step1' : set_step(time = 10e-6, qubits = qubits, operations = ['X', 'Y']),
            'step2' : set_step(time = 10e-6, qubits = qubits, operations = ['X', 'Y']),
            'step3' : [],
            }

    read_cfg = {
            'step1' : set_step(time = 10e-6, qubits = qubits, voltages = [0.3, 0.2]),
            'step2' : set_step(time = 10e-6, qubits = qubits, voltages = [0.3, 0.2]),
            'step3' : set_step(time = 10e-6, qubits = qubits, voltages = [0.3, 0.2]),
            }

    experiment.sequence_cfg = [init_cfg, manip_cfg, read_cfg]


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
